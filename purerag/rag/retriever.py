"""
Stages 4, 5 & 6: Hybrid Dense + BM25 Retriever with Reciprocal Rank Fusion (RRF)
and Second-Stage Cross-Encoder Reranking.

Supports 3 embedding tiers with automatic graceful fallback:
1. `fastembed` (ONNX local BAAI/bge-* embeddings — SOTA speed & MTEB accuracy)
2. `ollama` (Local Ollama `/api/embeddings` e.g. `nomic-embed-text`)
3. `hashed_ngram` (Zero-download deterministic character/word n-gram dense vectors
   so Pure RAG works out-of-the-box offline even before any model download is approved).

Supports Cross-Encoder Reranking via `flashrank` (ONNX `ms-marco-MiniLM-L-12-v2`)
plus a lexical-proximity cross-feature scorer fallback.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from rank_bm25 import BM25Okapi

from purerag.rag.chunker import DocumentChunk


TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_/-]{2,}\b")

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "have", "has", "had", "not", "but", "what", "which", "who", "whom", "whose",
    "where", "when", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "than", "too", "very", "can", "will", "just",
    "should", "now", "into", "about", "over", "after", "before", "between", "under",
}


def tokenize_for_bm25(text: str) -> List[str]:
    tokens = [t.lower() for t in TOKEN_PATTERN.findall(text)]
    return [t for t in tokens if t not in STOPWORDS]


@dataclass
class RetrievedCandidate:
    chunk: DocumentChunk
    final_score: float
    rrf_score: float
    dense_score: float
    bm25_score: float
    rerank_score: Optional[float]
    dense_rank: Optional[int]
    bm25_rank: Optional[int]

    def to_citation_dict(self, rank: int) -> Dict:
        return {
            "rank": rank,
            "chunk_id": self.chunk.chunk_id,
            "filename": self.chunk.filename,
            "page_number": self.chunk.page_number,
            "section_path": self.chunk.section_path,
            "snippet": self.chunk.raw_text[:480] + ("..." if len(self.chunk.raw_text) > 480 else ""),
            "full_text": self.chunk.raw_text,
            "final_score": round(float(self.final_score), 4),
            "rrf_score": round(float(self.rrf_score), 4),
            "dense_score": round(float(self.dense_score), 4),
            "bm25_score": round(float(self.bm25_score), 4),
            "rerank_score": round(float(self.rerank_score), 4) if self.rerank_score is not None else None,
        }


class LocalEmbedder:
    """
    Unified local embedding provider supporting FastEmbed (ONNX BGE),
    Ollama Embeddings, and a deterministic Hashed N-Gram vector fallback.
    """

    def __init__(
        self,
        backend: str = "fastembed",
        model_name: str = "BAAI/bge-small-en-v1.5",
        ollama_base_url: str = "http://127.0.0.1:11434",
        allow_download: bool = False,
    ) -> None:
        self.requested_backend = backend
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.allow_download = allow_download
        self._fastembed_model = None
        self._st_model = None
        self.active_backend = "hashed_ngram"
        self._init_backend()

    def _init_backend(self) -> None:
        # 1. Check if the requested model is already cached locally in HuggingFace Hub (~/.cache/huggingface/hub)
        #    so we can load it with local_files_only=True WITHOUT any network download!
        hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{self.model_name.replace('/', '--')}"
        is_locally_cached_hf = hf_cache_dir.exists()

        if is_locally_cached_hf and self.requested_backend != "hashed_ngram":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._st_model = SentenceTransformer(self.model_name, local_files_only=True)
                self.active_backend = f"local_hf:{self.model_name}"
                return
            except Exception:
                pass
            try:
                from fastembed import TextEmbedding  # type: ignore

                self._fastembed_model = TextEmbedding(model_name=self.model_name, local_files_only=True)
                self.active_backend = f"fastembed_local:{self.model_name}"
                return
            except Exception:
                pass

        # 1b. If the configured model wasn't cached, check if ANY embedding model is already cached in ~/.cache/huggingface/hub
        if self.requested_backend != "hashed_ngram":
            try:
                from purerag.hardware import _scan_huggingface_and_local_caches

                cached_embeds, _, _, _ = _scan_huggingface_and_local_caches()
                for alt_embed in cached_embeds:
                    try:
                        from sentence_transformers import SentenceTransformer  # type: ignore

                        self._st_model = SentenceTransformer(alt_embed, local_files_only=True)
                        self.model_name = alt_embed
                        self.active_backend = f"local_hf:{alt_embed}"
                        return
                    except Exception:
                        pass
            except Exception:
                pass

        # 2. If user approved downloads, allow FastEmbed or SentenceTransformers
        if self.requested_backend in ("fastembed", "local_hf_or_fastembed") and self.allow_download:
            try:
                from fastembed import TextEmbedding  # type: ignore

                self._fastembed_model = TextEmbedding(model_name=self.model_name)
                self.active_backend = f"fastembed:{self.model_name}"
                return
            except Exception:
                pass
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._st_model = SentenceTransformer(self.model_name)
                self.active_backend = f"sentence_transformers:{self.model_name}"
                return
            except Exception:
                pass

        # 3. Check local Ollama embeddings (never downloads automatically, uses installed model)
        if self.requested_backend == "ollama":
            try:
                resp = requests.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": "warmup"},
                    timeout=3.0,
                )
                if resp.status_code == 200 and resp.json().get("embedding"):
                    self.active_backend = f"ollama:{self.model_name}"
                    return
            except Exception:
                pass

        # 4. Deterministic 512-dim signed feature-hashing embedder over word + char n-grams
        self.active_backend = "hashed_ngram_512d"

    @staticmethod
    def _hashed_ngram_embed(texts: List[str], dim: int = 512) -> np.ndarray:
        mat = np.zeros((len(texts), dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            words = tokenize_for_bm25(text)
            features: List[Tuple[str, float]] = []
            # Unigrams (weight 1.5)
            for w in words:
                features.append((f"w:{w}", 1.5))
                # Character 4-grams for morphological/stemming resilience (weight 0.6)
                if len(w) >= 4:
                    for i in range(len(w) - 3):
                        features.append((f"c:{w[i:i+4]}", 0.6))
            # Bigrams (weight 2.0)
            for i in range(len(words) - 1):
                features.append((f"b:{words[i]}_{words[i+1]}", 2.0))

            vec = np.zeros(dim, dtype=np.float32)
            for feat, weight in features:
                h = int(hashlib.md5(feat.encode("utf-8")).hexdigest()[:8], 16)
                idx = h % dim
                sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
                vec[idx] += sign * weight

            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec /= norm
            mat[row_idx] = vec
        return mat

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 512), dtype=np.float32)

        if self._st_model is not None:
            try:
                vecs = self._st_model.encode(
                    texts,
                    batch_size=64,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return np.asarray(vecs, dtype=np.float32)
            except Exception:
                pass

        if self._fastembed_model is not None:
            try:
                vecs = list(self._fastembed_model.embed(texts))
                arr = np.asarray(vecs, dtype=np.float32)
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms < 1e-8] = 1.0
                return arr / norms
            except Exception:
                pass

        if self.active_backend.startswith("ollama:"):
            try:
                vecs = []
                for t in texts:
                    r = requests.post(
                        f"{self.ollama_base_url}/api/embeddings",
                        json={"model": self.model_name, "prompt": t},
                        timeout=15.0,
                    )
                    r.raise_for_status()
                    vecs.append(r.json()["embedding"])
                arr = np.asarray(vecs, dtype=np.float32)
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms < 1e-8] = 1.0
                return arr / norms
            except Exception:
                pass

        return self._hashed_ngram_embed(texts)


class CrossEncoderReranker:
    """
    Second-stage precision reranker.
    Uses FlashRank ONNX Cross-Encoder (`ms-marco-MiniLM-L-12-v2`) when installed and approved,
    falling back to a high-precision Query-Coverage + Phrase-Proximity Cross-Scorer.
    """

    def __init__(
        self,
        enabled: bool = True,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        allow_download: bool = False,
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.allow_download = allow_download
        self._ranker = None
        self._st_cross_encoder = None
        self.active_reranker = "disabled"
        if self.enabled:
            self._init_ranker()

    def _init_ranker(self) -> None:
        # 1. Check if reranker is already cached locally in HuggingFace Hub (~/.cache/huggingface/hub)
        hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{self.model_name.replace('/', '--')}"
        if hf_cache_dir.exists():
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                self._st_cross_encoder = CrossEncoder(self.model_name, local_files_only=True)
                self.active_reranker = f"local_hf_cross_encoder:{self.model_name}"
                return
            except Exception:
                pass

        # 1b. If the configured reranker wasn't cached, check if ANY reranker is already cached in ~/.cache/huggingface/hub
        try:
            from purerag.hardware import _scan_huggingface_and_local_caches

            _, cached_rerankers, _, _ = _scan_huggingface_and_local_caches()
            for alt_reranker in cached_rerankers:
                try:
                    from sentence_transformers import CrossEncoder  # type: ignore

                    self._st_cross_encoder = CrossEncoder(alt_reranker, local_files_only=True)
                    self.model_name = alt_reranker
                    self.active_reranker = f"local_hf_cross_encoder:{alt_reranker}"
                    return
                except Exception:
                    pass
        except Exception:
            pass

        # 2. Check FlashRank local cache or download if approved
        fr_cached = (Path("/tmp") / self.model_name).exists() or (Path.home() / ".cache" / "flashrank" / self.model_name).exists()
        if fr_cached or self.allow_download:
            try:
                from flashrank import Ranker  # type: ignore

                self._ranker = Ranker(model_name=self.model_name)
                self.active_reranker = f"flashrank:{self.model_name}"
                return
            except Exception:
                pass

        if self.allow_download:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                self._st_cross_encoder = CrossEncoder(self.model_name)
                self.active_reranker = f"cross_encoder:{self.model_name}"
                return
            except Exception:
                pass

        self.active_reranker = "cross_proximity_scorer"

    @staticmethod
    def _proximity_cross_score(query: str, passage: str) -> float:
        q_tokens = tokenize_for_bm25(query)
        if not q_tokens:
            return 0.0
        p_lower = passage.lower()
        p_tokens = tokenize_for_bm25(passage)
        if not p_tokens:
            return 0.0

        p_set = set(p_tokens)
        unique_q = list(dict.fromkeys(q_tokens))
        matched_q = [t for t in unique_q if t in p_set]
        coverage = len(matched_q) / len(unique_q)

        # Exact phrase bonus
        q_clean = " ".join(q_tokens)
        exact_phrase_bonus = 0.25 if (len(q_tokens) >= 2 and q_clean in " ".join(p_tokens)) else 0.0

        # Window proximity bonus: minimum span containing matched query tokens
        positions = [i for i, tok in enumerate(p_tokens) if tok in set(matched_q)]
        proximity_bonus = 0.0
        if len(positions) >= 2:
            min_gap = min(positions[i + 1] - positions[i] for i in range(len(positions) - 1))
            proximity_bonus = 0.20 / max(1.0, float(min_gap))

        # Section header match bonus
        header_line = passage.splitlines()[0].lower() if passage else ""
        header_hits = sum(1 for t in unique_q if t in header_line)
        header_bonus = 0.15 * (header_hits / len(unique_q))

        return float(min(1.0, 0.55 * coverage + exact_phrase_bonus + proximity_bonus + header_bonus))

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedCandidate],
        top_k: int = 6,
    ) -> List[RetrievedCandidate]:
        if not candidates or not self.enabled:
            return candidates[:top_k]

        if self._st_cross_encoder is not None:
            try:
                pairs = [(query, c.chunk.contextualized_text) for c in candidates]
                scores = self._st_cross_encoder.predict(pairs, show_progress_bar=False)
                for cand, raw_s in zip(candidates, scores):
                    # Sigmoid normalize raw logits to [0, 1]
                    norm_s = float(1.0 / (1.0 + math.exp(-float(raw_s))))
                    cand.rerank_score = norm_s
                    cand.final_score = 0.70 * norm_s + 0.30 * min(1.0, cand.rrf_score * 30.0)
                candidates.sort(key=lambda x: x.final_score, reverse=True)
                return candidates[:top_k]
            except Exception:
                pass

        if self._ranker is not None:
            try:
                from flashrank import RerankRequest  # type: ignore

                passages = [
                    {"id": idx, "text": c.chunk.contextualized_text}
                    for idx, c in enumerate(candidates)
                ]
                req = RerankRequest(query=query, passages=passages)
                results = self._ranker.rerank(req)
                reordered: List[RetrievedCandidate] = []
                for item in results:
                    orig_idx = int(item["id"])
                    cand = candidates[orig_idx]
                    rerank_s = float(item.get("score", 0.0))
                    cand.rerank_score = rerank_s
                    # Blend normalized RRF with cross-encoder score
                    cand.final_score = 0.70 * rerank_s + 0.30 * (cand.rrf_score * 30.0)
                    reordered.append(cand)
                reordered.sort(key=lambda x: x.final_score, reverse=True)
                return reordered[:top_k]
            except Exception:
                pass

        # Fallback: Cross-proximity + RRF blend
        for cand in candidates:
            cross_s = self._proximity_cross_score(query, cand.chunk.contextualized_text)
            cand.rerank_score = cross_s
            cand.final_score = 0.55 * cross_s + 0.45 * min(1.0, cand.rrf_score * 35.0)

        candidates.sort(key=lambda x: x.final_score, reverse=True)
        return candidates[:top_k]


class HybridRetriever:
    """
    Maintains both the Dense Vector Matrix and Sparse BM25 Okapi Index over
    contextualized chunks, performs Parallel Hybrid Retrieval, fuses rankings via
    Reciprocal Rank Fusion (RRF), and runs Cross-Encoder Reranking.
    """

    def __init__(
        self,
        embedder: LocalEmbedder,
        reranker: CrossEncoderReranker,
        rrf_k: int = 60,
        dense_weight: float = 0.55,
        bm25_weight: float = 0.45,
    ) -> None:
        self.embedder = embedder
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []

    def build_index(
        self,
        chunks: List[DocumentChunk],
        precomputed_embeddings: Optional[np.ndarray] = None,
    ) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            self.embeddings = np.zeros((0, 512), dtype=np.float32)
            self.bm25 = None
            self.tokenized_corpus = []
            return

        texts = [c.contextualized_text for c in self.chunks]
        if precomputed_embeddings is not None and precomputed_embeddings.shape[0] == len(self.chunks):
            self.embeddings = precomputed_embeddings
        else:
            self.embeddings = self.embedder.embed_texts(texts)

        self.tokenized_corpus = [tokenize_for_bm25(t) for t in texts]
        # Guard against all-empty token lists
        if any(len(toks) > 0 for toks in self.tokenized_corpus):
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def retrieve(
        self,
        query: str,
        dense_top_k: int = 25,
        bm25_top_k: int = 25,
        final_top_k: int = 6,
        filter_filenames: Optional[List[str]] = None,
    ) -> List[RetrievedCandidate]:
        if not self.chunks or self.embeddings is None or len(self.chunks) == 0:
            return []

        allowed_indices = set(range(len(self.chunks)))
        if filter_filenames:
            allowed_set = set(filter_filenames)
            allowed_indices = {
                i for i, c in enumerate(self.chunks) if c.filename in allowed_set
            }
            if not allowed_indices:
                return []

        # 1. Dense Semantic Search (Cosine Similarity)
        q_vec = self.embedder.embed_texts([query])[0]
        if self.embeddings.shape[1] != q_vec.shape[0]:
            # Re-embed corpus if backend changed dimensions
            self.embeddings = self.embedder.embed_texts([c.contextualized_text for c in self.chunks])

        dense_scores_all = np.dot(self.embeddings, q_vec)
        dense_ranked_indices = [
            int(i)
            for i in np.argsort(-dense_scores_all)
            if int(i) in allowed_indices
        ][:dense_top_k]

        dense_rank_map: Dict[int, int] = {
            doc_idx: rank_pos + 1 for rank_pos, doc_idx in enumerate(dense_ranked_indices)
        }

        # 2. Sparse Lexical Search (BM25 Okapi)
        bm25_scores_all = np.zeros(len(self.chunks), dtype=np.float32)
        bm25_rank_map: Dict[int, int] = {}
        if self.bm25 is not None:
            q_tokens = tokenize_for_bm25(query)
            if q_tokens:
                raw_bm25 = self.bm25.get_scores(q_tokens)
                bm25_scores_all = np.asarray(raw_bm25, dtype=np.float32)
                bm25_ranked_indices = [
                    int(i)
                    for i in np.argsort(-bm25_scores_all)
                    if int(i) in allowed_indices and bm25_scores_all[int(i)] > 0
                ][:bm25_top_k]
                bm25_rank_map = {
                    doc_idx: rank_pos + 1 for rank_pos, doc_idx in enumerate(bm25_ranked_indices)
                }

        # 3. Reciprocal Rank Fusion (RRF)
        candidate_indices = set(dense_rank_map.keys()) | set(bm25_rank_map.keys())
        if not candidate_indices:
            candidate_indices = set(dense_ranked_indices[:final_top_k])

        fused_candidates: List[RetrievedCandidate] = []
        for idx in candidate_indices:
            d_rank = dense_rank_map.get(idx)
            b_rank = bm25_rank_map.get(idx)

            rrf_s = 0.0
            if d_rank is not None:
                rrf_s += self.dense_weight / (self.rrf_k + d_rank)
            if b_rank is not None:
                rrf_s += self.bm25_weight / (self.rrf_k + b_rank)

            fused_candidates.append(
                RetrievedCandidate(
                    chunk=self.chunks[idx],
                    final_score=rrf_s,
                    rrf_score=rrf_s,
                    dense_score=float(dense_scores_all[idx]),
                    bm25_score=float(bm25_scores_all[idx]),
                    rerank_score=None,
                    dense_rank=d_rank,
                    bm25_rank=b_rank,
                )
            )

        fused_candidates.sort(key=lambda x: x.rrf_score, reverse=True)
        pool_for_rerank = fused_candidates[: max(final_top_k * 3, 18)]

        # 4. Cross-Encoder Second-Stage Reranking
        reranked = self.reranker.rerank(query=query, candidates=pool_for_rerank, top_k=final_top_k)
        return reranked
