"""
Pure RAG Orchestration Engine.
Watches the designated `pdfs/` folder, incrementally parses & indexes new or changed
documents, persists chunks + embeddings to `data/`, and serves hybrid + reranked queries.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional

import numpy as np

from purerag.config import PureRAGConfig
from purerag.rag.chunker import DocumentChunk, chunk_document
from purerag.rag.generator import GroundedGenerator, PromptOptions
from purerag.rag.parser import SUPPORTED_EXTENSIONS, compute_file_hash, parse_document
from purerag.rag.retriever import CrossEncoderReranker, HybridRetriever, LocalEmbedder, RetrievedCandidate


@dataclass
class IndexedDocumentMeta:
    filename: str
    filepath: str
    file_hash: str
    file_size_bytes: int
    total_pages: int
    chunk_count: int
    parser_used: str
    indexed_at: float


class PureRAGEngine:
    def __init__(self, config: PureRAGConfig) -> None:
        self.config = config
        self.config.ensure_dirs()
        self._lock = threading.RLock()

        self.documents_meta: Dict[str, IndexedDocumentMeta] = {}
        self.chunks: List[DocumentChunk] = []

        # Auto-adopt already-present local models on first run before initializing components
        if not self.config.hardware_profiled and self.config.embedding_backend != "hashed_ngram":
            try:
                from purerag.hardware import analyze_hardware

                hw = analyze_hardware(
                    workspace_dir=Path(self.config.pdfs_dir).parent,
                    ollama_base_url=self.config.ollama_base_url,
                )
                disc = hw.discovered_local_models or {}
                if disc.get("best_local_llm"):
                    self.config.llm_model = disc["best_local_llm"]
                if disc.get("best_local_embedder"):
                    self.config.embedding_model = disc["best_local_embedder"]
                    if disc.get("best_local_embedder_backend"):
                        self.config.embedding_backend = disc["best_local_embedder_backend"]
                if disc.get("best_local_reranker"):
                    self.config.reranker_model = disc["best_local_reranker"]
            except Exception:
                pass

        self.embedder = LocalEmbedder(
            backend=self.config.embedding_backend,
            model_name=self.config.embedding_model,
            ollama_base_url=self.config.ollama_base_url,
            allow_download=self.config.setup_approved,
        )
        self.reranker = CrossEncoderReranker(
            enabled=self.config.use_reranker,
            model_name=self.config.reranker_model,
            allow_download=self.config.setup_approved,
        )
        self.retriever = HybridRetriever(
            embedder=self.embedder,
            reranker=self.reranker,
            rrf_k=self.config.rrf_k,
            dense_weight=self.config.dense_weight,
            bm25_weight=self.config.bm25_weight,
        )
        self.generator = GroundedGenerator(
            provider=self.config.llm_provider,
            ollama_base_url=self.config.ollama_base_url,
            llm_model=self.config.llm_model,
            openai_api_base=self.config.openai_api_base,
            openai_api_key=self.config.openai_api_key,
        )

        self._load_persisted_index()

    @property
    def index_meta_path(self) -> Path:
        return Path(self.config.data_dir) / "index_meta.json"

    @property
    def chunks_path(self) -> Path:
        return Path(self.config.data_dir) / "chunks.json"

    @property
    def embeddings_path(self) -> Path:
        return Path(self.config.data_dir) / "embeddings.npy"

    def reload_components_from_config(self) -> None:
        with self._lock:
            self.embedder = LocalEmbedder(
                backend=self.config.embedding_backend,
                model_name=self.config.embedding_model,
                ollama_base_url=self.config.ollama_base_url,
                allow_download=self.config.setup_approved,
            )
            self.reranker = CrossEncoderReranker(
                enabled=self.config.use_reranker,
                model_name=self.config.reranker_model,
                allow_download=self.config.setup_approved,
            )
            self.retriever = HybridRetriever(
                embedder=self.embedder,
                reranker=self.reranker,
                rrf_k=self.config.rrf_k,
                dense_weight=self.config.dense_weight,
                bm25_weight=self.config.bm25_weight,
            )
            self.generator = GroundedGenerator(
                provider=self.config.llm_provider,
                ollama_base_url=self.config.ollama_base_url,
                llm_model=self.config.llm_model,
                openai_api_base=self.config.openai_api_base,
                openai_api_key=self.config.openai_api_key,
            )
            if self.chunks:
                self.retriever.build_index(self.chunks)
                self._persist_index()

    def _load_persisted_index(self) -> None:
        with self._lock:
            if not self.index_meta_path.exists() or not self.chunks_path.exists():
                return
            try:
                with open(self.index_meta_path, "r", encoding="utf-8") as f:
                    raw_meta = json.load(f)
                self.documents_meta = {
                    k: IndexedDocumentMeta(**v) for k, v in raw_meta.get("documents", {}).items()
                }
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    raw_chunks = json.load(f)
                self.chunks = [DocumentChunk.from_dict(c) for c in raw_chunks]

                pre_emb = None
                if self.embeddings_path.exists():
                    loaded = np.load(str(self.embeddings_path))
                    if loaded.shape[0] == len(self.chunks):
                        pre_emb = loaded

                self.retriever.build_index(self.chunks, precomputed_embeddings=pre_emb)
            except Exception:
                self.documents_meta = {}
                self.chunks = []

    def _persist_index(self) -> None:
        with self._lock:
            self.config.ensure_dirs()
            meta_payload = {
                "updated_at": time.time(),
                "embedding_backend": self.embedder.active_backend,
                "documents": {k: asdict(v) for k, v in self.documents_meta.items()},
            }
            with open(self.index_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_payload, f, indent=2)

            with open(self.chunks_path, "w", encoding="utf-8") as f:
                json.dump([c.to_dict() for c in self.chunks], f, indent=2)

            if self.retriever.embeddings is not None and len(self.chunks) > 0:
                np.save(str(self.embeddings_path), self.retriever.embeddings)

    def scan_and_ingest(self, force_reindex: bool = False) -> Dict:
        """
        Scans `config.pdfs_dir` for `.pdf`, `.md`, `.txt` files.
        Incrementally parses only new or modified files, removes deleted files,
        and updates the hybrid dense + BM25 index.
        """
        with self._lock:
            pdfs_folder = Path(self.config.pdfs_dir)
            pdfs_folder.mkdir(parents=True, exist_ok=True)

            discovered_files: Dict[str, Path] = {}
            for p in sorted(pdfs_folder.iterdir()):
                if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    discovered_files[p.name] = p

            added: List[str] = []
            updated: List[str] = []
            removed: List[str] = []
            unchanged: List[str] = []
            errors: Dict[str, str] = {}

            # Detect removed files
            for existing_name in list(self.documents_meta.keys()):
                if existing_name not in discovered_files:
                    removed.append(existing_name)
                    del self.documents_meta[existing_name]

            if force_reindex:
                self.documents_meta.clear()
                self.chunks.clear()

            # Keep chunks belonging to unchanged documents
            retained_chunks_by_file: Dict[str, List[DocumentChunk]] = {}
            for c in self.chunks:
                if c.filename in self.documents_meta:
                    retained_chunks_by_file.setdefault(c.filename, []).append(c)

            new_all_chunks: List[DocumentChunk] = []

            for fname, fpath in discovered_files.items():
                try:
                    current_hash = compute_file_hash(fpath)
                    existing = self.documents_meta.get(fname)
                    if (
                        not force_reindex
                        and existing is not None
                        and existing.file_hash == current_hash
                        and fname in retained_chunks_by_file
                    ):
                        unchanged.append(fname)
                        new_all_chunks.extend(retained_chunks_by_file[fname])
                        continue

                    parsed_doc = parse_document(fpath)
                    doc_chunks = chunk_document(
                        parsed_doc,
                        chunk_size_chars=self.config.chunk_size_chars,
                        chunk_overlap_chars=self.config.chunk_overlap_chars,
                    )
                    new_all_chunks.extend(doc_chunks)

                    if existing is None:
                        added.append(fname)
                    else:
                        updated.append(fname)

                    self.documents_meta[fname] = IndexedDocumentMeta(
                        filename=fname,
                        filepath=str(fpath.resolve()),
                        file_hash=current_hash,
                        file_size_bytes=parsed_doc.file_size_bytes,
                        total_pages=parsed_doc.total_pages,
                        chunk_count=len(doc_chunks),
                        parser_used=parsed_doc.parser_used,
                        indexed_at=time.time(),
                    )
                except Exception as exc:
                    errors[fname] = str(exc)

            changed = bool(added or updated or removed or force_reindex)
            if changed or (len(new_all_chunks) != len(self.chunks)):
                self.chunks = new_all_chunks
                self.retriever.build_index(self.chunks)
                self._persist_index()

            return {
                "added": added,
                "updated": updated,
                "removed": removed,
                "unchanged": unchanged,
                "errors": errors,
                "total_documents": len(self.documents_meta),
                "total_chunks": len(self.chunks),
                "active_embedder": self.embedder.active_backend,
                "active_reranker": self.reranker.active_reranker,
            }

    def query(
        self,
        question: str,
        length: Optional[str] = None,
        style: Optional[str] = None,
        temperature: Optional[float] = None,
        custom_instruction: Optional[str] = None,
        top_k: Optional[int] = None,
        filter_filenames: Optional[List[str]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        with self._lock:
            opts = PromptOptions(
                length=length or self.config.default_length,
                style=style or self.config.default_style,
                temperature=temperature if temperature is not None else self.config.default_temperature,
                custom_instruction=custom_instruction if custom_instruction is not None else self.config.custom_system_prompt,
                top_k=top_k or self.config.final_top_k,
            )

            t0 = time.perf_counter()
            candidates = self.retriever.retrieve(
                query=question,
                dense_top_k=self.config.dense_top_k,
                bm25_top_k=self.config.bm25_top_k,
                final_top_k=opts.top_k,
                filter_filenames=filter_filenames,
            )
            retrieval_ms = round((time.perf_counter() - t0) * 1000.0, 1)

            t1 = time.perf_counter()
            answer, model_used = self.generator.generate(
                query=question,
                candidates=candidates,
                options=opts,
                chat_history=chat_history,
            )
            generation_ms = round((time.perf_counter() - t1) * 1000.0, 1)

            citations = [cand.to_citation_dict(rank=i + 1) for i, cand in enumerate(candidates)]

            return {
                "question": question,
                "answer": answer,
                "citations": citations,
                "promptability": {
                    "length": opts.length,
                    "style": opts.style,
                    "temperature": opts.temperature,
                    "top_k": opts.top_k,
                },
                "pipeline_metrics": {
                    "retrieval_ms": retrieval_ms,
                    "generation_ms": generation_ms,
                    "embedder": self.embedder.active_backend,
                    "reranker": self.reranker.active_reranker,
                    "generator": model_used,
                    "chunks_searched": len(self.chunks),
                },
            }

    def stream_query(
        self,
        question: str,
        length: Optional[str] = None,
        style: Optional[str] = None,
        temperature: Optional[float] = None,
        custom_instruction: Optional[str] = None,
        top_k: Optional[int] = None,
        filter_filenames: Optional[List[str]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[str, None, None]:
        opts = PromptOptions(
            length=length or self.config.default_length,
            style=style or self.config.default_style,
            temperature=temperature if temperature is not None else self.config.default_temperature,
            custom_instruction=custom_instruction if custom_instruction is not None else self.config.custom_system_prompt,
            top_k=top_k or self.config.final_top_k,
        )

        t0 = time.perf_counter()
        with self._lock:
            candidates = self.retriever.retrieve(
                query=question,
                dense_top_k=self.config.dense_top_k,
                bm25_top_k=self.config.bm25_top_k,
                final_top_k=opts.top_k,
                filter_filenames=filter_filenames,
            )
        retrieval_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        citations = [cand.to_citation_dict(rank=i + 1) for i, cand in enumerate(candidates)]

        # 1. Emit retrieval metadata & citations immediately
        meta_event = {
            "type": "metadata",
            "citations": citations,
            "promptability": {
                "length": opts.length,
                "style": opts.style,
                "temperature": opts.temperature,
                "top_k": opts.top_k,
            },
            "pipeline_metrics": {
                "retrieval_ms": retrieval_ms,
                "embedder": self.embedder.active_backend,
                "reranker": self.reranker.active_reranker,
                "chunks_searched": len(self.chunks),
            },
        }
        yield f"data: {json.dumps(meta_event)}\n\n"

        # 2. Stream tokens
        full_tokens: List[str] = []
        used_model = "extractive_grounded_synthesizer"
        t1 = time.perf_counter()
        for tok_event in self.generator.stream_generate(
            query=question,
            candidates=candidates,
            options=opts,
            chat_history=chat_history,
        ):
            used_model = tok_event.get("model", used_model)
            full_tokens.append(tok_event.get("content", ""))
            yield f"data: {json.dumps(tok_event)}\n\n"

        gen_ms = round((time.perf_counter() - t1) * 1000.0, 1)
        done_event = {
            "type": "done",
            "answer": "".join(full_tokens),
            "generator": used_model,
            "generation_ms": gen_ms,
        }
        yield f"data: {json.dumps(done_event)}\n\n"

    def status_summary(self) -> Dict:
        with self._lock:
            docs = [asdict(v) for v in self.documents_meta.values()]
            docs.sort(key=lambda x: x["filename"].lower())
            return {
                "pdfs_dir": str(Path(self.config.pdfs_dir).resolve()),
                "data_dir": str(Path(self.config.data_dir).resolve()),
                "total_documents": len(docs),
                "total_chunks": len(self.chunks),
                "documents": docs,
                "active_embedder": self.embedder.active_backend,
                "active_reranker": self.reranker.active_reranker,
                "llm_provider": self.config.llm_provider,
                "llm_model": self.config.llm_model,
                "setup_approved": self.config.setup_approved,
                "selected_tier": self.config.selected_tier,
            }
