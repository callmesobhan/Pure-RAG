"""
Stage 7: Promptability & Grounded Generation Engine.
Gives users complete control over response Length, Style/Format, Grounding strictness,
and Domain Persona, while citing exact documents using clean numeric brackets `[1]`, `[2]`.
Supports:
1. Local Ollama (`/api/chat` streaming & non-streaming)
2. Any OpenAI-Compatible API (`vLLM`, `LM Studio`, `llama.cpp`, `OpenAI`)
3. Smart Extractive Synthesis Fallback (works immediately even if no LLM server is running yet!)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional

import requests

from purerag.rag.retriever import RetrievedCandidate, tokenize_for_bm25


LENGTH_MODIFIERS: Dict[str, str] = {
    "shorter": (
        "RESPONSE LENGTH: CONCISE & SHORT. Keep your answer brief, direct, and under 100 words "
        "(2 to 4 crisp sentences). Strip all filler."
    ),
    "balanced": (
        "RESPONSE LENGTH: BALANCED. Provide a clear, well-structured, easy-to-read answer of moderate length "
        "(roughly 150–280 words) that directly answers the question with key supporting details."
    ),
    "longer": (
        "RESPONSE LENGTH: COMPREHENSIVE & DETAILED. Provide an in-depth, well-organized breakdown "
        "covering all relevant nuances, figures, definitions, and context from the retrieved passages."
    ),
}

STYLE_MODIFIERS: Dict[str, str] = {
    "standard": (
        "RESPONSE STYLE: Natural, clean, highly readable AI assistant prose with clear paragraphs."
    ),
    "precise": (
        "RESPONSE STYLE: HIGHLY PRECISE & TECHNICAL. Prioritize exact terminology, numbers, equations, "
        "dates, thresholds, and verbatim facts from the source text."
    ),
    "bulleted": (
        "RESPONSE STYLE: BULLETED KEY POINTS. Structure the response as clean, scannable "
        "Markdown bullet points (`- `) with one clear takeaway per bullet."
    ),
    "executive": (
        "RESPONSE STYLE: EXECUTIVE BRIEF. Start with a bold **Bottom Line** sentence, followed "
        "by 3–5 concise **Key Takeaways**."
    ),
    "step_by_step": (
        "RESPONSE STYLE: STEP-BY-STEP GUIDE. Present the answer as a clean numbered list "
        "(`1.`, `2.`, `3.`) explaining the process or logical sequence clearly."
    ),
    "eli5": (
        "RESPONSE STYLE: SIMPLE & ACCESSIBLE (ELI5). Explain complex concepts in plain, everyday "
        "language while staying faithful to the documents."
    ),
    "citations_only": (
        "RESPONSE STYLE: DIRECT QUOTES & EVIDENCE. Answer using clean blockquotes from the retrieved sources, "
        "ending each quote with its numeric citation `[1]`, `[2]`, etc."
    ),
}


@dataclass
class PromptOptions:
    length: str = "balanced"         # "shorter", "balanced", "longer"
    style: str = "standard"          # "standard", "precise", "bulleted", "executive", "step_by_step", "eli5", "citations_only"
    temperature: float = 0.2
    custom_instruction: str = ""
    top_k: int = 6


def build_grounded_messages(
    query: str,
    candidates: List[RetrievedCandidate],
    options: PromptOptions,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    length_rule = LENGTH_MODIFIERS.get(options.length, LENGTH_MODIFIERS["balanced"])
    style_rule = STYLE_MODIFIERS.get(options.style, STYLE_MODIFIERS["standard"])

    context_blocks: List[str] = []
    for idx, cand in enumerate(candidates, start=1):
        c = cand.chunk
        block = (
            f"[{idx}] Source: {c.filename} (Page {c.page_number}, Section: {c.section_path})\n"
            f"{c.raw_text}"
        )
        context_blocks.append(block)

    joined_context = "\n\n".join(context_blocks) if context_blocks else "(No relevant documents found in index.)"

    custom_block = ""
    if options.custom_instruction and options.custom_instruction.strip():
        custom_block = f"\nUSER CUSTOM DOMAIN / PERSONA INSTRUCTION:\n{options.custom_instruction.strip()}\n"

    system_prompt = (
        "You are Pure RAG, a state-of-the-art local Retrieval-Augmented Generation AI assistant.\n"
        "Write clear, natural, effortlessly readable answers grounded strictly in the user's documents.\n\n"
        "CITATION & READABILITY RULES:\n"
        "1. Base your answer strictly on the numbered RETRIEVED SOURCES `[1]`, `[2]`, etc. provided below.\n"
        "2. Cite sources inline using ONLY short bracketed numbers like `[1]` or `[1][2]` at the end of a sentence or claim. "
        "NEVER write long filenames or page numbers inline inside the prose.\n"
        "3. Write clean, well-spaced Markdown that is easy to scan and read.\n"
        "4. If the retrieved context does not contain the answer, state that clearly without guessing.\n\n"
        f"{length_rule}\n"
        f"{style_rule}"
        f"{custom_block}"
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if chat_history:
        for turn in chat_history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    user_message = (
        f"RETRIEVED SOURCES:\n"
        f"{joined_context}\n\n"
        f"---\n"
        f"QUESTION: {query}\n\n"
        f"Answer clearly following the requested length ({options.length}) and style ({options.style}), "
        f"using only clean bracketed numbers (`[1]`, `[2]`, ...) for citations."
    )
    messages.append({"role": "user", "content": user_message})
    return messages


def _clean_sentence_for_display(text: str) -> str:
    s = text.strip()
    # Strip leading markdown list markers or headings
    s = re.sub(r"^(?:[-*•]+|\d+\.)\s+", "", s).strip()
    s = re.sub(r"^#{1,6}\s+", "", s).strip()
    # Collapse internal newlines into clean spaces
    s = re.sub(r"\s*\n+\s*", " ", s).strip()
    return s


_INCOMPLETE_ENDING_WORDS = {
    "and", "or", "the", "a", "an", "of", "in", "to", "for", "with", "on", "at",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "that", "which", "who", "whom", "whose", "where",
    "when", "why", "how", "than", "then", "into", "onto", "upon", "within",
    "without", "between", "among", "through", "during", "before", "after",
}


def _is_complete_sentence(s: str) -> bool:
    if len(s) < 40:
        return False
    if s[0].islower() and not s.startswith("`"):
        return False
    # Must end with sentence punctuation
    if s[-1] not in ".!?":
        return False
    # Check last word before punctuation isn't a dangling preposition/conjunction/auxiliary
    words = re.findall(r"[a-zA-Z]+", s)
    if len(words) < 6:
        return False
    if words[-1].lower() in _INCOMPLETE_ENDING_WORDS:
        return False
    return True


def synthesize_extractive_response(
    query: str,
    candidates: List[RetrievedCandidate],
    options: PromptOptions,
) -> str:
    """
    Clean, readable Promptability-Aware Extractive Synthesizer used when no local LLM
    is active. Uses concise bracketed numbers `[1]`, `[2]` mapped to the retrieved sources.
    """
    if not candidates:
        return (
            "No matching documents were found in your `pdfs/` folder. "
            "Drop a PDF, Markdown, or TXT file into the sidebar or `./pdfs/` folder to begin."
        )

    q_tokens = set(tokenize_for_bm25(query))

    # Extract and score clean sentences across top retrieved chunks; keep their 1-based source rank
    scored_sentences: List[tuple[float, str, int, str]] = []
    for rank_idx, cand in enumerate(candidates, start=1):
        c = cand.chunk
        # Split on sentence boundaries or bullet lines
        raw_parts = re.split(r"(?<=[.!?])\s+|\n+", c.raw_text)
        for part in raw_parts:
            raw_line = part.strip()
            if not raw_line or raw_line.startswith("#") or raw_line.startswith("---"):
                continue
            s_clean = _clean_sentence_for_display(raw_line)
            if not _is_complete_sentence(s_clean):
                continue
            s_tokens = set(tokenize_for_bm25(s_clean))
            overlap = len(q_tokens & s_tokens) / max(1, len(q_tokens))
            rank_weight = 1.0 / (1.0 + 0.22 * (rank_idx - 1))
            score = (0.65 * overlap + 0.35 * cand.final_score) * rank_weight
            scored_sentences.append((score, s_clean, rank_idx, c.section_path))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    selected: List[tuple[str, int, str]] = []
    seen_prefixes = set()
    max_sents = 3 if options.length == "shorter" else (5 if options.length == "balanced" else 8)

    for _, text, src_num, section in scored_sentences:
        prefix = re.sub(r"\W+", "", text[:45].lower())
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        # Ensure sentence ends with clean punctuation before adding [N]
        if text[-1] not in ".!?":
            text = text + "."
        selected.append((text, src_num, section))
        if len(selected) >= max_sents:
            break

    if not selected:
        top_c = candidates[0].chunk
        fallback_txt = _clean_sentence_for_display(top_c.raw_text[:320])
        selected = [(fallback_txt, 1, top_c.section_path)]

    # Format cleanly according to options.style using ONLY [1], [2] bracket citations
    if options.style == "bulleted":
        return "\n".join(f"- {text} [{src_num}]" for text, src_num, _ in selected)

    if options.style == "executive":
        lead_text, lead_num, _ = selected[0]
        lines = [f"**Bottom Line:** {lead_text} [{lead_num}]"]
        if len(selected) > 1:
            lines.append("\n**Key Takeaways:**")
            for text, src_num, _ in selected[1:]:
                lines.append(f"- {text} [{src_num}]")
        return "\n".join(lines)

    if options.style == "step_by_step":
        return "\n".join(
            f"{idx}. {text} [{src_num}]"
            for idx, (text, src_num, _) in enumerate(selected, start=1)
        )

    if options.style == "citations_only":
        return "\n\n".join(f'> "{text}" [{src_num}]' for text, src_num, _ in selected)

    if options.style == "precise":
        return "\n".join(f"- {text} [{src_num}]" for text, src_num, _ in selected)

    # Standard or ELI5: clean, readable paragraphs with [1], [2] citations
    cited_sentences = [f"{text} [{src_num}]" for text, src_num, _ in selected]
    if options.length == "shorter" or len(cited_sentences) <= 2:
        return " ".join(cited_sentences)

    first_para = " ".join(cited_sentences[:2])
    bullet_points = "\n".join(f"- {s}" for s in cited_sentences[2:])
    return f"{first_para}\n\n{bullet_points}"


class GroundedGenerator:
    def __init__(
        self,
        provider: str = "ollama",
        ollama_base_url: str = "http://127.0.0.1:11434",
        llm_model: str = "qwen3:4b",
        openai_api_base: str = "https://api.openai.com/v1",
        openai_api_key: str = "",
    ) -> None:
        self.provider = provider
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.llm_model = llm_model
        self.openai_api_base = openai_api_base.rstrip("/")
        self.openai_api_key = openai_api_key

    def _ensure_ollama_daemon_running(self) -> bool:
        """
        Checks if local Ollama HTTP API is reachable; if not, and `ollama` binary
        is installed on the machine, automatically launches `ollama serve` in the background!
        """
        try:
            resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=1.5)
            if resp.status_code == 200:
                return True
        except Exception:
            pass

        ollama_bin = shutil.which("ollama") or "/usr/local/bin/ollama"
        if not shutil.which("ollama") and not os.path.exists(ollama_bin):
            return False

        try:
            subprocess.Popen(
                [ollama_bin, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            for _ in range(12):
                time.sleep(0.35)
                try:
                    r = requests.get(f"{self.ollama_base_url}/api/tags", timeout=1.0)
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
        except Exception:
            pass
        return False

    def _resolve_active_ollama_model(self) -> Optional[str]:
        """Checks if Ollama is reachable (starting it if needed) and returns the configured model."""
        if not self._ensure_ollama_daemon_running():
            return None
        try:
            resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2.0)
            if resp.status_code != 200:
                return None
            models = [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
            if not models:
                return None
            if self.llm_model in models:
                return self.llm_model
            base_prefix = self.llm_model.split(":")[0]
            for m in models:
                if m.startswith(base_prefix) and "embed" not in m.lower():
                    return m
            non_embed = [m for m in models if "embed" not in m.lower()]
            return non_embed[0] if non_embed else None
        except Exception:
            return None

    def generate(
        self,
        query: str,
        candidates: List[RetrievedCandidate],
        options: PromptOptions,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> tuple[str, str]:
        """Returns (response_text, generator_used_label)."""
        messages = build_grounded_messages(query, candidates, options, chat_history)

        if self.provider == "ollama":
            active_model = self._resolve_active_ollama_model()
            if active_model:
                try:
                    num_predict = 250 if options.length == "shorter" else (600 if options.length == "balanced" else 1400)
                    r = requests.post(
                        f"{self.ollama_base_url}/api/chat",
                        json={
                            "model": active_model,
                            "messages": messages,
                            "stream": False,
                            "options": {
                                "temperature": options.temperature,
                                "num_predict": num_predict,
                            },
                        },
                        timeout=90.0,
                    )
                    if r.status_code == 200:
                        content = r.json().get("message", {}).get("content", "").strip()
                        if content:
                            return content, f"ollama:{active_model}"
                except Exception:
                    pass

        if self.provider == "openai_compat" and self.openai_api_base:
            try:
                headers = {"Content-Type": "application/json"}
                if self.openai_api_key:
                    headers["Authorization"] = f"Bearer {self.openai_api_key}"
                max_tokens = 250 if options.length == "shorter" else (600 if options.length == "balanced" else 1400)
                r = requests.post(
                    f"{self.openai_api_base}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.llm_model,
                        "messages": messages,
                        "temperature": options.temperature,
                        "max_tokens": max_tokens,
                    },
                    timeout=90.0,
                )
                if r.status_code == 200:
                    content = r.json()["choices"][0]["message"]["content"].strip()
                    if content:
                        return content, f"openai_compat:{self.llm_model}"
            except Exception:
                pass

        synthesized = synthesize_extractive_response(query, candidates, options)
        return synthesized, "extractive_synthesizer"

    def stream_generate(
        self,
        query: str,
        candidates: List[RetrievedCandidate],
        options: PromptOptions,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[Dict, None, None]:
        """
        Yields streaming SSE event dicts:
        `{"type": "token", "content": "...", "model": "..."}`
        """
        messages = build_grounded_messages(query, candidates, options, chat_history)

        if self.provider == "ollama":
            active_model = self._resolve_active_ollama_model()
            if active_model:
                try:
                    num_predict = 250 if options.length == "shorter" else (600 if options.length == "balanced" else 1400)
                    with requests.post(
                        f"{self.ollama_base_url}/api/chat",
                        json={
                            "model": active_model,
                            "messages": messages,
                            "stream": True,
                            "options": {
                                "temperature": options.temperature,
                                "num_predict": num_predict,
                            },
                        },
                        stream=True,
                        timeout=90.0,
                    ) as resp:
                        if resp.status_code == 200:
                            for raw_line in resp.iter_lines():
                                if not raw_line:
                                    continue
                                data = json.loads(raw_line.decode("utf-8"))
                                token = data.get("message", {}).get("content", "")
                                if token:
                                    yield {
                                        "type": "token",
                                        "content": token,
                                        "model": f"ollama:{active_model}",
                                    }
                            return
                except Exception:
                    pass

        text, used_model = self.generate(query, candidates, options, chat_history)
        words = re.findall(r"\S+\s*", text)
        batch = []
        for w in words:
            batch.append(w)
            if len(batch) >= 3:
                yield {"type": "token", "content": "".join(batch), "model": used_model}
                batch = []
        if batch:
            yield {"type": "token", "content": "".join(batch), "model": used_model}
