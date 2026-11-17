"""
Stage 7: Promptability & Grounded Generation Engine.
Gives users complete control over response Length, Style/Format, Grounding strictness,
and Domain Persona, while citing exact documents and page numbers.
Supports:
1. Local Ollama (`/api/chat` streaming & non-streaming)
2. Any OpenAI-Compatible API (`vLLM`, `LM Studio`, `llama.cpp`, `OpenAI`)
3. Smart Extractive Synthesis Fallback (works immediately even if no LLM server is running yet!)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional

import requests

from purerag.rag.retriever import RetrievedCandidate, tokenize_for_bm25


LENGTH_MODIFIERS: Dict[str, str] = {
    "shorter": (
        "RESPONSE LENGTH: CONCISE & SHORT. Keep your answer brief, direct, and under 120 words "
        "(2 to 4 crisp sentences or a tight summary). Strip all filler."
    ),
    "balanced": (
        "RESPONSE LENGTH: BALANCED. Provide a clear, well-structured answer of moderate length "
        "(roughly 150–300 words) that directly answers the question with key supporting details."
    ),
    "longer": (
        "RESPONSE LENGTH: COMPREHENSIVE & DETAILED. Provide an in-depth, thorough breakdown "
        "covering all relevant nuances, figures, definitions, and context from the retrieved passages."
    ),
}

STYLE_MODIFIERS: Dict[str, str] = {
    "standard": (
        "RESPONSE STYLE: Natural, clear, helpful AI assistant tone with clean Markdown formatting."
    ),
    "precise": (
        "RESPONSE STYLE: HIGHLY PRECISE & TECHNICAL. Prioritize exact terminology, numbers, equations, "
        "dates, thresholds, and verbatim facts from the source text. Avoid vague generalizations."
    ),
    "bulleted": (
        "RESPONSE STYLE: BULLETED KEY POINTS. Structure the entire response as organized, scannable "
        "Markdown bullet points (`- `) grouped by logical subheadings where helpful."
    ),
    "executive": (
        "RESPONSE STYLE: EXECUTIVE BRIEF. Start with a bold **Bottom Line (BLUF)** sentence, followed "
        "by **Key Takeaways** and **Supporting Evidence**."
    ),
    "step_by_step": (
        "RESPONSE STYLE: STEP-BY-STEP GUIDE. Present the answer as a numbered, sequential breakdown "
        "(`1.`, `2.`, `3.`) explaining the process or logical chain clearly."
    ),
    "eli5": (
        "RESPONSE STYLE: SIMPLE & ACCESSIBLE (ELI5). Explain complex concepts in plain, everyday "
        "language with intuitive analogies while staying 100% faithful to the documents."
    ),
    "citations_only": (
        "RESPONSE STYLE: STRICT EVIDENCE & QUOTES. Answer only with direct findings and verbatim quotes "
        "from the retrieved sources, attributing every single statement to its `[Source: filename, p. X]`."
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
            f"--- [Source #{idx}: {c.filename} | Page {c.page_number} | Section: {c.section_path}] ---\n"
            f"{c.raw_text}"
        )
        context_blocks.append(block)

    joined_context = "\n\n".join(context_blocks) if context_blocks else "(No relevant documents found in index.)"

    custom_block = ""
    if options.custom_instruction and options.custom_instruction.strip():
        custom_block = f"\nUSER CUSTOM DOMAIN / PERSONA INSTRUCTION:\n{options.custom_instruction.strip()}\n"

    system_prompt = (
        "You are Pure RAG, a state-of-the-art local Retrieval-Augmented Generation AI assistant.\n"
        "Your primary obligation is truthfulness and strict grounding in the user's local documents.\n\n"
        "CORE GROUNDING RULES:\n"
        "1. Base your answer strictly on the RETRIEVED DOCUMENT CONTEXT provided below.\n"
        "2. Always cite your sources inline using the format `[filename, p. X]` (for example: `[paper.pdf, p. 4]`).\n"
        "3. If the retrieved context does not contain enough information to answer a part of the question, "
        "state clearly what is and isn't covered in the uploaded documents rather than guessing or hallucinating.\n\n"
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
        f"RETRIEVED DOCUMENT CONTEXT:\n"
        f"{joined_context}\n\n"
        f"---\n"
        f"USER QUESTION: {query}\n\n"
        f"Remember to follow the requested length ({options.length}) and style ({options.style}) "
        f"and include inline `[filename, p. X]` citations."
    )
    messages.append({"role": "user", "content": user_message})
    return messages


def synthesize_extractive_response(
    query: str,
    candidates: List[RetrievedCandidate],
    options: PromptOptions,
) -> str:
    """
    Intelligent Promptability-Aware Extractive Synthesizer used when no local LLM
    (Ollama / OpenAI endpoint) is active yet, so Pure RAG works 100% out of the box!
    Formats retrieved evidence according to the user's chosen `length` and `style`.
    """
    if not candidates:
        return (
            "I couldn't find any indexed documents matching your query. "
            "Please add PDF, Markdown, or TXT files to the `pdfs/` folder (or drag & drop them in the sidebar) "
            "and click **Sync Documents**."
        )

    q_tokens = set(tokenize_for_bm25(query))

    # Extract and score individual sentences across top retrieved chunks
    scored_sentences: List[tuple[float, str, str, int, str]] = []
    for rank_idx, cand in enumerate(candidates):
        c = cand.chunk
        raw_sents = re.split(r"(?<=[.!?])\s+|\n+", c.raw_text)
        for sent in raw_sents:
            s_clean = re.sub(r"^[-*•]\s+", "", sent.strip()).strip()
            if len(s_clean) < 25 or s_clean.startswith("#"):
                continue
            s_tokens = set(tokenize_for_bm25(s_clean))
            overlap = len(q_tokens & s_tokens) / max(1, len(q_tokens))
            rank_weight = 1.0 / (1.0 + 0.25 * rank_idx)
            score = (0.65 * overlap + 0.35 * cand.final_score) * rank_weight
            scored_sentences.append((score, s_clean, c.filename, c.page_number, c.section_path))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate near-identical sentences
    selected: List[tuple[str, str, int, str]] = []
    seen_prefixes = set()
    max_sents = 3 if options.length == "shorter" else (6 if options.length == "balanced" else 10)

    for _, text, fname, page, section in scored_sentences:
        prefix = text[:50].lower()
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        selected.append((text, fname, page, section))
        if len(selected) >= max_sents:
            break

    if not selected:
        top_c = candidates[0].chunk
        selected = [(top_c.raw_text[:350], top_c.filename, top_c.page_number, top_c.section_path)]

    # Format according to options.style
    if options.style == "bulleted":
        lines = [f"### Key Points from Your Documents\n"]
        for text, fname, page, _ in selected:
            lines.append(f"- {text} **`[{fname}, p. {page}]`**")
        return "\n".join(lines)

    if options.style == "executive":
        lead_text, lead_file, lead_page, _ = selected[0]
        lines = [
            f"**Bottom Line (BLUF):** {lead_text} **`[{lead_file}, p. {lead_page}]`**\n",
            "**Key Supporting Evidence:**",
        ]
        for text, fname, page, sec in selected[1:]:
            lines.append(f"- *({sec})* {text} **`[{fname}, p. {page}]`**")
        return "\n".join(lines)

    if options.style == "step_by_step":
        lines = ["### Step-by-Step Breakdown\n"]
        for idx, (text, fname, page, _) in enumerate(selected, start=1):
            lines.append(f"{idx}. {text} **`[{fname}, p. {page}]`**")
        return "\n".join(lines)

    if options.style == "citations_only":
        lines = ["### Direct Document Evidence & Citations\n"]
        for text, fname, page, sec in selected:
            lines.append(f'> "{text}"\n> — **`[{fname}, p. {page} | {sec}]`**\n')
        return "\n".join(lines)

    if options.style == "precise":
        lines = ["### Precise Findings\n"]
        for text, fname, page, sec in selected:
            lines.append(f"- **[{sec}]**: {text} **`[{fname}, p. {page}]`**")
        return "\n".join(lines)

    # Standard or ELI5 prose paragraph synthesis
    paragraphs: List[str] = []
    for text, fname, page, _ in selected:
        paragraphs.append(f"{text} `[{fname}, p. {page}]`")

    if options.length == "shorter":
        return " ".join(paragraphs)

    intro = f"Based on the retrieved sections from your documents, here is what the sources state:\n\n"
    body = "\n\n".join(paragraphs[:2])
    if len(paragraphs) > 2:
        bullets = "\n".join(f"- {p}" for p in paragraphs[2:])
        return f"{intro}{body}\n\n**Additional Context:**\n{bullets}"
    return f"{intro}{body}"


class GroundedGenerator:
    def __init__(
        self,
        provider: str = "ollama",
        ollama_base_url: str = "http://127.0.0.1:11434",
        llm_model: str = "llama3.2:3b",
        openai_api_base: str = "https://api.openai.com/v1",
        openai_api_key: str = "",
    ) -> None:
        self.provider = provider
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.llm_model = llm_model
        self.openai_api_base = openai_api_base.rstrip("/")
        self.openai_api_key = openai_api_key

    def _resolve_active_ollama_model(self) -> Optional[str]:
        """Checks if Ollama is reachable and returns the configured model or first available LLM."""
        try:
            resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2.0)
            if resp.status_code != 200:
                return None
            models = [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
            if not models:
                return None
            if self.llm_model in models:
                return self.llm_model
            # Match prefix (e.g., "qwen2.5:3b" vs "qwen2.5:3b-instruct")
            base_prefix = self.llm_model.split(":")[0]
            for m in models:
                if m.startswith(base_prefix) and "embed" not in m.lower():
                    return m
            # Fallback to any non-embedding model installed in Ollama
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

        # Instant Grounded Extractive Synthesis Fallback
        synthesized = synthesize_extractive_response(query, candidates, options)
        return synthesized, "extractive_grounded_synthesizer"

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

        # Fallback to extractive synthesizer streamed smoothly
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
