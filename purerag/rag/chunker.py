"""
Stage 2 & 3: Structure-Aware Hierarchical Chunker + Contextual Breadcrumb Enrichment.
Implements Anthropic-inspired Contextual Retrieval by prepending document & section
breadcrumbs (`[Document: ... | Page: ... | Section: ...]`) to every chunk before
dense embedding and BM25 indexing.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List

from purerag.rag.parser import ParsedDocument


@dataclass
class DocumentChunk:
    chunk_id: str
    filename: str
    file_hash: str
    page_number: int
    section_path: str
    raw_text: str
    contextualized_text: str
    char_count: int
    word_count: int

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "DocumentChunk":
        return cls(**data)


def _split_into_semantic_units(text: str) -> List[tuple[str, str]]:
    """
    Splits page text into (heading_context, paragraph_or_table_block) pairs.
    Preserves Markdown tables and lists intact where possible.
    """
    lines = text.splitlines()
    units: List[tuple[str, str]] = []
    current_h1 = ""
    current_h2 = ""
    current_h3 = ""
    buffer: List[str] = []

    def current_breadcrumb() -> str:
        parts = [p for p in (current_h1, current_h2, current_h3) if p]
        return " > ".join(parts) if parts else "General"

    def flush_buffer() -> None:
        nonlocal buffer
        joined = "\n".join(buffer).strip()
        if joined:
            units.append((current_breadcrumb(), joined))
        buffer = []

    for line in lines:
        stripped = line.strip()
        m = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if m:
            flush_buffer()
            level = len(m.group(1))
            title = re.sub(r"[*_`#]+", "", m.group(2)).strip()
            if level == 1:
                current_h1, current_h2, current_h3 = title, "", ""
            elif level == 2:
                current_h2, current_h3 = title, ""
            else:
                current_h3 = title
            buffer.append(line)
            continue

        # Detect numbered academic/technical headings (e.g., "2.1 System Architecture")
        if (
            len(stripped) < 85
            and re.match(r"^(\d+(\.\d+){0,2}|ABSTRACT|INTRODUCTION|METHODOLOGY|RESULTS|DISCUSSION|CONCLUSION|REFERENCES)\b", stripped, re.IGNORECASE)
            and not stripped.endswith(".")
        ):
            flush_buffer()
            current_h2 = stripped
            buffer.append(line)
            continue

        if not stripped:
            # Blank line marks paragraph boundary unless buffer is tiny
            if sum(len(x) for x in buffer) >= 280:
                flush_buffer()
            elif buffer:
                buffer.append("")
        else:
            buffer.append(line)

    flush_buffer()
    return units


def _window_split_long_block(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    # Split along sentence boundaries first
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    chunks: List[str] = []
    current = ""

    for sent in sentences:
        if not sent.strip():
            continue
        if len(current) + len(sent) + 1 <= max_chars:
            current = f"{current} {sent}".strip() if current else sent
        else:
            if current:
                chunks.append(current)
                # Keep tail overlap for continuity
                tail = current[-overlap_chars:] if overlap_chars > 0 else ""
                current = f"{tail} {sent}".strip()
            else:
                # Single sentence exceeds max_chars; hard-slice safely
                for i in range(0, len(sent), max(200, max_chars - overlap_chars)):
                    chunks.append(sent[i : i + max_chars])
                current = ""

    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_document(
    doc: ParsedDocument,
    chunk_size_chars: int = 1200,
    chunk_overlap_chars: int = 220,
) -> List[DocumentChunk]:
    """
    Produces Contextualized Chunks from a ParsedDocument.
    Each chunk has both `raw_text` (for clean display) and `contextualized_text`
    (with document title, page, and hierarchical section path prepended for SOTA retrieval).
    """
    all_chunks: List[DocumentChunk] = []
    chunk_counter = 0
    active_section = "Overview"

    for page in doc.pages:
        if page.detected_headings and active_section == "Overview":
            active_section = page.detected_headings[0]

        semantic_units = _split_into_semantic_units(page.text)
        current_raw = ""
        current_section = active_section

        def emit_chunk(raw_body: str, section_label: str) -> None:
            nonlocal chunk_counter
            cleaned_body = raw_body.strip()
            if len(cleaned_body) < 25:
                return
            chunk_counter += 1
            chunk_id = f"{doc.file_hash[:10]}_p{page.page_number}_c{chunk_counter}"
            context_prefix = (
                f"[Document: {doc.filename} | Page {page.page_number} of {doc.total_pages} "
                f"| Section: {section_label}]\n"
            )
            contextualized = f"{context_prefix}{cleaned_body}"
            words = len(cleaned_body.split())
            all_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    filename=doc.filename,
                    file_hash=doc.file_hash,
                    page_number=page.page_number,
                    section_path=section_label,
                    raw_text=cleaned_body,
                    contextualized_text=contextualized,
                    char_count=len(cleaned_body),
                    word_count=words,
                )
            )

        for unit_section, unit_text in semantic_units:
            if unit_section != "General":
                active_section = unit_section
                current_section = unit_section

            if len(unit_text) > chunk_size_chars:
                if current_raw:
                    emit_chunk(current_raw, current_section)
                    current_raw = ""
                for sub in _window_split_long_block(unit_text, chunk_size_chars, chunk_overlap_chars):
                    emit_chunk(sub, current_section)
                continue

            if len(current_raw) + len(unit_text) + 2 <= chunk_size_chars:
                current_raw = f"{current_raw}\n\n{unit_text}".strip() if current_raw else unit_text
            else:
                emit_chunk(current_raw, current_section)
                # Carry over only complete trailing sentence(s) for overlap so chunks never start mid-word!
                overlap_tail = ""
                if chunk_overlap_chars > 0:
                    sents = re.split(r"(?<=[.!?])\s+", current_raw.strip())
                    if len(sents) > 1 and len(sents[-1]) <= chunk_overlap_chars:
                        overlap_tail = sents[-1].strip()
                if overlap_tail and len(overlap_tail) + len(unit_text) + 2 <= chunk_size_chars:
                    current_raw = f"{overlap_tail}\n\n{unit_text}".strip()
                else:
                    current_raw = unit_text

        if current_raw:
            emit_chunk(current_raw, current_section)

    return all_chunks
