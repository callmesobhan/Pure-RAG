"""
Stage 1: Layout-Aware Document & PDF Parser.
Uses PyMuPDF4LLM when installed to preserve Markdown headers, tables, bolding,
and multi-column reading order, with graceful fallback to pypdf for PDFs
and native parsing for Markdown/Text documents.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ParsedPage:
    page_number: int
    text: str
    detected_headings: List[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    filename: str
    filepath: str
    file_hash: str
    file_size_bytes: int
    total_pages: int
    parser_used: str
    pages: List[ParsedPage] = field(default_factory=list)


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".markdown"}


def compute_file_hash(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _extract_headings_from_text(text: str) -> List[str]:
    headings: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # Markdown style headings
        m = re.match(r"^(#{1,6})\s+(.+)$", s)
        if m:
            clean = re.sub(r"[*_`#]+", "", m.group(2)).strip()
            if clean and len(clean) <= 120:
                headings.append(clean)
            continue
        # Numbered section headers in plain PDF extraction (e.g. "1.2 Methodology" or "CHAPTER 3: RESULTS")
        if re.match(r"^(\d+(\.\d+){0,3}|CHAPTER\s+\d+|SECTION\s+\d+)[\s:.-]+[A-Z][A-Za-z0-9\s,&/-]{2,75}$", s):
            headings.append(s)
    return headings[:10]


def _clean_extracted_text(text: str) -> str:
    # Normalize excessive blank lines while keeping paragraph and table breaks
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf_file(filepath: Path) -> ParsedDocument:
    file_hash = compute_file_hash(filepath)
    file_size = filepath.stat().st_size
    pages: List[ParsedPage] = []
    parser_used = "pypdf"

    # Try SOTA layout-aware Markdown extraction via pymupdf4llm first
    try:
        import pymupdf4llm  # type: ignore

        page_chunks = pymupdf4llm.to_markdown(str(filepath), page_chunks=True)
        if isinstance(page_chunks, list) and len(page_chunks) > 0:
            parser_used = "pymupdf4llm (Layout-Aware Markdown)"
            for idx, item in enumerate(page_chunks):
                md_text = _clean_extracted_text(item.get("text", ""))
                meta = item.get("metadata", {}) or {}
                page_num = int(meta.get("page", idx + 1))
                if md_text:
                    pages.append(
                        ParsedPage(
                            page_number=page_num,
                            text=md_text,
                            detected_headings=_extract_headings_from_text(md_text),
                        )
                    )
            if pages:
                return ParsedDocument(
                    filename=filepath.name,
                    filepath=str(filepath.resolve()),
                    file_hash=file_hash,
                    file_size_bytes=file_size,
                    total_pages=len(page_chunks),
                    parser_used=parser_used,
                    pages=pages,
                )
    except Exception:
        pass

    # Fallback to pypdf
    from pypdf import PdfReader

    reader = PdfReader(str(filepath))
    total_pages = len(reader.pages)
    for idx, page in enumerate(reader.pages):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""
        cleaned = _clean_extracted_text(raw_text)
        if cleaned:
            pages.append(
                ParsedPage(
                    page_number=idx + 1,
                    text=cleaned,
                    detected_headings=_extract_headings_from_text(cleaned),
                )
            )

    return ParsedDocument(
        filename=filepath.name,
        filepath=str(filepath.resolve()),
        file_hash=file_hash,
        file_size_bytes=file_size,
        total_pages=max(total_pages, len(pages)),
        parser_used=parser_used,
        pages=pages,
    )


def parse_text_or_markdown_file(filepath: Path) -> ParsedDocument:
    file_hash = compute_file_hash(filepath)
    file_size = filepath.stat().st_size
    raw = filepath.read_text(encoding="utf-8", errors="replace")
    cleaned = _clean_extracted_text(raw)

    # Split large text/markdown files into logical "pages" (~3000 chars each) for accurate citations
    blocks = re.split(r"(?m)(?=^#{1,2}\s+)", cleaned)
    pages: List[ParsedPage] = []
    buffer = ""
    page_num = 1

    for block in blocks:
        if not block.strip():
            continue
        if len(buffer) + len(block) > 3200 and buffer.strip():
            pages.append(
                ParsedPage(
                    page_number=page_num,
                    text=buffer.strip(),
                    detected_headings=_extract_headings_from_text(buffer),
                )
            )
            page_num += 1
            buffer = block
        else:
            buffer = f"{buffer}\n\n{block}".strip() if buffer else block

    if buffer.strip():
        pages.append(
            ParsedPage(
                page_number=page_num,
                text=buffer.strip(),
                detected_headings=_extract_headings_from_text(buffer),
            )
        )

    return ParsedDocument(
        filename=filepath.name,
        filepath=str(filepath.resolve()),
        file_hash=file_hash,
        file_size_bytes=file_size,
        total_pages=len(pages),
        parser_used="markdown-native",
        pages=pages,
    )


def parse_document(filepath: Path) -> ParsedDocument:
    ext = filepath.suffix.lower()
    if ext == ".pdf":
        return parse_pdf_file(filepath)
    elif ext in {".md", ".markdown", ".txt"}:
        return parse_text_or_markdown_file(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
