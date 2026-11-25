"""
Deterministic Corpus Ingestion Blind-Spot Certificate & Adversarial Counter-Evidence Pass.

Two verification capabilities absent from commercial local/cloud RAG tools:
1. CorpusBlindSpotCertificate: Audits every file in `pdfs/` (and per-page parser diagnostics)
   to surface silent ingestion failures — scanned/image-only pages yielding ~0 characters,
   garbled CID/mojibake encoding pages, low-density chart/formula pages, and unsupported files
   dropped in the folder — with a deterministic SHA-256 certificate and coverage percentage.
2. Adversarial Counter-Evidence Retrieval Pass: Runs a second hybrid retrieval pass using
   epistemic negation/exception/caveat probes (excluding already-cited primary chunks) to
   surface contradictions, contraindications, caveats, and numeric discrepancies that
   confirmation-biased first-pass retrieval hides.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from purerag.rag.parser import SUPPORTED_EXTENSIONS, ParsedDocument, parse_document
from purerag.rag.retriever import RetrievedCandidate, tokenize_for_bm25

if TYPE_CHECKING:
    from purerag.rag.engine import PureRAGEngine


_CID_PATTERN = re.compile(r"\(cid:\d+\)")
_COUNTER_CUES = re.compile(
    r"\b(?:however|although|whereas|contrary|contradict\w*|disagree\w*|conflict\w*|"
    r"except(?:ion)?s?|unless|caveat\w*|limitation\w*|warning|caution|contraindicat\w*|"
    r"not\s+recommended|do\s+not|never|avoid|prohibit\w*|deprecated|unsafe|invalid|"
    r"failed\s+to|no\s+significant|ineffective|adverse|toxicity|risk\s+of|erratum|retracted)\b",
    re.IGNORECASE,
)
_NEGATION_CUES = re.compile(
    r"\b(?:not\s+recommended|contraindicat\w*|do\s+not|never|prohibit\w*|unsafe|invalid|ineffective|failed\s+to|no\s+significant)\b",
    re.IGNORECASE,
)
_NUMBER_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<num>-?\d+(?:\.\d+)?)\s*(?P<unit>%|mg|mcg|g|kg|ml|l|gy|cgy|mev|kev|mm|cm|m|s|ms|min|h|hr|hours|days|kelvin|k|mw|w|hz|khz|mhz|ghz)?\b",
    re.IGNORECASE,
)


@dataclass
class PageBlindSpot:
    filename: str
    page_number: int
    issue: str  # "empty_or_scanned" | "low_density" | "garbled_encoding"
    char_count: int
    cid_count: int
    replacement_char_count: int
    detail: str


@dataclass
class DocumentAuditSummary:
    filename: str
    file_hash: str
    parser_used: str
    total_pages: int
    healthy_pages: int
    blind_spot_pages: int
    coverage_pct: float
    issues: list[PageBlindSpot] = field(default_factory=list)


@dataclass
class CorpusBlindSpotCertificate:
    pdfs_dir: str
    corpus_sha256: str
    total_files_seen: int
    supported_files: int
    unsupported_files: list[str]
    parse_errors: dict[str, str]
    total_pages: int
    healthy_pages: int
    blind_spot_pages: int
    overall_coverage_pct: float
    documents: list[DocumentAuditSummary] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# Pure RAG — Corpus Ingestion Blind-Spot Certificate",
            "",
            f"- **Corpus Directory:** `{self.pdfs_dir}`",
            f"- **Deterministic Corpus SHA-256:** `{self.corpus_sha256[:16]}...`",
            f"- **Supported Documents Audited:** {self.supported_files} (Total files in folder: {self.total_files_seen})",
            f"- **Page Coverage:** {self.healthy_pages}/{self.total_pages} healthy pages (**{self.overall_coverage_pct:.1f}%**)",
            f"- **Silent Blind-Spot Pages Detected:** {self.blind_spot_pages}",
        ]
        if self.unsupported_files:
            lines.append(f"- **Unsupported / Ignored Files ({len(self.unsupported_files)}):** {', '.join(self.unsupported_files)}")
        if self.parse_errors:
            lines.append(f"- **Unreadable / Failed Files ({len(self.parse_errors)}):** {', '.join(self.parse_errors.keys())}")
        lines.append("")

        if any(d.issues for d in self.documents):
            lines.append("## Per-Document Blind Spots")
            lines.append("")
            for d in self.documents:
                if not d.issues:
                    continue
                lines.append(
                    f"### `{d.filename}` — {d.healthy_pages}/{d.total_pages} healthy ({d.coverage_pct:.1f}%, parser: `{d.parser_used}`)"
                )
                for iss in d.issues[:25]:
                    lines.append(f"- **Page {iss.page_number}** [`{iss.issue}`]: {iss.detail}")
                if len(d.issues) > 25:
                    lines.append(f"- *... and {len(d.issues) - 25} more blind-spot pages.*")
                lines.append("")
        else:
            lines.append("No page-level ingestion blind spots detected across supported documents.")
            lines.append("")

        if self.recommendations:
            lines.append("## Recommended Actions")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        return "\n".join(lines)


def diagnose_page_text(filename: str, page_number: int, text: str, is_pdf: bool = True) -> PageBlindSpot | None:
    """Deterministically inspect one extracted page for silent ingestion failure modes."""
    stripped = (text or "").strip()
    char_count = len(stripped)
    cid_count = len(_CID_PATTERN.findall(stripped))
    repl_count = stripped.count("\ufffd")

    if char_count < 40:
        return PageBlindSpot(
            filename=filename,
            page_number=page_number,
            issue="empty_or_scanned",
            char_count=char_count,
            cid_count=cid_count,
            replacement_char_count=repl_count,
            detail=f"Extracted only {char_count} characters — likely a scanned image-only page or blank/figure page invisible to RAG.",
        )

    garbage_ratio = (cid_count * 6 + repl_count) / max(1, char_count)
    if cid_count >= 5 or repl_count >= 5 or garbage_ratio > 0.08:
        return PageBlindSpot(
            filename=filename,
            page_number=page_number,
            issue="garbled_encoding",
            char_count=char_count,
            cid_count=cid_count,
            replacement_char_count=repl_count,
            detail=(
                f"Detected {cid_count} unresolved PDF CID glyph(s) and {repl_count} replacement char(s) "
                f"in {char_count} chars — font encoding is partially unreadable."
            ),
        )

    if is_pdf and char_count < 160:
        return PageBlindSpot(
            filename=filename,
            page_number=page_number,
            issue="low_density",
            char_count=char_count,
            cid_count=cid_count,
            replacement_char_count=repl_count,
            detail=f"Only {char_count} characters extracted — likely figure-heavy, slide, or vector-equation page with minimal searchable text.",
        )

    return None


def audit_parsed_document(doc: ParsedDocument) -> DocumentAuditSummary:
    """Build a per-document blind-spot summary from a ParsedDocument (including missing empty PDF pages)."""
    is_pdf = doc.filename.lower().endswith(".pdf")
    by_page = {p.page_number: p.text for p in doc.pages}
    total_pages = max(doc.total_pages, len(doc.pages), max(by_page.keys(), default=0))
    issues: list[PageBlindSpot] = []

    if total_pages == 0:
        issues.append(
            PageBlindSpot(
                filename=doc.filename,
                page_number=1,
                issue="empty_or_scanned",
                char_count=0,
                cid_count=0,
                replacement_char_count=0,
                detail="Document produced 0 readable pages (empty or scanned image-only file).",
            )
        )
        return DocumentAuditSummary(
            filename=doc.filename,
            file_hash=doc.file_hash,
            parser_used=doc.parser_used,
            total_pages=1,
            healthy_pages=0,
            blind_spot_pages=1,
            coverage_pct=0.0,
            issues=issues,
        )

    for page_num in range(1, total_pages + 1):
        txt = by_page.get(page_num, "")
        spot = diagnose_page_text(doc.filename, page_num, txt, is_pdf=is_pdf)
        if spot is not None:
            issues.append(spot)

    blind = len(issues)
    healthy = max(0, total_pages - blind)
    cov = round(100.0 * healthy / total_pages, 1) if total_pages else 100.0
    return DocumentAuditSummary(
        filename=doc.filename,
        file_hash=doc.file_hash,
        parser_used=doc.parser_used,
        total_pages=total_pages,
        healthy_pages=healthy,
        blind_spot_pages=blind,
        coverage_pct=cov,
        issues=issues,
    )


def generate_blind_spot_certificate(pdfs_dir: Path | str) -> CorpusBlindSpotCertificate:
    """Audit every file in `pdfs_dir` and return a deterministic CorpusBlindSpotCertificate."""
    folder = Path(pdfs_dir)
    folder.mkdir(parents=True, exist_ok=True)

    all_files = [p for p in sorted(folder.iterdir()) if p.is_file() and not p.name.startswith(".")]
    unsupported: list[str] = []
    parse_errors: dict[str, str] = {}
    doc_summaries: list[DocumentAuditSummary] = []

    hasher = hashlib.sha256()
    for p in all_files:
        ext = p.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            unsupported.append(p.name)
            hasher.update(f"UNSUPPORTED:{p.name}:{p.stat().st_size}\n".encode())
            continue
        try:
            parsed = parse_document(p)
            summary = audit_parsed_document(parsed)
            doc_summaries.append(summary)
            hasher.update(f"DOC:{summary.filename}:{summary.file_hash}:{summary.blind_spot_pages}\n".encode())
        except Exception as exc:  # noqa: BLE001
            parse_errors[p.name] = str(exc)
            hasher.update(f"ERR:{p.name}:{exc}\n".encode())

    total_pages = sum(d.total_pages for d in doc_summaries)
    healthy_pages = sum(d.healthy_pages for d in doc_summaries)
    blind_pages = sum(d.blind_spot_pages for d in doc_summaries)
    overall_cov = round(100.0 * healthy_pages / total_pages, 1) if total_pages > 0 else 100.0

    recs: list[str] = []
    if unsupported:
        recs.append(
            f"Convert or remove {len(unsupported)} unsupported file(s) ignored by indexing: {', '.join(unsupported[:5])}."
        )
    if parse_errors:
        recs.append(
            f"Repair or decrypt {len(parse_errors)} unreadable file(s): {', '.join(list(parse_errors.keys())[:5])}."
        )
    for d in doc_summaries:
        scanned = [i.page_number for i in d.issues if i.issue == "empty_or_scanned"]
        garbled = [i.page_number for i in d.issues if i.issue == "garbled_encoding"]
        if scanned:
            pg = ", ".join(str(x) for x in scanned[:10]) + ("..." if len(scanned) > 10 else "")
            recs.append(f"Run OCR (e.g. ocrmypdf) on `{d.filename}` page(s) {pg} — currently invisible to retrieval.")
        if garbled:
            pg = ", ".join(str(x) for x in garbled[:10]) + ("..." if len(garbled) > 10 else "")
            recs.append(f"Re-export or OCR `{d.filename}` page(s) {pg} to fix garbled CID/Unicode font mapping.")

    return CorpusBlindSpotCertificate(
        pdfs_dir=str(folder.resolve()),
        corpus_sha256=hasher.hexdigest(),
        total_files_seen=len(all_files),
        supported_files=len(doc_summaries),
        unsupported_files=unsupported,
        parse_errors=parse_errors,
        total_pages=total_pages,
        healthy_pages=healthy_pages,
        blind_spot_pages=blind_pages,
        overall_coverage_pct=overall_cov,
        documents=doc_summaries,
        recommendations=recs,
    )


# ==================== ADVERSARIAL COUNTER-EVIDENCE RETRIEVAL PASS ====================

@dataclass
class CounterEvidenceItem:
    rank: int
    kind: str  # "negation_or_contraindication" | "exception_or_caveat" | "numeric_discrepancy"
    chunk_id: str
    filename: str
    page_number: int
    section_path: str
    counter_score: float
    topic_overlap: float
    matched_cues: list[str]
    reason: str
    snippet: str

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_unit_numbers(text: str) -> dict[str, set[float]]:
    out: dict[str, set[float]] = {}
    for m in _NUMBER_UNIT_RE.finditer(text):
        unit = (m.group("unit") or "").lower()
        if not unit:
            continue
        try:
            val = float(m.group("num"))
        except ValueError:
            continue
        out.setdefault(unit, set()).add(val)
    return out


def find_counter_evidence(
    engine: PureRAGEngine,
    question: str,
    primary_candidates: Sequence[RetrievedCandidate],
    max_items: int = 3,
    filter_filenames: list[str] | None = None,
) -> list[CounterEvidenceItem]:
    """
    Adversarial second-pass retrieval that actively hunts for caveats, exceptions,
    contraindications, and numeric discrepancies on the same topic outside the primary chunks.
    """
    if not engine.chunks or not question.strip():
        return []

    q_tokens = set(tokenize_for_bm25(question))
    primary_ids = {c.chunk.chunk_id for c in primary_candidates}
    primary_nums: dict[str, set[float]] = {}
    for c in primary_candidates:
        for unit, vals in _extract_unit_numbers(c.chunk.raw_text).items():
            primary_nums.setdefault(unit, set()).update(vals)

    # Expand topic anchor tokens with top shared non-stopword terms from primary top chunk
    topic_tokens = set(q_tokens)
    if primary_candidates:
        top_toks = set(tokenize_for_bm25(primary_candidates[0].chunk.raw_text))
        topic_tokens |= (q_tokens & top_toks)
    if not topic_tokens:
        return []

    # Build adversarial counter-query probing for exceptions, contraindications, and limitations
    core_terms = " ".join(sorted(topic_tokens)[:8])
    probe_query = (
        f"{core_terms} however exception limitation caveat contraindicated "
        f"not recommended warning caution contrary failed adverse"
    )

    # Retrieve a wider candidate pool using the existing hybrid retriever
    pool = engine.retriever.retrieve(
        query=probe_query,
        dense_top_k=max(30, len(engine.chunks)),
        bm25_top_k=max(30, len(engine.chunks)),
        final_top_k=min(max(18, max_items * 6), len(engine.chunks)),
        filter_filenames=filter_filenames,
    )

    # Also scan any remaining corpus chunks directly if the corpus is modest (<= 500 chunks)
    # so counter-evidence with lower dense similarity is never missed.
    seen_ids = {c.chunk.chunk_id for c in pool}
    extra_chunks = []
    if len(engine.chunks) <= 500:
        allowed_files = set(filter_filenames) if filter_filenames else None
        for ch in engine.chunks:
            if ch.chunk_id in seen_ids or ch.chunk_id in primary_ids:
                continue
            if allowed_files and ch.filename not in allowed_files:
                continue
            extra_chunks.append(
                RetrievedCandidate(
                    chunk=ch,
                    final_score=0.0,
                    rrf_score=0.0,
                    dense_score=0.0,
                    bm25_score=0.0,
                    rerank_score=0.0,
                    dense_rank=None,
                    bm25_rank=None,
                )
            )

    scored: list[tuple[float, CounterEvidenceItem]] = []
    for cand in [*pool, *extra_chunks]:
        ch = cand.chunk
        if ch.chunk_id in primary_ids:
            continue
        ch_tokens = set(tokenize_for_bm25(ch.contextualized_text))
        overlap_toks = topic_tokens & ch_tokens
        topic_overlap = len(overlap_toks) / max(1, len(topic_tokens))
        if topic_overlap < 0.20:
            continue

        cue_matches = sorted({m.group(0).lower() for m in _COUNTER_CUES.finditer(ch.raw_text)})
        neg_matches = sorted({m.group(0).lower() for m in _NEGATION_CUES.finditer(ch.raw_text)})

        # Check for numeric discrepancy on the same unit & topic (e.g. primary says 0.72%, counter says 1.4%)
        cand_nums = _extract_unit_numbers(ch.raw_text)
        num_diffs: list[str] = []
        for unit, vals in cand_nums.items():
            if unit not in primary_nums:
                continue
            p_vals = primary_nums[unit]
            for v in vals:
                if all(abs(v - pv) > max(1e-6, 0.15 * max(abs(v), abs(pv))) for pv in p_vals):
                    p_str = ", ".join(f"{pv:g}{unit}" for pv in sorted(p_vals)[:2])
                    num_diffs.append(f"{v:g}{unit} vs primary {p_str}")
                    break

        if not cue_matches and not num_diffs:
            continue

        if neg_matches:
            kind = "negation_or_contraindication"
            reason = f"Contains explicit restriction/negation ({', '.join(neg_matches[:3])}) on matching topic."
            cue_bonus = 0.45 + 0.08 * min(len(neg_matches), 3)
        elif num_diffs:
            kind = "numeric_discrepancy"
            reason = f"Reports differing quantitative value ({'; '.join(num_diffs[:2])}) on matching topic."
            cue_bonus = 0.40 + 0.08 * min(len(cue_matches), 3)
        else:
            kind = "exception_or_caveat"
            reason = f"Adds caveat/exception ({', '.join(cue_matches[:3])}) not present in top primary chunks."
            cue_bonus = 0.30 + 0.06 * min(len(cue_matches), 4)

        score = round(min(1.0, 0.45 * topic_overlap + cue_bonus + 0.10 * float(cand.final_score)), 4)
        scored.append(
            (
                score,
                CounterEvidenceItem(
                    rank=0,
                    kind=kind,
                    chunk_id=ch.chunk_id,
                    filename=ch.filename,
                    page_number=ch.page_number,
                    section_path=ch.section_path,
                    counter_score=score,
                    topic_overlap=round(topic_overlap, 3),
                    matched_cues=(neg_matches or cue_matches or num_diffs)[:5],
                    reason=reason,
                    snippet=ch.raw_text[:400] + ("..." if len(ch.raw_text) > 400 else ""),
                ),
            )
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[CounterEvidenceItem] = []
    for i, (_, item) in enumerate(scored[:max_items], start=1):
        item.rank = i
        out.append(item)
    return out
