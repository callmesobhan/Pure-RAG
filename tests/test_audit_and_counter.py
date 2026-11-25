"""Tests for Pure RAG Corpus Ingestion Blind-Spot Certificate and Adversarial Counter-Evidence Pass."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from purerag.cli import main as cli_main
from purerag.config import PureRAGConfig
from purerag.rag.audit import generate_blind_spot_certificate
from purerag.rag.engine import PureRAGEngine
from purerag.server import create_app


def _write_blank_pdf(path: Path, num_pages: int = 2) -> None:
    """Create a real valid PDF with blank/image-only pages (0 extractable text)."""
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=612, height=792)
    with open(path, "wb") as fh:
        writer.write(fh)


def test_blind_spot_certificate_detects_scanned_pdf_garbled_text_and_unsupported_files(tmp_path: Path) -> None:
    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()

    # 1. Healthy markdown document
    (pdfs_dir / "healthy_protocol.md").write_text(
        "# Dosimetry Protocol\n\n"
        "Standard kidney absorbed dose limits for Lu-177 radionuclide therapy are monitored across cycles "
        "using quantitative SPECT/CT imaging and verified time-activity curves.\n",
        encoding="utf-8",
    )

    # 2. Scanned/image-only 2-page PDF (0 extracted characters)
    _write_blank_pdf(pdfs_dir / "scanned_appendix.pdf", num_pages=2)

    # 3. Document with garbled PDF CID encoding artifacts
    (pdfs_dir / "garbled_ocr.txt").write_text(
        "Clinical Table 1: (cid:12)(cid:44)(cid:19)(cid:88)(cid:91)(cid:103)(cid:12)(cid:14) \ufffd\ufffd\ufffd\ufffd\ufffd "
        "dosage summary unreadable font encoding.",
        encoding="utf-8",
    )

    # 4. Unsupported file silently dropped in pdfs/
    (pdfs_dir / "deck.pptx").write_bytes(b"PK\x03\x04fake-pptx")

    cert = generate_blind_spot_certificate(pdfs_dir)
    assert cert.total_files_seen == 4
    assert cert.supported_files == 3
    assert cert.unsupported_files == ["deck.pptx"]
    assert cert.blind_spot_pages >= 3  # 2 blank PDF pages + 1 garbled page
    assert cert.overall_coverage_pct < 50.0
    assert len(cert.corpus_sha256) == 64

    # Deterministic hash across repeated runs
    cert2 = generate_blind_spot_certificate(pdfs_dir)
    assert cert2.corpus_sha256 == cert.corpus_sha256

    issues_by_file = {d.filename: [i.issue for i in d.issues] for d in cert.documents}
    assert issues_by_file["healthy_protocol.md"] == []
    assert issues_by_file["scanned_appendix.pdf"] == ["empty_or_scanned", "empty_or_scanned"]
    assert issues_by_file["garbled_ocr.txt"] == ["garbled_encoding"]
    assert any("ocrmypdf" in r for r in cert.recommendations)
    assert any("deck.pptx" in r for r in cert.recommendations)
    assert "Corpus Ingestion Blind-Spot Certificate" in cert.to_markdown()


def test_adversarial_counter_evidence_surfaces_contraindications_and_numeric_discrepancies(tmp_path: Path) -> None:
    pdfs_dir = tmp_path / "pdfs"
    data_dir = tmp_path / "data"
    pdfs_dir.mkdir()
    data_dir.mkdir()

    (pdfs_dir / "efficacy_summary.md").write_text(
        "# Therapy Overview\n\n"
        "## Primary Efficacy of Compound-Z99\n"
        "Compound-Z99 is recommended at 50 mg daily and achieved an 82% remission rate in adult patients.\n",
        encoding="utf-8",
    )
    (pdfs_dir / "safety_warnings.md").write_text(
        "# Safety & Limitations\n\n"
        "## Renal Contraindications for Compound-Z99\n"
        "However, Compound-Z99 is contraindicated and not recommended in patients with severe renal impairment "
        "due to nephrotoxicity risk.\n",
        encoding="utf-8",
    )
    (pdfs_dir / "elderly_cohort.md").write_text(
        "# Subgroup Analysis\n\n"
        "## Elderly Response to Compound-Z99\n"
        "In contrast to younger adults, elderly patients taking Compound-Z99 had only a 28% remission rate "
        "and experienced higher adverse event rates.\n",
        encoding="utf-8",
    )

    cfg = PureRAGConfig(
        pdfs_dir=str(pdfs_dir),
        data_dir=str(data_dir),
        llm_provider="extractive_fallback",
        embedding_backend="hashed_ngram",
        setup_approved=False,
    )
    engine = PureRAGEngine(cfg)
    engine.scan_and_ingest()

    # Restrict primary top_k=1 so safety_warnings and elderly_cohort are excluded from primary citations
    res = engine.query(
        question="What is the recommended dosage and remission rate of Compound-Z99?",
        top_k=1,
        check_counter_evidence=True,
    )
    assert res["citations"][0]["filename"] == "efficacy_summary.md"
    ce = res["counter_evidence"]
    assert len(ce) >= 1
    ce_files = {item["filename"] for item in ce}
    assert "safety_warnings.md" in ce_files or "elderly_cohort.md" in ce_files
    kinds = {item["kind"] for item in ce}
    assert kinds & {"negation_or_contraindication", "numeric_discrepancy", "exception_or_caveat"}


def test_api_and_cli_expose_blindspot_audit_and_counter_evidence(tmp_path: Path) -> None:
    pdfs_dir = tmp_path / "pdfs"
    data_dir = tmp_path / "data"
    pdfs_dir.mkdir()
    data_dir.mkdir()

    (pdfs_dir / "guide.md").write_text(
        "# Cryogenic Controller\n\n"
        "The CryoCtrl-9 amplifier runs at 4 Kelvin with 1.2 mW power draw.\n",
        encoding="utf-8",
    )
    (pdfs_dir / "errata.md").write_text(
        "# Errata for CryoCtrl-9\n\n"
        "Warning: CryoCtrl-9 is not recommended above 4 Kelvin and failed stability testing at 9.5 mW.\n",
        encoding="utf-8",
    )
    _write_blank_pdf(pdfs_dir / "blank.pdf", num_pages=1)

    cfg = PureRAGConfig(
        pdfs_dir=str(pdfs_dir),
        data_dir=str(data_dir),
        llm_provider="extractive_fallback",
        embedding_backend="hashed_ngram",
    )
    app = create_app(cfg)
    client = TestClient(app)

    r_aud = client.get("/api/audit/blindspots")
    assert r_aud.status_code == 200
    aud_data = r_aud.json()
    assert aud_data["blind_spot_pages"] == 1
    assert aud_data["total_pages"] == 3

    r_chat = client.post(
        "/api/chat",
        json={
            "question": "What is the CryoCtrl-9 amplifier operating temperature and power draw?",
            "top_k": 1,
            "counter_evidence": True,
        },
    )
    assert r_chat.status_code == 200
    body = r_chat.json()
    assert "counter_evidence" in body
    assert len(body["counter_evidence"]) >= 1
    assert body["counter_evidence"][0]["filename"] == "errata.md"

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={
            "question": "What is the CryoCtrl-9 amplifier power draw?",
            "top_k": 1,
            "counter_evidence": True,
        },
    ) as stream_resp:
        assert stream_resp.status_code == 200
        meta_events = [
            json.loads(line[6:])
            for line in stream_resp.iter_lines()
            if line.startswith("data: ") and '"metadata"' in line
        ]
        assert meta_events and "counter_evidence" in meta_events[0]

    # CLI audit subcommand writes JSON certificate
    out_file = tmp_path / "cert.json"
    rc = cli_main(["audit", "--pdfs-dir", str(pdfs_dir), "--json", "--out", str(out_file)])
    assert rc == 0
    cert_json = json.loads(out_file.read_text(encoding="utf-8"))
    assert cert_json["blind_spot_pages"] == 1
