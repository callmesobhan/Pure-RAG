"""
Comprehensive Automated Test Suite for Pure RAG:
- Hardware Profiler & Tier Recommendation tests
- Document Parser & Contextual Breadcrumb Chunker tests
- Hybrid Dense + BM25 + RRF + Cross-Encoder Reranker tests
- Promptability (Length & Style modifiers) & Grounded Synthesis tests
- FastAPI REST & Streaming SSE API tests
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from purerag.config import PureRAGConfig
from purerag.hardware import HARDWARE_TIERS, analyze_hardware, discover_local_models, select_recommended_tier
from purerag.rag.chunker import chunk_document
from purerag.rag.engine import PureRAGEngine
from purerag.rag.parser import parse_document
from purerag.server import create_app


def test_hardware_profiler_and_tier_selection():
    profile = analyze_hardware()
    assert profile.os_name in ("macOS", "Windows", "Linux", "Darwin")
    assert profile.ram_total_gb > 0
    assert profile.cpu_cores_logical >= 1
    assert profile.recommended_tier in HARDWARE_TIERS
    assert profile.discovered_local_models is not None
    assert profile.active_zero_download_plan is not None

    # Verify local model scoring & discovery reuses existing models
    disc = discover_local_models(
        ollama_models=["mistral:latest", "gemma4:12b", "nomic-embed-text:latest"],
        ram_total_gb=16.0,
    )
    assert disc.best_local_llm == "gemma4:12b"
    assert disc.best_local_embedder == "nomic-embed-text:latest"
    assert disc.best_local_embedder_backend == "ollama"

    assert select_recommended_tier(4.0, False, "cpu_only", None) == "ultra_light"
    assert select_recommended_tier(12.0, False, "cpu_only", None) == "balanced"
    assert select_recommended_tier(16.0, True, "apple_metal", 12.0) == "performance"
    assert select_recommended_tier(64.0, True, "nvidia_cuda", 24.0) == "workstation"


def test_parser_and_contextual_chunker(tmp_path: Path):
    doc_file = tmp_path / "clinical_study.md"
    doc_file.write_text(
        "# Phase 3 Clinical Trial Protocol\n\n"
        "## 1. Dosage and Administration\n"
        "Patients in Cohort A received 250mg of Compound-X42 twice daily for 14 weeks.\n\n"
        "## 2. Efficacy Outcomes\n"
        "Biomarker reduction reached 78.4% (p < 0.001) compared to placebo.\n",
        encoding="utf-8",
    )

    parsed = parse_document(doc_file)
    assert parsed.filename == "clinical_study.md"
    assert parsed.total_pages >= 1

    chunks = chunk_document(parsed, chunk_size_chars=300, chunk_overlap_chars=50)
    assert len(chunks) >= 1
    # Verify Contextual Breadcrumb enrichment
    for c in chunks:
        assert "[Document: clinical_study.md" in c.contextualized_text
        assert "Page 1" in c.contextualized_text
        assert "Section:" in c.contextualized_text


def test_hybrid_rag_engine_and_promptability_modes(tmp_path: Path):
    pdfs_dir = tmp_path / "pdfs"
    data_dir = tmp_path / "data"
    pdfs_dir.mkdir()
    data_dir.mkdir()

    (pdfs_dir / "quantum_computing.md").write_text(
        "# Quantum Error Correction Architecture\n\n"
        "## Surface Code Thresholds\n"
        "The fault-tolerant threshold of the rotated surface code under circuit-level noise is approximately 0.72%.\n"
        "Logical qubit overhead scales quadratically with the code distance d.\n\n"
        "## Cryogenic Control Interface\n"
        "Cryogenic CMOS controllers operate at 4 Kelvin inside the dilution refrigerator stage, "
        "dissipating less than 1.2 milliwatts per channel.\n",
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
    summary = engine.scan_and_ingest()
    assert summary["total_documents"] == 1
    assert summary["total_chunks"] >= 1

    # Query for exact technical keyword ("4 Kelvin" / "0.72%")
    res_bulleted = engine.query(
        question="What is the surface code threshold and cryogenic operating temperature?",
        length="shorter",
        style="bulleted",
    )
    assert "0.72%" in res_bulleted["answer"] or "4 Kelvin" in res_bulleted["answer"]
    assert "- " in res_bulleted["answer"]
    assert len(res_bulleted["citations"]) >= 1
    assert res_bulleted["citations"][0]["filename"] == "quantum_computing.md"

    # Query in Executive style
    res_exec = engine.query(
        question="How do cryogenic CMOS controllers work?",
        length="balanced",
        style="executive",
    )
    assert "Bottom Line" in res_exec["answer"]
    assert "[1]" in res_exec["answer"] or "[2]" in res_exec["answer"]

    # Query in Step-by-Step style
    res_steps = engine.query(
        question="Explain the surface code and cryogenic interface",
        length="longer",
        style="step_by_step",
    )
    assert "1." in res_steps["answer"]


def test_fastapi_endpoints_and_sse_streaming(tmp_path: Path):
    pdfs_dir = tmp_path / "pdfs"
    data_dir = tmp_path / "data"
    pdfs_dir.mkdir()
    data_dir.mkdir()

    (pdfs_dir / "handbook.txt").write_text(
        "Pure RAG supports macOS, Windows, and Linux with hybrid BM25 and dense vector search.",
        encoding="utf-8",
    )

    cfg = PureRAGConfig(
        pdfs_dir=str(pdfs_dir),
        data_dir=str(data_dir),
        llm_provider="extractive_fallback",
        embedding_backend="hashed_ngram",
    )
    app = create_app(cfg)
    client = TestClient(app)

    # 1. Health check
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["engine"]["total_documents"] == 1

    # 2. Hardware check
    r_hw = client.get("/api/hardware")
    assert r_hw.status_code == 200
    assert "recommended_tier" in r_hw.json()["hardware"]

    # 3. Chat synchronous endpoint
    r_chat = client.post(
        "/api/chat",
        json={
            "question": "Which operating systems does Pure RAG support?",
            "length": "shorter",
            "style": "citations_only",
        },
    )
    assert r_chat.status_code == 200
    chat_data = r_chat.json()
    assert "macOS" in chat_data["answer"]
    assert len(chat_data["citations"]) >= 1

    # 4. Chat SSE Streaming endpoint
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={
            "question": "What search method does Pure RAG use?",
            "length": "balanced",
            "style": "bulleted",
        },
    ) as stream_resp:
        assert stream_resp.status_code == 200
        events = []
        for line in stream_resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
        event_types = {e["type"] for e in events}
        assert "metadata" in event_types
        assert "token" in event_types
        assert "done" in event_types
