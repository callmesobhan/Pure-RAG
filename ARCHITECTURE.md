# Pure RAG — Deep Architectural Specification

## 1. Design Philosophy

Pure RAG is architected around four core principles:
1. **SOTA Retrieval Fidelity Over Naive Splitting**: Combines layout-aware Markdown extraction, Contextual Breadcrumb Enrichment, Hybrid Dense + Sparse (BM25 Okapi) search with Reciprocal Rank Fusion (RRF), and Cross-Encoder second-stage reranking.
2. **Hardware Empathy & Permissioned Setup**: Never assumes the user has an H100 GPU or silently downloads multi-gigabyte weights. Profiles the host machine (OS, CPU, RAM, Disk, Apple Metal / NVIDIA CUDA / AMD ROCm) and presents tailored model tiers for user approval.
3. **Human & AI-Agent Parity**: Every workflow available in the Web UI is also available via a deterministic, scriptable CLI (`purerag`) with `--yes` and `--json` flags.
4. **Resilient Zero-Dependency Fallback**: Works out-of-the-box offline immediately upon install via a 512-dimensional signed feature-hashed N-Gram vector embedder, BM25 Okapi lexical search, Cross-Proximity reranker, and Promptability-Aware Extractive Synthesizer, seamlessly upgrading to ONNX (`fastembed`, `flashrank`) and local LLMs (`Ollama`, `vLLM`, `LM Studio`) when enabled.

---

## 2. Module Map

```text
Pure-RAG/
├── purerag/
│   ├── __init__.py          # Package metadata & version
│   ├── config.py            # Persistent config & hardware tier state (data/config.json)
│   ├── hardware.py          # Cross-platform OS/CPU/RAM/GPU/Ollama analyzer & tier recommender
│   ├── server.py            # FastAPI REST + Server-Sent Events (SSE) streaming backend
│   ├── cli.py               # Rich interactive & AI-agent CLI (`purerag`)
│   ├── rag/
│   │   ├── parser.py        # Stage 1: PyMuPDF4LLM / pypdf / Markdown page-aware extractor
│   │   ├── chunker.py       # Stage 2 & 3: Structure-aware chunker + Contextual Breadcrumbs
│   │   ├── retriever.py     # Stage 4, 5 & 6: Dense + BM25 + RRF + Cross-Encoder Reranker
│   │   ├── generator.py     # Stage 7: Promptability modifiers + Ollama/OpenAI/Extractive synthesis
│   │   └── engine.py        # Incremental folder synchronization & persistence orchestrator
│   └── static/
│       ├── index.html       # Chatbot UI with Voice, Promptability Bar, & Hardware Wizard
│       ├── styles.css       # Responsive dark/light glassmorphic design system
│       └── app.js           # Streaming SSE client, Web Speech STT/TTS, & drag-and-drop uploader
├── pdfs/                    # Designated drop-folder for user PDFs, Markdown, and TXT files
├── tests/
│   └── test_purerag.py      # End-to-end automated unit & API test suite
├── install.sh / run.sh      # macOS & Linux one-step installer & launcher
├── install.ps1 / run.bat    # Windows PowerShell & CMD installer & launcher
├── AGENTS.md                # AI-Agent installation & JSON CLI guide
└── README.md                # User & developer documentation
```
