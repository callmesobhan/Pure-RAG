# AGENTS.md — AI Agent Installation & Operation Guide for Pure RAG

This repository (`Pure-RAG`) is explicitly engineered to be **zero-friction for AI agents** (Claude Code, Antigravity, Cursor, Codex, OpenDevin, Devin, SWE-agent, etc.) as well as human users across **macOS, Windows, and Linux**.

---

## 1. Non-Interactive One-Command Setup (AI Agents)

Never block on interactive prompts when running as an AI agent. Always pass `--yes` (or `-Yes` on Windows PowerShell).

### macOS & Linux
```bash
bash ./install.sh --yes
```

### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Yes
```

### Direct Python Virtual Environment Setup (Cross-Platform)
```bash
python3 -m venv .venv
./.venv/bin/pip install -e .
./.venv/bin/purerag setup --yes --tier auto --ingest
```

---

## 2. CLI Commands for AI Agents (JSON Output Supported)

Once installed, use `./.venv/bin/purerag` (or `.venv\Scripts\purerag.exe` on Windows):

1. **Inspect Hardware & Active Configuration (JSON)**:
   ```bash
   ./.venv/bin/purerag doctor --json
   ```

2. **Ingest / Re-index the Designated `pdfs/` Folder (JSON)**:
   ```bash
   ./.venv/bin/purerag ingest --json
   ```
   *To index a custom folder:*
   ```bash
   ./.venv/bin/purerag ingest --pdfs-dir /path/to/pdfs --json
   ```

3. **Query the RAG Pipeline Programmatically (JSON)**:
   ```bash
   ./.venv/bin/purerag query "What are the main findings?" --length shorter --style bulleted --top-k 5 --format json
   ```
   *Supported `--length` values:* `shorter`, `balanced`, `longer`
   *Supported `--style` values:* `standard`, `precise`, `bulleted`, `executive`, `step_by_step`, `eli5`, `citations_only`

4. **Start the Local Web App & REST API Server**:
   ```bash
   ./.venv/bin/purerag serve --host 127.0.0.1 --port 8000 --no-browser --yes
   ```

---

## 3. REST API Endpoints (When Server is Running)

- `GET /api/health` — Returns server health and indexed document/chunk counts.
- `GET /api/hardware` — Returns full OS/CPU/RAM/GPU hardware profile and tier recommendations.
- `POST /api/setup/approve` — Programmatically approve a hardware tier (`ultra_light`, `balanced`, `performance`, `workstation`).
- `GET /api/documents` — Lists all indexed PDFs/docs in `pdfs/`.
- `POST /api/ingest` — Triggers incremental folder scan & indexing.
- `POST /api/upload` — Multipart file upload into `pdfs/` + automatic indexing.
- `POST /api/chat` — Synchronous grounded RAG query returning `{ "answer": "...", "citations": [...], "pipeline_metrics": {...} }`.
- `POST /api/chat/stream` — Server-Sent Events (SSE) streaming RAG endpoint.
