"""
FastAPI Backend Server for Pure RAG.
Provides REST & Server-Sent Events (SSE) endpoints for:
- Hardware profiling & permissioned model setup (`/api/hardware`, `/api/setup/approve`, `/api/ollama/pull`)
- Folder scanning, incremental PDF ingestion & drag-and-drop upload (`/api/documents`, `/api/ingest`, `/api/upload`)
- Grounded Chat with Promptability modifiers & SSE streaming (`/api/chat`, `/api/chat/stream`)
- Static Web UI serving (`/`)
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from purerag.config import PureRAGConfig
from purerag.hardware import HARDWARE_TIERS, analyze_hardware
from purerag.rag.engine import PureRAGEngine
from purerag.rag.parser import SUPPORTED_EXTENSIONS


STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question or prompt")
    session_id: Optional[str] = Field(default=None, description="Optional chat session ID to group turns")
    length: Optional[str] = Field(default=None, description="shorter | balanced | longer")
    style: Optional[str] = Field(
        default=None,
        description="standard | precise | bulleted | executive | step_by_step | eli5 | citations_only",
    )
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.5)
    custom_instruction: Optional[str] = Field(default=None)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    filter_filenames: Optional[List[str]] = Field(default=None)
    chat_history: Optional[List[Dict[str, str]]] = Field(default=None)


class RenameSessionRequest(BaseModel):
    title: str = Field(..., min_length=1)


class SetupApprovalRequest(BaseModel):
    tier_id: str = Field(default="balanced")
    approved: bool = Field(default=True)
    llm_provider: Optional[str] = Field(default=None)
    llm_model: Optional[str] = Field(default=None)
    embedding_backend: Optional[str] = Field(default=None)
    embedding_model: Optional[str] = Field(default=None)
    use_reranker: Optional[bool] = Field(default=None)
    pull_ollama_model: bool = Field(default=False)
    openai_api_base: Optional[str] = Field(default=None)
    openai_api_key: Optional[str] = Field(default=None)


class ConfigUpdateRequest(BaseModel):
    default_length: Optional[str] = None
    default_style: Optional[str] = None
    default_temperature: Optional[float] = None
    custom_system_prompt: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    ollama_base_url: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_api_key: Optional[str] = None
    use_reranker: Optional[bool] = None
    final_top_k: Optional[int] = None


class OllamaPullRequest(BaseModel):
    model: str


def create_app(config: Optional[PureRAGConfig] = None) -> FastAPI:
    cfg = config or PureRAGConfig.load()
    engine = PureRAGEngine(cfg)
    # Initial non-blocking scan of the pdfs/ folder on startup
    engine.scan_and_ingest()

    app = FastAPI(
        title="Pure RAG — Local SOTA RAG Pipeline & AI Chatbot",
        version="1.0.0",
        description="High-Precision Hybrid RAG with Cross-Encoder Reranking, Promptability, Voice, and Hardware-Aware Setup.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.config = cfg
    app.state.engine = engine

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index_page():
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return HTMLResponse(index_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Pure RAG API Server is running.</h1>")

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "version": "1.0.0",
            "engine": app.state.engine.status_summary(),
        }

    @app.get("/api/hardware")
    async def get_hardware_profile():
        profile = analyze_hardware(
            workspace_dir=Path(app.state.config.pdfs_dir).parent,
            ollama_base_url=app.state.config.ollama_base_url,
        )
        return {
            "hardware": asdict(profile),
            "config": app.state.config.to_public_dict(),
        }

    @app.post("/api/setup/approve")
    async def approve_setup(req: SetupApprovalRequest):
        c: PureRAGConfig = app.state.config
        tier = HARDWARE_TIERS.get(req.tier_id, HARDWARE_TIERS["balanced"])

        c.hardware_profiled = True
        c.setup_approved = req.approved
        c.selected_tier = tier.tier_id
        c.llm_model = req.llm_model or tier.llm_model
        c.embedding_backend = req.embedding_backend or tier.embedding_backend
        c.embedding_model = req.embedding_model or tier.embedding_model
        c.use_reranker = req.use_reranker if req.use_reranker is not None else tier.reranker_enabled
        c.reranker_model = tier.reranker_model
        c.chunk_size_chars = tier.chunk_size_chars
        c.final_top_k = tier.final_top_k

        if req.llm_provider:
            c.llm_provider = req.llm_provider
        if req.openai_api_base:
            c.openai_api_base = req.openai_api_base
        if req.openai_api_key is not None and not req.openai_api_key.endswith("***"):
            c.openai_api_key = req.openai_api_key

        c.save()
        app.state.engine.reload_components_from_config()

        pull_result = None
        if req.approved and req.pull_ollama_model and c.llm_provider == "ollama":
            try:
                r = requests.post(
                    f"{c.ollama_base_url.rstrip('/')}/api/pull",
                    json={"name": c.llm_model, "stream": False},
                    timeout=300.0,
                )
                pull_result = {"status": "pulled" if r.status_code == 200 else "error", "code": r.status_code}
            except Exception as exc:
                pull_result = {"status": "skipped_ollama_unreachable", "detail": str(exc)}

        return {
            "status": "saved",
            "config": c.to_public_dict(),
            "engine": app.state.engine.status_summary(),
            "ollama_pull": pull_result,
        }

    @app.post("/api/ollama/pull")
    async def pull_ollama_model_stream(req: OllamaPullRequest):
        base_url = app.state.config.ollama_base_url.rstrip("/")

        def event_stream():
            try:
                with requests.post(
                    f"{base_url}/api/pull",
                    json={"name": req.model, "stream": True},
                    stream=True,
                    timeout=600.0,
                ) as resp:
                    if resp.status_code != 200:
                        yield f"data: {json.dumps({'status': 'error', 'detail': f'HTTP {resp.status_code}'})}\n\n"
                        return
                    for line in resp.iter_lines():
                        if line:
                            yield f"data: {line.decode('utf-8')}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'status': 'error', 'detail': str(exc)})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/config")
    async def get_config():
        return app.state.config.to_public_dict()

    @app.post("/api/config")
    async def update_config(req: ConfigUpdateRequest):
        c: PureRAGConfig = app.state.config
        updates = req.model_dump(exclude_none=True)
        for k, v in updates.items():
            if k == "openai_api_key" and str(v).endswith("***"):
                continue
            if hasattr(c, k):
                setattr(c, k, v)
        c.save()
        app.state.engine.reload_components_from_config()
        return {"status": "updated", "config": c.to_public_dict()}

    @app.get("/api/documents")
    async def list_documents():
        return app.state.engine.status_summary()

    @app.post("/api/ingest")
    async def trigger_ingest(force: bool = False):
        summary = app.state.engine.scan_and_ingest(force_reindex=force)
        return {
            "status": "indexed",
            "ingest_summary": summary,
            "engine": app.state.engine.status_summary(),
        }

    @app.post("/api/upload")
    async def upload_files(files: List[UploadFile] = File(...)):
        pdfs_dir = Path(app.state.config.pdfs_dir)
        pdfs_dir.mkdir(parents=True, exist_ok=True)
        saved_files: List[str] = []
        rejected_files: List[str] = []

        for upload in files:
            fname = Path(upload.filename or "document.pdf").name
            ext = Path(fname).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                rejected_files.append(fname)
                continue
            dest = pdfs_dir / fname
            with open(dest, "wb") as out_f:
                shutil.copyfileobj(upload.file, out_f)
            saved_files.append(fname)

        ingest_summary = app.state.engine.scan_and_ingest(force_reindex=False)
        return {
            "saved_files": saved_files,
            "rejected_files": rejected_files,
            "ingest_summary": ingest_summary,
            "engine": app.state.engine.status_summary(),
        }

    @app.delete("/api/documents/{filename}")
    async def delete_document(filename: str):
        safe_name = Path(filename).name
        target = Path(app.state.config.pdfs_dir) / safe_name
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"File '{safe_name}' not found in pdfs/ folder.")
        target.unlink()
        ingest_summary = app.state.engine.scan_and_ingest(force_reindex=False)
        return {
            "deleted": safe_name,
            "ingest_summary": ingest_summary,
            "engine": app.state.engine.status_summary(),
        }

    @app.get("/api/documents/{filename}/view")
    async def view_document(filename: str):
        safe_name = Path(filename).name
        target = Path(app.state.config.pdfs_dir) / safe_name
        if not target.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(str(target), filename=safe_name)

    @app.post("/api/chat")
    async def chat_endpoint(req: ChatRequest):
        result = app.state.engine.query(
            question=req.question,
            length=req.length,
            style=req.style,
            temperature=req.temperature,
            custom_instruction=req.custom_instruction,
            top_k=req.top_k,
            filter_filenames=req.filter_filenames,
            chat_history=req.chat_history,
            session_id=req.session_id,
        )
        return result

    @app.post("/api/chat/stream")
    async def chat_stream_endpoint(req: ChatRequest):
        generator = app.state.engine.stream_query(
            question=req.question,
            length=req.length,
            style=req.style,
            temperature=req.temperature,
            custom_instruction=req.custom_instruction,
            top_k=req.top_k,
            filter_filenames=req.filter_filenames,
            chat_history=req.chat_history,
            session_id=req.session_id,
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    # ==================== Q&A HISTORY & METADATA MANAGER API ====================
    @app.get("/api/history")
    async def list_history(q: Optional[str] = None):
        sessions = app.state.engine.history.list_sessions(search_query=q)
        turns = app.state.engine.history.list_all_turns(search_query=q)
        return {
            "total_sessions": len(sessions),
            "total_turns": len(turns),
            "sessions": sessions,
            "turns": turns,
        }

    @app.get("/api/history/export")
    async def export_history(format: str = "json", session_id: Optional[str] = None):
        if format.lower() in ("md", "markdown"):
            md_text = app.state.engine.history.export_markdown(session_id=session_id)
            return PlainTextResponse(
                md_text,
                headers={"Content-Disposition": 'attachment; filename="purerag_qa_history.md"'},
            )
        if session_id:
            sess = app.state.engine.history.get_session(session_id)
            if not sess:
                raise HTTPException(status_code=404, detail="Session not found")
            return sess
        return {
            "sessions": [
                app.state.engine.history.get_session(s["session_id"])
                for s in app.state.engine.history.list_sessions()
            ]
        }

    @app.get("/api/history/{session_id}")
    async def get_history_session(session_id: str):
        sess = app.state.engine.history.get_session(session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        return sess

    @app.patch("/api/history/{session_id}")
    async def rename_history_session(session_id: str, req: RenameSessionRequest):
        updated = app.state.engine.history.rename_session(session_id, req.title)
        if not updated:
            raise HTTPException(status_code=404, detail="Session not found")
        return updated

    @app.delete("/api/history/{session_id}")
    async def delete_history_session(session_id: str):
        ok = app.state.engine.history.delete_session(session_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": session_id}

    @app.delete("/api/history/{session_id}/turns/{turn_id}")
    async def delete_history_turn(session_id: str, turn_id: str):
        ok = app.state.engine.history.delete_turn(session_id, turn_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Turn not found")
        return {"deleted_turn": turn_id, "session_id": session_id}

    @app.delete("/api/history")
    async def clear_all_history():
        cleared = app.state.engine.history.clear_all()
        return {"cleared_sessions": cleared}

    return app
