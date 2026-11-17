"""
Persistent Chat & Q&A History Manager for Pure RAG.
Stores every conversation session and every individual Q&A turn along with
its full retrieval/generation metadata:
- Session ID, auto-generated title, creation & update timestamps
- Question text, Answer text, Timestamp
- Promptability settings (`length`, `style`, `temperature`, `top_k`, `custom_instruction`, `filter_filenames`)
- Full numbered Citations list (`rank`, `filename`, `page_number`, `section_path`, `snippet`, `final_score`, `rrf_score`, `dense_score`, `bm25_score`, `rerank_score`)
- Pipeline Execution Metrics (`retrieval_ms`, `generation_ms`, `embedder`, `reranker`, `generator`, `chunks_searched`)

Supports:
- Listing, loading, renaming, and deleting chat sessions
- Searching across all past questions, answers, and cited filenames
- Exporting any session or the entire Q&A log as JSON or Markdown
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class QATurnRecord:
    turn_id: str
    session_id: str
    created_at: float
    created_at_iso: str
    question: str
    answer: str
    citations: List[Dict] = field(default_factory=list)
    promptability: Dict = field(default_factory=dict)
    pipeline_metrics: Dict = field(default_factory=dict)
    filter_filenames: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ChatSessionRecord:
    session_id: str
    title: str
    created_at: float
    created_at_iso: str
    updated_at: float
    updated_at_iso: str
    turns: List[QATurnRecord] = field(default_factory=list)

    def to_summary_dict(self) -> Dict:
        cited_docs = sorted(
            {
                c.get("filename", "")
                for t in self.turns
                for c in (t.citations or [])
                if c.get("filename")
            }
        )
        last_question = self.turns[-1].question if self.turns else ""
        last_generator = (
            self.turns[-1].pipeline_metrics.get("generator", "")
            if self.turns and self.turns[-1].pipeline_metrics
            else ""
        )
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "created_at_iso": self.created_at_iso,
            "updated_at": self.updated_at,
            "updated_at_iso": self.updated_at_iso,
            "turn_count": len(self.turns),
            "cited_documents": cited_docs,
            "last_question": last_question,
            "last_generator": last_generator,
        }

    def to_full_dict(self) -> Dict:
        data = self.to_summary_dict()
        data["turns"] = [t.to_dict() for t in self.turns]
        return data


class HistoryManager:
    """
    Thread-safe persistent manager for all Pure RAG chat sessions, questions,
    answers, citations, and pipeline metadata.
    Persisted at `<data_dir>/chat_history.json`.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.data_dir / "chat_history.json"
        self._lock = threading.RLock()
        self.sessions: Dict[str, ChatSessionRecord] = {}
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not self.history_path.exists():
                return
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                loaded_sessions: Dict[str, ChatSessionRecord] = {}
                for s_data in raw.get("sessions", []):
                    turns = [
                        QATurnRecord(
                            turn_id=t.get("turn_id", str(uuid.uuid4())[:8]),
                            session_id=s_data["session_id"],
                            created_at=float(t.get("created_at", time.time())),
                            created_at_iso=t.get("created_at_iso", _iso_now()),
                            question=t.get("question", ""),
                            answer=t.get("answer", ""),
                            citations=t.get("citations", []),
                            promptability=t.get("promptability", {}),
                            pipeline_metrics=t.get("pipeline_metrics", {}),
                            filter_filenames=t.get("filter_filenames"),
                        )
                        for t in s_data.get("turns", [])
                    ]
                    sess = ChatSessionRecord(
                        session_id=s_data["session_id"],
                        title=s_data.get("title", "Untitled Chat"),
                        created_at=float(s_data.get("created_at", time.time())),
                        created_at_iso=s_data.get("created_at_iso", _iso_now()),
                        updated_at=float(s_data.get("updated_at", time.time())),
                        updated_at_iso=s_data.get("updated_at_iso", _iso_now()),
                        turns=turns,
                    )
                    loaded_sessions[sess.session_id] = sess
                self.sessions = loaded_sessions
            except Exception:
                self.sessions = {}

    def _save(self) -> None:
        with self._lock:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            ordered = sorted(
                self.sessions.values(), key=lambda s: s.updated_at, reverse=True
            )
            payload = {
                "updated_at": time.time(),
                "total_sessions": len(ordered),
                "total_turns": sum(len(s.turns) for s in ordered),
                "sessions": [s.to_full_dict() for s in ordered],
            }
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

    @staticmethod
    def _generate_title(question: str) -> str:
        clean = " ".join(question.strip().split())
        if not clean:
            return "New Conversation"
        return clean[:58] + ("…" if len(clean) > 58 else "")

    def create_session(self, title: Optional[str] = None, session_id: Optional[str] = None) -> ChatSessionRecord:
        with self._lock:
            sid = session_id or f"sess_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
            if sid in self.sessions:
                return self.sessions[sid]
            now = time.time()
            iso = _iso_now()
            sess = ChatSessionRecord(
                session_id=sid,
                title=title or "New Chat",
                created_at=now,
                created_at_iso=iso,
                updated_at=now,
                updated_at_iso=iso,
                turns=[],
            )
            self.sessions[sid] = sess
            self._save()
            return sess

    def record_turn(
        self,
        question: str,
        answer: str,
        citations: List[Dict],
        promptability: Dict,
        pipeline_metrics: Dict,
        session_id: Optional[str] = None,
        filter_filenames: Optional[List[str]] = None,
    ) -> tuple[ChatSessionRecord, QATurnRecord]:
        with self._lock:
            now = time.time()
            iso = _iso_now()
            if session_id and session_id in self.sessions:
                sess = self.sessions[session_id]
                if not sess.turns and sess.title in ("New Chat", "Untitled Chat"):
                    sess.title = self._generate_title(question)
            else:
                sid = session_id or f"sess_{int(now * 1000)}_{uuid.uuid4().hex[:6]}"
                sess = ChatSessionRecord(
                    session_id=sid,
                    title=self._generate_title(question),
                    created_at=now,
                    created_at_iso=iso,
                    updated_at=now,
                    updated_at_iso=iso,
                    turns=[],
                )
                self.sessions[sid] = sess

            turn = QATurnRecord(
                turn_id=f"turn_{int(now * 1000)}_{uuid.uuid4().hex[:5]}",
                session_id=sess.session_id,
                created_at=now,
                created_at_iso=iso,
                question=question,
                answer=answer,
                citations=citations,
                promptability=promptability,
                pipeline_metrics=pipeline_metrics,
                filter_filenames=filter_filenames,
            )
            sess.turns.append(turn)
            sess.updated_at = now
            sess.updated_at_iso = iso
            self._save()
            return sess, turn

    def list_sessions(self, search_query: Optional[str] = None) -> List[Dict]:
        with self._lock:
            ordered = sorted(
                self.sessions.values(), key=lambda s: s.updated_at, reverse=True
            )
            if not search_query or not search_query.strip():
                return [s.to_summary_dict() for s in ordered]

            q = search_query.strip().lower()
            matched: List[Dict] = []
            for s in ordered:
                if q in s.title.lower():
                    matched.append(s.to_summary_dict())
                    continue
                for t in s.turns:
                    if (
                        q in t.question.lower()
                        or q in t.answer.lower()
                        or any(q in c.get("filename", "").lower() for c in t.citations)
                    ):
                        matched.append(s.to_summary_dict())
                        break
            return matched

    def list_all_turns(self, search_query: Optional[str] = None) -> List[Dict]:
        """Returns a flat chronological list of every Q&A turn and its metadata across all sessions."""
        with self._lock:
            all_turns: List[Dict] = []
            for s in self.sessions.values():
                for t in s.turns:
                    item = t.to_dict()
                    item["session_title"] = s.title
                    all_turns.append(item)

            all_turns.sort(key=lambda x: x["created_at"], reverse=True)

            if not search_query or not search_query.strip():
                return all_turns

            q = search_query.strip().lower()
            return [
                t
                for t in all_turns
                if q in t["question"].lower()
                or q in t["answer"].lower()
                or q in t.get("session_title", "").lower()
                or any(q in c.get("filename", "").lower() for c in t.get("citations", []))
                or q in str(t.get("pipeline_metrics", {}).get("generator", "")).lower()
            ]

    def get_session(self, session_id: str) -> Optional[Dict]:
        with self._lock:
            sess = self.sessions.get(session_id)
            return sess.to_full_dict() if sess else None

    def rename_session(self, session_id: str, new_title: str) -> Optional[Dict]:
        with self._lock:
            sess = self.sessions.get(session_id)
            if not sess:
                return None
            sess.title = new_title.strip() or sess.title
            sess.updated_at = time.time()
            sess.updated_at_iso = _iso_now()
            self._save()
            return sess.to_summary_dict()

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id not in self.sessions:
                return False
            del self.sessions[session_id]
            self._save()
            return True

    def delete_turn(self, session_id: str, turn_id: str) -> bool:
        with self._lock:
            sess = self.sessions.get(session_id)
            if not sess:
                return False
            before = len(sess.turns)
            sess.turns = [t for t in sess.turns if t.turn_id != turn_id]
            if len(sess.turns) == before:
                return False
            if not sess.turns:
                del self.sessions[session_id]
            else:
                sess.updated_at = time.time()
                sess.updated_at_iso = _iso_now()
            self._save()
            return True

    def clear_all(self) -> int:
        with self._lock:
            count = len(self.sessions)
            self.sessions.clear()
            self._save()
            return count

    def export_markdown(self, session_id: Optional[str] = None) -> str:
        with self._lock:
            if session_id:
                sess = self.sessions.get(session_id)
                target_sessions = [sess] if sess else []
            else:
                target_sessions = sorted(
                    self.sessions.values(), key=lambda s: s.updated_at, reverse=True
                )

            lines: List[str] = ["# Pure RAG — Q&A History & Metadata Export\n"]
            for s in target_sessions:
                if not s:
                    continue
                lines.append(f"## Session: {s.title}")
                lines.append(f"- **Session ID:** `{s.session_id}`")
                lines.append(f"- **Created:** `{s.created_at_iso}` | **Updated:** `{s.updated_at_iso}`")
                lines.append(f"- **Total Q&A Turns:** {len(s.turns)}\n")

                for idx, t in enumerate(s.turns, start=1):
                    pm = t.pipeline_metrics or {}
                    pr = t.promptability or {}
                    lines.append(f"### Turn #{idx} — {t.created_at_iso}")
                    lines.append(
                        f"**Metadata:** Length=`{pr.get('length', 'balanced')}` | "
                        f"Style=`{pr.get('style', 'standard')}` | "
                        f"Top-K=`{pr.get('top_k', 6)}` | "
                        f"Generator=`{pm.get('generator', 'N/A')}` | "
                        f"Embedder=`{pm.get('embedder', 'N/A')}` | "
                        f"Reranker=`{pm.get('reranker', 'N/A')}` | "
                        f"Retrieval=`{pm.get('retrieval_ms', 0)}ms` | "
                        f"Generation=`{pm.get('generation_ms', 0)}ms`\n"
                    )
                    lines.append(f"**Question:**\n> {t.question}\n")
                    lines.append(f"**Answer:**\n{t.answer}\n")
                    if t.citations:
                        lines.append("**Cited Sources:**")
                        for c in t.citations:
                            lines.append(
                                f"- `[{c.get('rank')}]` **{c.get('filename')}** "
                                f"(Page {c.get('page_number')} — {c.get('section_path')}) "
                                f"• Score: `{c.get('final_score')}` "
                                f"(Dense: `{c.get('dense_score')}`, BM25: `{c.get('bm25_score')}`, Rerank: `{c.get('rerank_score')}`)"
                            )
                    lines.append("\n---\n")
            return "\n".join(lines)
