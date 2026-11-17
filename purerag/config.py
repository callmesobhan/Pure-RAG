"""
Persistent configuration and runtime settings for Pure RAG.
Stores user permissions, selected hardware tier, model choices, and directory paths.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


PACKAGE_ROOT_DIR = Path(__file__).resolve().parent.parent


def _resolve_default_workspace() -> Path:
    if os.environ.get("PURERAG_HOME"):
        return Path(os.environ["PURERAG_HOME"]).resolve()
    cwd = Path.cwd().resolve()
    # If running from filesystem root '/' or an unwritable directory, use the repo root
    if str(cwd) == "/" or not os.access(str(cwd), os.W_OK):
        return PACKAGE_ROOT_DIR
    # If repo root has pdfs/ or pyproject.toml and cwd doesn't, prefer repo root
    if (PACKAGE_ROOT_DIR / "pyproject.toml").exists() and not (cwd / "pdfs").exists():
        return PACKAGE_ROOT_DIR
    return cwd


DEFAULT_WORKSPACE_DIR = _resolve_default_workspace()


@dataclass
class PureRAGConfig:
    # Directories
    pdfs_dir: str = field(default_factory=lambda: str(DEFAULT_WORKSPACE_DIR / "pdfs"))
    data_dir: str = field(default_factory=lambda: str(DEFAULT_WORKSPACE_DIR / "data"))

    # Hardware & Permission state
    hardware_profiled: bool = False
    setup_approved: bool = False
    selected_tier: str = "balanced"

    # LLM settings
    llm_provider: str = "ollama"  # "ollama", "openai_compat", "extractive_fallback"
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen2.5:3b"
    openai_api_base: str = "https://api.openai.com/v1"
    openai_api_key: str = ""

    # Embedding & Retrieval settings (SOTA Hybrid + Rerank)
    embedding_backend: str = "fastembed"  # "fastembed", "ollama", "hashed_ngram"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    ollama_embedding_model: str = "nomic-embed-text"
    use_reranker: bool = True
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"

    # Chunking & Hybrid RRF parameters
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 220
    dense_top_k: int = 25
    bm25_top_k: int = 25
    rrf_k: int = 60
    dense_weight: float = 0.55
    bm25_weight: float = 0.45
    final_top_k: int = 6

    # Default Promptability settings
    default_length: str = "balanced"  # "shorter", "balanced", "longer"
    default_style: str = "standard"   # "standard", "precise", "bulleted", "executive", "step_by_step", "eli5", "citations_only"
    default_temperature: float = 0.2
    custom_system_prompt: str = ""

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def config_path(self) -> Path:
        return Path(self.data_dir) / "config.json"

    def ensure_dirs(self) -> None:
        Path(self.pdfs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

    def save(self, path: Optional[Path] = None) -> Path:
        self.ensure_dirs()
        target = path or self.config_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        return target

    @classmethod
    def load(cls, data_dir: Optional[str] = None) -> "PureRAGConfig":
        resolved_data_dir = Path(data_dir) if data_dir else (DEFAULT_WORKSPACE_DIR / "data")
        cfg_file = resolved_data_dir / "config.json"
        if cfg_file.exists():
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                valid_keys = {k for k in cls.__dataclass_fields__.keys()}
                filtered = {k: v for k, v in raw.items() if k in valid_keys}
                cfg = cls(**filtered)
                cfg.ensure_dirs()
                return cfg
            except Exception:
                pass
        cfg = cls(
            pdfs_dir=str(DEFAULT_WORKSPACE_DIR / "pdfs"),
            data_dir=str(resolved_data_dir),
        )
        cfg.ensure_dirs()
        return cfg

    def to_public_dict(self) -> Dict:
        data = asdict(self)
        if data.get("openai_api_key"):
            key = data["openai_api_key"]
            data["openai_api_key"] = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "***"
        return data
