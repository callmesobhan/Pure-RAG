"""
Cross-Platform Hardware Profiler, Local Model Discovery Engine, & Model Recommender.

Key Capabilities:
1. Hardware Profiling across macOS (Apple Silicon Metal / Intel), Windows (NVIDIA CUDA / DirectML / CPU),
   and Linux (NVIDIA CUDA / AMD ROCm / CPU).
2. Local Model Auto-Discovery (Zero-Download Reuse):
   - Scans Ollama models via HTTP API (`/api/tags`), CLI (`ollama list`), AND direct disk manifests
     (`~/.ollama/models/manifests/...` or `$OLLAMA_MODELS`) so installed Ollama models are detected
     even inside sandboxes or when the Ollama daemon isn't running yet.
   - Scans HuggingFace local hub cache (`~/.cache/huggingface/hub/models--*` or `$HF_HOME`) for already
     downloaded embedding models (`e5-*`, `bge-*`, `nomic-*`, `snowflake-arctic-embed-*`, `qwen3-embedding-*`,
     `gte-*`, `minilm-*`) and cross-encoder rerankers (`bge-reranker-v2-m3`, `bge-reranker-base`, `ms-marco-*`).
   - Scans FastEmbed & FlashRank local caches (`/tmp/fastembed_cache`, `~/.cache/fastembed`, `/tmp/flashrank`).
   - Scans LM Studio / GGUF local model directories (`~/.lmstudio/models`, `~/.cache/lm-studio/models`).
3. Up-to-Date 2025–2026 Frontier Open-Weight RAM Tier Recommendations:
   - Ultra-Light (<8 GB RAM): `qwen3:1.7b` (alt: `gemma3:1b`, `qwen2.5:1.5b`, `llama3.2:1b`) + `Snowflake/snowflake-arctic-embed-s` / `BAAI/bge-small-en-v1.5`
   - Balanced (8–15 GB RAM): `qwen3:4b` (alt: `gemma3:4b`, `deepseek-r1:8b`, `llama3.2:3b`) + `Snowflake/snowflake-arctic-embed-m` / `BAAI/bge-base-en-v1.5` + `BAAI/bge-reranker-base`
   - Performance (16–31 GB RAM): `gemma4:12b` / `qwen3:8b` (alt: `gemma3:12b`, `phi4:14b`, `deepseek-r1:14b`, `mistral:latest`) + `BAAI/bge-m3` / `intfloat/e5-large-v2` + `BAAI/bge-reranker-v2-m3`
   - Workstation (32+ GB RAM): `qwen3:30b-a3b` / `qwen3:32b` (alt: `gemma3:27b`, `deepseek-r1:32b`, `llama3.3:70b`) + `Qwen/Qwen3-Embedding-4B` / `BAAI/bge-m3` + `BAAI/bge-reranker-v2-m3`
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import requests


@dataclass
class DiscoveredLocalModels:
    ollama_llms: List[str] = field(default_factory=list)
    ollama_embeddings: List[str] = field(default_factory=list)
    hf_embeddings: List[str] = field(default_factory=list)
    hf_rerankers: List[str] = field(default_factory=list)
    hf_other_models: List[str] = field(default_factory=list)
    lmstudio_ggufs: List[str] = field(default_factory=list)
    best_local_llm: Optional[str] = None
    best_local_embedder: Optional[str] = None
    best_local_embedder_backend: Optional[str] = None
    best_local_reranker: Optional[str] = None


@dataclass
class TierRecommendation:
    tier_id: str
    name: str
    badge: str
    min_ram_gb: float
    llm_model: str
    llm_alternatives: List[str]
    llm_size_gb: float
    embedding_backend: str
    embedding_model: str
    embedding_alternatives: List[str]
    embedding_size_mb: int
    reranker_enabled: bool
    reranker_model: str
    reranker_alternatives: List[str]
    reranker_size_mb: int
    chunk_size_chars: int
    final_top_k: int
    description: str
    why_recommended: str


HARDWARE_TIERS: Dict[str, TierRecommendation] = {
    "ultra_light": TierRecommendation(
        tier_id="ultra_light",
        name="Ultra-Light (Edge / <8GB RAM)",
        badge="⚡ Fast & Compact (2026 SOTA Small)",
        min_ram_gb=4.0,
        llm_model="qwen3:1.7b",
        llm_alternatives=["gemma3:1b", "qwen3:0.6b", "llama3.2:1b", "qwen2.5:1.5b"],
        llm_size_gb=1.1,
        embedding_backend="fastembed",
        embedding_model="Snowflake/snowflake-arctic-embed-s",
        embedding_alternatives=["BAAI/bge-small-en-v1.5", "nomic-ai/nomic-embed-text-v1.5"],
        embedding_size_mb=130,
        reranker_enabled=False,
        reranker_model="ms-marco-MiniLM-L-12-v2",
        reranker_alternatives=["BAAI/bge-reranker-base"],
        reranker_size_mb=0,
        chunk_size_chars=1000,
        final_top_k=4,
        description=(
            "Engineered for <8 GB RAM machines. Pairs Qwen3 1.7B (32K context, hybrid thinking) "
            "or Gemma 3 1B with Snowflake Arctic Embed + RRF Hybrid Search."
        ),
        why_recommended="Keeps total RAM footprint under 2 GB while leveraging modern 2025–2026 small reasoning weights.",
    ),
    "balanced": TierRecommendation(
        tier_id="balanced",
        name="Balanced (8–15GB RAM Laptop / Desktop)",
        badge="⚖️ Best Everyday SOTA",
        min_ram_gb=8.0,
        llm_model="qwen3:4b",
        llm_alternatives=["gemma3:4b", "deepseek-r1:8b", "llama3.2:3b", "mistral:latest"],
        llm_size_gb=2.6,
        embedding_backend="fastembed",
        embedding_model="Snowflake/snowflake-arctic-embed-m",
        embedding_alternatives=["BAAI/bge-base-en-v1.5", "nomic-ai/nomic-embed-text-v1.5"],
        embedding_size_mb=270,
        reranker_enabled=True,
        reranker_model="BAAI/bge-reranker-base",
        reranker_alternatives=["ms-marco-MiniLM-L-12-v2", "BAAI/bge-reranker-v2-m3"],
        reranker_size_mb=275,
        chunk_size_chars=1200,
        final_top_k=5,
        description=(
            "Ideal for 8–15 GB RAM systems. Uses Qwen3 4B or Gemma 3 4B (which rival older 14B models) "
            "+ Arctic Embed + Cross-Encoder Reranking."
        ),
        why_recommended="Delivers high instruction-following fidelity and 3-stage hybrid reranking without memory pressure.",
    ),
    "performance": TierRecommendation(
        tier_id="performance",
        name="Performance (16–31GB RAM / Apple Silicon / GPU)",
        badge="🚀 Frontier Local RAG (16GB+)",
        min_ram_gb=16.0,
        llm_model="gemma4:12b",
        llm_alternatives=["qwen3:8b", "qwen3:14b", "gemma3:12b", "phi4:14b", "deepseek-r1:14b", "mistral:latest"],
        llm_size_gb=7.5,
        embedding_backend="fastembed",
        embedding_model="BAAI/bge-m3",
        embedding_alternatives=["intfloat/e5-large-v2", "Snowflake/snowflake-arctic-embed-l", "BAAI/bge-large-en-v1.5"],
        embedding_size_mb=560,
        reranker_enabled=True,
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_alternatives=["BAAI/bge-reranker-base", "ms-marco-MiniLM-L-12-v2"],
        reranker_size_mb=560,
        chunk_size_chars=1400,
        final_top_k=6,
        description=(
            "Recommended for 16–31 GB RAM, Apple M-series Macs, or 8–16 GB VRAM GPUs. "
            "Runs Gemma 4 12B / Qwen3 8B–14B / Phi-4 14B + BGE-M3 / E5-Large + BGE-Reranker-v2-M3."
        ),
        why_recommended="Your 16GB+ unified/GPU memory effortlessly runs 8B–14B frontier open models and multilingual M3 reranking.",
    ),
    "workstation": TierRecommendation(
        tier_id="workstation",
        name="Workstation Pro (32GB+ RAM / High VRAM)",
        badge="🏆 Maximum MoE & Deep Reasoning",
        min_ram_gb=32.0,
        llm_model="qwen3:30b-a3b",
        llm_alternatives=["qwen3:32b", "gemma3:27b", "deepseek-r1:32b", "llama3.3:70b"],
        llm_size_gb=18.0,
        embedding_backend="fastembed",
        embedding_model="BAAI/bge-m3",
        embedding_alternatives=["Qwen/Qwen3-Embedding-4B", "intfloat/e5-large-v2", "BAAI/bge-large-en-v1.5"],
        embedding_size_mb=1200,
        reranker_enabled=True,
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_alternatives=["BAAI/bge-reranker-large", "BAAI/bge-reranker-base"],
        reranker_size_mb=560,
        chunk_size_chars=1600,
        final_top_k=8,
        description=(
            "Built for 32+ GB RAM workstations and 24GB+ VRAM GPUs. Uses Qwen3 30B-A3B (MoE: 30B capacity at 3B active speed!), "
            "Qwen3 32B, or Gemma 3 27B + BGE-M3 + BGE-Reranker-v2-M3."
        ),
        why_recommended="Unlocks frontier-class multi-document synthesis, MoE inference speed, and deep table/citation reasoning.",
    ),
}


@dataclass
class HardwareProfile:
    os_name: str               # "macOS", "Windows", "Linux"
    os_release: str
    architecture: str          # "arm64", "x86_64", etc.
    cpu_brand: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    gpu_detected: bool
    gpu_name: str
    gpu_acceleration: str      # "apple_metal", "nvidia_cuda", "amd_rocm", "cpu_only"
    gpu_vram_gb: Optional[float]
    ollama_installed: bool
    ollama_running: bool
    ollama_models: List[str] = field(default_factory=list)
    discovered_local_models: Optional[Dict] = None
    recommended_tier: str = "balanced"
    recommendation_details: Optional[Dict] = None
    active_zero_download_plan: Optional[Dict] = None
    all_tiers: Dict[str, Dict] = field(default_factory=dict)


def _run_cmd(cmd: List[str], timeout: float = 3.5) -> str:
    try:
        out = subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, timeout=timeout, text=True
        )
        return out.strip()
    except Exception:
        return ""


def _detect_cpu_brand(sys_name: str) -> str:
    if sys_name == "Darwin":
        brand = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            return brand
    elif sys_name == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    elif sys_name == "Windows":
        brand = _run_cmd(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
        if brand:
            return brand
    return platform.processor() or platform.machine() or "Generic CPU"


def _detect_gpu(sys_name: str, arch: str, cpu_brand: str, ram_total_gb: float) -> tuple[bool, str, str, Optional[float]]:
    """Returns (gpu_detected, gpu_name, gpu_acceleration, gpu_vram_gb)."""
    if sys_name == "Darwin" and (arch in ("arm64", "aarch64") or "Apple" in cpu_brand):
        usable_vram = round(ram_total_gb * 0.75, 1)
        return True, f"{cpu_brand} (Apple Metal Unified GPU)", "apple_metal", usable_vram

    if shutil.which("nvidia-smi"):
        smi = _run_cmd([
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        if smi:
            first_line = smi.splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            gpu_name = parts[0] if parts else "NVIDIA GPU"
            vram_gb = None
            if len(parts) > 1:
                try:
                    vram_gb = round(float(parts[1]) / 1024.0, 1)
                except ValueError:
                    pass
            return True, gpu_name, "nvidia_cuda", vram_gb

    if sys_name == "Linux" and shutil.which("rocm-smi"):
        return True, "AMD Radeon GPU (ROCm)", "amd_rocm", None

    if sys_name == "Windows":
        win_gpu = _run_cmd([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_VideoController | Select-Object -First 1).Name",
        ])
        if win_gpu and not any(x in win_gpu.lower() for x in ("basic", "microsoft remote")):
            accel = (
                "nvidia_cuda"
                if "nvidia" in win_gpu.lower()
                else ("amd_rocm" if "amd" in win_gpu.lower() or "radeon" in win_gpu.lower() else "directml")
            )
            return True, win_gpu, accel, None

    return False, "CPU Only (No Dedicated GPU Detected)", "cpu_only", None


def get_live_telemetry() -> Dict:
    """
    Returns real-time system resource utilization across macOS, Windows, and Linux:
    - CPU utilization (%)
    - GPU utilization (%) & GPU memory used (GB) [Apple Metal via ioreg, NVIDIA via nvidia-smi, AMD via rocm-smi]
    - System RAM used / total (GB) & (%)
    - Swap RAM used / total (GB) & (%)
    """
    # 1. CPU % (non-blocking instant sample from psutil interval state)
    cpu_pct = round(float(psutil.cpu_percent(interval=None)), 1)

    # 2. System RAM
    vm = psutil.virtual_memory()
    ram_total_gb = round(vm.total / (1024 ** 3), 1)
    ram_used_gb = round((vm.total - vm.available) / (1024 ** 3), 1)
    ram_pct = round(float(vm.percent), 1)

    # 3. Swap RAM
    sw = psutil.swap_memory()
    swap_total_gb = round(sw.total / (1024 ** 3), 2)
    swap_used_gb = round(sw.used / (1024 ** 3), 2)
    swap_pct = round(float(sw.percent), 1) if sw.total > 0 else 0.0

    # 4. GPU Utilization (%) & Memory
    sys_name = platform.system()
    gpu_pct: Optional[float] = None
    gpu_mem_used_gb: Optional[float] = None
    gpu_mem_total_gb: Optional[float] = None
    gpu_label = "CPU"

    if sys_name == "Darwin":
        gpu_label = "Metal GPU"
        out = _run_cmd(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], timeout=1.2)
        if not out or "PerformanceStatistics" not in out:
            out = _run_cmd(["ioreg", "-l"], timeout=1.5)
        if out and "PerformanceStatistics" in out:
            m_util = re.search(r'"Device Utilization %"\s*=\s*(\d+)', out)
            if not m_util:
                m_util = re.search(r'"Renderer Utilization %"\s*=\s*(\d+)', out)
            if m_util:
                gpu_pct = float(m_util.group(1))
            m_mem = re.search(r'"In use system memory"\s*=\s*(\d+)', out)
            if m_mem:
                gpu_mem_used_gb = round(int(m_mem.group(1)) / (1024 ** 3), 2)
        if gpu_pct is None:
            gpu_pct = 0.0
    elif shutil.which("nvidia-smi"):
        gpu_label = "CUDA GPU"
        smi = _run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=1.5,
        )
        if smi:
            first = smi.splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            try:
                gpu_pct = float(parts[0])
                if len(parts) >= 3:
                    gpu_mem_used_gb = round(float(parts[1]) / 1024.0, 2)
                    gpu_mem_total_gb = round(float(parts[2]) / 1024.0, 2)
            except Exception:
                pass

    return {
        "cpu_percent": cpu_pct,
        "gpu_percent": gpu_pct,
        "gpu_label": gpu_label,
        "gpu_mem_used_gb": gpu_mem_used_gb,
        "gpu_mem_total_gb": gpu_mem_total_gb,
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "ram_percent": ram_pct,
        "swap_used_gb": swap_used_gb,
        "swap_total_gb": swap_total_gb,
        "swap_percent": swap_pct,
    }


def _scan_ollama_manifests_on_disk() -> List[str]:
    """
    Directly inspects `~/.ollama/models/manifests` (or `$OLLAMA_MODELS/manifests`) on disk
    so Pure RAG detects installed Ollama models even when the Ollama HTTP server isn't running
    or when running inside a restricted sandbox!
    """
    candidates: List[Path] = []
    if os.environ.get("OLLAMA_MODELS"):
        candidates.append(Path(os.environ["OLLAMA_MODELS"]) / "manifests")
    candidates.append(Path.home() / ".ollama" / "models" / "manifests")
    if platform.system() == "Windows":
        candidates.append(Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".ollama" / "models" / "manifests")

    discovered: List[str] = []
    seen = set()

    for manifest_root in candidates:
        if not manifest_root.exists():
            continue
        # Standard structure: manifests/registry.ollama.ai/library/<model>/<tag>
        # or manifests/registry.ollama.ai/<namespace>/<model>/<tag>
        try:
            for file_path in manifest_root.rglob("*"):
                if not file_path.is_file() or file_path.name.startswith("."):
                    continue
                rel_parts = file_path.relative_to(manifest_root).parts
                # e.g. ('registry.ollama.ai', 'library', 'gemma4', '12b')
                if len(rel_parts) >= 3:
                    tag = rel_parts[-1]
                    model_name = rel_parts[-2]
                    namespace = rel_parts[-3]
                    if namespace == "library":
                        full_name = f"{model_name}:{tag}"
                    else:
                        full_name = f"{namespace}/{model_name}:{tag}"
                    if full_name not in seen:
                        seen.add(full_name)
                        discovered.append(full_name)
        except Exception:
            pass

    return discovered


def check_ollama_status(base_url: str = "http://127.0.0.1:11434") -> tuple[bool, bool, List[str]]:
    """Returns (installed, running, list_of_installed_models) using API + CLI + Disk Manifests."""
    installed = shutil.which("ollama") is not None
    if not installed:
        common_paths = [
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
            str(Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"),
        ]
        installed = any(Path(p).exists() for p in common_paths)

    models: List[str] = []
    running = False

    # 1. Try Ollama HTTP API
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=1.8)
        if resp.status_code == 200:
            running = True
            installed = True
            data = resp.json()
            for m in data.get("models", []):
                name = m.get("name", "")
                if name and name not in models:
                    models.append(name)
    except Exception:
        pass

    # 2. Try `ollama list` CLI if HTTP returned nothing
    if not models and installed:
        cli_out = _run_cmd(["ollama", "list"], timeout=2.5)
        if cli_out:
            running = True
            for line in cli_out.splitlines()[1:]:
                parts = line.split()
                if parts:
                    name = parts[0].strip()
                    if name and name not in models:
                        models.append(name)

    # 3. Always merge with direct disk manifest scan (`~/.ollama/models/manifests`)
    disk_models = _scan_ollama_manifests_on_disk()
    if disk_models:
        installed = True
        for dm in disk_models:
            if dm not in models:
                models.append(dm)

    return installed, running, models


def _scan_huggingface_and_local_caches() -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Scans HuggingFace Hub cache (`~/.cache/huggingface/hub`), FastEmbed cache,
    FlashRank cache, and LM Studio models directory for already-downloaded models on disk.
    Returns (hf_embeddings, hf_rerankers, hf_other_models, lmstudio_ggufs).
    """
    hf_dirs: List[Path] = []
    if os.environ.get("HF_HUB_CACHE"):
        hf_dirs.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        hf_dirs.append(Path(os.environ["HF_HOME"]) / "hub")
    hf_dirs.append(Path.home() / ".cache" / "huggingface" / "hub")
    hf_dirs.append(Path("/tmp/fastembed_cache"))
    hf_dirs.append(Path.home() / ".cache" / "fastembed")

    hf_embeddings: List[str] = []
    hf_rerankers: List[str] = []
    hf_other: List[str] = []
    seen_hf = set()

    for hub_dir in hf_dirs:
        if not hub_dir.exists():
            continue
        try:
            for entry in sorted(hub_dir.iterdir()):
                if not entry.is_dir() or not entry.name.startswith("models--"):
                    continue
                # Ensure the cached model actually has snapshot weights (not an empty folder)
                snapshots_dir = entry / "snapshots"
                blobs_dir = entry / "blobs"
                has_weights = (snapshots_dir.exists() and any(snapshots_dir.iterdir())) or (
                    blobs_dir.exists() and any(blobs_dir.iterdir())
                )
                if not has_weights:
                    continue

                repo_id = entry.name[len("models--") :].replace("--", "/")
                if repo_id in seen_hf:
                    continue
                seen_hf.add(repo_id)

                lower_id = repo_id.lower()
                if "rerank" in lower_id or "ms-marco" in lower_id or "cross-encoder" in lower_id:
                    hf_rerankers.append(repo_id)
                elif any(
                    kw in lower_id
                    for kw in (
                        "embed",
                        "bge-",
                        "e5-",
                        "gte-",
                        "minilm",
                        "mpnet",
                        "arctic-embed",
                        "nomic",
                        "instructor",
                    )
                ):
                    hf_embeddings.append(repo_id)
                else:
                    hf_other.append(repo_id)
        except Exception:
            pass

    # Also check FlashRank cache in /tmp or ~/.cache/flashrank
    for fr_dir in [Path("/tmp"), Path.home() / ".cache" / "flashrank"]:
        if fr_dir.exists():
            for model_name in ("ms-marco-MiniLM-L-12-v2", "ms-marco-MultiBERT-L-12", "rank-T5-flan"):
                if (fr_dir / model_name).is_dir() and model_name not in hf_rerankers:
                    hf_rerankers.append(model_name)

    # Scan LM Studio local GGUF models (`~/.lmstudio/models` or `~/.cache/lm-studio/models`)
    lmstudio_ggufs: List[str] = []
    for lms_dir in [Path.home() / ".lmstudio" / "models", Path.home() / ".cache" / "lm-studio" / "models"]:
        if not lms_dir.exists():
            continue
        try:
            for gguf_file in lms_dir.rglob("*.gguf"):
                rel = str(gguf_file.relative_to(lms_dir))
                if rel not in lmstudio_ggufs:
                    lmstudio_ggufs.append(rel)
        except Exception:
            pass

    return hf_embeddings, hf_rerankers, hf_other, lmstudio_ggufs


def _score_local_llm_for_rag(model_name: str, ram_total_gb: float) -> float:
    """
    Scores an already-installed local Ollama LLM based on model generation modernity
    (Gemma 4 / Qwen 3 / Llama 3.3 / DeepSeek R1 / Phi 4 / Gemma 3 / Qwen 2.5 / Mistral)
    and whether its parameter size fits comfortably in the device's RAM.
    """
    lower = model_name.lower()
    if any(k in lower for k in ("embed", "bge", "nomic-embed", "arctic-embed", "e5-")):
        return -1.0

    # Estimate parameter size in billions from tag (e.g., ":12b", ":7b", ":1.5b")
    param_b = 7.0  # default assumption for `:latest`
    m = re.search(r"[:_-](\d+(?:\.\d+)?)b\b", lower)
    if m:
        try:
            param_b = float(m.group(1))
        except ValueError:
            pass

    # Estimate Q4 memory requirement (~0.65 GB per 1B parameters + 1.2GB KV cache overhead)
    est_ram_gb = param_b * 0.65 + 1.2
    if est_ram_gb > ram_total_gb * 0.88:
        # Penalize models that might swap heavily on this RAM tier
        fit_score = 10.0
    else:
        # Prefer the most capable model that fits comfortably in RAM
        fit_score = min(45.0, param_b * 3.0)

    # Family quality bonus (2025-2026 frontier open weights ranked first)
    family_bonus = 20.0
    if any(f in lower for f in ("qwen3", "gemma4", "llama4")):
        family_bonus = 55.0
    elif any(f in lower for f in ("gemma3", "llama3.3", "phi4", "deepseek-r1", "qwen2.5")):
        family_bonus = 48.0
    elif any(f in lower for f in ("llama3.2", "llama3.1", "mistral", "mixtral", "command-r")):
        family_bonus = 40.0

    return family_bonus + fit_score


def discover_local_models(
    ollama_models: List[str],
    ram_total_gb: float,
) -> DiscoveredLocalModels:
    ollama_llms: List[str] = []
    ollama_embeds: List[str] = []

    for m in ollama_models:
        low = m.lower()
        if any(k in low for k in ("embed", "bge-", "all-minilm", "snowflake-arctic-embed", "mxbai-embed")):
            ollama_embeds.append(m)
        else:
            ollama_llms.append(m)

    hf_embeddings, hf_rerankers, hf_other, lmstudio_ggufs = _scan_huggingface_and_local_caches()

    # Rank local Ollama LLMs to pick the best already-installed LLM
    best_llm: Optional[str] = None
    if ollama_llms:
        scored = sorted(
            ollama_llms,
            key=lambda name: _score_local_llm_for_rag(name, ram_total_gb),
            reverse=True,
        )
        best_llm = scored[0]

    # Pick the best already-downloaded local embedding model
    best_embedder: Optional[str] = None
    best_embedder_backend: Optional[str] = None
    if ollama_embeds:
        best_embedder = ollama_embeds[0]
        best_embedder_backend = "ollama"
    elif hf_embeddings:
        # Prefer bge-m3, qwen3-embedding, arctic-embed, e5-large-v2, bge-large, bge-base, bge-small
        priority_order = ("bge-m3", "qwen3-embedding", "arctic-embed", "e5-large", "bge-large", "bge-base", "bge-small")
        chosen = hf_embeddings[0]
        for pref in priority_order:
            match = next((x for x in hf_embeddings if pref in x.lower()), None)
            if match:
                chosen = match
                break
        best_embedder = chosen
        best_embedder_backend = "local_hf_or_fastembed"

    # Pick the best already-downloaded local cross-encoder reranker
    best_reranker: Optional[str] = None
    if hf_rerankers:
        rerank_priority = ("bge-reranker-v2-m3", "bge-reranker-large", "bge-reranker-base", "ms-marco-minilm")
        chosen_r = hf_rerankers[0]
        for pref in rerank_priority:
            match = next((x for x in hf_rerankers if pref in x.lower()), None)
            if match:
                chosen_r = match
                break
        best_reranker = chosen_r

    return DiscoveredLocalModels(
        ollama_llms=ollama_llms,
        ollama_embeddings=ollama_embeds,
        hf_embeddings=hf_embeddings,
        hf_rerankers=hf_rerankers,
        hf_other_models=hf_other,
        lmstudio_ggufs=lmstudio_ggufs,
        best_local_llm=best_llm,
        best_local_embedder=best_embedder,
        best_local_embedder_backend=best_embedder_backend,
        best_local_reranker=best_reranker,
    )


def select_recommended_tier(
    ram_total_gb: float,
    gpu_detected: bool,
    gpu_acceleration: str,
    gpu_vram_gb: Optional[float],
) -> str:
    if gpu_acceleration == "nvidia_cuda" and gpu_vram_gb is not None:
        if gpu_vram_gb >= 16.0 and ram_total_gb >= 24.0:
            return "workstation"
        if gpu_vram_gb >= 8.0 and ram_total_gb >= 14.0:
            return "performance"

    if ram_total_gb >= 30.0 and gpu_detected:
        return "workstation"
    if ram_total_gb >= 15.0:
        return "performance"
    if ram_total_gb >= 7.5:
        return "balanced"
    return "ultra_light"


def analyze_hardware(
    workspace_dir: Optional[Path] = None,
    ollama_base_url: str = "http://127.0.0.1:11434",
) -> HardwareProfile:
    sys_raw = platform.system()
    os_map = {"Darwin": "macOS", "Windows": "Windows", "Linux": "Linux"}
    os_name = os_map.get(sys_raw, sys_raw)
    os_release = platform.release()
    arch = platform.machine()

    vm = psutil.virtual_memory()
    ram_total_gb = round(vm.total / (1024 ** 3), 1)
    ram_available_gb = round(vm.available / (1024 ** 3), 1)

    target_dir = workspace_dir or Path.cwd()
    try:
        disk = psutil.disk_usage(str(target_dir))
        disk_free_gb = round(disk.free / (1024 ** 3), 1)
    except Exception:
        disk_free_gb = 50.0

    cpu_cores_physical = psutil.cpu_count(logical=False) or 2
    cpu_cores_logical = psutil.cpu_count(logical=True) or cpu_cores_physical
    cpu_brand = _detect_cpu_brand(sys_raw)

    gpu_detected, gpu_name, gpu_acceleration, gpu_vram_gb = _detect_gpu(
        sys_raw, arch, cpu_brand, ram_total_gb
    )

    ollama_installed, ollama_running, ollama_models = check_ollama_status(ollama_base_url)

    # Auto-discover all local models already on the device
    discovered = discover_local_models(ollama_models=ollama_models, ram_total_gb=ram_total_gb)

    rec_tier_id = select_recommended_tier(
        ram_total_gb=ram_total_gb,
        gpu_detected=gpu_detected,
        gpu_acceleration=gpu_acceleration,
        gpu_vram_gb=gpu_vram_gb,
    )

    # Customize the tier recommendation to prioritize ALREADY-INSTALLED local models first!
    base_tier = HARDWARE_TIERS[rec_tier_id]
    rec_dict = asdict(base_tier)

    reused_notes: List[str] = []
    if discovered.best_local_llm:
        rec_dict["llm_model"] = discovered.best_local_llm
        rec_dict["llm_already_installed"] = True
        reused_notes.append(f"Reusing local Ollama LLM '{discovered.best_local_llm}' (0 GB download)")
    else:
        rec_dict["llm_already_installed"] = False

    if discovered.best_local_embedder:
        rec_dict["embedding_model"] = discovered.best_local_embedder
        rec_dict["embedding_already_installed"] = True
        reused_notes.append(f"Reusing cached local embedder '{discovered.best_local_embedder}' (0 MB download)")
    else:
        rec_dict["embedding_already_installed"] = False

    if discovered.best_local_reranker:
        rec_dict["reranker_model"] = discovered.best_local_reranker
        rec_dict["reranker_enabled"] = True
        rec_dict["reranker_already_installed"] = True
        reused_notes.append(f"Reusing cached local reranker '{discovered.best_local_reranker}' (0 MB download)")
    else:
        rec_dict["reranker_already_installed"] = False

    if reused_notes:
        rec_dict["why_recommended"] = (
            f"✨ Local Models Detected! {' • '.join(reused_notes)}. "
            + base_tier.why_recommended
        )

    zero_download_plan = {
        "has_local_llm": discovered.best_local_llm is not None,
        "local_llm_to_use": discovered.best_local_llm,
        "all_local_llms": discovered.ollama_llms,
        "has_local_embedder": discovered.best_local_embedder is not None,
        "local_embedder_to_use": discovered.best_local_embedder,
        "has_local_reranker": discovered.best_local_reranker is not None,
        "local_reranker_to_use": discovered.best_local_reranker,
        "requires_any_download": not (
            discovered.best_local_llm and discovered.best_local_embedder
        ),
    }

    return HardwareProfile(
        os_name=os_name,
        os_release=os_release,
        architecture=arch,
        cpu_brand=cpu_brand,
        cpu_cores_physical=cpu_cores_physical,
        cpu_cores_logical=cpu_cores_logical,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        disk_free_gb=disk_free_gb,
        gpu_detected=gpu_detected,
        gpu_name=gpu_name,
        gpu_acceleration=gpu_acceleration,
        gpu_vram_gb=gpu_vram_gb,
        ollama_installed=ollama_installed,
        ollama_running=ollama_running,
        ollama_models=ollama_models,
        discovered_local_models=asdict(discovered),
        recommended_tier=rec_tier_id,
        recommendation_details=rec_dict,
        active_zero_download_plan=zero_download_plan,
        all_tiers={k: asdict(v) for k, v in HARDWARE_TIERS.items()},
    )
