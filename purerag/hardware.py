"""
Cross-Platform Hardware Profiler & Model Recommendation Engine.
Works across macOS (Apple Silicon Metal / Intel), Windows (NVIDIA CUDA / DirectML / CPU),
and Linux (NVIDIA CUDA / AMD ROCm / CPU).
Analyzes RAM, CPU cores, Disk space, GPU acceleration, and local Ollama availability
to recommend the optimal RAG stack and asks for user permission before installing.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import psutil
import requests


@dataclass
class TierRecommendation:
    tier_id: str
    name: str
    badge: str
    min_ram_gb: float
    llm_model: str
    llm_size_gb: float
    embedding_backend: str
    embedding_model: str
    embedding_size_mb: int
    reranker_enabled: bool
    reranker_model: str
    reranker_size_mb: int
    chunk_size_chars: int
    final_top_k: int
    description: str
    why_recommended: str


HARDWARE_TIERS: Dict[str, TierRecommendation] = {
    "ultra_light": TierRecommendation(
        tier_id="ultra_light",
        name="Ultra-Light (Edge / Low-RAM)",
        badge="⚡ Fast & Compact",
        min_ram_gb=4.0,
        llm_model="qwen2.5:1.5b",
        llm_size_gb=1.0,
        embedding_backend="fastembed",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_size_mb=130,
        reranker_enabled=False,
        reranker_model="ms-marco-MiniLM-L-12-v2",
        reranker_size_mb=0,
        chunk_size_chars=1000,
        final_top_k=4,
        description="Designed for machines with <8 GB RAM or older CPUs. Uses Qwen 2.5 1.5B + BGE-Small + RRF Hybrid Search.",
        why_recommended="Keeps total memory footprint under 2.2 GB while preserving hybrid semantic + BM25 search.",
    ),
    "balanced": TierRecommendation(
        tier_id="balanced",
        name="Balanced (Everyday Laptop / Desktop)",
        badge="⚖️ Best Balance",
        min_ram_gb=8.0,
        llm_model="llama3.2:3b",
        llm_size_gb=2.0,
        embedding_backend="fastembed",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_size_mb=130,
        reranker_enabled=True,
        reranker_model="ms-marco-MiniLM-L-12-v2",
        reranker_size_mb=120,
        chunk_size_chars=1200,
        final_top_k=5,
        description="Ideal for 8–15 GB RAM systems. Enables full 3-stage SOTA retrieval (Dense + BM25 + Cross-Encoder Reranker) with Llama 3.2 3B.",
        why_recommended="Delivers fast token generation and high retrieval precision without slowing down other desktop apps.",
    ),
    "performance": TierRecommendation(
        tier_id="performance",
        name="Performance (Apple Silicon / GPU / 16GB+ RAM)",
        badge="🚀 High Accuracy (SOTA)",
        min_ram_gb=16.0,
        llm_model="qwen2.5:7b",
        llm_size_gb=4.7,
        embedding_backend="fastembed",
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_size_mb=420,
        reranker_enabled=True,
        reranker_model="ms-marco-MiniLM-L-12-v2",
        reranker_size_mb=120,
        chunk_size_chars=1400,
        final_top_k=6,
        description="Recommended for 16–31 GB RAM or dedicated GPU / Apple M-series Macs. Runs Qwen 2.5 7B Instruct + BGE-Base + Cross-Encoder Reranking.",
        why_recommended="Your system has sufficient RAM/acceleration to run a 7B parameter model with deep cross-encoder reranking for near-zero hallucination.",
    ),
    "workstation": TierRecommendation(
        tier_id="workstation",
        name="Workstation Pro (32GB+ RAM / High VRAM)",
        badge="🏆 Maximum Reasoning",
        min_ram_gb=32.0,
        llm_model="qwen2.5:14b",
        llm_size_gb=9.0,
        embedding_backend="fastembed",
        embedding_model="BAAI/bge-large-en-v1.5",
        embedding_size_mb=1200,
        reranker_enabled=True,
        reranker_model="ms-marco-MiniLM-L-12-v2",
        reranker_size_mb=120,
        chunk_size_chars=1600,
        final_top_k=8,
        description="Built for 32+ GB RAM workstations and high-VRAM GPUs. Uses Qwen 2.5 14B + BGE-Large embeddings + Cross-Encoder Reranking.",
        why_recommended="Unlocks deep multi-document synthesis, complex table reasoning, and maximum MTEB embedding recall.",
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
    recommended_tier: str = "balanced"
    recommendation_details: Optional[Dict] = None
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
    # 1. Check Apple Silicon (Unified Memory Metal GPU)
    if sys_name == "Darwin" and (arch in ("arm64", "aarch64") or "Apple" in cpu_brand):
        # Apple Silicon uses unified memory; ~75% is usable by Metal GPU
        usable_vram = round(ram_total_gb * 0.75, 1)
        return True, f"{cpu_brand} (Apple Metal Unified GPU)", "apple_metal", usable_vram

    # 2. Check NVIDIA GPU via nvidia-smi (Windows & Linux)
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

    # 3. Check AMD ROCm on Linux
    if sys_name == "Linux" and shutil.which("rocm-smi"):
        return True, "AMD Radeon GPU (ROCm)", "amd_rocm", None

    # 4. Check Windows GPU via PowerShell if nvidia-smi wasn't found
    if sys_name == "Windows":
        win_gpu = _run_cmd([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_VideoController | Select-Object -First 1).Name",
        ])
        if win_gpu and not any(x in win_gpu.lower() for x in ("basic", "microsoft remote")):
            accel = "nvidia_cuda" if "nvidia" in win_gpu.lower() else ("amd_rocm" if "amd" in win_gpu.lower() or "radeon" in win_gpu.lower() else "directml")
            return True, win_gpu, accel, None

    return False, "CPU Only (No Dedicated GPU Detected)", "cpu_only", None


def check_ollama_status(base_url: str = "http://127.0.0.1:11434") -> tuple[bool, bool, List[str]]:
    """Returns (installed, running, list_of_installed_models)."""
    installed = shutil.which("ollama") is not None
    # Also check common macOS/Windows install paths if not on PATH
    if not installed:
        common_paths = [
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
            str(Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"),
        ]
        installed = any(Path(p).exists() for p in common_paths)

    models: List[str] = []
    running = False
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            running = True
            installed = True
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        pass

    return installed, running, models


def select_recommended_tier(
    ram_total_gb: float,
    gpu_detected: bool,
    gpu_acceleration: str,
    gpu_vram_gb: Optional[float],
) -> str:
    effective_mem = ram_total_gb
    if gpu_acceleration == "nvidia_cuda" and gpu_vram_gb is not None:
        # If dedicated NVIDIA GPU has 16GB+ VRAM or system has 32GB+ RAM
        if gpu_vram_gb >= 12.0 and ram_total_gb >= 24.0:
            return "workstation"
        if gpu_vram_gb >= 6.0 and ram_total_gb >= 14.0:
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

    rec_tier_id = select_recommended_tier(
        ram_total_gb=ram_total_gb,
        gpu_detected=gpu_detected,
        gpu_acceleration=gpu_acceleration,
        gpu_vram_gb=gpu_vram_gb,
    )
    rec_tier = HARDWARE_TIERS[rec_tier_id]

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
        recommended_tier=rec_tier_id,
        recommendation_details=asdict(rec_tier),
        all_tiers={k: asdict(v) for k, v in HARDWARE_TIERS.items()},
    )
