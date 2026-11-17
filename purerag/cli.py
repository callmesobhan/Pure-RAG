"""
Pure RAG — Unified Command-Line Interface & Permissioned Hardware Installer.
Designed for both Human interactive use and AI-Agent non-interactive automation (`--yes`, `--json`).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from purerag.config import PureRAGConfig
from purerag.hardware import HARDWARE_TIERS, analyze_hardware
from purerag.rag.engine import PureRAGEngine


console = Console()


def _print_hardware_report(hw_dict: dict) -> None:
    table = Table(title="🖥️  Pure RAG — Device Capability & Local Model Analysis", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="bold")
    table.add_column("Detected Value")

    table.add_row("Operating System", f"{hw_dict['os_name']} {hw_dict['os_release']} ({hw_dict['architecture']})")
    table.add_row("Processor (CPU)", f"{hw_dict['cpu_brand']} ({hw_dict['cpu_cores_physical']} physical / {hw_dict['cpu_cores_logical']} logical cores)")
    table.add_row("System Memory (RAM)", f"{hw_dict['ram_total_gb']} GB Total ({hw_dict['ram_available_gb']} GB Available)")
    table.add_row("Free Disk Space", f"{hw_dict['disk_free_gb']} GB Free")
    table.add_row("GPU Acceleration", f"{hw_dict['gpu_name']} [{hw_dict['gpu_acceleration']}]")

    disc = hw_dict.get("discovered_local_models") or {}
    local_llms = disc.get("ollama_llms") or []
    local_embeds = (disc.get("ollama_embeddings") or []) + (disc.get("hf_embeddings") or [])
    local_rerankers = disc.get("hf_rerankers") or []

    table.add_row(
        "Installed Local LLMs",
        f"✅ {', '.join(local_llms)} (Best: [bold green]{disc.get('best_local_llm')}[/bold green])"
        if local_llms
        else "ℹ️ None found on disk",
    )
    table.add_row(
        "Cached Local Embedders",
        f"✅ {', '.join(local_embeds)} (Best: [bold green]{disc.get('best_local_embedder')}[/bold green])"
        if local_embeds
        else "ℹ️ None cached yet (Built-in Hashed N-Gram ready)",
    )
    table.add_row(
        "Cached Local Rerankers",
        f"✅ {', '.join(local_rerankers)} (Best: [bold green]{disc.get('best_local_reranker')}[/bold green])"
        if local_rerankers
        else "ℹ️ None cached yet (Built-in Cross-Proximity ready)",
    )
    console.print(table)

    rec = hw_dict["recommendation_details"]
    llm_tag = " [bold green](Already Installed — 0 GB Download!)[/bold green]" if rec.get("llm_already_installed") else f" (~{rec['llm_size_gb']} GB if downloaded)"
    emb_tag = " [bold green](Already Cached — 0 MB Download!)[/bold green]" if rec.get("embedding_already_installed") else f" (~{rec['embedding_size_mb']} MB if downloaded)"
    rerank_tag = (
        " [bold green](Already Cached — 0 MB Download!)[/bold green]"
        if rec.get("reranker_already_installed")
        else (f" (~{rec['reranker_size_mb']} MB if downloaded)" if rec["reranker_enabled"] else "")
    )

    console.print(
        Panel(
            f"[bold green]Recommended Tier:[/bold green] {rec['name']} ({rec['badge']})\n"
            f"[bold]Active/Recommended Local LLM:[/bold] {rec['llm_model']}{llm_tag}\n"
            f"[bold]Dense Embeddings:[/bold] {rec['embedding_model']}{emb_tag}\n"
            f"[bold]Cross-Encoder Reranker:[/bold] {'Enabled (' + rec['reranker_model'] + ')' + rerank_tag if rec['reranker_enabled'] else 'Disabled'}\n\n"
            f"[italic]{rec['why_recommended']}[/italic]",
            title="🎯 Optimal RAG Stack (Local Models Prioritized)",
            border_style="green",
        )
    )


def run_interactive_or_agent_setup(
    cfg: PureRAGConfig,
    yes: bool = False,
    tier_override: str = "auto",
    pull_ollama: bool = False,
) -> PureRAGConfig:
    hw = analyze_hardware(workspace_dir=Path(cfg.pdfs_dir).parent, ollama_base_url=cfg.ollama_base_url)
    hw_dict = asdict(hw)
    _print_hardware_report(hw_dict)

    chosen_tier_id = hw.recommended_tier if tier_override == "auto" else tier_override
    if chosen_tier_id not in HARDWARE_TIERS:
        chosen_tier_id = hw.recommended_tier

    if not yes and sys.stdin.isatty():
        console.print("\n[bold]Available Hardware Tiers:[/bold]")
        for tid, t in HARDWARE_TIERS.items():
            marker = " (Recommended)" if tid == hw.recommended_tier else ""
            console.print(f"  • [cyan]{tid}[/cyan]: {t.name}{marker} — LLM: {t.llm_model}, Embed: {t.embedding_model}")

        chosen_tier_id = Prompt.ask(
            "\nSelect your preferred RAG tier",
            choices=list(HARDWARE_TIERS.keys()),
            default=chosen_tier_id,
        )

        tier_obj = HARDWARE_TIERS[chosen_tier_id]
        approved = Confirm.ask(
            f"\nGrant permission to configure Pure RAG with '{tier_obj.name}' "
            f"(Embedder: {tier_obj.embedding_model} ~{tier_obj.embedding_size_mb}MB, "
            f"Reranker: {'Yes' if tier_obj.reranker_enabled else 'No'})?",
            default=True,
        )
    else:
        approved = True

    tier_obj = HARDWARE_TIERS[chosen_tier_id]
    disc = hw_dict.get("discovered_local_models") or {}

    # Always prefer already-present local models over downloading new ones!
    selected_llm = disc.get("best_local_llm") or tier_obj.llm_model
    selected_embedder = disc.get("best_local_embedder") or tier_obj.embedding_model
    selected_embed_backend = (
        disc.get("best_local_embedder_backend")
        if disc.get("best_local_embedder")
        else (tier_obj.embedding_backend if approved else "hashed_ngram")
    )
    selected_reranker = disc.get("best_local_reranker") or tier_obj.reranker_model

    cfg.hardware_profiled = True
    cfg.setup_approved = approved
    cfg.selected_tier = tier_obj.tier_id
    cfg.llm_model = selected_llm
    cfg.embedding_backend = selected_embed_backend or "fastembed"
    cfg.embedding_model = selected_embedder
    cfg.use_reranker = True if disc.get("best_local_reranker") else tier_obj.reranker_enabled
    cfg.reranker_model = selected_reranker
    cfg.chunk_size_chars = tier_obj.chunk_size_chars
    cfg.final_top_k = tier_obj.final_top_k
    cfg.save()

    console.print(f"[bold green]✓ Saved configuration to {cfg.config_path} (LLM: {cfg.llm_model}, Embedder: {cfg.embedding_model}, Reranker: {cfg.reranker_model})[/bold green]")

    # Only offer to pull an Ollama model if NO local LLM is already installed!
    if approved and not yes and sys.stdin.isatty():
        if hw.ollama_running and not disc.get("best_local_llm") and cfg.llm_model not in hw.ollama_models:
            want_pull = Confirm.ask(
                f"No local Ollama LLM was found. Would you like to pull '{cfg.llm_model}' (~{tier_obj.llm_size_gb} GB) via Ollama now?",
                default=False,
            )
            if want_pull:
                pull_ollama = True

    if approved and pull_ollama and hw.ollama_installed and cfg.llm_model not in hw.ollama_models:
        console.print(f"[cyan]Pulling Ollama model '{cfg.llm_model}'...[/cyan]")
        subprocess.run(["ollama", "pull", cfg.llm_model], check=False)

    return cfg


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = PureRAGConfig.load()
    hw = analyze_hardware(workspace_dir=Path(cfg.pdfs_dir).parent, ollama_base_url=cfg.ollama_base_url)
    if args.json:
        print(json.dumps({"hardware": asdict(hw), "config": cfg.to_public_dict()}, indent=2))
    else:
        _print_hardware_report(asdict(hw))
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    cfg = PureRAGConfig.load()
    run_interactive_or_agent_setup(
        cfg=cfg,
        yes=args.yes,
        tier_override=args.tier,
        pull_ollama=args.pull_llm,
    )
    if args.ingest:
        engine = PureRAGEngine(cfg)
        summary = engine.scan_and_ingest(force_reindex=False)
        console.print(
            f"[bold green]✓ Indexed {summary['total_documents']} document(s) "
            f"({summary['total_chunks']} contextual chunks) from {cfg.pdfs_dir}[/bold green]"
        )
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = PureRAGConfig.load()
    if args.pdfs_dir:
        cfg.pdfs_dir = str(Path(args.pdfs_dir).resolve())
        cfg.save()
    engine = PureRAGEngine(cfg)
    summary = engine.scan_and_ingest(force_reindex=args.force)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        console.print(
            Panel(
                f"[bold]PDFs Folder:[/bold] {cfg.pdfs_dir}\n"
                f"[bold]Added:[/bold] {len(summary['added'])} | [bold]Updated:[/bold] {len(summary['updated'])} | "
                f"[bold]Unchanged:[/bold] {len(summary['unchanged'])}\n"
                f"[bold]Total Indexed:[/bold] {summary['total_documents']} docs ({summary['total_chunks']} contextual chunks)\n"
                f"[bold]Active Embedder:[/bold] {summary['active_embedder']} | [bold]Reranker:[/bold] {summary['active_reranker']}",
                title="📚 Pure RAG Ingestion Complete",
                border_style="green",
            )
        )
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    cfg = PureRAGConfig.load()
    engine = PureRAGEngine(cfg)
    engine.scan_and_ingest(force_reindex=False)
    result = engine.query(
        question=args.question,
        length=args.length,
        style=args.style,
        top_k=args.top_k,
    )
    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        console.print(
            Panel(
                result["answer"],
                title=f"💬 Pure RAG Answer (Length={result['promptability']['length']}, Style={result['promptability']['style']})",
                border_style="cyan",
            )
        )
        if result["citations"]:
            console.print("[bold]📚 Citations:[/bold]")
            for c in result["citations"]:
                console.print(
                    f"  [{c['rank']}] [bold]{c['filename']}[/bold] (p. {c['page_number']} | {c['section_path']}) — score: {c['final_score']}"
                )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    from purerag.server import create_app

    cfg = PureRAGConfig.load()
    if args.pdfs_dir:
        cfg.pdfs_dir = str(Path(args.pdfs_dir).resolve())
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port

    # First-time hardware profiling if never run before
    if not cfg.hardware_profiled:
        run_interactive_or_agent_setup(cfg, yes=args.yes, tier_override="auto", pull_ollama=False)
    else:
        cfg.save()

    app = create_app(cfg)
    url = f"http://{cfg.host}:{cfg.port}"
    console.print(
        Panel(
            f"[bold green]Pure RAG Web App is live at:[/bold green] [underline cyan]{url}[/underline cyan]\n"
            f"[bold]Designated PDFs Folder:[/bold] {Path(cfg.pdfs_dir).resolve()}\n"
            f"[dim]Drop any PDF, Markdown, or TXT file into that folder or drag-and-drop in the browser UI![/dim]",
            title="🚀 Pure RAG Local Server",
            border_style="green",
        )
    )

    if not args.no_browser:
        def _open():
            time.sleep(1.2)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="purerag",
        description="Pure RAG — State-of-the-Art Local RAG Pipeline & AI Chatbot (macOS, Windows, Linux)",
    )
    sub = parser.add_subparsers(dest="command")

    # serve (also default when running `purerag` with no subcommand)
    p_serve = sub.add_parser("serve", help="Start the local Pure RAG Web UI & API server")
    p_serve.add_argument("--host", default=None, help="Bind host (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=None, help="Bind port (default: 8000)")
    p_serve.add_argument("--pdfs-dir", default=None, help="Custom path to designated PDFs folder")
    p_serve.add_argument("--no-browser", action="store_true", help="Do not auto-open browser tab")
    p_serve.add_argument("--yes", "-y", action="store_true", help="Non-interactive mode (auto-accept recommended hardware tier)")

    # setup
    p_setup = sub.add_parser("setup", help="Analyze hardware capability and configure recommended models with permission")
    p_setup.add_argument("--tier", default="auto", choices=["auto", "ultra_light", "balanced", "performance", "workstation"])
    p_setup.add_argument("--yes", "-y", action="store_true", help="Non-interactive approval for AI agents & scripts")
    p_setup.add_argument("--pull-llm", action="store_true", help="Also pull the recommended Ollama model if Ollama is installed")
    p_setup.add_argument("--ingest", action="store_true", default=True, help="Scan and index the pdfs/ folder after setup")

    # doctor
    p_doc = sub.add_parser("doctor", help="Inspect OS, CPU, RAM, GPU, and Ollama status")
    p_doc.add_argument("--json", action="store_true", help="Output machine-readable JSON for AI agents")

    # ingest
    p_ing = sub.add_parser("ingest", help="Scan the designated pdfs/ folder and incrementally index documents")
    p_ing.add_argument("--pdfs-dir", default=None, help="Override path to pdfs/ folder")
    p_ing.add_argument("--force", action="store_true", help="Force full re-chunking and re-embedding of all files")
    p_ing.add_argument("--json", action="store_true", help="Output JSON summary")

    # query
    p_q = sub.add_parser("query", help="Run a grounded hybrid + reranked RAG query from the CLI")
    p_q.add_argument("question", help="The question to ask your documents")
    p_q.add_argument("--length", default="balanced", choices=["shorter", "balanced", "longer"])
    p_q.add_argument(
        "--style",
        default="standard",
        choices=["standard", "precise", "bulleted", "executive", "step_by_step", "eli5", "citations_only"],
    )
    p_q.add_argument("--top-k", type=int, default=6, help="Number of reranked chunks to retrieve")
    p_q.add_argument("--format", default="text", choices=["text", "json"], help="Output format (text or json)")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        # Default action when user runs `purerag` directly: launch serve!
        serve_args = parser.parse_args(["serve"])
        return cmd_serve(serve_args)

    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "setup":
        return cmd_setup(args)
    if args.command == "doctor":
        return cmd_doctor(args)
    if args.command == "ingest":
        return cmd_ingest(args)
    if args.command == "query":
        return cmd_query(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
