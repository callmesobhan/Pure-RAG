#!/usr/bin/env bash
# ============================================================================
# Pure RAG — Cross-Platform Installer & Setup Wizard (macOS & Linux)
# Supports both Interactive Human Setup and AI-Agent Automation (--yes)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

AUTO_YES=0
INSTALL_SOTA=1
START_SERVER=0

for arg in "$@"; do
  case "$arg" in
    --yes|-y) AUTO_YES=1 ;;
    --minimal) INSTALL_SOTA=0 ;;
    --run|--serve) START_SERVER=1 ;;
  esac
done

echo "=================================================================="
echo "  🚀 Pure RAG — Local SOTA RAG Pipeline Installer (macOS / Linux)"
echo "=================================================================="

# 1. Check Python >= 3.9
if command -v python3 >/dev/null 2>&1; then
  PY_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PY_BIN="python"
else
  echo "❌ Error: Python 3.9+ is required but was not found on PATH."
  echo "   Install Python from https://www.python.org/downloads/ and re-run."
  exit 1
fi

echo "✓ Found Python: $($PY_BIN --version)"

# 2. Create virtual environment (.venv) if not present
if [ ! -d ".venv" ]; then
  echo "📦 Creating Python virtual environment in ./.venv (with access to system ML packages)..."
  "$PY_BIN" -m venv --system-site-packages .venv
fi

VENV_PY="./.venv/bin/python"
VENV_PIP="./.venv/bin/pip"

# 3. Ask permission (unless --yes was passed) before installing packages
if [ "$AUTO_YES" -eq 0 ] && [ -t 0 ]; then
  echo ""
  read -r -p "Grant permission to install Pure RAG Python dependencies into ./.venv? [Y/n] " resp
  resp="${resp:-Y}"
  if [[ ! "$resp" =~ ^[Yy]$ ]]; then
    echo "Installation cancelled by user."
    exit 0
  fi
fi

echo "⬇️  Installing core Pure RAG packages..."
"$VENV_PIP" install --upgrade pip wheel setuptools >/dev/null
"$VENV_PIP" install -e .

if [ "$INSTALL_SOTA" -eq 1 ]; then
  if [ "$AUTO_YES" -eq 0 ] && [ -t 0 ]; then
    echo ""
    read -r -p "Install optional SOTA PDF & ONNX accelerators (pymupdf4llm, fastembed, flashrank)? [Y/n] " sota_resp
    sota_resp="${sota_resp:-Y}"
    if [[ "$sota_resp" =~ ^[Yy]$ ]]; then
      "$VENV_PIP" install ".[sota]" || echo "ℹ️ Continuing with core hybrid retriever."
    fi
  else
    "$VENV_PIP" install ".[sota]" || echo "ℹ️ Continuing with core hybrid retriever."
  fi
fi

# 4. Run Hardware Capability Analyzer & Permissioned Setup
mkdir -p pdfs data
if [ "$AUTO_YES" -eq 1 ]; then
  "$VENV_PY" -m purerag.cli setup --yes --ingest
else
  "$VENV_PY" -m purerag.cli setup --ingest
fi

echo ""
echo "=================================================================="
echo "  ✅ Pure RAG is installed and ready!"
echo "  📂 Put your PDFs inside: $SCRIPT_DIR/pdfs/"
echo "  ▶️  Start the app anytime with: ./run.sh   (or ./.venv/bin/purerag)"
echo "=================================================================="

if [ "$START_SERVER" -eq 1 ]; then
  exec "./.venv/bin/purerag" serve
fi
