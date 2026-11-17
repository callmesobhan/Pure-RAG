#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
  echo "⚙️  First-time run detected. Launching Pure RAG installer..."
  bash ./install.sh "$@"
fi

exec ./.venv/bin/purerag serve "$@"
