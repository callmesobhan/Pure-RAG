#!/usr/bin/env bash
export PATH="/usr/local/bin:/opt/homebrew/bin:/Users/sobhan/.local/bin:$PATH"
PURERAG_DIR="/Users/sobhan/.gemini/antigravity/scratch/Pure-RAG"
cd "$PURERAG_DIR"

if lsof -iTCP:8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
  open "http://127.0.0.1:8000"
  exit 0
fi

nohup "$PURERAG_DIR/.venv/bin/purerag" serve --host 127.0.0.1 --port 8000 --yes > "$PURERAG_DIR/data/purerag_app.log" 2>&1 &
exit 0
