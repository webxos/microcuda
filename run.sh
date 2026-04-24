#!/usr/bin/env bash
# MICROCUDA v2.2 - Launcher

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/venv"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}MICROCUDA v2.2 - CPU-Accelerated AI Harness${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is required but not installed. Exiting.${NC}"
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
if ! python -c "import numpy" &> /dev/null; then
    echo "Installing Python dependencies (numpy missing)..."
    pip install --upgrade pip
    pip install fastapi uvicorn httpx pydantic psutil textual rich numpy requests apscheduler
fi

if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Ollama does not appear to be running.${NC}"
    echo "Start it with: sudo systemctl start ollama  (or 'ollama serve')"
    echo "Pull a model, e.g.: ollama pull qwen2.5:0.5b"
fi

exec python microcuda.py "$@"
