#!/usr/bin/env python3
"""
MICROCUDA v2.1 – CPU-Accelerated AI Harness
Main entry point – orchestrates install, backend, TUI.
"""
import os
import sys
import subprocess
import socket
import time
import argparse
from pathlib import Path

# --- Constants -------------------------------------------------
TARGET_DIR = Path.home() / "microcuda"
VENV_DIR   = TARGET_DIR / "venv"
BACKEND_PORT = 8472
DEFAULT_MODEL = "qwen2.5:0.5b"

# --- Helper functions -----------------------------------------
def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False

def is_installed() -> bool:
    return (TARGET_DIR / "microcuda_core.py").exists()

def deploy_files():
    """Copy the four Python files to ~/microcuda/ if not already there."""
    print(f"[microcuda] Deploying to {TARGET_DIR} ...")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ["microcuda.py", "microcuda_core.py", "microcuda_tui.py", "microcuda_extra.py"]:
        src = Path(__file__).parent / fname
        dst = TARGET_DIR / fname
        if src.resolve() != dst.resolve():
            import shutil
            shutil.copy2(str(src), str(dst))
            print(f"  copied {fname}")
        else:
            print(f"  already in place: {fname}")

def create_venv():
    if VENV_DIR.exists():
        print("[microcuda] venv already exists.")
        return
    print("[microcuda] Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)

def install_deps():
    pip = str(VENV_DIR / "bin" / "pip")
    print("[microcuda] Installing Python dependencies...")
    subprocess.run([pip, "install", "--upgrade", "pip"], check=True)
    subprocess.run([pip, "install", "fastapi", "uvicorn", "httpx", "pydantic",
                    "psutil", "textual", "rich", "numpy", "requests"], check=True)

def ensure_deps():
    """Verify that numpy is importable inside the venv; install deps if missing."""
    if not VENV_DIR.exists():
        create_venv()
        install_deps()
        return
    # Try to import numpy using the venv's python
    python = str(VENV_DIR / "bin" / "python")
    try:
        subprocess.run([python, "-c", "import numpy"], check=True, capture_output=True)
        print("[microcuda] Dependencies already satisfied.")
    except subprocess.CalledProcessError:
        print("[microcuda] Missing dependencies (numpy not found). Installing...")
        install_deps()

def start_backend(model: str):
    backend_dir = TARGET_DIR
    env = os.environ.copy()
    env["MICROCUDA_MODEL"] = model
    python = str(VENV_DIR / "bin" / "python")
    proc = subprocess.Popen(
        [python, "-c", "from microcuda_core import start; start()"],
        cwd=str(backend_dir), env=env
    )
    for _ in range(30):
        if port_open(BACKEND_PORT):
            print(f"[microcuda] Backend running on port {BACKEND_PORT}")
            return proc
        time.sleep(1)
    print("[microcuda] WARNING: backend did not start in time.")
    return proc

def launch_tui():
    python = str(VENV_DIR / "bin" / "python")
    subprocess.run([python, str(TARGET_DIR / "microcuda_tui.py")])

# --- Main -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MICROCUDA v2.1")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--backend", action="store_true", help="Start backend only")
    parser.add_argument("--tui", action="store_true", help="Launch TUI only")
    parser.add_argument("--redeploy", action="store_true", help="Force redeploy files")
    args = parser.parse_args()

    # Deploy if needed
    if args.redeploy or not is_installed():
        deploy_files()
        create_venv()
        install_deps()
    else:
        print(f"[microcuda] Installation found at {TARGET_DIR}")

    # Ensure dependencies are installed (in case venv exists but numpy missing)
    ensure_deps()

    # Change to target directory for imports
    os.chdir(TARGET_DIR)
    sys.path.insert(0, str(TARGET_DIR))

    if args.backend:
        start_backend(args.model)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        return

    if args.tui:
        launch_tui()
        return

    # Default: start backend + TUI
    if port_open(BACKEND_PORT):
        print(f"[microcuda] Backend already running on port {BACKEND_PORT}")
        backend_proc = None
    else:
        backend_proc = start_backend(args.model)

    try:
        launch_tui()
    finally:
        if backend_proc:
            print("[microcuda] Shutting down backend...")
            backend_proc.terminate()

if __name__ == "__main__":
    main()
