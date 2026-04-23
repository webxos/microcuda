# MICROCUDA v2.1 – CPU-Accelerated AI Harness

![Alt Text](https://github.com/webxos/microcuda/blob/main/assets/logo.jpeg)

**MicroCUDA** is a lightweight framework that brings GPU‑like compute abstractions and a local AI agent to your CPU. It simulates a CUDA runtime using OpenMP and SIMD (AVX2/AVX‑512), runs large language models via Ollama, and provides a terminal user interface (TUI) for interacting with the AI agent, running CPU‑accelerated benchmarks, and compiling “CUDA” kernels directly to C++/OpenMP.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Ollama](https://img.shields.io/badge/ollama-required-green)

---

## Features

- **CPU‑Accelerated “CUDA” Bridge**  
  Translates a subset of CUDA C++ (``__global__``, ``__device__``, ``threadIdx.x``, etc.) into OpenMP‑parallelised C++ with SIMD intrinsics.

- **Local AI Agent**  
  Chat with any Ollama model (e.g., `qwen2.5:0.5b`, `llama3.2:1b`). The agent uses chain‑of‑thought reasoning inside `<thinking>...</thinking>` tags.

- **Terminal UI (TUI)**  
  Built with [Textual](https://textual.textualize.io/). Monitor CPU/RAM usage, switch models, run benchmarks, compile kernels, and converse with the AI – all from the terminal.

- **CPU Kernel Benchmarks**  
  - Matrix multiplication (simulated TFLOPS)  
  - Memory bandwidth (GB/s)  
  - Vector addition (GB/s)

- **“CUDA” Kernel Compilation**  
  Write a simple CUDA kernel (e.g., vector addition), click “Compile CUDA”, and MicroCUDA generates a native binary using `g++` with OpenMP and AVX2/AVX‑512 flags.

- **No GPU Required** – runs on any modern x86‑64 CPU with Linux (and probably WSL2).

---

## 📦 Requirements

- **Operating System**: Linux (tested on Ubuntu 22.04/24.04).  
  *MacOS/Windows (WSL2) may work but are not officially tested.*
- **Python** 3.8 or newer
- **Ollama** installed and running (see [ollama.com](https://ollama.com))
- **C++ compiler** with OpenMP support (`g++`/`clang++`, typically via `build-essential`)
- **Python virtual environment** (automatically created by the launcher)

---

## 🔧 Installation

### Download all files to a folder on your system:

Then bash:

```bash
cd ~/microcuda
chmod +x run.sh && ./run.sh
```

On first launch, the script will:
- Create a Python virtual environment in `./venv`
- Install dependencies (`fastapi`, `uvicorn`, `textual`, `numpy`, …)
- Start the backend (FastAPI on port `8472`) and open the TUI.

> 💡 The launcher works from any directory and passes all arguments to `microcuda.py`.

---

## Usage

### Command‑line arguments

| Argument | Effect |
|----------|--------|
| `--model NAME` | Set the default Ollama model (e.g. `llama3.2:1b`) |
| `--backend`   | Start only the FastAPI backend (no TUI) |
| `--tui`       | Launch only the TUI (assumes backend already running) |
| `--redeploy`  | Force‑copy Python files to `~/microcuda/` and reinstall dependencies |

Example:
```bash
./run.sh --model llama3.2:1b
```

### TUI Controls

- **Mouse / keyboard** – fully interactive (click tabs, buttons, input field)
- **Commands** in the Agent tab (type `/command` in the input box)

| Command | Description |
|---------|-------------|
| `/help` | Show full help text |
| `/status` | Display backend CPU / RAM / model info |
| `/models` | List all models available in Ollama |
| `/model NAME` | Switch to a different model (e.g. `/model llama3.2:1b`) |
| `/clear` | Clear the Agent log |
| `/bench` | Run all three CPU benchmarks |
| `/compile` | Compile the example vector‑add CUDA kernel |
| `/version` | Show version information |

- **Hotkeys**  
  `Ctrl+Q` – quit  
  `Ctrl+L` – clear active log  
  `F1` – show help

### Kernel tab (CPU Kernels)

Click any button to run a benchmark or compile the example kernel:
- **MatMul** – matrix multiplication (1024×1024, float32)
- **Bandwidth** – memory bandwidth test (512 MiB buffer)
- **Vector Add** – simple vector addition (10M elements)
- **Run All** – executes all three benchmarks sequentially
- **Compile CUDA** – builds a native binary from the example kernel in `microcuda_extra.py`

Each result is printed as JSON with measured time, TFLOPS/GB/s, and the detected SIMD flags.

---

## Architecture

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────┐
│  microcuda_tui  │◄───►│   FastAPI backend   │◄───►│   Ollama     │
│   (Textual)     │     │   (port 8472)       │     │   (port 11434)│
└─────────────────┘     └─────────────────────┘     └──────────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │  MicroCUDABridge    │
                        │  - CUDA → C++/OMP   │
                        │  - SIMD (AVX2/512)  │
                        │  - numpy benchmarks │
                        └─────────────────────┘
```

- **`microcuda.py`** – orchestration: deploys files, creates venv, starts backend and/or TUI.
- **`microcuda_core.py`** – FastAPI application with the bridge, Ollama agent, `/status`, `/agent`, `/kernel/*`, and `/compile` endpoints.
- **`microcuda_tui.py`** – Textual TUI that consumes the backend API.
- **`microcuda_extra.py`** – static data (help text, example CUDA kernel).

### The “CUDA” Bridge

MicroCUDA does **not** use a real GPU. Instead, it performs source‑to‑source translation of a limited CUDA dialect:

| CUDA keyword | Translation |
|--------------|--------------|
| `__global__` | removed, function becomes host‑callable |
| `__device__` | becomes `inline` |
| `threadIdx.x` | `(omp_get_thread_num() % 32)` |
| `blockIdx.x` | global variable (set before parallel region) |
| `blockDim.x` | constant (256) |

The resulting C++ code is compiled with `-fopenmp` and the most advanced SIMD flags available on your CPU (AVX‑512 > AVX2 > SSE4.2). For more complex kernels you can extend the translator inside `MicroCUDABridge._translate()`.

---

## Example: Compiling a CUDA Kernel

Inside the **CPU Kernels** tab, click **Compile CUDA**. The following kernel will be compiled and linked:

```cpp
__global__ void vecAdd(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}
```

MicroCUDA generates a complete `main()` that launches blocks/threads using OpenMP, then prints a success message. The binary is saved in `/tmp/microcuda/mc_<counter>.bin`.

---

## 🛠️ Development & Customisation Ideas (Use at your own risk)

### Adding a new kernel benchmark

1. In `microcuda_core.py`, add a method to the `MicroCUDABridge` class (e.g. `benchmark_convolution`).
2. Add a corresponding FastAPI endpoint (`@app.post("/kernel/conv")`).
3. In `microcuda_tui.py`, add a new button and call the endpoint in `_run_kernel`.

### Changing the system prompt

Edit `SYSTEM_PROMPT` in `microcuda_core.py`. The tokenisation pattern for `<thinking>` is fixed in the TUI – make sure your prompt retains the `<thinking>... </thinking>` structure.

### Using a different Ollama host

Set the environment variable `OLLAMA_HOST` before starting MicroCUDA, or modify `OllamaAgent.__init__`.

---

## ❓ Troubleshooting

| Problem | Likely solution |
|---------|----------------|
| `ImportError: No module named 'numpy'` | Run `./run.sh --redeploy` to reinstall dependencies. |
| Backend doesn’t start / port 8472 refused | Check that no other process uses port 8472. Start with `./run.sh --backend` manually. |
| `OllamaAgent` errors | Ensure Ollama is running: `curl http://localhost:11434/api/tags`. Pull a model: `ollama pull qwen2.5:0.5b`. |
| Compilation fails (“g++: command not found”) | Install `build-essential` (Ubuntu) or equivalent. |
| TUI shows `[backend unreachable]` | Start the backend first on another terminal: `./run.sh --backend`, then `./run.sh --tui`. |

---

## 📄 License

MIT 

## SCREENSHOTS:

![Alt Text](https://github.com/webxos/microcuda/blob/main/assets/screen4.png)

![Alt Text](https://github.com/webxos/microcuda/blob/main/assets/screen3.png)
