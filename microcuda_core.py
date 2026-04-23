#!/usr/bin/env python3
"""
MICROCUDA v2.1 – Backend (FastAPI, bridge, Ollama agent)
"""
import os
import re
import subprocess
import tempfile
import numpy as np
import time
import psutil
import httpx
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ---------- MicroCUDABridge (CPU‑accelerated "CUDA") ----------
class MicroCUDABridge:
    def __init__(self, cpu_features: List[str]):
        self.cpu_features = cpu_features
        self.temp_dir = Path(tempfile.gettempdir()) / "microcuda"
        self.temp_dir.mkdir(exist_ok=True)
        self.count = 0

    def _translate(self, cuda_code: str, target: str) -> str:
        cpp = cuda_code
        reps = {
            r"__global__\s*": "",
            r"__device__\s*": "inline ",
            r"__host__\s*": "",
            r"__shared__\s*": "static ",
            r"threadIdx\.x": "(omp_get_thread_num() % 32)",
            r"blockIdx\.x": "blockIdx_x",
            r"blockDim\.x": "blockDim_x",
        }
        for p, r in reps.items():
            cpp = re.sub(p, r, cpp)
        hdr = (
            "// MICROCUDA GENERATED\n"
            "#include <iostream>\n"
            "#include <omp.h>\n"
            "int blockIdx_x=0, blockDim_x=256;\n"
        )
        if "int main" not in cpp:
            launcher = (
                "\nint main(){\n"
                "    const int blocks=4, threads=256;\n"
                "    std::cout << \"[MICROCUDA] Launch \" << blocks << \" blocks\\n\";\n"
                "    #pragma omp parallel for num_threads(blocks*threads)\n"
                "    for(int i=0; i<blocks*threads; i++) blockIdx_x=i/threads;\n"
                "    return 0;\n"
                "}\n"
            )
            cpp = hdr + cpp + launcher
        else:
            cpp = hdr + cpp
        return cpp

    def compile(self, code: str, target: str = "avx512", opt: int = 3) -> Dict:
        self.count += 1
        flags = [f"-O{opt}", "-fopenmp", "-std=c++17"]
        if target == "avx512" and "avx512f" in self.cpu_features:
            flags += ["-mavx512f", "-mavx512vl"]
        elif target == "avx2" and "avx2" in self.cpu_features:
            flags += ["-mavx2"]
        else:
            flags += ["-msse4.2"]
        cpp = self._translate(code, target)
        src = self.temp_dir / f"mc_{self.count}.cpp"
        bin_path = self.temp_dir / f"mc_{self.count}.bin"
        src.write_text(cpp)
        result = subprocess.run(
            ["g++"] + flags + [str(src), "-o", str(bin_path)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        return {
            "binary_path": str(bin_path),
            "log": f"Compiled: {bin_path}",
            "flags": flags,
            "simulated_blocks": 4,
            "threads_per_block": 256,
        }

    def benchmark_matmul(self, size: int = 1024, iters: int = 10) -> Dict:
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        for _ in range(3): np.dot(a, b)
        start = time.perf_counter()
        for _ in range(iters): np.dot(a, b)
        ms = (time.perf_counter() - start) / iters * 1000
        return {"type": "matmul", "size": size, "iters": iters,
                "ms": round(ms, 3),
                "tflops": round((2.0 * size ** 3) / (ms * 1e9), 3),
                "simd": self.cpu_features}

    def benchmark_bandwidth(self, mb: int = 512, iters: int = 20) -> Dict:
        n = mb * 1024 * 1024 // 4
        src = np.random.randn(n).astype(np.float32)
        dst = np.empty_like(src)
        for _ in range(3): dst[:] = src
        start = time.perf_counter()
        for _ in range(iters): dst[:] = src
        ms = (time.perf_counter() - start) / iters * 1000
        return {"type": "bandwidth", "buffer_mb": mb, "iters": iters,
                "ms": round(ms, 3),
                "gb_per_sec": round((n * 4) / (ms * 1e6), 2),
                "simd": self.cpu_features}

    def benchmark_vector_add(self, n: int = 10_000_000, iters: int = 20) -> Dict:
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        c = np.empty_like(a)
        for _ in range(3): c[:] = a + b
        start = time.perf_counter()
        for _ in range(iters): c[:] = a + b
        ms = (time.perf_counter() - start) / iters * 1000
        return {"type": "vector_add", "n": n, "iters": iters,
                "ms": round(ms, 3),
                "gb_per_sec": round((3 * n * 4) / (ms * 1e6), 2),
                "simd": self.cpu_features}

# ---------- Ollama Agent ----------
SYSTEM_PROMPT = (
    "You are MICROCUDA, a CPU-accelerated AI reasoning engine. "
    "Always reason step-by-step inside <thinking>...</thinking> tags, "
    "then give your final answer after </thinking>. "
    "Be concise, accurate, and helpful."
)

class OllamaAgent:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=120.0)

    async def list_models(self) -> List[Dict]:
        try:
            r = await self.client.get("/api/tags")
            r.raise_for_status()
            return [{"name": m["name"]} for m in r.json().get("models", [])]
        except Exception:
            return []

    async def generate(self, prompt: str, model: str = "qwen2.5:0.5b",
                       temperature: float = 0.1, num_predict: int = 1024) -> Dict[str, Any]:
        full_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            + SYSTEM_PROMPT
            + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            + prompt
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n<thinking>"
        )
        try:
            r = await self.client.post("/api/generate", json={
                "model": model, "prompt": full_prompt, "stream": False,
                "options": {"temperature": temperature, "num_predict": num_predict},
            })
            r.raise_for_status()
            d = r.json()
            return {"model": model, "text": d.get("response", ""), "tokens": d.get("eval_count", 0)}
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")

    async def close(self):
        await self.client.aclose()

# ---------- FastAPI App ----------
def get_flags():
    try:
        with open("/proc/cpuinfo") as f:
            d = f.read().lower()
        return [flag for flag in ["avx512f", "avx2", "sse4_2"] if flag in d] or ["sse2"]
    except Exception:
        return ["sse2"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    flags = get_flags()
    app.state.bridge = MicroCUDABridge(flags)
    app.state.agent = OllamaAgent()
    app.state.model = os.environ.get("MICROCUDA_MODEL", "qwen2.5:0.5b")
    print(f"[Backend] Ready | CPU flags: {', '.join(flags)} | Model: {app.state.model}")
    yield
    await app.state.agent.close()

app = FastAPI(title="MICROCUDA v2.1", version="2.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class CompileReq(BaseModel):
    code: str
    target: str = "avx512"

class AgentReq(BaseModel):
    prompt: str
    model: Optional[str] = None
    temperature: float = 0.1
    num_predict: int = 1024

class KernelMatmulReq(BaseModel):
    size: int = 1024
    iters: int = 10

class KernelBandwidthReq(BaseModel):
    mb: int = 512
    iters: int = 20

class KernelVectorReq(BaseModel):
    n: int = 10_000_000
    iters: int = 20

@app.get("/")
async def root():
    return {"service": "MICROCUDA", "version": "2.1.0", "cpu": get_flags()}

@app.get("/status")
async def status():
    mem = psutil.virtual_memory()
    cpus = psutil.cpu_count(logical=True)
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": mem.percent,
        "memory_used_gb": round(mem.used / 1e9, 2),
        "memory_total_gb": round(mem.total / 1e9, 2),
        "cpu_cores": cpus,
        "simulated_cuda_cores": cpus * 32,
        "cpu_flags": get_flags(),
        "active_model": getattr(app.state, "model", "unknown"),
    }

@app.get("/models")
async def list_models():
    if not hasattr(app.state, "agent"):
        return {"models": []}
    return {"models": await app.state.agent.list_models()}

@app.post("/compile")
async def compile_cuda(req: CompileReq):
    if not hasattr(app.state, "bridge"):
        raise HTTPException(503, "Bridge not ready")
    try:
        return {"status": "success", **app.state.bridge.compile(req.code, req.target)}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/agent")
async def run_agent(req: AgentReq):
    if not hasattr(app.state, "agent"):
        raise HTTPException(503, "Agent not ready")
    model = req.model or getattr(app.state, "model", "qwen2.5:0.5b")
    try:
        r = await app.state.agent.generate(req.prompt, model, req.temperature, req.num_predict)
        return {"status": "success", "output": r["text"], "model": r["model"], "tokens": r["tokens"]}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/kernel/matmul")
async def kernel_matmul(req: KernelMatmulReq):
    if not hasattr(app.state, "bridge"):
        raise HTTPException(503, "Bridge not ready")
    return {"status": "success", **app.state.bridge.benchmark_matmul(req.size, req.iters)}

@app.post("/kernel/bandwidth")
async def kernel_bandwidth(req: KernelBandwidthReq):
    if not hasattr(app.state, "bridge"):
        raise HTTPException(503, "Bridge not ready")
    return {"status": "success", **app.state.bridge.benchmark_bandwidth(req.mb, req.iters)}

@app.post("/kernel/vector")
async def kernel_vector(req: KernelVectorReq):
    if not hasattr(app.state, "bridge"):
        raise HTTPException(503, "Bridge not ready")
    return {"status": "success", **app.state.bridge.benchmark_vector_add(req.n, req.iters)}

def start():
    uvicorn.run(app, host="0.0.0.0", port=8472)

if __name__ == "__main__":
    start()
