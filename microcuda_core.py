#!/usr/bin/env python3
"""
MICROCUDA v2.2 – Backend with model-driven tool use, memory, proactive tasks, skills.
"""
import os
import re
import json
import sqlite3
import subprocess
import tempfile
import asyncio
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import psutil
import httpx
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# ---------- MicroCUDABridge (unchanged from original) ----------
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

# ---------- Memory Manager (SQLite) ----------
class MemoryManager:
    def __init__(self, db_path: str = "microcuda_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS core_memory (key TEXT PRIMARY KEY, value TEXT, updated_at TIMESTAMP)")
            conn.execute("CREATE TABLE IF NOT EXISTS conversation_history (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, content TEXT, timestamp TIMESTAMP)")
            conn.execute("CREATE TABLE IF NOT EXISTS skills (name TEXT PRIMARY KEY, command TEXT, created_at TIMESTAMP)")

    # Core memory (key-value)
    def set_memory(self, key: str, value: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("REPLACE INTO core_memory (key, value, updated_at) VALUES (?, ?, ?)",
                         (key, value, datetime.utcnow().isoformat()))

    def get_memory(self, key: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT value FROM core_memory WHERE key=?", (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def delete_memory(self, key: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM core_memory WHERE key=?", (key,))

    def list_memory(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT key, value FROM core_memory")
            return [{"key": k, "value": v} for k, v in cur.fetchall()]

    # Conversation history
    def add_message(self, role: str, content: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO conversation_history (role, content, timestamp) VALUES (?, ?, ?)",
                         (role, content, datetime.utcnow().isoformat()))

    def get_recent_history(self, limit: int = 20) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT role, content FROM conversation_history ORDER BY id DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
            return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    def clear_history(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM conversation_history")

    # Skills
    def add_skill(self, name: str, command: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("REPLACE INTO skills (name, command, created_at) VALUES (?, ?, ?)",
                         (name, command, datetime.utcnow().isoformat()))

    def get_skill(self, name: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT command FROM skills WHERE name=?", (name,))
            row = cur.fetchone()
            return row[0] if row else None

    def list_skills(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT name, command FROM skills")
            return [{"name": n, "command": c} for n, c in cur.fetchall()]

    def delete_skill(self, name: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM skills WHERE name=?", (name,))

# ---------- Webhook Manager ----------
class WebhookManager:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=5.0)

    async def send_webhook(self, url: str, payload: Any):
        try:
            await self.client.post(url, json=payload)
        except Exception as e:
            print(f"Webhook failed: {e}")

    async def close(self):
        await self.client.aclose()

# ---------- Scheduler Manager (proactive tasks) ----------
class SchedulerManager:
    def __init__(self, webhook_manager: WebhookManager, memory_mgr: MemoryManager):
        self.scheduler = BackgroundScheduler()
        self.webhook_mgr = webhook_manager
        self.memory_mgr = memory_mgr
        self.tasks = {}  # name -> job
        self.scheduler.start()

    def add_task(self, name: str, interval_seconds: int, action: str, action_args: dict = None):
        # action: "send_webhook", "run_skill", "call_api"
        def job_func():
            asyncio.run(self._execute_action(action, action_args or {}))
        job = self.scheduler.add_job(job_func, trigger=IntervalTrigger(seconds=interval_seconds), id=name, replace_existing=True)
        self.tasks[name] = job

    async def _execute_action(self, action: str, args: dict):
        if action == "send_webhook":
            url = args.get("url")
            payload = args.get("payload", {})
            if url:
                await self.webhook_mgr.send_webhook(url, payload)
        elif action == "run_skill":
            skill_name = args.get("skill_name")
            if skill_name:
                cmd = self.memory_mgr.get_skill(skill_name)
                if cmd:
                    subprocess.run(cmd, shell=True, capture_output=True)
        elif action == "call_api":
            # For future expansion
            pass

    def remove_task(self, name: str):
        if name in self.tasks:
            self.tasks[name].remove()
            del self.tasks[name]

    def list_tasks(self) -> List[Dict]:
        return [{"name": job.id, "next_run": str(job.next_run_time)} for job in self.scheduler.get_jobs()]

    def shutdown(self):
        self.scheduler.shutdown()

# ---------- Enhanced Ollama Agent with Tool Calling ----------
SYSTEM_PROMPT = (
    "You are MICROCUDA, a CPU-accelerated AI reasoning engine with tool use. "
    "You have access to tools: run_benchmark, compile_cuda, memory_set, memory_get, "
    "memory_delete, schedule_task, list_tasks, send_webhook, add_skill, list_skills. "
    "Always reason inside <thinking> tags. When you need to use a tool, output:\n"
    "<tool name='tool_name' args='{\"arg1\":\"value\"}'></tool>\n"
    "After the tool result, continue thinking and answer. Be helpful and proactive."
)

class OllamaAgent:
    def __init__(self, base_url: str = "http://localhost:11434", memory_mgr: MemoryManager = None):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=120.0)
        self.memory = memory_mgr

    async def list_models(self) -> List[Dict]:
        try:
            r = await self.client.get("/api/tags")
            r.raise_for_status()
            return [{"name": m["name"]} for m in r.json().get("models", [])]
        except Exception:
            return []

    async def generate_with_tools(self, prompt: str, model: str = "qwen2.5:0.5b",
                                  temperature: float = 0.1, num_predict: int = 1024,
                                  tool_executor: Callable = None) -> Dict[str, Any]:
        # Build conversation from history
        history = self.memory.get_recent_history(10) if self.memory else []
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history:
            messages.append(msg)
        messages.append({"role": "user", "content": prompt})

        full_prompt = self._format_chat(messages)  # simple fallback for /generate
        try:
            r = await self.client.post("/api/generate", json={
                "model": model, "prompt": full_prompt, "stream": False,
                "options": {"temperature": temperature, "num_predict": num_predict},
            })
            r.raise_for_status()
            d = r.json()
            response_text = d.get("response", "")
            # Check for tool calls
            tool_calls = self._parse_tool_calls(response_text)
            tool_results = []
            if tool_calls and tool_executor:
                for tc in tool_calls:
                    result = await tool_executor(tc["name"], tc.get("args", {}))
                    tool_results.append(result)
                    # Inject result into conversation as system message
                    result_msg = f"Tool {tc['name']} returned: {json.dumps(result)}"
                    if self.memory:
                        self.memory.add_message("system", result_msg)
                # Re-run generation with tool results appended
                if tool_results:
                    extended_prompt = f"{prompt}\n\n[Tool results: {json.dumps(tool_results)}]\nNow continue your answer."
                    return await self.generate_with_tools(extended_prompt, model, temperature, num_predict, tool_executor)
            # Store interaction in memory
            if self.memory:
                self.memory.add_message("user", prompt)
                self.memory.add_message("assistant", response_text)
            return {"model": model, "text": response_text, "tokens": d.get("eval_count", 0), "tool_calls": tool_calls}
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")

    def _format_chat(self, messages: List[Dict]) -> str:
        # Simple concatenation for small models, works with qwen/llama
        out = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                out += f"<|system|>\n{content}\n"
            elif role == "user":
                out += f"<|user|>\n{content}\n"
            else:
                out += f"<|assistant|>\n{content}\n"
        out += "<|assistant|>\n"
        return out

    def _parse_tool_calls(self, text: str) -> List[Dict]:
        # Extract <tool name="..." args='...'></tool>
        pattern = r'<tool\s+name="([^"]+)"\s+args=\'([^\']*)\'\s*></tool>'
        matches = re.findall(pattern, text)
        calls = []
        for name, args_str in matches:
            try:
                args = json.loads(args_str) if args_str else {}
            except:
                args = {}
            calls.append({"name": name, "args": args})
        return calls

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
    app.state.memory = MemoryManager()
    app.state.webhook_mgr = WebhookManager()
    app.state.scheduler = SchedulerManager(app.state.webhook_mgr, app.state.memory)
    app.state.agent = OllamaAgent(memory_mgr=app.state.memory)
    app.state.model = os.environ.get("MICROCUDA_MODEL", "qwen2.5:0.5b")
    print(f"[Backend] Ready | CPU flags: {', '.join(flags)} | Model: {app.state.model}")
    yield
    await app.state.agent.close()
    await app.state.webhook_mgr.close()
    app.state.scheduler.shutdown()

app = FastAPI(title="MICROCUDA v2.2", version="2.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Pydantic models
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

class MemorySetReq(BaseModel):
    key: str
    value: str

class SkillAddReq(BaseModel):
    name: str
    command: str

class TaskAddReq(BaseModel):
    name: str
    interval_seconds: int
    action: str
    action_args: dict = {}

class WebhookSendReq(BaseModel):
    url: str
    payload: dict = {}

# ---- Tool executor for LLM ----
async def execute_tool(name: str, args: dict) -> Any:
    """Called by the agent when a tool tag is detected."""
    if name == "run_benchmark":
        bench_type = args.get("type", "matmul")
        if bench_type == "matmul":
            return app.state.bridge.benchmark_matmul()
        elif bench_type == "bandwidth":
            return app.state.bridge.benchmark_bandwidth()
        elif bench_type == "vector":
            return app.state.bridge.benchmark_vector_add()
        else:
            return {"error": "unknown benchmark type"}
    elif name == "compile_cuda":
        code = args.get("code", "")
        return app.state.bridge.compile(code)
    elif name == "memory_set":
        key = args.get("key")
        value = args.get("value")
        if key and value:
            app.state.memory.set_memory(key, value)
            return {"status": "stored", "key": key}
        return {"error": "missing key or value"}
    elif name == "memory_get":
        key = args.get("key")
        val = app.state.memory.get_memory(key)
        return {"key": key, "value": val}
    elif name == "memory_delete":
        key = args.get("key")
        app.state.memory.delete_memory(key)
        return {"status": "deleted", "key": key}
    elif name == "schedule_task":
        name_task = args.get("name")
        interval = args.get("interval_seconds")
        action = args.get("action")
        action_args = args.get("action_args", {})
        if name_task and interval and action:
            app.state.scheduler.add_task(name_task, interval, action, action_args)
            return {"status": "scheduled", "task": name_task}
        return {"error": "missing fields"}
    elif name == "list_tasks":
        return app.state.scheduler.list_tasks()
    elif name == "send_webhook":
        url = args.get("url")
        payload = args.get("payload", {})
        if url:
            await app.state.webhook_mgr.send_webhook(url, payload)
            return {"status": "sent", "url": url}
        return {"error": "missing url"}
    elif name == "add_skill":
        skill_name = args.get("name")
        command = args.get("command")
        if skill_name and command:
            app.state.memory.add_skill(skill_name, command)
            return {"status": "skill added", "name": skill_name}
        return {"error": "missing name or command"}
    elif name == "list_skills":
        return app.state.memory.list_skills()
    else:
        return {"error": f"unknown tool: {name}"}

# ---- API routes ----
@app.get("/")
async def root():
    return {"service": "MICROCUDA", "version": "2.2.0", "cpu": get_flags()}

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
        "memory_entries": len(app.state.memory.list_memory()) if app.state.memory else 0,
        "scheduled_tasks": len(app.state.scheduler.list_tasks()) if app.state.scheduler else 0,
    }

@app.get("/models")
async def list_models():
    return {"models": await app.state.agent.list_models()}

@app.post("/agent")
async def run_agent(req: AgentReq, background_tasks: BackgroundTasks):
    model = req.model or getattr(app.state, "model", "qwen2.5:0.5b")
    # Tool executor bound to this request
    async def tool_callback(name, args):
        return await execute_tool(name, args)
    try:
        result = await app.state.agent.generate_with_tools(
            req.prompt, model, req.temperature, req.num_predict, tool_callback
        )
        return {"status": "success", "output": result["text"], "model": result["model"],
                "tokens": result["tokens"], "tool_calls": result.get("tool_calls", [])}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/compile")
async def compile_cuda(req: CompileReq):
    try:
        return {"status": "success", **app.state.bridge.compile(req.code, req.target)}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/kernel/matmul")
async def kernel_matmul(req: KernelMatmulReq):
    return {"status": "success", **app.state.bridge.benchmark_matmul(req.size, req.iters)}

@app.post("/kernel/bandwidth")
async def kernel_bandwidth(req: KernelBandwidthReq):
    return {"status": "success", **app.state.bridge.benchmark_bandwidth(req.mb, req.iters)}

@app.post("/kernel/vector")
async def kernel_vector(req: KernelVectorReq):
    return {"status": "success", **app.state.bridge.benchmark_vector_add(req.n, req.iters)}

# Memory endpoints
@app.get("/memory")
async def list_memory():
    return {"memory": app.state.memory.list_memory()}

@app.post("/memory")
async def set_memory(req: MemorySetReq):
    app.state.memory.set_memory(req.key, req.value)
    return {"status": "ok"}

@app.delete("/memory/{key}")
async def delete_memory(key: str):
    app.state.memory.delete_memory(key)
    return {"status": "ok"}

@app.get("/memory/{key}")
async def get_memory(key: str):
    val = app.state.memory.get_memory(key)
    return {"key": key, "value": val}

# Conversation history
@app.get("/history")
async def get_history(limit: int = 20):
    return {"history": app.state.memory.get_recent_history(limit)}

@app.delete("/history")
async def clear_history():
    app.state.memory.clear_history()
    return {"status": "cleared"}

# Skills
@app.get("/skills")
async def list_skills():
    return {"skills": app.state.memory.list_skills()}

@app.post("/skills")
async def add_skill(req: SkillAddReq):
    app.state.memory.add_skill(req.name, req.command)
    return {"status": "added"}

@app.delete("/skills/{name}")
async def delete_skill(name: str):
    app.state.memory.delete_skill(name)
    return {"status": "deleted"}

# Tasks
@app.get("/tasks")
async def list_tasks():
    return {"tasks": app.state.scheduler.list_tasks()}

@app.post("/tasks")
async def add_task(req: TaskAddReq):
    app.state.scheduler.add_task(req.name, req.interval_seconds, req.action, req.action_args)
    return {"status": "added"}

@app.delete("/tasks/{name}")
async def delete_task(name: str):
    app.state.scheduler.remove_task(name)
    return {"status": "deleted"}

# Webhook
@app.post("/webhook/send")
async def send_webhook(req: WebhookSendReq, background_tasks: BackgroundTasks):
    background_tasks.add_task(app.state.webhook_mgr.send_webhook, req.url, req.payload)
    return {"status": "sending"}

def start():
    uvicorn.run(app, host="0.0.0.0", port=8472)

if __name__ == "__main__":
    start()
