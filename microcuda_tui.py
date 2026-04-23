#!/usr/bin/env python3
"""MICROCUDA v2.1 - Textual TUI (Agent + CPU Kernels)"""
import asyncio
import httpx
import json
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Input, Static, Label, ProgressBar, RichLog, Button, TabbedContent, TabPane

from microcuda_extra import HELP_TEXT, EXAMPLE_KERNEL

BASE_URL = "http://localhost:8472"

class MonitorBar(Static):
    def compose(self) -> ComposeResult:
        with Horizontal(id="mon_row"):
            with Vertical(classes="mon_col"):
                yield Label("CPU %", classes="mon_lbl")
                yield ProgressBar(id="cpu_bar", total=100, show_percentage=True)
            with Vertical(classes="mon_col"):
                yield Label("RAM %", classes="mon_lbl")
                yield ProgressBar(id="ram_bar", total=100, show_percentage=True)
            with Vertical(classes="mon_col"):
                yield Label("Sim CUDA Cores", classes="mon_lbl")
                yield Label("--", id="cores_lbl", classes="mon_val")
            with Vertical(classes="mon_col"):
                yield Label("Active Model", classes="mon_lbl")
                yield Label("--", id="model_lbl", classes="mon_val")

    def on_mount(self) -> None:
        self.set_interval(2.5, self.refresh_stats)

    async def refresh_stats(self) -> None:
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get(f"{BASE_URL}/status", timeout=2.0)
                d = r.json()
                self.query_one("#cpu_bar", ProgressBar).progress = d.get("cpu_percent", 0)
                self.query_one("#ram_bar", ProgressBar).progress = d.get("memory_percent", 0)
                self.query_one("#cores_lbl", Label).update(str(d.get("simulated_cuda_cores", "--")))
                self.query_one("#model_lbl", Label).update(d.get("active_model", "--"))
        except Exception:
            pass

class MicroCUDATUI(App):
    CSS = """
    Screen { background: #080808; color: #e0e0e0; }
    Header { background: #001800; color: #00ff41; text-style: bold; }
    Footer { background: #001800; color: #00cc33; }
    MonitorBar { height: 6; border: solid #00aa22; margin: 0 1; padding: 0 1; }
    .mon_col { width: 1fr; padding: 0 1; }
    .mon_lbl { color: #00ff41; text-style: bold; }
    .mon_val { color: #ffaa00; text-style: bold; text-align: center; }
    TabbedContent { margin: 1; height: 1fr; }
    RichLog { background: #080808; border: solid #1a3a1a; height: 1fr; }
    Input { background: #111; border: solid #00aa22; color: #e0e0e0; margin: 1 0 0 0; }
    Input:focus { border: solid #00ff41; }
    Button { margin: 0 1 1 0; min-width: 18; }
    Button.-primary { background: #002800; color: #00ff41; border: solid #005500; }
    Button.-success { background: #001828; color: #00aaff; border: solid #003366; }
    Button:hover { background: #003300; }
    .btn_row { height: 5; padding: 1 0; }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "clear_active", "Clear"),
        Binding("f1", "show_help", "Help"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_model: str = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield MonitorBar()
        with TabbedContent(initial="agent"):
            with TabPane("🤖 Agent", id="agent"):
                yield RichLog(id="agent_log", markup=True, highlight=True)
                yield Input(placeholder="Ask the model... (type /help for commands)", id="agent_input")
            with TabPane("⚙️ CPU Kernels", id="kernels"):
                yield RichLog(id="kernel_log", markup=True)
                with Horizontal(classes="btn_row"):
                    yield Button("MatMul", id="btn_matmul", variant="primary")
                    yield Button("Bandwidth", id="btn_bw", variant="primary")
                    yield Button("Vector Add", id="btn_vec", variant="primary")
                    yield Button("Run All", id="btn_all", variant="success")
                    yield Button("Compile CUDA", id="btn_compile", variant="success")
        yield Footer()

    async def on_mount(self) -> None:
        log = self.query_one("#agent_log", RichLog)
        log.write("[bold green]MICROCUDA v2.1  NemoClaw Reasoning Ready[/bold green]")
        log.write("[dim]Backend: http://localhost:8472  |  Type [bold]/help[/bold] for commands[/dim]")

    def action_clear_active(self) -> None:
        try:
            active = self.query_one("TabbedContent").active
            log_id = {"agent": "#agent_log", "kernels": "#kernel_log"}.get(active, "#agent_log")
            self.query_one(log_id, RichLog).clear()
        except Exception:
            pass

    def action_show_help(self) -> None:
        try:
            self.query_one("#agent_log", RichLog).write(HELP_TEXT)
        except Exception:
            pass

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "agent_input":
            return
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.startswith("/"):
            await self._handle_command(text)
        else:
            await self._do_agent(text)

    async def _handle_command(self, cmd: str) -> None:
        log = self.query_one("#agent_log", RichLog)
        lower = cmd.strip().lower()

        if lower in ("/help", "/h", "/?"):
            log.write(HELP_TEXT)
        elif lower == "/clear":
            log.clear()
            log.write("[green]Log cleared.[/green]")
        elif lower == "/version":
            log.write("[bold green]MICROCUDA v2.1[/bold green]  CPU-Accelerated AI Harness")
            log.write("NemoClaw Reasoning Engine | OpenMP/AVX | Ollama LLM Backend")
        elif lower == "/status":
            try:
                async with httpx.AsyncClient() as c:
                    r = await c.get(f"{BASE_URL}/status", timeout=3.0)
                    d = r.json()
                log.write("[bold cyan]-- Backend Status ---------------------[/bold cyan]")
                log.write(f"  CPU usage  : [yellow]{d.get('cpu_percent', '?')}%[/yellow]")
                log.write(f"  RAM usage  : [yellow]{d.get('memory_percent', '?')}%[/yellow]"
                          f"  ({d.get('memory_used_gb', '?')} / {d.get('memory_total_gb', '?')} GB)")
                log.write(f"  CPU cores  : [yellow]{d.get('cpu_cores', '?')}[/yellow]")
                log.write(f"  Sim. CUDA  : [yellow]{d.get('simulated_cuda_cores', '?')} cores[/yellow]")
                log.write(f"  CPU flags  : [yellow]{', '.join(d.get('cpu_flags', []))}[/yellow]")
                log.write(f"  Model      : [yellow]{d.get('active_model', '?')}[/yellow]")
            except Exception as e:
                log.write(f"[red]Backend unreachable: {e}[/red]")
        elif lower == "/models":
            try:
                async with httpx.AsyncClient() as c:
                    r = await c.get(f"{BASE_URL}/models", timeout=5.0)
                    models = r.json().get("models", [])
                if models:
                    log.write("[bold cyan]-- Available Models (from Ollama) -----[/bold cyan]")
                    for m in models:
                        active = "  [green]<-- active[/green]" if m["name"] == self.selected_model else ""
                        log.write(f"  [cyan]{m['name']}[/cyan]{active}")
                else:
                    log.write("[yellow]No models found. Is Ollama running? Pull one: ollama pull qwen2.5:0.5b[/yellow]")
            except Exception as e:
                log.write(f"[red]Failed to fetch models: {e}[/red]")
        elif lower.startswith("/model "):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                log.write("[red]Usage: /model <name>   e.g. /model llama3.2:1b[/red]")
            else:
                self.selected_model = parts[1].strip()
                log.write(f"[yellow]Switched to model: [bold]{self.selected_model}[/bold][/yellow]")
        elif lower == "/bench":
            log.write("[yellow]Running all benchmarks...[/yellow]")
            await self._run_kernel("matmul", log)
            await self._run_kernel("bandwidth", log)
            await self._run_kernel("vector", log)
        elif lower == "/compile":
            await self._compile_example(log)
        else:
            log.write(f"[red]Unknown command:[/red] [bold]{cmd}[/bold]")
            log.write("[dim]Type /help to see all commands.[/dim]")

    async def _do_agent(self, prompt: str) -> None:
        log = self.query_one("#agent_log", RichLog)
        log.write(f"[bold white]You:[/bold white] {prompt}")
        payload = {"prompt": prompt, "num_predict": 1024}
        if self.selected_model:
            payload["model"] = self.selected_model
        try:
            async with httpx.AsyncClient() as c:
                log.write("[dim]Thinking...[/dim]")
                r = await c.post(f"{BASE_URL}/agent", json=payload, timeout=120.0)
                data = r.json()
            raw = data.get("output", "")
            used = data.get("model", "?")
            tok = data.get("tokens", "?")
            if "</thinking>" in raw:
                think, answer = raw.split("</thinking>", 1)
                think = think.replace("<thinking>", "").strip()
                preview = think[:300] + ("..." if len(think) > 300 else "")
                log.write(f"[dim]<thinking> {preview} </thinking>[/dim]")
                log.write(f"[bold green]Agent ({used}, {tok} tok):[/bold green] {answer.strip()}")
            else:
                log.write(f"[bold green]Agent ({used}, {tok} tok):[/bold green] {raw.strip()}")
        except Exception as e:
            log.write(f"[red]Error: {e}[/red]")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button.id
        if btn == "btn_matmul":
            await self._run_kernel("matmul")
        elif btn == "btn_bw":
            await self._run_kernel("bandwidth")
        elif btn == "btn_vec":
            await self._run_kernel("vector")
        elif btn == "btn_all":
            log = self.query_one("#kernel_log", RichLog)
            await self._run_kernel("matmul", log)
            await self._run_kernel("bandwidth", log)
            await self._run_kernel("vector", log)
        elif btn == "btn_compile":
            await self._compile_example(self.query_one("#kernel_log", RichLog))

    async def _run_kernel(self, kind: str, log: RichLog = None) -> None:
        if log is None:
            log = self.query_one("#kernel_log", RichLog)
        log.write(f"[yellow]Running [bold]{kind}[/bold] benchmark...[/yellow]")
        try:
            async with httpx.AsyncClient() as c:
                r = await c.post(f"{BASE_URL}/kernel/{kind}", json={}, timeout=60.0)
                data = r.json()
            log.write(f"[green]{json.dumps(data, indent=2)}[/green]")
        except Exception as e:
            log.write(f"[red]Error: {e}[/red]")

    async def _compile_example(self, log: RichLog) -> None:
        log.write("[yellow]Compiling example CUDA kernel...[/yellow]")
        try:
            async with httpx.AsyncClient() as c:
                r = await c.post(f"{BASE_URL}/compile",
                                 json={"code": EXAMPLE_KERNEL, "target": "avx2"},
                                 timeout=30.0)
                data = r.json()
            if data.get("status") == "success":
                log.write(f"[green]Compiled OK -> {data.get('binary_path', '?')}[/green]")
                log.write(f"[dim]Flags: {data.get('flags', [])}[/dim]")
            else:
                log.write(f"[red]Compile failed: {data}[/red]")
        except Exception as e:
            log.write(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    MicroCUDATUI().run()
