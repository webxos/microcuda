#!/usr/bin/env python3
"""MICROCUDA v2.2 - Textual TUI with Memory, Skills, and Proactive Tasks"""
import asyncio
import httpx
import json
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Input, Static, Label, ProgressBar, RichLog, Button, TabbedContent, TabPane, DataTable
from textual import work

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
                yield Label("Memory Keys", classes="mon_lbl")
                yield Label("--", id="mem_keys_lbl", classes="mon_val")
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
                self.query_one("#mem_keys_lbl", Label).update(str(d.get("memory_entries", 0)))
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
    RichLog:focus { border: solid #00ff41; }
    Input { background: #111; border: solid #00aa22; color: #e0e0e0; margin: 1 0 0 0; }
    Input:focus { border: solid #00ff41; }
    Button { margin: 0 1 1 0; min-width: 18; }
    Button.-primary { background: #002800; color: #00ff41; border: solid #005500; }
    Button.-success { background: #001828; color: #00aaff; border: solid #003366; }
    Button:hover { background: #003300; }
    .btn_row { height: 5; padding: 1 0; }
    DataTable { background: #0a0a0a; border: solid #1a3a1a; }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_active", "Clear log", show=True),
        Binding("f1", "show_help", "Help", show=True),
        Binding("pgup", "scroll_log_up", "Scroll Up", show=False),
        Binding("pgdn", "scroll_log_down", "Scroll Down", show=False),
        Binding("up", "focus_up", "Focus Up", show=False),
        Binding("down", "focus_down", "Focus Down", show=False),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_model: str = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield MonitorBar()
        with TabbedContent(initial="agent"):
            with TabPane("🤖 Agent", id="agent"):
                yield RichLog(id="agent_log", markup=True, highlight=True, auto_scroll=True, wrap=True)
                yield Input(placeholder="Ask the model... (type /help for commands)", id="agent_input")
            with TabPane("⚙️ Kernels", id="kernels"):
                yield RichLog(id="kernel_log", markup=True, auto_scroll=True, wrap=True)
                with Horizontal(classes="btn_row"):
                    yield Button("MatMul", id="btn_matmul", variant="primary")
                    yield Button("Bandwidth", id="btn_bw", variant="primary")
                    yield Button("Vector Add", id="btn_vec", variant="primary")
                    yield Button("Run All", id="btn_all", variant="success")
                    yield Button("Compile CUDA", id="btn_compile", variant="success")
            with TabPane("💾 Memory", id="memory"):
                yield DataTable(id="memory_table")
                with Horizontal():
                    yield Input(placeholder="Key", id="mem_key")
                    yield Input(placeholder="Value", id="mem_value")
                    yield Button("Set", id="mem_set", variant="primary")
                    yield Button("Refresh", id="mem_refresh", variant="default")
            with TabPane("🛠️ Skills", id="skills"):
                yield DataTable(id="skills_table")
                with Horizontal():
                    yield Input(placeholder="Skill name", id="skill_name")
                    yield Input(placeholder="Command (shell/python)", id="skill_cmd")
                    yield Button("Add", id="skill_add", variant="primary")
                    yield Button("Refresh Skills", id="skill_refresh", variant="default")
        yield Footer()

    async def on_mount(self) -> None:
        log = self.query_one("#agent_log", RichLog)
        log.write("[bold green]MICROCUDA v2.2 – Model‑Driven Tools & Memory[/bold green]")
        log.write("[dim]Agent can now use tools (benchmarks, memory, tasks). Type /help[/dim]")
        await self.refresh_memory_table()
        await self.refresh_skills_table()

    def action_quit(self) -> None:
        self.exit()

    def action_clear_active(self) -> None:
        try:
            active = self.query_one("TabbedContent").active
            log_id = {"agent": "#agent_log", "kernels": "#kernel_log"}.get(active, "#agent_log")
            log = self.query_one(log_id, RichLog)
            log.clear()
            if active == "agent":
                log.write("[green]Log cleared.[/green]")
        except Exception:
            pass

    def action_show_help(self) -> None:
        try:
            self.query_one("#agent_log", RichLog).write(HELP_TEXT)
        except Exception:
            pass

    def action_scroll_log_up(self) -> None:
        """Scroll the active log up one page."""
        active = self.query_one("TabbedContent").active
        log_id = {"agent": "#agent_log", "kernels": "#kernel_log"}.get(active, "#agent_log")
        log = self.query_one(log_id, RichLog)
        log.scroll_page_up()

    def action_scroll_log_down(self) -> None:
        """Scroll the active log down one page."""
        active = self.query_one("TabbedContent").active
        log_id = {"agent": "#agent_log", "kernels": "#kernel_log"}.get(active, "#agent_log")
        log = self.query_one(log_id, RichLog)
        log.scroll_page_down()

    def action_focus_up(self) -> None:
        """Move focus up (to input or previous widget)."""
        self.screen.focus_previous()

    def action_focus_down(self) -> None:
        """Move focus down (to input or next widget)."""
        self.screen.focus_next()

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
            log.write("[bold green]MICROCUDA v2.2[/bold green] – CPU-Accelerated AI Harness with Memory, Tools & Proactivity")
        elif lower == "/status":
            try:
                async with httpx.AsyncClient() as c:
                    r = await c.get(f"{BASE_URL}/status", timeout=3.0)
                    d = r.json()
                log.write("[bold cyan]-- Backend Status ---------------------[/bold cyan]")
                log.write(f"  CPU usage  : [yellow]{d.get('cpu_percent', '?')}%[/yellow]")
                log.write(f"  RAM usage  : [yellow]{d.get('memory_percent', '?')}%[/yellow]")
                log.write(f"  Memory keys: [yellow]{d.get('memory_entries', 0)}[/yellow]")
                log.write(f"  Tasks      : [yellow]{d.get('scheduled_tasks', 0)}[/yellow]")
                log.write(f"  Model      : [yellow]{d.get('active_model', '?')}[/yellow]")
                log.write(f"  Flags      : [yellow]{', '.join(d.get('cpu_flags', []))}[/yellow]")
            except Exception as e:
                log.write(f"[red]Backend unreachable: {e}[/red]")
        elif lower == "/models":
            try:
                async with httpx.AsyncClient() as c:
                    r = await c.get(f"{BASE_URL}/models", timeout=5.0)
                    models = r.json().get("models", [])
                if models:
                    log.write("[bold cyan]-- Available Models -----[/bold cyan]")
                    for m in models:
                        log.write(f"  [cyan]{m['name']}[/cyan]")
                else:
                    log.write("[yellow]No models found.[/yellow]")
            except Exception as e:
                log.write(f"[red]{e}[/red]")
        elif lower.startswith("/model "):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                log.write("[red]Usage: /model <name>[/red]")
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
        elif lower.startswith("/memory list"):
            await self.refresh_memory_table()
            log.write("[green]Memory table refreshed.[/green]")
        elif lower.startswith("/memory set "):
            parts = cmd.split(maxsplit=3)
            if len(parts) >= 4:
                key = parts[2]
                val = parts[3]
                async with httpx.AsyncClient() as c:
                    await c.post(f"{BASE_URL}/memory", json={"key": key, "value": val})
                log.write(f"[green]Memory set: {key} = {val}[/green]")
                await self.refresh_memory_table()
            else:
                log.write("[red]Usage: /memory set key value[/red]")
        else:
            log.write(f"[red]Unknown command: {cmd}[/red]")

    async def _do_agent(self, prompt: str) -> None:
        log = self.query_one("#agent_log", RichLog)
        log.write(f"[bold white]You:[/bold white] {prompt}")
        payload = {"prompt": prompt, "num_predict": 1024}
        if self.selected_model:
            payload["model"] = self.selected_model
        try:
            async with httpx.AsyncClient() as c:
                log.write("[dim]Agent thinking...[/dim]")
                r = await c.post(f"{BASE_URL}/agent", json=payload, timeout=180.0)
                data = r.json()
            raw = data.get("output", "")
            used = data.get("model", "?")
            tok = data.get("tokens", "?")
            tool_calls = data.get("tool_calls", [])
            if tool_calls:
                log.write(f"[dim]🛠️ Executed tools: {', '.join(tc['name'] for tc in tool_calls)}[/dim]")
            if "</thinking>" in raw:
                think, answer = raw.split("</thinking>", 1)
                think = think.replace("<thinking>", "").strip()
                preview = think[:300] + ("..." if len(think) > 300 else "")
                log.write(f"[dim]<thinking> {preview} </thinking>[/dim]")
                log.write(f"[bold green]Agent ({used}, {tok} tok):[/bold green] {answer.strip()}")
            else:
                log.write(f"[bold green]Agent ({used}, {tok} tok):[/bold green] {raw.strip()}")
            # Refresh tables after possible memory changes
            await self.refresh_memory_table()
            await self.refresh_skills_table()
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
        elif btn == "mem_set":
            key = self.query_one("#mem_key", Input).value.strip()
            val = self.query_one("#mem_value", Input).value.strip()
            if key and val:
                async with httpx.AsyncClient() as c:
                    await c.post(f"{BASE_URL}/memory", json={"key": key, "value": val})
                await self.refresh_memory_table()
                self.query_one("#agent_log", RichLog).write(f"[green]Memory stored: {key} = {val}[/green]")
        elif btn == "mem_refresh":
            await self.refresh_memory_table()
        elif btn == "skill_add":
            name = self.query_one("#skill_name", Input).value.strip()
            cmd = self.query_one("#skill_cmd", Input).value.strip()
            if name and cmd:
                async with httpx.AsyncClient() as c:
                    await c.post(f"{BASE_URL}/skills", json={"name": name, "command": cmd})
                await self.refresh_skills_table()
                self.query_one("#agent_log", RichLog).write(f"[green]Skill added: {name}[/green]")
        elif btn == "skill_refresh":
            await self.refresh_skills_table()

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
            else:
                log.write(f"[red]Compile failed: {data}[/red]")
        except Exception as e:
            log.write(f"[red]Error: {e}[/red]")

    async def refresh_memory_table(self):
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get(f"{BASE_URL}/memory", timeout=2.0)
                data = r.json()
            table = self.query_one("#memory_table", DataTable)
            table.clear(columns=True)
            table.add_columns("Key", "Value")
            for item in data.get("memory", []):
                table.add_row(item["key"], item["value"])
        except Exception:
            pass

    async def refresh_skills_table(self):
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get(f"{BASE_URL}/skills", timeout=2.0)
                data = r.json()
            table = self.query_one("#skills_table", DataTable)
            table.clear(columns=True)
            table.add_columns("Name", "Command")
            for item in data.get("skills", []):
                table.add_row(item["name"], item["command"])
        except Exception:
            pass

if __name__ == "__main__":
    MicroCUDATUI().run()
