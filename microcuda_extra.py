# MICROCUDA v2.2 – Extra data (help, skill examples, kernel template)

HELP_TEXT = """\
[bold green]╔══════════════════════════════════════════════════════════════════════╗
║              MICROCUDA v2.2 – Model‑Driven Tools & Memory                ║
╚══════════════════════════════════════════════════════════════════════════╝[/bold green]

[bold cyan]  /help[/bold cyan]           Show this help
[bold cyan]  /status[/bold cyan]         Show backend stats
[bold cyan]  /models[/bold cyan]         List Ollama models
[bold cyan]  /model NAME[/bold cyan]     Switch model
[bold cyan]  /clear[/bold cyan]          Clear agent log
[bold cyan]  /bench[/bold cyan]          Run all benchmarks
[bold cyan]  /compile[/bold cyan]        Compile example CUDA kernel

[bold yellow]⚡ Model‑Driven Tools (Agent can request)[/bold yellow]
  When you ask the agent to perform actions, it will output:
  <tool name="run_benchmark" args='{"type":"matmul"}'></tool>
  Supported tools:
    • run_benchmark  – type: matmul, bandwidth, vector
    • compile_cuda   – code: (CUDA kernel string)
    • memory_set     – key, value (store facts)
    • memory_get     – key
    • memory_delete  – key
    • schedule_task  – name, interval_seconds, action (e.g., "send_webhook")
    • list_tasks     – no args
    • send_webhook   – url, payload
    • add_skill      – name, command (shell or python expression)
    • list_skills    – no args

[bold cyan]💾 Long‑term memory[/bold cyan]
  Use `/memory list` or `/memory set key=value` in chat.
  The agent can also set/recall facts automatically.

[bold cyan]⏰ Proactive Tasks[/bold cyan]
  Use `/task add "task name" interval_seconds command`
  Example: `/task add "status report" 60 "send_webhook http://localhost:9000 {'msg':'alive'}"`

[dim]Tips:
  The agent thinks in <thinking> tags, then executes tools and refines answers.
  Any tool call is executed automatically – you see the result in chat.
  Use /memory recall to see all stored core memories.[/dim]
"""

EXAMPLE_KERNEL = """\
__global__ void vecAdd(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}
"""

DEFAULT_SKILLS = {
    "greet": "echo 'Hello from skill!'",
    "status_skill": "python -c 'import psutil; print(psutil.cpu_percent())'"
}
