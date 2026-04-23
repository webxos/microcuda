# MICROCUDA v2.1 – Extra data (help text, kernel examples)

HELP_TEXT = """\
[bold green]╔══════════════════════════════════════════╗
║     MICROCUDA v2.1  Agent Commands       ║
╚══════════════════════════════════════════╝[/bold green]

[bold cyan]  /help[/bold cyan]           Show this help message
[bold cyan]  /status[/bold cyan]         Show backend CPU/RAM/core stats
[bold cyan]  /models[/bold cyan]         List available Ollama models
[bold cyan]  /model NAME[/bold cyan]     Switch active model (any from your 'ollama list')
[bold cyan]  /clear[/bold cyan]          Clear the agent log
[bold cyan]  /bench[/bold cyan]          Run all three CPU benchmarks
[bold cyan]  /compile[/bold cyan]        Compile the example CUDA kernel
[bold cyan]  /version[/bold cyan]        Show version info

[bold yellow]  Any other text[/bold yellow]  Sent to the AI agent as a prompt

[dim]Tips:
  The agent wraps responses in <thinking>...</thinking> for CoT.
  Use /model to switch between any model you have pulled locally.
  Ctrl+Q = quit   Ctrl+L = clear active log   F1 = help[/dim]
"""

EXAMPLE_KERNEL = """\
__global__ void vecAdd(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}
"""
