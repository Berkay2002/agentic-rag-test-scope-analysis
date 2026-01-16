"""Display utilities for interactive chat."""

import json
import time
from datetime import datetime

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

# Modern minimalist color palette
COLORS = {
    "primary": "#00D4AA",  # Green accent for success/responses
    "accent": "#5EB0FF",   # Blue accent for info/tool calls
    "warning": "#FFB800",  # Yellow accent for warnings/pending
    "error": "#FF5B5B",    # Red accent for errors/critical
    "neutral": "#8B8B8B",  # Subtle gray for secondary text
    "bg_subtle": "#2B2B2B", # Dark background for code blocks
    "text": "#EAEAEA",     # Primary text color
}


def print_welcome(
    console: Console,
    thread_id: str,
    checkpointer_backend: str,
    enable_hitl: bool,
) -> None:
    """Print welcome message.

    Args:
        console: Rich console for output.
        thread_id: Current session thread ID.
        checkpointer_backend: Backend type ("postgres", "memory", or empty).
        enable_hitl: Whether HITL mode is enabled.
    """
    # Create session info badge
    session_info = Text()
    session_info.append(" ", style="white")
    session_info.append(f"{thread_id}", style=COLORS["accent"])

    # Create mode badge
    mode_text = "Safe Mode" if enable_hitl else "YOLO Mode"
    mode_color = COLORS["primary"] if enable_hitl else COLORS["warning"]
    mode_icon = "🛡️" if enable_hitl else "⚡"
    mode_badge = Text()
    mode_badge.append(f"{mode_icon} {mode_text}", style=mode_color)

    # Create persistence info
    persistence = "PostgreSQL (durable)" if checkpointer_backend == "postgres" else "In-memory (session only)"
    persistence_color = COLORS["primary"] if checkpointer_backend == "postgres" else COLORS["warning"]

    # Build commands table
    commands_table = Table(show_header=False, box=None, padding=(0, 1, 0, 0))
    commands_table.add_column("Command", style=COLORS["accent"], width=12)
    commands_table.add_column("Description")
    commands_table.add_row("/help", "Show available commands")
    commands_table.add_row("/clear", "Clear the screen")
    commands_table.add_row("/history", "Show message history")
    commands_table.add_row("/stats", "Show conversation statistics")
    commands_table.add_row("/reset", "Start new conversation")
    commands_table.add_row("/save", "Save conversation to file")
    commands_table.add_row("/export", "Export transcript (add --verbose for tool details)")
    commands_table.add_row("/verbose", "Toggle tool call arguments")
    commands_table.add_row("/thinking", "View or set thinking configuration")
    commands_table.add_row("/exit", "Exit the chat")

    # Build tips
    tips_text = """\
**Tips:**

• Ask about test coverage, dependencies, and requirements

• The agent has access to vector search, keyword search, \
graph traversal, and hybrid search tools

• Type naturally—the agent will understand your intent

**Persistence:**
""" + f"{persistence}"

    if checkpointer_backend == "postgres":
        tips_text += f"""

**Resume Session:**
`agrag chat --thread-id {thread_id}`"""

    # Create the layout
    content = Group(
        Text("AgRAG Interactive Chat", style=f"bold {COLORS['primary']}", justify="center"),
        Text("Agentic GraphRAG for Test Scope Analysis", style=COLORS["neutral"], justify="center"),
        Text("\n"),  # Spacing
        commands_table,
        Text("\n"),  # Spacing
        Markdown(tips_text)
    )

    # Create the main panel
    panel = Panel(
        content,
        title=f"{mode_badge} • Session",
        subtitle=session_info,
        border_style=COLORS["primary"],
        padding=(1, 2),
    )

    console.print(panel)
    console.print()


def print_agent_response(console: Console, response: str) -> None:
    """Print agent response in a formatted panel.

    Args:
        console: Rich console for output.
        response: The agent's response text.
    """
    # Add subtle header with timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")
    header = Text()
    header.append("● ", style=COLORS["primary"])
    header.append("Assistant", style=f"bold {COLORS['primary']}")
    header.append(f"  {timestamp}", style="dim")

    console.print()
    console.print(header)
    console.print(
        Panel(
            Markdown(response),
            title=None,
            border_style=COLORS["primary"],
            padding=(1, 2),
        )
    )


def print_tool_call(
    console: Console, tool_name: str, tool_call_id: str, arguments: dict
) -> None:
    """Print tool call details when verbose mode is enabled."""
    # Header with timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")
    header = Text()
    header.append("⚡ ", style=COLORS["accent"])
    header.append("Tool Call  ", style=f"bold {COLORS['accent']}")
    header.append(f"{timestamp}", style="dim")

    console.print()
    console.print(header)

    # Tool name and ID
    tool_info = Text()
    tool_info.append("Tool: ", style=COLORS["neutral"])
    tool_info.append(f"{tool_name}\n", style=COLORS["accent"])
    tool_info.append("ID: ", style=COLORS["neutral"])
    tool_info.append(tool_call_id or "n/a", style="dim")

    args_json = json.dumps(arguments, indent=2, ensure_ascii=True, default=str)

    content = Group(
        tool_info,
        Text("\n"),  # Spacing
        Markdown(f"**Arguments:**\n```json\n{args_json}\n```"),
    )

    console.print(
        Panel(
            content,
            title=None,
            border_style=COLORS["accent"],
            padding=(1, 2),
        )
    )


def print_query_stats(console: Console, tool_calls: int, model_calls: int) -> None:
    """Print mini statistics for a single query.

    Args:
        console: Rich console for output.
        tool_calls: Number of tool calls in this query.
        model_calls: Number of model calls in this query.
    """
    # Create chip-style stats
    stats_line = Text()
    stats_line.append(" ", style="white")
    stats_line.append("⚡", style=COLORS["accent"])
    stats_line.append(f" {tool_calls}", style=COLORS["accent"])
    stats_line.append(" tools", style=COLORS["neutral"])

    stats_line.append(" • ", style="dim")

    stats_line.append("\N{brain}", style=COLORS["primary"])
    stats_line.append(f" {model_calls}", style=COLORS["primary"])
    stats_line.append(" models", style=COLORS["neutral"])

    console.print(stats_line)
    console.print()


def print_error(console: Console, message: str, traceback_str: str | None = None) -> None:
    """Print an error message.

    Args:
        console: Rich console for output.
        message: Error message.
        traceback_str: Optional traceback string.
    """
    # Header
    header = Text()
    header.append("✗ ", style=COLORS["error"])
    header.append("Error", style=f"bold {COLORS['error']}")

    console.print()


def format_summary_table(
    summary,
    include_ragas: bool = False,
    include_trials: bool = False,
) -> str:
    """Format evaluation summary as an ASCII table."""

    def _format_value(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{value:.3f}"

    mean_metrics = {}
    std_metrics = {}
    if include_trials and getattr(summary, "trial_statistics", None):
        trial_stats = summary.trial_statistics or {}
        mean_metrics = trial_stats.get("mean_metrics", {})
        std_metrics = trial_stats.get("std_metrics", {})

    k_values = sorted(
        set(summary.avg_precision_at_k.keys())
        | set(summary.avg_recall_at_k.keys())
        | set(summary.avg_f1_at_k.keys())
    )

    rows = []

    rows.append(
        (
            "MAP",
            _format_value(mean_metrics.get("average_precision", summary.map_score)),
            _format_value(std_metrics.get("average_precision")) if include_trials else "-",
        )
    )
    rows.append(
        (
            "MRR",
            _format_value(mean_metrics.get("reciprocal_rank", summary.mrr_score)),
            _format_value(std_metrics.get("reciprocal_rank")) if include_trials else "-",
        )
    )

    for k in k_values:
        rows.append(
            (
                f"Precision@{k}",
                _format_value(mean_metrics.get(f"precision@{k}", summary.avg_precision_at_k.get(k))),
                _format_value(std_metrics.get(f"precision@{k}")) if include_trials else "-",
            )
        )
        rows.append(
            (
                f"Recall@{k}",
                _format_value(mean_metrics.get(f"recall@{k}", summary.avg_recall_at_k.get(k))),
                _format_value(std_metrics.get(f"recall@{k}")) if include_trials else "-",
            )
        )
        rows.append(
            (
                f"F1@{k}",
                _format_value(mean_metrics.get(f"f1@{k}", summary.avg_f1_at_k.get(k))),
                _format_value(std_metrics.get(f"f1@{k}")) if include_trials else "-",
            )
        )

    if include_ragas and summary.avg_ragas_metrics:
        for metric_key, metric_value in summary.avg_ragas_metrics.items():
            label = metric_key.replace("_", " ").title()
            rows.append((label, _format_value(metric_value), "-"))

    headers = ("Metric", "Mean", "Std Dev")
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def _line(left: str, mid: str, right: str, fill: str) -> str:
        return (
            left
            + mid.join(fill * (w + 2) for w in col_widths)
            + right
        )

    lines = [_line("┌", "┬", "┐", "─")]
    header_cells = [f" {headers[i].ljust(col_widths[i])} " for i in range(3)]
    lines.append("│" + "│".join(header_cells) + "│")
    lines.append(_line("├", "┼", "┤", "─"))

    for row in rows:
        cells = [f" {row[i].ljust(col_widths[i])} " for i in range(3)]
        lines.append("│" + "│".join(cells) + "│")

    lines.append(_line("└", "┴", "┘", "─"))

    stats_lines = []
    if summary.total_queries:
        success_rate = summary.successful_queries / max(1, summary.total_queries)
        stats_lines.append(
            f"Success Rate: {success_rate:.1%} ({summary.successful_queries}/{summary.total_queries} queries)"
        )

    if summary.avg_execution_time_ms:
        avg_seconds = summary.avg_execution_time_ms / 1000
        stats_lines.append(f"Avg Execution Time: {avg_seconds:.2f}s")

    if include_trials and summary.trial_statistics:
        trial_stats = summary.trial_statistics
        pass_at_k = trial_stats.get("pass_at_k")
        pass_pow_k = trial_stats.get("pass_pow_k")
        stability = trial_stats.get("stability_score")
        if pass_at_k is not None:
            stats_lines.append(f"Pass@k: {pass_at_k:.1%}")
        if pass_pow_k is not None:
            stats_lines.append(f"Pass^k: {pass_pow_k:.1%}")
        if stability is not None:
            stats_lines.append(f"Stability Score: {stability:.2f}")

    if stats_lines:
        lines.append("")
        lines.extend(stats_lines)

    return "\n".join(lines)
    console.print(header)
    console.print(
        Panel(
            Text(message, style=COLORS["text"]),
            border_style=COLORS["error"],
            padding=(1, 2),
        )
    )

    if traceback_str:
        console.print()
        console.print(
            Panel(
                Text(traceback_str, style="dim"),
                title="Traceback",
                border_style="dim",
                padding=(1, 2),
            )
        )

    console.print()
