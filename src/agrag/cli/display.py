"""Display utilities for interactive chat."""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


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
    welcome = """
# AgRAG Interactive Chat

Welcome to the Agentic GraphRAG Test Scope Analysis system!

**Available Commands:**
- `/help` - Show this help message
- `/clear` - Clear the screen
- `/history` - Show conversation history
- `/stats` - Show conversation statistics
- `/reset` - Reset conversation (start fresh)
- `/save` - Save conversation to file
- `/exit` or `/quit` - Exit the chat

**Tips:**
- Ask questions about test coverage, dependencies, and requirements
- The agent has access to vector search, keyword search, graph traversal, and hybrid search tools
- Type naturally - the agent will understand your intent and select the right tools
"""
    if checkpointer_backend == "postgres":
        welcome += f"\n**Session ID:** `{thread_id}`\n"
        welcome += "Your conversation is automatically saved and can be resumed later.\n"
        welcome += f"Resume with: `agrag chat --thread-id {thread_id}`\n"
    elif checkpointer_backend == "memory":
        welcome += f"\n**Session ID:** `{thread_id}`\n"
        welcome += (
            "Session data is stored in memory for this run only (cannot be resumed after exit).\n"
        )

    if enable_hitl:
        welcome += "\n**🚦 Safe Mode (HITL)**\n"
        welcome += "The agent will ask for your approval before executing each tool.\n"
        welcome += "This lets you see and control exactly what the agent does.\n"
    else:
        welcome += "\n**⚡ YOLO Mode Active**\n"
        welcome += "The agent is executing autonomously without asking for approval.\n"
        welcome += "Use this mode only when you trust the agent completely.\n"

    console.print(Panel(Markdown(welcome), title="AgRAG Chat", border_style="green"))
    console.print()


def print_agent_response(console: Console, response: str) -> None:
    """Print agent response in a formatted panel.

    Args:
        console: Rich console for output.
        response: The agent's response text.
    """
    console.print()
    console.print(
        Panel(
            Markdown(response),
            title="Agent Response",
            border_style="green",
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
    from rich.text import Text

    stats_text = Text()
    stats_text.append("Tool calls: ", style="dim")
    stats_text.append(str(tool_calls), style="cyan")
    stats_text.append(" | Model calls: ", style="dim")
    stats_text.append(str(model_calls), style="cyan")
    console.print(stats_text)
    console.print()


def print_error(console: Console, message: str, traceback_str: str | None = None) -> None:
    """Print an error message.

    Args:
        console: Rich console for output.
        message: Error message.
        traceback_str: Optional traceback string.
    """
    console.print(f"\n[red]✗ {message}[/red]\n")
    if traceback_str:
        console.print(f"[dim]{traceback_str}[/dim]")
