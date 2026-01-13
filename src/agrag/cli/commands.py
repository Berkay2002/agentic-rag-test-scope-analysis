"""Command handling for interactive chat."""

from datetime import datetime
from typing import Protocol, Optional
import uuid

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agrag.cli.thinking import handle_thinking_command, format_thinking_budget


class ChatSessionProtocol(Protocol):
    """Protocol for chat session state that commands can modify."""

    thread_id: str
    message_count: int
    tool_calls_total: int
    model_calls_total: int
    start_time: datetime
    thinking_budget: Optional[int]
    enable_hitl: bool
    verbose: bool

    def _persistence_label(self) -> str: ...
    def export_conversation(
        self, filename: Optional[str] = None, include_tool_details: bool = False
    ) -> str: ...
    def reset_conversation(self) -> None: ...
    def set_verbose(self, enabled: bool) -> None: ...


def print_help(console: Console) -> None:
    """Print help message.

    Args:
        console: Rich console for output.
    """
    help_text = """
**Commands:**
- `/help` - Show this help
- `/clear` - Clear screen
- `/history` - Show message history
- `/stats` - Show statistics
- `/reset` - Start new conversation
- `/save` - Save conversation to file
- `/export` - Export conversation transcript (use --verbose for tool details)
- `/verbose` - Toggle tool call arguments and logs in output
- `/thinking [preset]` - View or set thinking budget
- `/exit`, `/quit` - Exit chat

**Example Queries:**
- "What tests cover handover requirements?"
- "Find all test cases related to authentication"
- "Show me functions called by initiate_handover"
- "Which requirements depend on REQ_AUTH_005?"
"""
    console.print(Panel(Markdown(help_text), title="Help", border_style="blue"))


def print_stats(console: Console, session: ChatSessionProtocol) -> None:
    """Print conversation statistics.

    Args:
        console: Rich console for output.
        session: Chat session with statistics.
    """
    duration = datetime.now() - session.start_time
    mode = (
        "🚦 Safe Mode (you approve each tool)"
        if session.enable_hitl
        else "⚡ YOLO Mode (autonomous)"
    )
    stats = f"""
**Session Statistics:**
- Session ID: `{session.thread_id}`
- Messages: {session.message_count}
- Total Tool Calls: {session.tool_calls_total}
- Total Model Calls: {session.model_calls_total}
- Duration: {duration.seconds // 60}m {duration.seconds % 60}s
- Persistence: {session._persistence_label()}
- Mode: {mode}
- Thinking Budget: {format_thinking_budget(session.thinking_budget)}
"""
    console.print(Panel(Markdown(stats), title="Statistics", border_style="cyan"))


def save_conversation(console: Console, thread_id: str) -> None:
    """Save conversation to file.

    Args:
        console: Rich console for output.
        thread_id: Thread ID for filename.
    """
    filename = f"conversation_{thread_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        with open(filename, "w") as f:
            f.write("AgRAG Conversation\n")
            f.write(f"Session ID: {thread_id}\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            # Note: Full conversation history retrieval not yet implemented
            f.write("(Conversation history retrieval not yet implemented)\n")
        console.print(f"[green]✓ Conversation saved to {filename}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to save: {e}[/red]")


class CommandHandler:
    """Handles chat commands."""

    def __init__(self, console: Console, session: ChatSessionProtocol):
        """Initialize command handler.

        Args:
            console: Rich console for output.
            session: Chat session to operate on.
        """
        self.console = console
        self.session = session
        self._print_welcome_callback = None

    def set_welcome_callback(self, callback) -> None:
        """Set callback for printing welcome message (used by /clear).

        Args:
            callback: Function to call for printing welcome.
        """
        self._print_welcome_callback = callback

    def handle(self, user_input: str) -> bool:
        """Handle special commands.

        Args:
            user_input: The user's input.

        Returns:
            True if should continue, False if should exit.
        """
        raw_command = user_input.strip()
        command = raw_command.lower()

        if command in ["/exit", "/quit"]:
            self.console.print("\n[green]Goodbye! 👋[/green]\n")
            return False

        elif command == "/help":
            print_help(self.console)

        elif command == "/clear":
            self.console.clear()
            if self._print_welcome_callback:
                self._print_welcome_callback()

        elif command == "/history":
            self.console.print("[yellow]History viewing not yet implemented[/yellow]")

        elif command == "/stats":
            print_stats(self.console, self.session)

        elif command == "/reset":
            self.session.thread_id = f"chat-{uuid.uuid4().hex[:8]}"
            self.session.message_count = 0
            self.session.tool_calls_total = 0
            self.session.model_calls_total = 0
            self.session.start_time = datetime.now()
            self.session.reset_conversation()
            self.console.print(
                f"[green]✓ Conversation reset. New session: {self.session.thread_id}[/green]"
            )

        elif command == "/save":
            save_conversation(self.console, self.session.thread_id)

        elif command.startswith("/export"):
            parts = raw_command.split()
            filename: Optional[str] = None
            include_tool_details = False
            for token in parts[1:]:
                token_lower = token.lower()
                if token_lower in {"--verbose", "--debug", "verbose", "debug"}:
                    include_tool_details = True
                    continue
                if filename is None:
                    filename = token
                    continue
                self.console.print(
                    "[red]Invalid /export usage. Use /export [filename] [--verbose].[/red]"
                )
                return True
            try:
                output_path = self.session.export_conversation(
                    filename=filename, include_tool_details=include_tool_details
                )
                self.console.print(f"[green]✓ Conversation exported to {output_path}[/green]")
            except Exception as e:
                self.console.print(f"[red]✗ Failed to export: {e}[/red]")

        elif command.startswith("/verbose"):
            parts = raw_command.split(maxsplit=1)
            if len(parts) == 1:
                self.session.set_verbose(not self.session.verbose)
            else:
                value = parts[1].strip().lower()
                if value in {"on", "true", "1", "yes"}:
                    self.session.set_verbose(True)
                elif value in {"off", "false", "0", "no"}:
                    self.session.set_verbose(False)
                else:
                    self.console.print(
                        "[red]Invalid /verbose value. Use on/off or omit to toggle.[/red]"
                    )
                    return True
            state = "on" if self.session.verbose else "off"
            self.console.print(f"[green]✓ Verbose mode is now {state}[/green]")

        elif command.startswith("/thinking"):
            new_budget = handle_thinking_command(
                self.console, raw_command, self.session.thinking_budget
            )
            if new_budget is not None:
                self.session.thinking_budget = new_budget

        else:
            self.console.print(f"[red]Unknown command: {command}[/red]")
            self.console.print("[yellow]Type /help for available commands[/yellow]")

        return True
