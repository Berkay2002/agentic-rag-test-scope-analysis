"""Command handling for interactive chat."""

from datetime import datetime
from typing import Protocol, Optional
import uuid

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agrag.cli.thinking import handle_thinking_command, format_thinking_setting

# Import colors from display module
from agrag.cli.display import COLORS


class ChatSessionProtocol(Protocol):
    """Protocol for chat session state that commands can modify."""

    thread_id: str
    message_count: int
    tool_calls_total: int
    model_calls_total: int
    start_time: datetime
    thinking_budget: Optional[int]
    thinking_level: Optional[str]
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
    # Create commands table
    commands_table = Table(show_header=False, box=None, padding=(0, 1, 0, 0))
    commands_table.add_column("Command", style=COLORS["accent"], width=12, no_wrap=True)
    commands_table.add_column("Description", style="white")
    commands_table.add_row("/help", "Show this help message")
    commands_table.add_row("/clear", "Clear the screen")
    commands_table.add_row("/history", "Show message history")
    commands_table.add_row("/stats", "Show conversation statistics")
    commands_table.add_row("/reset", "Start new conversation")
    commands_table.add_row("/save", "Save conversation to file")
    commands_table.add_row("/export [file] [--verbose]", "Export conversation transcript")
    commands_table.add_row("/verbose [on|off]", "Toggle tool call details")
    commands_table.add_row("/thinking [level|budget]", "View or set thinking configuration")
    commands_table.add_row("/exit, /quit", "Exit the chat")

    # Example queries
    examples_text = """\
**Example Queries:**

• "What tests cover handover requirements?"

• "Find all test cases related to authentication"

• "Show me functions called by initiate_handover"

• "Which requirements depend on REQ_AUTH_005?"
"""

    # Title
    title = Text()
    title.append("❓ ", style=COLORS["accent"])
    title.append("Help", style=f"bold {COLORS['accent']}")

    content = Text()
    content.append(commands_table)
    content.append("\n")
    content.append(Markdown(examples_text))

    console.print()
    console.print(Panel(content, title=title, border_style=COLORS["accent"], padding=(1, 2)))
    console.print()


def print_stats(console: Console, session: ChatSessionProtocol) -> None:
    """Print conversation statistics.

    Args:
        console: Rich console for output.
        session: Chat session with statistics.
    """
    duration = datetime.now() - session.start_time

    # Create stats table
    stats_table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    stats_table.add_column("", style=COLORS["neutral"], width=20)
    stats_table.add_column("", style="white")

    stats_table.add_row("Session ID:", f"{session.thread_id}")
    stats_table.add_row("Messages:", f"{session.message_count}")
    stats_table.add_row("Total Tool Calls:", f"{session.tool_calls_total}")
    stats_table.add_row("Total Model Calls:", f"{session.model_calls_total}")
    stats_table.add_row("Duration:", f"{duration.seconds // 60}m {duration.seconds % 60}s")
    stats_table.add_row("Persistence:", f"{session._persistence_label()}")

    mode_text = "Safe Mode (you approve each tool)" if session.enable_hitl else "YOLO Mode (autonomous)"
    mode_icon = "🛡️" if session.enable_hitl else "⚡"
    mode_style = COLORS["primary"] if session.enable_hitl else COLORS["warning"]
    mode_display = Text()
    mode_display.append(f"{mode_icon} ", style=mode_style)
    mode_display.append(mode_text, style=mode_style)
    stats_table.add_row("Mode:", mode_display)

    stats_table.add_row(
        "Thinking:",
        f"{format_thinking_setting(session.thinking_level, session.thinking_budget)}",
    )

    # Title
    title = Text()
    title.append("📊 ", style=COLORS["primary"])
    title.append("Session Statistics", style=f"bold {COLORS['primary']}")

    console.print()
    console.print(Panel(stats_table, title=title, border_style=COLORS["primary"], padding=(1, 2)))
    console.print()



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
            new_setting = handle_thinking_command(
                self.console,
                raw_command,
                self.session.thinking_level,
                self.session.thinking_budget,
            )
            if new_setting is not None:
                self.session.thinking_level, self.session.thinking_budget = new_setting

        else:
            self.console.print(f"[red]Unknown command: {command}[/red]")
            self.console.print("[yellow]Type /help for available commands[/yellow]")

        return True
