"""Interactive chat interface for the AgRAG agent."""

import sys
import uuid
from typing import Optional, Dict, Any, List
import json
from datetime import datetime

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from langchain_core.messages import AIMessage

from agrag.core import create_agent_graph, create_initial_state
from agrag.core.checkpointing import initialize_checkpointer, summarize_error
from agrag.config import settings

THINKING_PRESETS = {
    "low": 256,
    "medium": 1024,
    "high": 4096,
    "dynamic": -1,
}


class InteractiveChat:
    """Interactive chat interface for the AgRAG agent."""

    def __init__(
        self,
        thread_id: Optional[str] = None,
        enable_hitl: bool = True,
    ):
        """Initialize the interactive chat.

        Args:
            thread_id: Thread ID for conversation persistence. If None, generates a new one.
            enable_hitl: Whether to require approval before executing tools (default: True for safety).
        """
        self.console = Console()
        self.thread_id = thread_id or f"chat-{uuid.uuid4().hex[:8]}"
        self.enable_hitl = enable_hitl

        # Initialize checkpointer (falls back to in-memory when Postgres is unreachable)
        init_result = initialize_checkpointer(enable_hitl=enable_hitl)
        self.checkpointer = init_result.checkpointer
        self.checkpointer_backend = init_result.backend
        self.checkpointer_persistent = init_result.persistent

        if self.checkpointer_backend == "memory":
            warning_prefix = "[yellow]Warning: Could not use Postgres checkpointer."
            if init_result.error:
                warning_prefix += f" Reason: {summarize_error(init_result.error)}"
            self.console.print(f"{warning_prefix}[/yellow]")
            self.console.print(
                "[yellow]Falling back to in-memory persistence for this session only.[/yellow]"
            )
        elif self.enable_hitl and not self.checkpointer:
            self.console.print(
                "[yellow]HITL enabled but no checkpointer available; approvals will be disabled.[/yellow]"
            )

        # Create agent graph (interrupt behavior is controlled by checkpointer presence)
        self.graph = create_agent_graph(
            checkpointer=self.checkpointer if self.checkpointer else None,
        )

        # Setup prompt toolkit
        self.history = InMemoryHistory()

        # Command completer
        commands = [
            "/help",
            "/clear",
            "/history",
            "/stats",
            "/exit",
            "/quit",
            "/reset",
            "/save",
            "/thinking",
        ]
        self.completer = WordCompleter(commands, ignore_case=True)

        # Style
        self.style = Style.from_dict(
            {
                "prompt": "#00aa00 bold",
            }
        )

        self.session = PromptSession(
            history=self.history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=self.completer,
            style=self.style,
        )

        # Conversation stats
        self.message_count = 0
        self.tool_calls_total = 0
        self.model_calls_total = 0
        self.start_time = datetime.now()
        self.thinking_budget = settings.google_thinking_budget

    def _get_config(self) -> Dict[str, Any]:
        """Get the config for graph execution."""
        config = {}
        if self.checkpointer:
            config["configurable"] = {"thread_id": self.thread_id}
        return config

    def _persistence_label(self) -> str:
        """Describe current persistence backend."""
        if not self.checkpointer:
            return "Disabled"
        if self.checkpointer_backend == "postgres":
            return "PostgreSQL (durable)"
        if self.checkpointer_backend == "memory":
            return "In-memory (session only)"
        return "Enabled"

    def _print_welcome(self):
        """Print welcome message."""
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
        if self.checkpointer_backend == "postgres":
            welcome += f"\n**Session ID:** `{self.thread_id}`\n"
            welcome += "Your conversation is automatically saved and can be resumed later.\n"
            welcome += f"Resume with: `agrag chat --thread-id {self.thread_id}`\n"
        elif self.checkpointer_backend == "memory":
            welcome += f"\n**Session ID:** `{self.thread_id}`\n"
            welcome += "Session data is stored in memory for this run only (cannot be resumed after exit).\n"

        if self.enable_hitl:
            welcome += "\n**🚦 Safe Mode (HITL)**\n"
            welcome += "The agent will ask for your approval before executing each tool.\n"
            welcome += "This lets you see and control exactly what the agent does.\n"
        else:
            welcome += "\n**⚡ YOLO Mode Active**\n"
            welcome += "The agent is executing autonomously without asking for approval.\n"
            welcome += "Use this mode only when you trust the agent completely.\n"

        self.console.print(Panel(Markdown(welcome), title="AgRAG Chat", border_style="green"))
        self.console.print()

    def _print_help(self):
        """Print help message."""
        help_text = """
**Commands:**
- `/help` - Show this help
- `/clear` - Clear screen
- `/history` - Show message history
- `/stats` - Show statistics
- `/reset` - Start new conversation
- `/save` - Save conversation to file
- `/thinking [preset]` - View or set thinking budget
- `/exit`, `/quit` - Exit chat

**Example Queries:**
- "What tests cover handover requirements?"
- "Find all test cases related to authentication"
- "Show me functions called by initiate_handover"
- "Which requirements depend on REQ_AUTH_005?"
"""
        self.console.print(Panel(Markdown(help_text), title="Help", border_style="blue"))

    def _print_stats(self):
        """Print conversation statistics."""
        duration = datetime.now() - self.start_time
        stats = f"""
**Session Statistics:**
- Session ID: `{self.thread_id}`
- Messages: {self.message_count}
- Total Tool Calls: {self.tool_calls_total}
- Total Model Calls: {self.model_calls_total}
- Duration: {duration.seconds // 60}m {duration.seconds % 60}s
- Persistence: {self._persistence_label()}
- Mode: {'🚦 Safe Mode (you approve each tool)' if self.enable_hitl else '⚡ YOLO Mode (autonomous)'}
- Thinking Budget: {self._format_thinking_budget(self.thinking_budget)}
"""
        self.console.print(Panel(Markdown(stats), title="Statistics", border_style="cyan"))

    def _format_thinking_budget(self, value: Optional[int]) -> str:
        """Return a friendly label for the current thinking budget."""
        if value is None:
            return "Default (model decides)"
        preset = next((name for name, budget in THINKING_PRESETS.items() if budget == value), None)
        if preset:
            label = preset.capitalize()
        else:
            label = str(value)
        if value == -1:
            label += " (dynamic)"
        elif value == 0:
            label += " (disabled)"
        return label

    def _print_thinking_help(self):
        """Display thinking configuration help."""
        preset_details = "\n".join(
            f"- `{name}` = {budget if budget != -1 else '-1 (dynamic)' } tokens"
            for name, budget in THINKING_PRESETS.items()
        )
        help_text = f"""
**Thinking Settings**
- `/thinking` - Show current setting
- `/thinking <preset>` - Apply preset (low, medium, high, dynamic) or an integer token budget

{preset_details}

Use `dynamic` (-1) to let the model decide, or provide a numeric token budget (e.g., `/thinking 512`).
"""
        self.console.print(Panel(Markdown(help_text), title="Thinking Configuration", border_style="magenta"))

    def _handle_thinking_command(self, raw_command: str):
        """Handle the /thinking command."""
        parts = raw_command.split()

        if len(parts) == 1:
            self.console.print(
                f"[cyan]Current thinking budget:[/cyan] {self._format_thinking_budget(self.thinking_budget)}"
            )
            self._print_thinking_help()
            return

        target = parts[1].lower()
        if target in THINKING_PRESETS:
            value = THINKING_PRESETS[target]
        else:
            try:
                value = int(target)
            except ValueError:
                self.console.print(
                    "[red]Invalid thinking value. Use a preset (low/medium/high/dynamic) or integer tokens.[/red]"
                )
                return

        settings.google_thinking_budget = value
        self.thinking_budget = value
        self.console.print(
            f"[green]✓ Thinking budget set to {self._format_thinking_budget(value)}[/green]"
        )

    def _handle_command(self, user_input: str) -> bool:
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
            self._print_help()

        elif command == "/clear":
            self.console.clear()
            self._print_welcome()

        elif command == "/history":
            # History viewing from checkpointer not yet implemented
            self.console.print("[yellow]History viewing not yet implemented[/yellow]")

        elif command == "/stats":
            self._print_stats()

        elif command == "/reset":
            # Generate new thread ID
            self.thread_id = f"chat-{uuid.uuid4().hex[:8]}"
            self.message_count = 0
            self.tool_calls_total = 0
            self.model_calls_total = 0
            self.start_time = datetime.now()
            self.console.print(f"[green]✓ Conversation reset. New session: {self.thread_id}[/green]")

        elif command == "/save":
            # Save conversation to file
            filename = f"conversation_{self.thread_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            try:
                with open(filename, "w") as f:
                    f.write(f"AgRAG Conversation\n")
                    f.write(f"Session ID: {self.thread_id}\n")
                    f.write(f"Date: {datetime.now().isoformat()}\n")
                    f.write("=" * 80 + "\n\n")
                    # Note: Full conversation history retrieval not yet implemented
                    f.write("(Conversation history retrieval not yet implemented)\n")
                self.console.print(f"[green]✓ Conversation saved to {filename}[/green]")
            except Exception as e:
                self.console.print(f"[red]✗ Failed to save: {e}[/red]")

        elif command.startswith("/thinking"):
            self._handle_thinking_command(raw_command)

        else:
            self.console.print(f"[red]Unknown command: {command}[/red]")
            self.console.print("[yellow]Type /help for available commands[/yellow]")

        return True

    def _process_query(self, query: str):
        """Process a user query through the agent.

        Args:
            query: The user's query.
        """
        config = self._get_config()

        try:
            # Create initial state
            pending_state: Optional[Dict[str, Any]] = create_initial_state(query)

            tool_calls_this_query = 0
            model_calls_this_query = 0
            last_state = None

            # Stream the agent's execution (resume after each interrupt)
            with self.console.status("[bold green]Agent is thinking...") as status:
                while True:
                    interrupted = False

                    for event in self.graph.stream(
                        pending_state, config=config, stream_mode="updates"
                    ):
                        # Initial state should only be sent once
                        pending_state = None

                        # Track last state
                        last_state = event

                        # Check for interrupts (HITL)
                        if "__interrupt__" in event:
                            status.stop()
                            decision = self._handle_interrupt(event, config)
                            # Resume streaming from the stored checkpoint
                            interrupted = True
                            status.start()
                            status.update("[bold blue] Agent is reasoning...")
                            break

                        # Process events
                        for node_name, node_state in event.items():
                            if node_name == "call_model":
                                model_calls_this_query += 1
                                status.update("[bold blue] Agent is reasoning...")

                            elif node_name == "execute_tools":
                                tool_calls_this_query += 1
                                # Extract tool names if available
                                if "messages" in node_state:
                                    messages = node_state["messages"]
                                    for msg in messages:
                                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                                            tool_names = [tc["name"] for tc in msg.tool_calls]
                                            status.update(
                                                f"[bold yellow]🔧 Executing tools: {', '.join(tool_names)}"
                                            )

                            elif node_name == "finalize_answer":
                                status.update("[bold green]📝 Finalizing answer...")

                    if interrupted:
                        # Continue loop to consume the next portion of the stream
                        continue

                    # Execution finished (END reached)
                    break

            # Get final state (only if checkpointer is available)
            if self.checkpointer:
                final_state = self.graph.get_state(config)
                final_answer = final_state.values.get("final_answer", "No answer generated")
            else:
                # Extract from last event when no checkpointer
                final_answer = "No answer generated"
                if last_state and isinstance(last_state, dict):
                    # Check for final_answer in any node state
                    for node_data in last_state.values():
                        if isinstance(node_data, dict) and "final_answer" in node_data:
                            final_answer = node_data["final_answer"]
                            break

            # Update stats
            self.tool_calls_total += tool_calls_this_query
            self.model_calls_total += model_calls_this_query

            # Display answer
            self.console.print()
            self.console.print(
                Panel(
                    Markdown(final_answer),
                    title="Agent Response",
                    border_style="green",
                    padding=(1, 2),
                )
            )

            # Display mini stats
            stats_text = Text()
            stats_text.append("Tool calls: ", style="dim")
            stats_text.append(str(tool_calls_this_query), style="cyan")
            stats_text.append(" | Model calls: ", style="dim")
            stats_text.append(str(model_calls_this_query), style="cyan")
            self.console.print(stats_text)
            self.console.print()

        except TimeoutError as e:
            self.console.print(f"\n[red]✗ LLM timeout: {e}[/red]\n")
        except Exception as e:
            self.console.print(f"\n[red]✗ Error: {e}[/red]\n")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")

    def _extract_tool_calls(self, event: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect proposed tool calls from an interrupt event."""
        tool_calls: List[Dict[str, Any]] = []
        for node_name, node_state in event.items():
            if node_name == "__interrupt__":
                continue
            messages = node_state.get("messages") if isinstance(node_state, dict) else None
            if not messages:
                continue
            for message in messages:
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for call in message.tool_calls:
                        tool_calls.append({
                            "node": node_name,
                            "name": call.get("name"),
                            "args": call.get("args", {}),
                        })

        # Fallback: inspect current graph state if event payload lacked tool calls
        if not tool_calls:
            current_state = self.graph.get_state(config)
            messages = current_state.values.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    for call in last_message.tool_calls:
                        tool_calls.append({
                            "node": "call_model",
                            "name": call.get("name"),
                            "args": call.get("args", {}),
                        })

        return tool_calls

    def _handle_interrupt(self, event: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Handle HITL interrupt.

        Args:
            event: The interrupt event.
            config: The graph config.
        """
        self.console.print("\n[bold yellow]🚦 Approval Required[/bold yellow]\n")
        tool_calls = self._extract_tool_calls(event, config)

        if tool_calls:
            self.console.print("[dim]The agent wants to execute the following tools:[/dim]\n")
            for idx, call in enumerate(tool_calls, start=1):
                args_json = json.dumps(call["args"], indent=2)
                self.console.print(
                    f"[cyan]{idx}. Node:[/cyan] {call['node']}\n"
                    f"   [cyan]Tool:[/cyan] {call['name']}\n"
                    f"   [cyan]Args:[/cyan]\n{args_json}\n"
                )
        else:
            self.console.print(
                "[dim]No structured tool call details available; approve to continue or reject to stop.[/dim]\n"
            )

        # Ask for approval
        while True:
            response = self.session.prompt(
                "Approve? (yes/no/edit): ",
                style=self.style,
            ).strip().lower()

            if response in ["yes", "y"]:
                self.console.print("[green]✓ Approved. Continuing...[/green]\n")
                return "approve"
            elif response in ["no", "n"]:
                self.graph.update_state(config, {"messages": [AIMessage("Action rejected by user")]})
                self.console.print("[red]✗ Rejected. Stopping execution.[/red]\n")
                return "reject"
            elif response == "edit":
                self.console.print("[yellow]Editing not yet implemented. Rejecting...[/yellow]")
                self.graph.update_state(config, {"messages": [AIMessage("Action rejected by user")]})
                return "reject"
            else:
                self.console.print("[yellow]Please respond with 'yes', 'no', or 'edit'[/yellow]")

    def run(self):
        """Run the interactive chat loop."""
        self._print_welcome()

        try:
            while True:
                try:
                    # Get user input
                    user_input = self.session.prompt(
                        [("class:prompt", "You: ")],
                    ).strip()

                    # Skip empty input
                    if not user_input:
                        continue

                    # Handle commands
                    if user_input.startswith("/"):
                        should_continue = self._handle_command(user_input)
                        if not should_continue:
                            break
                        continue

                    # Process query
                    self.message_count += 1
                    self._process_query(user_input)

                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Use /exit to quit[/yellow]")
                    continue

                except EOFError:
                    self.console.print("\n[green]Goodbye! 👋[/green]\n")
                    break

        except Exception as e:
            self.console.print(f"\n[red]Fatal error: {e}[/red]\n")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            sys.exit(1)


def start_interactive_chat(
    thread_id: Optional[str] = None,
    enable_hitl: bool = True,
):
    """Start an interactive chat session.

    Args:
        thread_id: Thread ID for conversation persistence (auto-generated if not provided).
        enable_hitl: Whether to require approval before executing tools (default: True for safety).
    """
    chat = InteractiveChat(
        thread_id=thread_id,
        enable_hitl=enable_hitl,
    )
    chat.run()
