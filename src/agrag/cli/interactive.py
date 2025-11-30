"""Interactive chat interface for the AgRAG agent."""

import sys
import traceback
import uuid
import warnings
from datetime import datetime
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import create_input
from prompt_toolkit.output import create_output
from prompt_toolkit.styles import Style
from rich.console import Console

from agrag.cli.commands import CommandHandler
from agrag.cli.display import (
    print_agent_response,
    print_error,
    print_query_stats,
    print_welcome,
)
from agrag.cli.hitl import HITLHandler
from agrag.config import settings
from agrag.core import create_agent_graph, create_initial_state
from agrag.core.checkpointing import initialize_checkpointer, summarize_error

# Suppress the CPR warning from prompt_toolkit in terminals that don't support it
warnings.filterwarnings(
    "ignore",
    message=".*cursor position.*",
    category=UserWarning,
)

# Available commands for auto-completion
CHAT_COMMANDS = [
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
            enable_hitl: Whether to require approval before executing tools (default: True).
        """
        self.console = Console()
        self.thread_id = thread_id or f"chat-{uuid.uuid4().hex[:8]}"
        self.enable_hitl = enable_hitl

        # Initialize checkpointer
        self._init_checkpointer()

        # Create agent graph
        self.graph = create_agent_graph(
            checkpointer=self.checkpointer if self.checkpointer else None,
            enable_hitl=self.enable_hitl,
        )

        # Setup prompt toolkit
        self._init_prompt_session()

        # Initialize handlers
        self.command_handler = CommandHandler(self.console, self)
        self.command_handler.set_welcome_callback(self._print_welcome)

        self.hitl_handler = HITLHandler(self.console, self.session, self.style, self.graph)

        # Conversation stats
        self.message_count = 0
        self.tool_calls_total = 0
        self.model_calls_total = 0
        self.start_time = datetime.now()
        self.thinking_budget = settings.google_thinking_budget

    def _init_checkpointer(self) -> None:
        """Initialize the checkpointer with fallback handling."""
        init_result = initialize_checkpointer(enable_hitl=self.enable_hitl)
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
                "[yellow]HITL enabled but no checkpointer available; "
                "approvals will be disabled.[/yellow]"
            )

    def _init_prompt_session(self) -> None:
        """Initialize prompt toolkit session."""
        self.history = InMemoryHistory()
        self.completer = WordCompleter(CHAT_COMMANDS, ignore_case=True)
        self.style = Style.from_dict({"prompt": "#00aa00 bold"})

        # Create input/output with CPR (cursor position request) disabled
        # to avoid the warning in terminals that don't support it (e.g., VS Code)
        try:
            pt_input = create_input(always_prefer_tty=True)
            pt_output = create_output(always_prefer_tty=True)
        except Exception:
            pt_input = None
            pt_output = None

        self.session = PromptSession(
            history=self.history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=self.completer,
            style=self.style,
            input=pt_input,
            output=pt_output,
        )

    def _get_config(self) -> RunnableConfig:
        """Get the config for graph execution."""
        config: RunnableConfig = {}
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

    def _print_welcome(self) -> None:
        """Print welcome message."""
        print_welcome(
            self.console,
            self.thread_id,
            self.checkpointer_backend,
            self.enable_hitl,
        )

    def _process_query(self, query: str) -> None:
        """Process a user query through the agent.

        Args:
            query: The user's query.
        """
        config = self._get_config()

        try:
            initial_state = create_initial_state(query)
            stats = {"tool_calls": 0, "model_calls": 0}
            final_answer = "No answer generated"

            with self.console.status("[bold green]Agent is thinking...") as status:
                result = self._stream_with_hitl(initial_state, config, status, stats)
                if result.get("answer"):
                    final_answer = result["answer"]
                elif result.get("cancelled"):
                    final_answer = "Query cancelled by user."

            # Update stats
            self.tool_calls_total += stats["tool_calls"]
            self.model_calls_total += stats["model_calls"]

            # Display response
            print_agent_response(self.console, final_answer)
            print_query_stats(self.console, stats["tool_calls"], stats["model_calls"])

        except TimeoutError as e:
            print_error(self.console, f"LLM timeout: {e}")
        except Exception as e:
            print_error(self.console, f"Error: {e}", traceback.format_exc())

    def _stream_with_hitl(
        self,
        input_state: Any,
        config: RunnableConfig,
        status: Any,
        stats: Dict[str, int],
    ) -> Dict[str, Any]:
        """Stream graph execution with HITL interrupt handling.

        This method handles the streaming loop and recursively processes
        any HITL interrupts that occur during execution.

        Args:
            input_state: Initial state or Command to resume with.
            config: Graph configuration.
            status: Rich status context for updates.
            stats: Mutable dict to accumulate tool_calls and model_calls.

        Returns:
            Dict with 'answer' (str) and/or 'cancelled' (bool).
        """
        result: Dict[str, Any] = {}

        for event in self.graph.stream(input_state, config=config, stream_mode="values"):
            # Check for interrupts (HITL)
            if "__interrupt__" in event:
                status.stop()
                hitl_result = self.hitl_handler.handle_interrupt(event, config)
                if hitl_result.decision_type == "reject":
                    result["cancelled"] = True
                    return result

                # Resume execution with the Command from HITL handler
                status.start()
                status.update("[bold blue]Resuming after approval...")

                # Recursively stream the resumed execution
                resume_result = self._stream_with_hitl(hitl_result.command, config, status, stats)

                # Propagate the final answer or cancellation
                if resume_result.get("cancelled"):
                    result["cancelled"] = True
                    return result
                if resume_result.get("answer"):
                    result["answer"] = resume_result["answer"]
                continue

            # Process messages from the event
            event_result = self._process_event(event, status)
            if event_result.get("tool_calls"):
                stats["tool_calls"] += event_result["tool_calls"]
            if event_result.get("model_calls"):
                stats["model_calls"] += event_result["model_calls"]
            if event_result.get("answer"):
                result["answer"] = event_result["answer"]

        return result

    def _process_event(self, event: Dict[str, Any], status: Any) -> Dict[str, Any]:
        """Process a single stream event.

        Args:
            event: The stream event.
            status: Rich status context for updates.

        Returns:
            Dict with tool_calls, model_calls, and answer counts/values.
        """
        result: Dict[str, Any] = {}
        messages = event.get("messages", [])

        if not messages:
            return result

        last_message = messages[-1]

        # Check for AI message with tool calls
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
            result["tool_calls"] = len(last_message.tool_calls)
            status.update(f"[bold yellow]🔧 Executing tools: {', '.join(tool_names)}")

        # Check for model response (AI message without tool calls)
        elif hasattr(last_message, "content") and last_message.content:
            if hasattr(last_message, "type") and last_message.type == "ai":
                result["model_calls"] = 1
                result["answer"] = self._extract_content(last_message.content)
                status.update("[bold blue]Agent is reasoning...")

        return result

    def _extract_content(self, content: Any) -> str:
        """Extract text content from message content.

        Args:
            content: Message content (string or list of blocks).

        Returns:
            Extracted text content.
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Handle content blocks (Gemini format)
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            return "\n".join(text_parts)
        return str(content)

    def run(self) -> None:
        """Run the interactive chat loop."""
        self._print_welcome()

        try:
            while True:
                try:
                    user_input = self.session.prompt(
                        [("class:prompt", "You: ")],
                    ).strip()

                    if not user_input:
                        continue

                    # Handle commands
                    if user_input.startswith("/"):
                        if not self.command_handler.handle(user_input):
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
            print_error(self.console, f"Fatal error: {e}", traceback.format_exc())
            sys.exit(1)


def start_interactive_chat(
    thread_id: Optional[str] = None,
    enable_hitl: bool = True,
) -> None:
    """Start an interactive chat session.

    Args:
        thread_id: Thread ID for conversation persistence (auto-generated if not provided).
        enable_hitl: Whether to require approval before executing tools (default: True).
    """
    chat = InteractiveChat(
        thread_id=thread_id,
        enable_hitl=enable_hitl,
    )
    chat.run()
