"""Interactive chat interface for the AgRAG agent."""

import json
import logging
import re
import sys
import traceback
import uuid
import warnings
from datetime import datetime
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, ToolMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import ConditionalCompleter, WordCompleter
from prompt_toolkit.filters import Condition
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
    print_tool_call,
    print_reasoning,
)
from agrag.cli.hitl import HITLHandler
from agrag.cli.utils import extract_message_content, extract_reasoning_and_answer
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
    "/export",
    "/verbose",
    "/thinking",
    "/threads",
    "/branches",
    "/fork",
    "/checkout",
]


class InteractiveChat:
    """Interactive chat interface for the AgRAG agent."""

    def __init__(
        self,
        thread_id: Optional[str] = None,
        enable_hitl: bool = True,
        verbose: bool = False,
    ):
        """Initialize the interactive chat.

        Args:
            thread_id: Thread ID for conversation persistence. If None, generates a new one.
            enable_hitl: Whether to require approval before executing tools (default: True).
        """
        self.console = Console()
        self.has_user_provided_thread_id = thread_id is not None
        self.thread_id = thread_id or f"chat-{uuid.uuid4().hex[:8]}"
        self.enable_hitl = enable_hitl
        self.verbose = verbose
        self.conversation_log: list[dict] = []
        self._tool_call_index: dict[str, dict] = {}
        self._base_log_level = logging.getLogger().level
        self._apply_logging_mode()

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
        self.thinking_level = settings.google_thinking_level
        self.thinking_budget = settings.google_thinking_budget

    def _init_checkpointer(self) -> None:
        """Initialize the checkpointer with fallback handling."""
        # Always enable persistence in interactive chat when available
        # This ensures consistent behavior across all modes (safe/YOLO)
        # and allows session resumption even for auto-generated thread IDs
        init_result = initialize_checkpointer(
            enable_hitl=self.enable_hitl,
            enable_persistence=True  # Always use persistence if available
        )
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
        # Only show command completions when typing a slash
        command_completer = WordCompleter(
            CHAT_COMMANDS,
            ignore_case=True,
            pattern=re.compile(r"^/.*"),  # Only match when line starts with /
        )
        def _should_complete_commands() -> bool:
            try:
                buffer_text = get_app().current_buffer.document.text
            except Exception:
                return False
            return buffer_text.lstrip().startswith("/")

        self.completer = ConditionalCompleter(
            command_completer,
            Condition(_should_complete_commands),
        )
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
            complete_while_typing=True,
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

    def _set_log_level(self, level: int) -> None:
        """Apply a log level to the root logger and its handlers."""
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)

    def _apply_logging_mode(self) -> None:
        """Adjust logging verbosity for chat UX."""
        if self.verbose:
            self._set_log_level(self._base_log_level)
        else:
            self._set_log_level(logging.ERROR)

    def set_verbose(self, enabled: bool) -> None:
        """Toggle verbose mode and update logging behavior."""
        self.verbose = enabled
        self._apply_logging_mode()

    def reset_conversation(self) -> None:
        """Reset stored conversation logs for this session."""
        self.conversation_log = []
        self._tool_call_index = {}

    def _log_event(self, entry: dict) -> None:
        """Append a structured event to the conversation log."""
        entry["timestamp"] = datetime.now().isoformat()
        self.conversation_log.append(entry)

    def export_conversation(
        self, filename: Optional[str] = None, include_tool_details: bool = False
    ) -> str:
        """Export the conversation log to a markdown or text file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = filename or f"conversation_{self.thread_id}_{timestamp}.md"
        if not output_name.endswith((".md", ".txt")):
            output_name = f"{output_name}.md"

        lines: list[str] = []

        for entry in self.conversation_log:
            entry_type = entry.get("type", "event")
            if entry_type == "user":
                lines.append(f"You: {entry.get('content', '')}")
                lines.append("")
                continue

            if entry_type == "assistant":
                lines.append(f"Assistant: {entry.get('content', '')}")
                lines.append("")
                continue

            if entry_type == "tool_call":
                tool_name = entry.get("tool_name", "unknown")
                if include_tool_details:
                    args_json = json.dumps(
                        entry.get("arguments", {}),
                        indent=2,
                        ensure_ascii=True,
                        default=str,
                    )
                    lines.append(f"Tool Call: {tool_name}")
                    lines.append("Tool Args:")
                    lines.append("```json")
                    lines.append(args_json)
                    lines.append("```")
                    lines.append("")
                else:
                    lines.append(f"Tool Call: {tool_name}")
                    lines.append("")
                continue

            if entry_type == "tool_result":
                if include_tool_details:
                    tool_name = entry.get("tool_name", "unknown")
                    lines.append(f"Tool Result: {tool_name}")
                    lines.append("Output:")
                    lines.append("```")
                    lines.append(entry.get("content", ""))
                    lines.append("```")
                    lines.append("")
                continue

            lines.append(entry.get("content", ""))
            lines.append("")

        with open(output_name, "w") as handle:
            handle.write("\n".join(lines))

        return output_name

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

        self._reset_counters_per_message(config)

        try:
            initial_state = create_initial_state(query)
            stats = {"tool_calls": 0, "model_calls": 0}
            final_answer = "No answer generated"
            self._log_event({"type": "user", "content": query})

            with self.console.status("[bold green]Agent is thinking...") as status:
                result = self._stream_with_hitl(initial_state, config, status, stats)
                if result.get("answer"):
                    final_answer = result["answer"]
                elif result.get("cancelled"):
                    final_answer = "Query cancelled by user."
                    self._log_event({"type": "assistant", "content": final_answer})

            if final_answer == "No answer generated":
                fallback_answer = self._extract_final_answer_from_state(config)
                if fallback_answer:
                    final_answer = fallback_answer

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
        last_tool_output: Optional[str] = None

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
            if event_result.get("tool_output"):
                last_tool_output = event_result["tool_output"]

        if not result.get("answer") and last_tool_output:
            result["answer"] = (
                "No final response was generated. "
                "Last tool output:\n\n"
                f"{last_tool_output}"
            )

        return result

    def _reset_counters_per_message(self, config: RunnableConfig) -> None:
        """Reset tool/model counters so limits apply per user message."""
        if not self.checkpointer:
            return

        try:
            self.graph.update_state(
                config=config,
                values={
                    "tool_call_count": 0,
                    "model_call_count": 0,
                    "final_answer": "",
                },
                as_node="call_model",
            )
        except Exception:
            return

    def _extract_final_answer_from_state(self, config: RunnableConfig) -> Optional[str]:
        """Fallback to retrieve the last AI response from graph state."""
        try:
            state = self.graph.get_state(config)
        except Exception:
            return None

        messages = state.values.get("messages", []) if state and state.values else []
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return extract_message_content(msg.content)

        if messages:
            last_message = messages[-1]
            if isinstance(last_message, ToolMessage) and last_message.content:
                tool_output = extract_message_content(last_message.content)
                return (
                    "No final response was generated. "
                    "Last tool output:\n\n"
                    f"{tool_output}"
                )

        return None

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
            if self.verbose or not self.enable_hitl:
                status.stop()
            for tool_call in last_message.tool_calls:
                tool_call_id = str(tool_call.get("id", ""))
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})
                self._tool_call_index[tool_call_id] = {
                    "name": tool_name,
                    "args": tool_args,
                }
                self._log_event(
                    {
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "arguments": tool_args,
                    }
                )
                if self.verbose or not self.enable_hitl:
                    print_tool_call(self.console, tool_name, tool_call_id, tool_args)
            if self.verbose or not self.enable_hitl:
                status.start()
                status.update(f"[bold yellow]🔧 Executing tools: {', '.join(tool_names)}")

        # Check for model response (AI message without tool calls)
        elif isinstance(last_message, ToolMessage):
            tool_call_id = str(last_message.tool_call_id or "")
            tool_info = self._tool_call_index.get(tool_call_id, {})
            tool_name = tool_info.get("name")
            tool_output = extract_message_content(last_message.content)
            result["tool_output"] = tool_output
            self._log_event(
                {
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "content": tool_output,
                }
            )

        elif isinstance(last_message, AIMessage) and last_message.content:
            result["model_calls"] = 1
            
            # Extract reasoning blocks and answer separately
            reasoning_blocks, answer_text = extract_reasoning_and_answer(last_message.content)
            
            # Display reasoning if present
            if reasoning_blocks:
                if self.verbose:
                    status.stop()
                    print_reasoning(self.console, reasoning_blocks, collapsed=False)
                    status.start()
                else:
                    # Show collapsed reasoning by default
                    status.stop()
                    print_reasoning(self.console, reasoning_blocks, collapsed=True)
                    status.start()
            
            result["answer"] = answer_text if answer_text else extract_message_content(last_message.content)
            self._log_event({"type": "assistant", "content": result["answer"]})
            status.update("[bold blue]Agent is reasoning...")

        return result

    # Content extraction is now handled by agrag.cli.utils.extract_message_content

    def run(self) -> None:
        """Run the interactive chat loop."""
        self._print_welcome()
        self._interrupt_count = 0

        try:
            while True:
                try:
                    user_input = self.session.prompt(
                        [("class:prompt", "You: ")],
                    ).strip()

                    if not user_input:
                        continue

                    # Reset interrupt count on successful input
                    self._interrupt_count = 0

                    # Handle commands
                    if user_input.startswith("/"):
                        if not self.command_handler.handle(user_input):
                            break
                        continue

                    # Process query
                    self.message_count += 1
                    self._process_query(user_input)

                except KeyboardInterrupt:
                    self._interrupt_count += 1
                    if self._interrupt_count >= 2:
                        self.console.print("\n[red]Interrupted by user[/red]")
                        break
                    else:
                        self.console.print("\n[yellow]Press Ctrl+C again to exit, or type /exit[/yellow]")
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
    verbose: bool = False,
) -> None:
    """Start an interactive chat session.

    Args:
        thread_id: Thread ID for conversation persistence (auto-generated if not provided).
        enable_hitl: Whether to require approval before executing tools (default: True).
        verbose: Whether to show tool call details in output.
    """
    chat = InteractiveChat(
        thread_id=thread_id,
        enable_hitl=enable_hitl,
        verbose=verbose,
    )
    chat.run()
