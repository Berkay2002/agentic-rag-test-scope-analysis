"""Human-in-the-loop (HITL) interrupt handling for interactive chat.

Aligned with LangChain / LangGraph HITL conventions:
- Extracts action_requests and review_configs from interrupt value
- Respects allowed_decisions per tool from review_configs
- Returns Command(resume={"decisions": [...]}) for caller to invoke
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from langchain_core.runnables import RunnableConfig
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from rich.console import Console
from langgraph.types import Command


class DecisionChoice(Enum):
    """Available decision choices for HITL."""

    APPROVE = "approve"
    APPROVE_REMEMBER = "approve_remember"
    REJECT = "reject"
    EDIT = "edit"
    COMMENT = "comment"


@dataclass
class HITLResult:
    """Result from HITL handler containing the resume command."""

    decision_type: str  # "approve", "edit", or "reject"
    command: Command  # The Command to resume execution
    remember_tools: List[str] = field(default_factory=list)  # Tools to auto-approve


@dataclass
class ActionRequest:
    """Represents a tool action pending approval."""

    name: str
    arguments: Dict[str, Any]
    description: str = ""
    allowed_decisions: Optional[Set[str]] = None

    def __post_init__(self):
        if self.allowed_decisions is None:
            self.allowed_decisions = {"approve", "edit", "reject"}


def extract_hitl_request(
    event: Dict[str, Any],
    graph: Any,
    config: RunnableConfig,
) -> List[ActionRequest]:
    """Extract action requests from an interrupt event.

    Parses the HITL request structure from LangChain:
    - action_requests: [{name, arguments, description}, ...]
    - review_configs: [{action_name, allowed_decisions}, ...]

    Args:
        event: The interrupt event containing __interrupt__ key.
        graph: The LangGraph graph instance.
        config: Graph configuration.

    Returns:
        List of ActionRequest objects with allowed_decisions populated.
    """
    action_requests: List[ActionRequest] = []

    interrupt_data = event.get("__interrupt__")
    if not interrupt_data:
        return _fallback_extract_from_state(graph, config)

    # Build lookup for allowed_decisions from review_configs
    allowed_decisions_map: Dict[str, Set[str]] = {}

    for interrupt in interrupt_data:
        if not (hasattr(interrupt, "value") and isinstance(interrupt.value, dict)):
            continue

        # Extract review_configs first to build allowed_decisions map
        review_configs = interrupt.value.get("review_configs", [])
        for rc in review_configs:
            action_name = rc.get("action_name")
            allowed = rc.get("allowed_decisions", ["approve", "edit", "reject"])
            if action_name:
                allowed_decisions_map[action_name] = set(allowed)

        # Extract action_requests
        for action in interrupt.value.get("action_requests", []):
            name = action.get("name", "")
            action_requests.append(
                ActionRequest(
                    name=name,
                    arguments=action.get("arguments", {}),
                    description=action.get("description", ""),
                    allowed_decisions=allowed_decisions_map.get(
                        name, {"approve", "edit", "reject"}
                    ),
                )
            )

    return action_requests or _fallback_extract_from_state(graph, config)


def _fallback_extract_from_state(
    graph: Any,
    config: RunnableConfig,
) -> List[ActionRequest]:
    """Fallback: extract tool calls from graph state if interrupt payload is empty."""
    action_requests: List[ActionRequest] = []
    try:
        current_state = graph.get_state(config)
        messages = current_state.values.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for call in last_message.tool_calls:
                    action_requests.append(
                        ActionRequest(
                            name=call.get("name", "unknown"),
                            arguments=call.get("args", {}),
                        )
                    )
    except Exception:
        pass
    return action_requests


def build_decisions(
    action_requests: List[ActionRequest],
    decision_type: str,
    message: Optional[str] = None,
    edited_action: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build decision list for Command(resume={"decisions": [...]}).

    Args:
        action_requests: List of actions to decide on.
        decision_type: "approve", "edit", or "reject".
        message: Rejection message (for reject decisions).
        edited_action: Edited action dict with name/args (for edit decisions).

    Returns:
        List of decision dicts in LangChain 1.1.0 format.
    """
    if not action_requests:
        # At least one decision required
        if decision_type == "reject":
            return [{"type": "reject", "message": message or "Action rejected by user"}]
        return [{"type": "approve"}]

    decisions: List[Dict[str, Any]] = []
    for i, action in enumerate(action_requests):
        if decision_type == "approve":
            decisions.append({"type": "approve"})
        elif decision_type == "reject":
            decisions.append(
                {
                    "type": "reject",
                    "message": message or "Action rejected by user",
                }
            )
        elif decision_type == "edit" and i == 0 and edited_action:
            # Only first action gets edited, rest are approved
            decisions.append(
                {
                    "type": "edit",
                    "edited_action": edited_action,
                }
            )
        else:
            # Default to approve for remaining actions
            decisions.append({"type": "approve"})

    return decisions


class HITLHandler:
    """Handles Human-in-the-Loop interrupts.

    Returns HITLResult with Command for caller to invoke, following
    LangChain 1.1.0 pattern where the caller controls graph execution.
    """

    def __init__(
        self,
        console: Console,
        session: PromptSession,
        style: Style,
        graph: Any,
    ):
        """Initialize HITL handler.

        Args:
            console: Rich console for output.
            session: Prompt toolkit session for input.
            style: Prompt style.
            graph: The LangGraph graph instance (used for state inspection fallback).
        """
        self.console = console
        self.session = session
        self.style = style
        self.graph = graph
        # Tools that are auto-approved for this session
        self.remembered_tools: Set[str] = set()

    def handle_interrupt(
        self,
        event: Dict[str, Any],
        config: RunnableConfig,
    ) -> HITLResult:
        """Handle HITL interrupt and return Command for caller to invoke.

        Args:
            event: The interrupt event containing action_requests and review_configs.
            config: The graph config.

        Returns:
            HITLResult with decision_type and Command to resume execution.
        """
        action_requests = extract_hitl_request(event, self.graph, config)

        # Check if all tools are remembered (auto-approve)
        if action_requests and all(
            action.name in self.remembered_tools for action in action_requests
        ):
            tool_names = [a.name for a in action_requests]
            self.console.print(
                f"\n[dim]🔄 Auto-approved (remembered): {', '.join(tool_names)}[/dim]"
            )
            return self._build_approve_result(action_requests)

        self.console.print("\n[bold yellow]🚦 Approval Required[/bold yellow]\n")
        self._display_action_requests(action_requests)
        return self._prompt_for_decision(action_requests)

    def _display_action_requests(self, action_requests: List[ActionRequest]) -> None:
        """Display the proposed tool calls to the user."""
        if action_requests:
            self.console.print("[dim]The agent wants to execute the following tools:[/dim]\n")
            for idx, action in enumerate(action_requests, start=1):
                args_json = json.dumps(action.arguments, indent=2)
                remembered = (
                    " [green](remembered)[/green]" if action.name in self.remembered_tools else ""
                )
                self.console.print(
                    f"[cyan]{idx}. Tool:[/cyan] [bold]{action.name}[/bold]{remembered}\n"
                    f"   [cyan]Args:[/cyan]\n{args_json}\n"
                )
                if action.description:
                    self.console.print(f"   [dim]{action.description}[/dim]\n")
        else:
            self.console.print(
                "[dim]No structured tool call details available; "
                "approve to continue or reject to stop.[/dim]\n"
            )

    def _build_menu_options(
        self,
        action_requests: List[ActionRequest],
    ) -> List[tuple[DecisionChoice, str, str]]:
        """Build menu options based on allowed decisions.

        Returns:
            List of (DecisionChoice, display_text, description) tuples.
        """
        options: List[tuple[DecisionChoice, str, str]] = []

        # Check allowed decisions from first action (use intersection for multiple)
        if action_requests:
            first_allowed = action_requests[0].allowed_decisions or {"approve", "edit", "reject"}
            all_allowed = first_allowed.copy()
            for action in action_requests[1:]:
                action_allowed = action.allowed_decisions or {"approve", "edit", "reject"}
                all_allowed &= action_allowed
        else:
            all_allowed = {"approve", "edit", "reject"}

        if "approve" in all_allowed:
            options.append((DecisionChoice.APPROVE, "✓ Yes, approve", "Execute the tool(s)"))
            # Only show remember option if there are named tools
            if action_requests:
                tool_names = ", ".join(a.name for a in action_requests)
                options.append(
                    (
                        DecisionChoice.APPROVE_REMEMBER,
                        "✓ Yes, and remember this session",
                        f"Auto-approve [{tool_names}] for rest of session",
                    )
                )

        if "reject" in all_allowed:
            options.append((DecisionChoice.REJECT, "✗ No, reject", "Cancel this action"))

        if "edit" in all_allowed:
            options.append((DecisionChoice.EDIT, "✎ Edit arguments", "Modify the tool arguments"))

        options.append(
            (DecisionChoice.COMMENT, "💬 Add a comment", "Provide feedback or instructions")
        )

        return options

    def _create_menu_app(
        self,
        options: List[tuple[DecisionChoice, str, str]],
    ) -> tuple[Application, list]:
        """Create an interactive menu application.

        Returns:
            Tuple of (Application, mutable result list).
        """
        selected_index = [0]  # Mutable to allow modification in closure
        result = [None]  # Store the selected option

        def get_formatted_text():
            lines = []
            lines.append(("class:header", "Use ↑↓ arrows to select, Enter to confirm:\n\n"))
            for i, (choice, text, desc) in enumerate(options):
                if i == selected_index[0]:
                    lines.append(("class:selected", f"  ▸ {text}\n"))
                    lines.append(("class:selected-desc", f"    {desc}\n"))
                else:
                    lines.append(("class:option", f"    {text}\n"))
            return lines

        kb = KeyBindings()

        @kb.add("up")
        def move_up(event):
            selected_index[0] = (selected_index[0] - 1) % len(options)

        @kb.add("down")
        def move_down(event):
            selected_index[0] = (selected_index[0] + 1) % len(options)

        @kb.add("enter")
        def select(event):
            result[0] = options[selected_index[0]][0]
            event.app.exit()

        @kb.add("c-c")
        def cancel(event):
            result[0] = DecisionChoice.REJECT
            event.app.exit()

        @kb.add("escape")
        def escape(event):
            result[0] = DecisionChoice.REJECT
            event.app.exit()

        # Quick keys
        @kb.add("y")
        def quick_yes(event):
            result[0] = DecisionChoice.APPROVE
            event.app.exit()

        @kb.add("n")
        def quick_no(event):
            result[0] = DecisionChoice.REJECT
            event.app.exit()

        @kb.add("e")
        def quick_edit(event):
            if any(c == DecisionChoice.EDIT for c, _, _ in options):
                result[0] = DecisionChoice.EDIT
                event.app.exit()

        menu_style = Style.from_dict(
            {
                "header": "#888888",
                "selected": "#00ff00 bold",
                "selected-desc": "#00aa00 italic",
                "option": "#cccccc",
            }
        )

        layout = Layout(
            HSplit(
                [
                    Window(content=FormattedTextControl(get_formatted_text)),
                ]
            )
        )

        app = Application(
            layout=layout,
            key_bindings=kb,
            style=menu_style,
            full_screen=False,
            mouse_support=True,
            erase_when_done=True,  # Clear menu from terminal after selection
        )

        return app, result

    def _prompt_for_decision(
        self,
        action_requests: List[ActionRequest],
    ) -> HITLResult:
        """Prompt user for approval decision using interactive menu."""
        options = self._build_menu_options(action_requests)

        while True:
            app, result = self._create_menu_app(options)
            app.run()

            choice = result[0]

            if choice == DecisionChoice.APPROVE:
                return self._build_approve_result(action_requests)

            elif choice == DecisionChoice.APPROVE_REMEMBER:
                # Remember these tools for the session
                for action in action_requests:
                    self.remembered_tools.add(action.name)
                tool_names = [a.name for a in action_requests]
                self.console.print(
                    f"[green]✓ Approved and remembered: {', '.join(tool_names)}[/green]\n"
                )
                decisions = build_decisions(action_requests, "approve")
                return HITLResult(
                    decision_type="approve",
                    command=Command(resume={"decisions": decisions}),
                    remember_tools=tool_names,
                )

            elif choice == DecisionChoice.REJECT:
                return self._build_reject_result(action_requests)

            elif choice == DecisionChoice.EDIT:
                edit_result = self._build_edit_result(action_requests)
                if edit_result:
                    return edit_result
                # Edit failed, show menu again

            elif choice == DecisionChoice.COMMENT:
                comment_result = self._handle_comment(action_requests)
                if comment_result:
                    return comment_result
                # Comment cancelled, show menu again

    def _handle_comment(
        self,
        action_requests: List[ActionRequest],
    ) -> Optional[HITLResult]:
        """Handle user comment/feedback.

        Returns:
            HITLResult if user wants to proceed with comment, None to return to menu.
        """
        self.console.print("\n[yellow]Enter your comment or feedback:[/yellow]")
        self.console.print(
            "[dim](This will be sent to the agent. Press Enter to submit, Ctrl+C to cancel)[/dim]"
        )

        try:
            comment = self.session.prompt("Comment: ").strip()
            if comment:
                self.console.print(
                    "[green]✓ Comment noted. Proceeding with execution...[/green]\n"
                )
                # Approve but include the comment in the decision
                decisions = build_decisions(action_requests, "approve")
                # Add comment to the first decision
                if decisions:
                    decisions[0]["comment"] = comment
                return HITLResult(
                    decision_type="approve",
                    command=Command(resume={"decisions": decisions}),
                )
            else:
                self.console.print("[dim]No comment provided. Returning to menu...[/dim]\n")
                return None
        except KeyboardInterrupt:
            self.console.print("\n[dim]Cancelled. Returning to menu...[/dim]\n")
            return None

    def _build_approve_result(
        self,
        action_requests: List[ActionRequest],
    ) -> HITLResult:
        """Build approval result."""
        self.console.print("[green]✓ Approved. Continuing...[/green]\n")
        decisions = build_decisions(action_requests, "approve")
        return HITLResult(
            decision_type="approve",
            command=Command(resume={"decisions": decisions}),
        )

    def _build_reject_result(
        self,
        action_requests: List[ActionRequest],
        message: str = "Action rejected by user",
    ) -> HITLResult:
        """Build rejection result."""
        self.console.print("[red]✗ Rejected. Stopping execution.[/red]\n")
        decisions = build_decisions(action_requests, "reject", message=message)
        return HITLResult(
            decision_type="reject",
            command=Command(resume={"decisions": decisions}),
        )

    def _build_edit_result(
        self,
        action_requests: List[ActionRequest],
    ) -> Optional[HITLResult]:
        """Build edit result.

        Returns:
            HITLResult if successful, None if edit failed and should retry.
        """
        self.console.print("\n[yellow]Edit mode - enter new arguments as JSON:[/yellow]")
        try:
            if action_requests:
                action = action_requests[0]
                self.console.print(f"[dim]Current args:[/dim] {json.dumps(action.arguments)}")
                self.console.print("[dim](Press Ctrl+C to cancel)[/dim]")
                new_args_str = self.session.prompt("New args (JSON): ").strip()
                new_args = json.loads(new_args_str)

                edited_action = {"name": action.name, "args": new_args}
                decisions = build_decisions(action_requests, "edit", edited_action=edited_action)

                self.console.print("[green]✓ Edited. Continuing...[/green]\n")
                return HITLResult(
                    decision_type="edit",
                    command=Command(resume={"decisions": decisions}),
                )
            else:
                self.console.print("[yellow]No tool calls to edit. Approving...[/yellow]")
                return HITLResult(
                    decision_type="approve",
                    command=Command(resume={"decisions": [{"type": "approve"}]}),
                )
        except KeyboardInterrupt:
            self.console.print("\n[dim]Edit cancelled. Returning to menu...[/dim]\n")
            return None
        except json.JSONDecodeError:
            self.console.print("[red]Invalid JSON. Returning to menu...[/red]\n")
            return None
        except Exception as e:
            self.console.print(f"[red]Edit failed: {e}. Returning to menu...[/red]\n")
            return None
