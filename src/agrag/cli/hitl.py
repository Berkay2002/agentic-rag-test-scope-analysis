"""Human-in-the-loop (HITL) interrupt handling for interactive chat.

Aligned with LangChain / LangGraph HITL conventions:
- Extracts action_requests and review_configs from interrupt value
- Respects allowed_decisions per tool from review_configs  
- Returns Command(resume={"decisions": [...]}) for caller to invoke
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from langchain_core.runnables import RunnableConfig
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from rich.console import Console
from langgraph.types import Command


@dataclass 
class HITLResult:
    """Result from HITL handler containing the resume command."""
    decision_type: str  # "approve", "edit", or "reject"
    command: Command  # The Command to resume execution


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
            action_requests.append(ActionRequest(
                name=name,
                arguments=action.get("arguments", {}),
                description=action.get("description", ""),
                allowed_decisions=allowed_decisions_map.get(
                    name, {"approve", "edit", "reject"}
                ),
            ))
    
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
                    action_requests.append(ActionRequest(
                        name=call.get("name", "unknown"),
                        arguments=call.get("args", {}),
                    ))
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
            decisions.append({
                "type": "reject",
                "message": message or "Action rejected by user",
            })
        elif decision_type == "edit" and i == 0 and edited_action:
            # Only first action gets edited, rest are approved
            decisions.append({
                "type": "edit",
                "edited_action": edited_action,
            })
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
        self.console.print("\n[bold yellow]🚦 Approval Required[/bold yellow]\n")
        action_requests = extract_hitl_request(event, self.graph, config)

        self._display_action_requests(action_requests)
        return self._prompt_for_decision(action_requests)
    
    def _display_action_requests(self, action_requests: List[ActionRequest]) -> None:
        """Display the proposed tool calls to the user."""
        if action_requests:
            self.console.print("[dim]The agent wants to execute the following tools:[/dim]\n")
            for idx, action in enumerate(action_requests, start=1):
                args_json = json.dumps(action.arguments, indent=2)
                allowed_set = action.allowed_decisions or {"approve", "edit", "reject"}
                allowed = ", ".join(sorted(allowed_set))
                self.console.print(
                    f"[cyan]{idx}. Tool:[/cyan] {action.name}\n"
                    f"   [cyan]Args:[/cyan]\n{args_json}\n"
                    f"   [dim]Allowed decisions: {allowed}[/dim]\n"
                )
                if action.description:
                    self.console.print(f"   [dim]{action.description}[/dim]\n")
        else:
            self.console.print(
                "[dim]No structured tool call details available; "
                "approve to continue or reject to stop.[/dim]\n"
            )
    
    def _get_prompt_options(self, action_requests: List[ActionRequest]) -> str:
        """Get prompt options based on allowed decisions."""
        if not action_requests:
            return "yes/no/edit"
        
        # Use intersection of all allowed decisions
        first_allowed = action_requests[0].allowed_decisions or {"approve", "edit", "reject"}
        all_allowed = first_allowed.copy()
        for action in action_requests[1:]:
            action_allowed = action.allowed_decisions or {"approve", "edit", "reject"}
            all_allowed &= action_allowed
        
        options = []
        if "approve" in all_allowed:
            options.append("yes")
        if "reject" in all_allowed:
            options.append("no")
        if "edit" in all_allowed:
            options.append("edit")
        
        return "/".join(options) if options else "yes/no"
    
    def _prompt_for_decision(
        self, 
        action_requests: List[ActionRequest],
    ) -> HITLResult:
        """Prompt user for approval decision."""
        prompt_options = self._get_prompt_options(action_requests)
        
        while True:
            response = (
                self.session.prompt(
                    f"Approve? ({prompt_options}): ",
                    style=self.style,
                )
                .strip()
                .lower()
            )

            if response in ["yes", "y"]:
                return self._build_approve_result(action_requests)
            elif response in ["no", "n"]:
                return self._build_reject_result(action_requests)
            elif response == "edit":
                result = self._build_edit_result(action_requests)
                if result:
                    return result
                # Edit failed, loop continues
            else:
                self.console.print(f"[yellow]Please respond with {prompt_options}[/yellow]")
    
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
        self.console.print("[yellow]Edit mode - enter new arguments as JSON:[/yellow]")
        try:
            if action_requests:
                action = action_requests[0]
                self.console.print(f"Current args: {json.dumps(action.arguments)}")
                new_args_str = self.session.prompt("New args (JSON): ").strip()
                new_args = json.loads(new_args_str)
                
                edited_action = {"name": action.name, "args": new_args}
                decisions = build_decisions(
                    action_requests, "edit", edited_action=edited_action
                )
                
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
        except json.JSONDecodeError:
            self.console.print("[red]Invalid JSON. Try again or use 'no' to reject.[/red]")
            return None
        except Exception as e:
            self.console.print(f"[red]Edit failed: {e}. Rejecting.[/red]")
            return HITLResult(
                decision_type="reject",
                command=Command(resume={"decisions": [{"type": "reject", "message": str(e)}]}),
            )
