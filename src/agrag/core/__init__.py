"""Core agent components.

This module provides the main agent functionality using a custom LangGraph StateGraph.

Key behaviors:
- Custom ReAct-style loop (`call_model` ↔ `execute_tools`)
- Human-in-the-loop approvals via `langgraph.types.interrupt` (Agent Inbox-style)
- Tool/model call limits enforced by graph routing
"""

from .state import AgentState
from .graph import create_agent_graph, create_initial_state, SYSTEM_PROMPT

# Note: `agrag.core.nodes` contains the graph node functions used by `create_agent_graph`.

__all__ = [
    "AgentState",
    "create_agent_graph",
    "create_initial_state",
    "SYSTEM_PROMPT",
]
