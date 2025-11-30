"""Core agent components.

This module provides the main agent functionality using LangChain's create_agent API.
The previous custom StateGraph implementation has been replaced with the new
high-level API that provides built-in middleware support including:
- HumanInTheLoopMiddleware for tool approval
- ModelCallLimitMiddleware for cost control
- ToolCallLimitMiddleware for execution limits
- PIIMiddleware for sensitive data handling (via agrag.middleware)
"""

from .state import AgentState
from .graph import create_agent_graph, create_initial_state, SYSTEM_PROMPT

# Note: nodes.py is kept for backwards compatibility but is no longer used
# by the create_agent API. The agent handles model calls and tool execution internally.

__all__ = [
    "AgentState",
    "create_agent_graph",
    "create_initial_state",
    "SYSTEM_PROMPT",
]
