"""State definition for LangGraph agent.

The project uses a custom `StateGraph` with a message-driven state.
This TypedDict keeps the agent state explicit and evaluation-friendly.
"""

import operator
from typing import TypedDict, List, Annotated, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """
    Legacy state for the agentic RAG system.

    State for the agentic RAG system.

    Attributes:
        messages: Conversation history (automatically managed by add_messages)
        tool_call_count: Number of tool calls made in this session
        model_call_count: Number of model calls made in this session
        final_answer: Optional final answer (not required for normal execution)
    """

    # Messages with automatic deduplication and appending
    messages: Annotated[List[BaseMessage], add_messages]

    # Counters for safety limits (enforced by graph routing)
    tool_call_count: int
    model_call_count: int

    # Final answer (when set, agent stops)
    final_answer: str

    # Retrieved contexts (used for evaluation and RAG metrics)
    retrieved_contexts: Annotated[List[Dict[str, Any]], operator.add]

    # Toggle for tracking contexts during tool execution
    enable_context_tracking: bool
