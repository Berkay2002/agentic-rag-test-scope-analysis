"""State definition for LangGraph agent."""

from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """
    State for the agentic RAG system.

    Attributes:
        messages: Conversation history (automatically managed by add_messages)
        tool_call_count: Number of tool calls made in this session
        model_call_count: Number of model calls made in this session
        final_answer: Final answer to return to user (if set, triggers termination)
    """

    # Messages with automatic deduplication and appending
    messages: Annotated[List[BaseMessage], add_messages]

    # Counters for safety limits
    tool_call_count: int
    model_call_count: int

    # Final answer (when set, agent stops)
    final_answer: str
