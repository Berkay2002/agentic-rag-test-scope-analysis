"""State definition for LangGraph agent.

Note: With the new create_agent API, the agent uses LangChain's built-in
AgentState which contains just 'messages'. This custom state definition
is kept for reference and backwards compatibility with evaluation code.

The create_agent API handles tool and model call limits via middleware:
- ModelCallLimitMiddleware
- ToolCallLimitMiddleware

For custom state extensions, use the state_schema parameter on create_agent
or define state via middleware.
"""

from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """
    Legacy state for the agentic RAG system.

    The new create_agent API uses a simpler state with just messages.
    This is kept for backwards compatibility with evaluation code.

    Attributes:
        messages: Conversation history (automatically managed by add_messages)
        tool_call_count: Number of tool calls made in this session
        model_call_count: Number of model calls made in this session
        final_answer: Final answer to return to user (if set, triggers termination)
    """

    # Messages with automatic deduplication and appending
    messages: Annotated[List[BaseMessage], add_messages]

    # Counters for safety limits (now handled by middleware)
    tool_call_count: int
    model_call_count: int

    # Final answer (when set, agent stops)
    final_answer: str
