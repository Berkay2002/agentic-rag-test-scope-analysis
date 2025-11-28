"""Graph nodes for StateGraph agent."""

import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import BaseTool

from agrag.core.state import AgentState
from agrag.models import get_llm
from agrag.config import settings

logger = logging.getLogger(__name__)


def call_model(state: AgentState, tools: List[BaseTool]) -> dict:
    """
    Call the LLM with available tools.

    Args:
        state: Current agent state
        tools: List of available tools

    Returns:
        State update with new message and incremented counter
    """
    logger.info("Calling model...")

    # Get LLM with tools bound
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)

    timeout = settings.llm_timeout_seconds

    def _invoke():
        return llm_with_tools.invoke(state["messages"])

    if timeout and timeout > 0:
        logger.info("Invoking LLM with timeout=%ss", timeout)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_invoke)
            try:
                response = future.result(timeout=timeout)
            except FuturesTimeoutError as exc:
                future.cancel()
                logger.error(
                    "LLM call exceeded timeout (%ss). Check connectivity or reduce workload.",
                    timeout,
                )
                raise TimeoutError(
                    f"LLM call timed out after {timeout}s. Verify GOOGLE_API_KEY connectivity."
                ) from exc
    else:
        response = _invoke()

    # Increment model call counter
    model_call_count = state.get("model_call_count", 0) + 1

    logger.info(
        f"Model response received (call {model_call_count}). "
        f"Tool calls: {len(response.tool_calls) if hasattr(response, 'tool_calls') else 0}"
    )

    return {
        "messages": [response],
        "model_call_count": model_call_count,
    }


def execute_tools(state: AgentState, tools: List[BaseTool]) -> dict:
    """
    Execute tool calls from the last message.

    Args:
        state: Current agent state
        tools: List of available tools

    Returns:
        State update with tool results and incremented counter
    """
    logger.info("Executing tools...")

    # Get last message (should be AIMessage with tool calls)
    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.warning("No tool calls found in last message")
        return {}

    # Build tool map
    tool_map = {tool.name: tool for tool in tools}

    # Execute each tool call
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

        # Execute tool
        if tool_name not in tool_map:
            error_msg = f"Tool '{tool_name}' not found"
            logger.error(error_msg)
            tool_messages.append(
                ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call_id,
                )
            )
            continue

        try:
            tool = tool_map[tool_name]
            result = tool.invoke(tool_args)

            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id,
                )
            )
            logger.info(f"Tool {tool_name} completed successfully")

        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            tool_messages.append(
                ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call_id,
                )
            )

    # Increment tool call counter
    tool_call_count = state.get("tool_call_count", 0) + len(tool_messages)

    logger.info(f"Executed {len(tool_messages)} tools (total: {tool_call_count})")

    return {
        "messages": tool_messages,
        "tool_call_count": tool_call_count,
    }


def finalize_answer(state: AgentState) -> dict:
    """
    Extract final answer from last message.

    Args:
        state: Current agent state

    Returns:
        State update with final answer set
    """
    logger.info("Finalizing answer...")

    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage):
        final_answer = last_message.content
    else:
        final_answer = str(last_message.content)

    logger.info(f"Final answer extracted ({len(final_answer)} chars)")

    return {
        "final_answer": final_answer,
    }
