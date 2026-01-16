"""Graph nodes for StateGraph agent."""

import logging
import re
import time
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from agrag.core.state import AgentState
from agrag.models import get_llm
from agrag.config import settings

logger = logging.getLogger(__name__)

_REL_LABEL_PATTERN = re.compile(r"\[([A-Za-z0-9_]+)\]")
_SEARCH_RESULT_HEADER_RE = re.compile(
    r"^\s*\d+\.\s+Entity ID:\s*(.+?)\s+\(([^:]+):\s*([0-9.]+)\)"
)
_SEARCH_RESULT_SNIPPET_RE = re.compile(r"^\s*Snippet:\s*(.+)$")
_SEARCH_RESULT_ENTITY_TYPE_RE = re.compile(r"^\s*Entity Type:\s*(.+)$")
_GRAPH_PATH_LINE_RE = re.compile(r"^\s*(Path|Sequence):\s*(.+)$")
_NODE_REF_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*):([A-Za-z0-9_\-]+)")


def _render_content(content: Any) -> str:
    """Convert LLM content (str, list, dict) into displayable text."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, (list, tuple)):
        rendered_parts = [_render_content(part) for part in content]
        return "\n".join([part for part in rendered_parts if part])

    if isinstance(content, dict):
        # Common Gemini formats
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        if "content" in content:
            return _render_content(content["content"])
        if "parts" in content:
            return _render_content(content["parts"])

    # Fallback to string conversion
    return str(content)


def _collect_tool_relationships(messages: List[Any]) -> set[str]:
    """Extract relationship labels from tool outputs after the most recent user message."""
    last_user_index = None
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            last_user_index = idx
            break

    start_index = (last_user_index + 1) if last_user_index is not None else 0
    relationships: set[str] = set()
    for message in messages[start_index:]:
        if not isinstance(message, ToolMessage):
            continue
        content = _render_content(message.content)
        for label in _REL_LABEL_PATTERN.findall(content):
            relationships.add(label.upper())

    return relationships


def _sanitize_graph_paths(content: str, allowed_relationships: set[str]) -> str:
    """Remove graph path lines that cite relationships not found in tool output."""
    if not content:
        return content

    lines = content.splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == "**Graph Paths:**":
            header_idx = idx
            break

    if header_idx is None:
        return content

    end_idx = len(lines)
    for idx in range(header_idx + 1, len(lines)):
        line = lines[idx].strip()
        if line.startswith("**") and line.endswith("**"):
            end_idx = idx
            break

    graph_lines = lines[header_idx + 1 : end_idx]
    filtered: List[str] = []
    for line in graph_lines:
        stripped = line.strip()
        if not stripped:
            continue
        labels = {label.upper() for label in _REL_LABEL_PATTERN.findall(line)}
        if not labels or labels.issubset(allowed_relationships):
            filtered.append(line)

    if not filtered:
        filtered = ["- n/a (no verified graph paths from tool output)"]

    updated_lines = lines[: header_idx + 1] + filtered + lines[end_idx:]
    return "\n".join(updated_lines)


def _update_message_content(message: AIMessage, content: str) -> AIMessage:
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"content": content})
    if hasattr(message, "copy"):
        return message.copy(update={"content": content})
    try:
        message.content = content
        return message
    except Exception:
        return AIMessage(content=content)


def _parse_search_results(content: str, tool_name: str) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for line in content.splitlines():
        header_match = _SEARCH_RESULT_HEADER_RE.match(line)
        if header_match:
            if current:
                contexts.append(current)
            entity_id = header_match.group(1).strip()
            score_str = header_match.group(3).strip()
            try:
                score = float(score_str)
            except ValueError:
                score = 0.0
            current = {
                "chunk_text": "",
                "source": entity_id,
                "entity_type": None,
                "score": score,
                "tool_name": tool_name,
                "timestamp": time.time(),
            }
            continue

        snippet_match = _SEARCH_RESULT_SNIPPET_RE.match(line)
        if snippet_match and current is not None:
            snippet = snippet_match.group(1).strip()
            if snippet.endswith("..."):
                snippet = snippet[:-3].rstrip()
            current["chunk_text"] = snippet
            continue

        entity_type_match = _SEARCH_RESULT_ENTITY_TYPE_RE.match(line)
        if entity_type_match and current is not None:
            current["entity_type"] = entity_type_match.group(1).strip()
            continue

        if not line.strip() and current is not None:
            contexts.append(current)
            current = None

    if current:
        contexts.append(current)

    return contexts


def _parse_graph_traversal(content: str, tool_name: str) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []

    for line in content.splitlines():
        match = _GRAPH_PATH_LINE_RE.match(line)
        if not match:
            continue

        path_text = match.group(2).strip()
        node_refs = _NODE_REF_RE.findall(path_text)
        entity_type = None
        source = None
        if node_refs:
            entity_type, source = node_refs[-1]

        contexts.append(
            {
                "chunk_text": path_text,
                "source": source or "graph_path",
                "entity_type": entity_type or "GraphPath",
                "score": 0.0,
                "tool_name": tool_name,
                "timestamp": time.time(),
            }
        )

    return contexts


def extract_contexts_from_tool_result(
    tool_name: str, tool_result: ToolMessage
) -> List[Dict[str, Any]]:
    """Extract structured contexts from tool results for evaluation."""
    content = _render_content(tool_result.content)
    if not content:
        return []

    if tool_name in {"vector_search", "keyword_search", "hybrid_search"}:
        return _parse_search_results(content, tool_name)

    if tool_name == "graph_traverse":
        return _parse_graph_traversal(content, tool_name)

    return []


def call_model(
    state: AgentState, tools: List[BaseTool], system_prompt: Optional[str] = None
) -> dict:
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

    messages = state.get("messages", [])
    messages_for_model = (
        [SystemMessage(content=system_prompt), *messages] if system_prompt else messages
    )

    def _invoke():
        return llm_with_tools.invoke(messages_for_model)

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

    if not getattr(response, "tool_calls", None) and isinstance(response, AIMessage):
        if isinstance(response.content, str):
            allowed_relationships = _collect_tool_relationships(state.get("messages", []))
            sanitized = _sanitize_graph_paths(response.content, allowed_relationships)
            if sanitized != response.content:
                response = _update_message_content(response, sanitized)

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


def execute_tools(state: AgentState, tools: List[BaseTool], enable_hitl: bool = False) -> dict:
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

    tool_calls = list(last_message.tool_calls)
    if enable_hitl:
        from langgraph.types import interrupt

        action_requests = [
            {
                "name": tc.get("name", ""),
                "arguments": tc.get("args", {}),
                "description": "",
            }
            for tc in tool_calls
        ]
        review_configs = [
            {
                "action_name": ar["name"],
                "allowed_decisions": ["approve", "edit", "reject"],
            }
            for ar in action_requests
            if ar["name"]
        ]

        approval = interrupt(
            {
                "action_requests": action_requests,
                "review_configs": review_configs,
            }
        )

        decisions: List[Dict[str, Any]] = []
        if isinstance(approval, dict):
            maybe_decisions = approval.get("decisions")
            if isinstance(maybe_decisions, list):
                decisions = [d for d in maybe_decisions if isinstance(d, dict)]

        # Apply decisions in-order (fallback: approve).
        for idx, tool_call in enumerate(tool_calls):
            decision = decisions[idx] if idx < len(decisions) else {"type": "approve"}
            decision_type = decision.get("type", "approve")
            if decision_type == "reject":
                tool_call["_hitl_rejected"] = True
                tool_call["_hitl_message"] = decision.get("message", "Action rejected by user")
                continue
            if decision_type == "edit":
                edited_action = decision.get("edited_action") or {}
                if isinstance(edited_action, dict):
                    if edited_action.get("name"):
                        tool_call["name"] = edited_action["name"]
                    if isinstance(edited_action.get("args"), dict):
                        tool_call["args"] = edited_action["args"]

    # Build tool map
    tool_map = {tool.name: tool for tool in tools}

    # Execute each tool call
    tool_messages = []
    for tool_call in tool_calls:
        if tool_call.get("_hitl_rejected"):
            tool_messages.append(
                ToolMessage(
                    content=str(tool_call.get("_hitl_message", "Action rejected by user")),
                    tool_call_id=str(tool_call.get("id", "")),
                )
            )
            continue

        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", "")

        if not tool_name:
            error_msg = "Tool call is missing a tool name"
            logger.error(error_msg)
            tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call_id))
            continue

        logger.info("Executing tool: %s with args: %s", tool_name, tool_args)

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
            error_msg = f"Error executing tool {tool_name}: {e}"
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

    retrieved_contexts: List[Dict[str, Any]] = []
    if state.get("enable_context_tracking", False):
        for tool_call, tool_message in zip(tool_calls, tool_messages):
            tool_name = tool_call.get("name")
            if not tool_name or not isinstance(tool_message, ToolMessage):
                continue
            retrieved_contexts.extend(
                extract_contexts_from_tool_result(
                    tool_name=tool_name,
                    tool_result=tool_message,
                )
            )

    return {
        "messages": tool_messages,
        "tool_call_count": tool_call_count,
        "retrieved_contexts": retrieved_contexts,
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
        final_answer = _render_content(last_message.content)
    else:
        final_answer = str(last_message.content)

    logger.info(f"Final answer extracted ({len(final_answer)} chars)")

    return {
        "final_answer": final_answer,
    }
