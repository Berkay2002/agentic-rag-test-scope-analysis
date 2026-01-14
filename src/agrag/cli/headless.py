"""Headless execution utilities for the AgRAG CLI."""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, ToolMessage

from agrag.config import settings
from agrag.core import create_agent_graph, create_initial_state
from agrag.core.checkpointing import initialize_checkpointer, summarize_error
from agrag.cli.utils import extract_message_content

logger = logging.getLogger(__name__)


def _iso_timestamp() -> str:
    """Return a UTC ISO timestamp with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# Content extraction is now handled by agrag.cli.utils.extract_message_content


def _is_tool_error(output: str) -> bool:
    """Heuristic to detect tool errors from textual output."""
    lowered = output.lower()
    return (
        lowered.startswith("error")
        or "error executing tool" in lowered
        or "not found" in lowered
        or "missing" in lowered
        or "failed" in lowered
    )


@dataclass
class ToolCallRecord:
    """Capture tool call metadata for stats aggregation."""

    tool_name: str
    arguments: Dict[str, Any]
    start_time: float


@dataclass
class ToolStats:
    """Aggregate tool usage statistics for headless mode."""

    total_calls: int = 0
    total_success: int = 0
    total_fail: int = 0
    total_duration_ms: float = 0.0
    by_name: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pending: Dict[str, ToolCallRecord] = field(default_factory=dict)

    def record_call(self, tool_call_id: str, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Record a new tool call."""
        self.total_calls += 1
        self.pending[tool_call_id] = ToolCallRecord(
            tool_name=tool_name,
            arguments=arguments,
            start_time=time.monotonic(),
        )
        slot = self.by_name.setdefault(
            tool_name,
            {
                "count": 0,
                "success": 0,
                "fail": 0,
                "durationMs": 0.0,
                "decisions": {"accept": 0, "reject": 0, "modify": 0, "auto_accept": 0},
            },
        )
        slot["count"] += 1
        slot["decisions"]["auto_accept"] += 1

    def record_result(self, tool_call_id: str, output: str) -> Dict[str, Any]:
        """Record a tool result and return resolved metadata."""
        now = time.monotonic()
        record = self.pending.pop(tool_call_id, None)
        tool_name = record.tool_name if record else "unknown"
        duration_ms = (now - record.start_time) * 1000 if record else 0.0
        success = not _is_tool_error(output)

        if success:
            self.total_success += 1
        else:
            self.total_fail += 1

        self.total_duration_ms += duration_ms

        slot = self.by_name.setdefault(
            tool_name,
            {
                "count": 0,
                "success": 0,
                "fail": 0,
                "durationMs": 0.0,
                "decisions": {"accept": 0, "reject": 0, "modify": 0, "auto_accept": 0},
            },
        )
        if success:
            slot["success"] += 1
        else:
            slot["fail"] += 1
        slot["durationMs"] += duration_ms

        return {
            "tool_name": tool_name,
            "duration_ms": duration_ms,
            "success": success,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tool stats for JSON output."""
        return {
            "totalCalls": self.total_calls,
            "totalSuccess": self.total_success,
            "totalFail": self.total_fail,
            "totalDurationMs": round(self.total_duration_ms),
            "totalDecisions": {
                "accept": 0,
                "reject": 0,
                "modify": 0,
                "auto_accept": self.total_calls,
            },
            "byName": self.by_name,
        }


def read_stdin() -> str:
    """Read stdin if it is piped, otherwise return an empty string."""
    if sys.stdin is None or sys.stdin.closed or sys.stdin.isatty():
        return ""
    content = sys.stdin.read()
    return content if content and content.strip() else ""


def build_prompt(prompt: Optional[str], stdin_text: str) -> str:
    """Combine prompt and stdin content into a single prompt."""
    if prompt and stdin_text:
        return f"{prompt.rstrip()}\n\n{stdin_text}"
    if prompt:
        return prompt
    if stdin_text:
        return stdin_text
    raise ValueError("No prompt provided. Use --prompt or pipe input via stdin.")


def _emit_json_event(event: Dict[str, Any]) -> None:
    """Emit a JSONL event to stdout."""
    payload = json.dumps(event, ensure_ascii=True, default=str)
    sys.stdout.write(payload + "\n")
    sys.stdout.flush()


def run_headless(
    prompt: str,
    output_format: str = "text",
    thread_id: Optional[str] = None,
    debug: bool = False,
) -> int:
    """Run a single prompt in headless mode."""
    if not debug:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.ERROR)
        for handler in root_logger.handlers:
            handler.setLevel(logging.ERROR)

    session_id = thread_id or f"headless-{uuid.uuid4().hex[:8]}"
    start_time = time.monotonic()
    tool_stats = ToolStats()
    model_calls = 0
    final_answer = ""
    error: Optional[Exception] = None

    checkpointer = None
    if thread_id:
        init_result = initialize_checkpointer(enable_hitl=False, enable_persistence=True)
        checkpointer = init_result.checkpointer
        if debug and init_result.backend == "memory" and init_result.error:
            logger.warning(
                "Headless persistence requested but Postgres unavailable: %s",
                summarize_error(init_result.error),
            )

    graph = create_agent_graph(checkpointer=checkpointer, enable_hitl=False)
    initial_state = create_initial_state(prompt)
    config: Dict[str, Any] = {}
    if checkpointer and thread_id:
        config["configurable"] = {"thread_id": thread_id}

    if output_format == "stream-json":
        _emit_json_event(
            {
                "type": "init",
                "timestamp": _iso_timestamp(),
                "session_id": session_id,
                "model": settings.google_model,
            }
        )
        _emit_json_event(
            {
                "type": "message",
                "role": "user",
                "content": prompt,
                "timestamp": _iso_timestamp(),
            }
        )

    try:
        for event in graph.stream(initial_state, config=config, stream_mode="values"):
            messages = event.get("messages", [])
            if not messages:
                continue

            last_message = messages[-1]

            if isinstance(last_message, AIMessage):
                model_calls += 1
                if getattr(last_message, "tool_calls", None):
                    for tool_call in last_message.tool_calls:
                        tool_call_id = str(tool_call.get("id", ""))
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})
                        tool_stats.record_call(tool_call_id, tool_name, tool_args)
                        if output_format == "stream-json":
                            _emit_json_event(
                                {
                                    "type": "tool_use",
                                    "tool_name": tool_name,
                                    "tool_id": tool_call_id,
                                    "parameters": tool_args,
                                    "timestamp": _iso_timestamp(),
                                }
                            )
                elif last_message.content:
                    final_answer = extract_message_content(last_message.content)
                    if output_format == "stream-json":
                        _emit_json_event(
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": final_answer,
                                "timestamp": _iso_timestamp(),
                            }
                        )

            elif isinstance(last_message, ToolMessage):
                tool_call_id = str(last_message.tool_call_id or "")
                output_text = extract_message_content(last_message.content)
                result_meta = tool_stats.record_result(tool_call_id, output_text)
                if output_format == "stream-json":
                    _emit_json_event(
                        {
                            "type": "tool_result",
                            "tool_id": tool_call_id,
                            "status": "success" if result_meta["success"] else "error",
                            "output": output_text,
                            "timestamp": _iso_timestamp(),
                        }
                    )

    except Exception as exc:
        error = exc
        logger.exception("Headless execution failed")
        if output_format == "stream-json":
            _emit_json_event(
                {
                    "type": "error",
                    "message": str(exc),
                    "error_type": exc.__class__.__name__,
                    "timestamp": _iso_timestamp(),
                }
            )

    duration_ms = round((time.monotonic() - start_time) * 1000)
    stats_payload = {
        "models": {
            settings.google_model: {
                "api": {
                    "totalRequests": model_calls,
                    "totalErrors": 1 if error else 0,
                    "totalLatencyMs": duration_ms,
                },
                "tokens": {
                    "prompt": 0,
                    "candidates": 0,
                    "total": 0,
                    "cached": 0,
                    "thoughts": 0,
                    "tool": 0,
                },
            }
        },
        "tools": tool_stats.to_dict(),
        "files": {"totalLinesAdded": 0, "totalLinesRemoved": 0},
    }

    if output_format == "stream-json":
        _emit_json_event(
            {
                "type": "result",
                "status": "error" if error else "success",
                "response": final_answer,
                "stats": stats_payload,
                "duration_ms": duration_ms,
                "timestamp": _iso_timestamp(),
            }
        )
        return 1 if error else 0

    if output_format == "json":
        response = {
            "response": final_answer,
            "stats": stats_payload,
        }
        if error:
            response["error"] = {
                "type": error.__class__.__name__,
                "message": str(error),
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=True, default=str) + "\n")
        sys.stdout.flush()
        return 1 if error else 0

    if error:
        sys.stderr.write(f"Error: {error}\n")
        sys.stderr.flush()
        return 1

    sys.stdout.write(final_answer + "\n")
    sys.stdout.flush()
    return 0
