"""Tool usage tracking for agentic evaluation.

This module provides utilities to track which tools the agent uses
for each query, enabling analysis of tool selection patterns.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Record of a single tool call."""

    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None
    execution_time_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ToolUsageStats:
    """Statistics about tool usage for a single query."""

    query_id: str
    query: str

    # Tool calls made
    tool_calls: List[ToolCall] = field(default_factory=list)

    # Aggregated metrics
    tools_used: List[str] = field(default_factory=list)
    tool_call_count: int = 0
    model_call_count: int = 0
    total_execution_time_ms: float = 0.0

    # Success indicators
    any_tool_succeeded: bool = False
    all_tools_succeeded: bool = True

    def add_tool_call(self, tool_call: ToolCall):
        """Add a tool call to the stats."""
        self.tool_calls.append(tool_call)
        self.tool_call_count += 1
        self.total_execution_time_ms += tool_call.execution_time_ms

        if tool_call.tool_name not in self.tools_used:
            self.tools_used.append(tool_call.tool_name)

        if tool_call.success:
            self.any_tool_succeeded = True
        else:
            self.all_tools_succeeded = False

    @property
    def unique_tools_count(self) -> int:
        """Get number of unique tools used."""
        return len(self.tools_used)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query_id": self.query_id,
            "query": self.query,
            "tools_used": self.tools_used,
            "tool_call_count": self.tool_call_count,
            "model_call_count": self.model_call_count,
            "total_execution_time_ms": self.total_execution_time_ms,
            "any_tool_succeeded": self.any_tool_succeeded,
            "all_tools_succeeded": self.all_tools_succeeded,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "success": tc.success,
                    "execution_time_ms": tc.execution_time_ms,
                }
                for tc in self.tool_calls
            ],
        }


@dataclass
class AggregateToolStats:
    """Aggregated tool usage statistics across multiple queries."""

    total_queries: int = 0
    total_tool_calls: int = 0
    total_model_calls: int = 0

    # Tool usage frequency
    tool_frequency: Dict[str, int] = field(default_factory=dict)

    # Tool combinations used
    tool_combinations: Dict[str, int] = field(default_factory=dict)

    # Average metrics
    avg_tools_per_query: float = 0.0
    avg_execution_time_ms: float = 0.0

    # Success rates by tool
    tool_success_rates: Dict[str, float] = field(default_factory=dict)

    def add_query_stats(self, stats: ToolUsageStats):
        """Add stats from a single query."""
        self.total_queries += 1
        self.total_tool_calls += stats.tool_call_count
        self.total_model_calls += stats.model_call_count

        # Update tool frequency
        for tool_name in stats.tools_used:
            self.tool_frequency[tool_name] = self.tool_frequency.get(tool_name, 0) + 1

        # Track tool combinations
        if stats.tools_used:
            combo_key = "+".join(sorted(stats.tools_used))
            self.tool_combinations[combo_key] = self.tool_combinations.get(combo_key, 0) + 1

        # Update averages
        if self.total_queries > 0:
            self.avg_tools_per_query = self.total_tool_calls / self.total_queries

    def compute_success_rates(self, all_stats: List[ToolUsageStats]):
        """Compute success rates for each tool."""
        tool_attempts: Dict[str, int] = {}
        tool_successes: Dict[str, int] = {}

        for stats in all_stats:
            for tc in stats.tool_calls:
                tool_attempts[tc.tool_name] = tool_attempts.get(tc.tool_name, 0) + 1
                if tc.success:
                    tool_successes[tc.tool_name] = tool_successes.get(tc.tool_name, 0) + 1

        for tool_name, attempts in tool_attempts.items():
            successes = tool_successes.get(tool_name, 0)
            self.tool_success_rates[tool_name] = successes / attempts if attempts > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_queries": self.total_queries,
            "total_tool_calls": self.total_tool_calls,
            "total_model_calls": self.total_model_calls,
            "tool_frequency": self.tool_frequency,
            "tool_combinations": self.tool_combinations,
            "avg_tools_per_query": round(self.avg_tools_per_query, 2),
            "avg_execution_time_ms": round(self.avg_execution_time_ms, 2),
            "tool_success_rates": {k: round(v, 4) for k, v in self.tool_success_rates.items()},
        }


class ToolTracker:
    """
    Tracks tool usage during agent execution.

    Can be used to analyze which tools the agent selects for different
    query types and how effective each tool is.
    """

    def __init__(self):
        self.query_stats: List[ToolUsageStats] = []
        self.aggregate_stats = AggregateToolStats()

    def start_query(self, query_id: str, query: str) -> ToolUsageStats:
        """Start tracking a new query."""
        stats = ToolUsageStats(query_id=query_id, query=query)
        return stats

    def record_query(self, stats: ToolUsageStats):
        """Record completed query stats."""
        self.query_stats.append(stats)
        self.aggregate_stats.add_query_stats(stats)

    def extract_tool_calls_from_messages(
        self,
        messages: List[Any],
        stats: ToolUsageStats,
    ):
        """
        Extract tool calls from message history.

        Args:
            messages: List of messages from agent execution
            stats: ToolUsageStats to populate
        """
        from langchain_core.messages import AIMessage, ToolMessage

        # Find AIMessages with tool_calls and corresponding ToolMessages
        pending_calls: Dict[str, ToolCall] = {}

        for msg in messages:
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_call = ToolCall(
                        tool_name=tc["name"],
                        arguments=tc.get("args", {}),
                    )
                    tc_id = tc.get("id")
                    if tc_id:
                        pending_calls[tc_id] = tool_call

            elif isinstance(msg, ToolMessage):
                tool_call_id = getattr(msg, "tool_call_id", None)
                if tool_call_id and tool_call_id in pending_calls:
                    tool_call = pending_calls[tool_call_id]
                    tool_call.result = str(msg.content)[:500]  # Truncate for storage

                    # Check for error indicators
                    content_lower = str(msg.content).lower()
                    if "error" in content_lower or "failed" in content_lower:
                        tool_call.success = False
                        tool_call.error_message = str(msg.content)[:200]

                    stats.add_tool_call(tool_call)
                    del pending_calls[tool_call_id]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked queries."""
        self.aggregate_stats.compute_success_rates(self.query_stats)

        return {
            "aggregate": self.aggregate_stats.to_dict(),
            "per_query": [stats.to_dict() for stats in self.query_stats],
        }

    def analyze_tool_selection_by_query_type(
        self,
        query_types: Dict[str, str],
    ) -> Dict[str, Dict[str, int]]:
        """
        Analyze which tools are selected for different query types.

        Args:
            query_types: Mapping of query_id to query_type

        Returns:
            Dict mapping query_type to tool frequency
        """
        result: Dict[str, Dict[str, int]] = {}

        for stats in self.query_stats:
            query_type = query_types.get(stats.query_id, "unknown")

            if query_type not in result:
                result[query_type] = {}

            for tool_name in stats.tools_used:
                result[query_type][tool_name] = result[query_type].get(tool_name, 0) + 1

        return result
