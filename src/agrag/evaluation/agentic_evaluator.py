"""Agentic evaluation for full agent pipeline testing.

This module evaluates the complete ReAct agent loop on test scope queries,
measuring how well the agent dynamically selects retrieval strategies
compared to static baseline approaches.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from agrag.evaluation.metrics import (
    evaluate_retrieval,
    average_precision,
    reciprocal_rank,
)
from agrag.evaluation.entity_extractor import (
    extract_entity_ids,
    extract_from_tool_results,
)
from agrag.evaluation.tool_tracker import (
    ToolTracker,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentEvaluationResult:
    """Result of evaluating a single query with the agent."""

    query_id: str
    query: str
    query_type: str
    difficulty: str

    # Retrieval results
    retrieved_ids: List[str] = field(default_factory=list)
    relevant_ids: Set[str] = field(default_factory=set)
    metrics: Dict[str, float] = field(default_factory=dict)

    # Agent behavior analysis
    tools_used: List[str] = field(default_factory=list)
    tool_call_count: int = 0
    model_call_count: int = 0
    execution_time_ms: float = 0.0

    # Agent response
    final_answer: str = ""

    # Success indicators
    found_any_relevant: bool = False
    first_relevant_rank: Optional[int] = None

    # Error handling
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query_id": self.query_id,
            "query": self.query,
            "query_type": self.query_type,
            "difficulty": self.difficulty,
            "retrieved_ids": self.retrieved_ids,
            "relevant_ids": list(self.relevant_ids),
            "metrics": self.metrics,
            "tools_used": self.tools_used,
            "tool_call_count": self.tool_call_count,
            "model_call_count": self.model_call_count,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "found_any_relevant": self.found_any_relevant,
            "first_relevant_rank": self.first_relevant_rank,
            "error": self.error,
        }


@dataclass
class AgentEvaluationSummary:
    """Summary of agent evaluation across all queries."""

    # Aggregate metrics
    map_score: float = 0.0
    mrr_score: float = 0.0
    avg_precision_at_k: Dict[int, float] = field(default_factory=dict)
    avg_recall_at_k: Dict[int, float] = field(default_factory=dict)

    # Tool usage statistics
    total_tool_calls: int = 0
    avg_tools_per_query: float = 0.0
    tool_frequency: Dict[str, int] = field(default_factory=dict)
    tool_combinations: Dict[str, int] = field(default_factory=dict)

    # Execution statistics
    total_queries: int = 0
    successful_queries: int = 0
    avg_execution_time_ms: float = 0.0

    # Per-query results
    results: List[AgentEvaluationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "map": round(self.map_score, 4),
            "mrr": round(self.mrr_score, 4),
            "avg_precision_at_k": {str(k): round(v, 4) for k, v in self.avg_precision_at_k.items()},
            "avg_recall_at_k": {str(k): round(v, 4) for k, v in self.avg_recall_at_k.items()},
            "tool_usage": {
                "total_tool_calls": self.total_tool_calls,
                "avg_tools_per_query": round(self.avg_tools_per_query, 2),
                "tool_frequency": self.tool_frequency,
                "tool_combinations": self.tool_combinations,
            },
            "execution_stats": {
                "total_queries": self.total_queries,
                "successful_queries": self.successful_queries,
                "success_rate": round(self.successful_queries / max(1, self.total_queries), 4),
                "avg_execution_time_ms": round(self.avg_execution_time_ms, 2),
            },
            "per_query_results": [r.to_dict() for r in self.results],
        }


class AgenticEvaluator:
    """
    Evaluates the full agent pipeline on test scope queries.

    Unlike static strategy evaluation, this:
    1. Runs the complete ReAct agent loop per query
    2. Lets the LLM decide which tool(s) to use
    3. Extracts entity IDs from the agent's final response
    4. Logs tool selection patterns for analysis
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        config: Optional[RunnableConfig] = None,
        k_values: Optional[List[int]] = None,
    ):
        """
        Initialize the agentic evaluator.

        Args:
            graph: Compiled agent graph (without HITL interrupts)
            config: Optional runnable config
            k_values: K values for metrics (default: [1, 3, 5, 10])
        """
        self.graph = graph
        self.config = config or {}
        self.k_values = k_values or [1, 3, 5, 10]
        self.tool_tracker = ToolTracker()

    def evaluate_query(
        self,
        query_id: str,
        query: str,
        relevant_ids: Set[str],
        query_type: str = "unknown",
        difficulty: str = "unknown",
    ) -> AgentEvaluationResult:
        """
        Run agent on a single query and evaluate results.

        Args:
            query_id: Unique query identifier
            query: The query text
            relevant_ids: Ground truth relevant entity IDs
            query_type: Type of query (for analysis)
            difficulty: Query difficulty (for analysis)

        Returns:
            AgentEvaluationResult with metrics and tool usage
        """
        result = AgentEvaluationResult(
            query_id=query_id,
            query=query,
            query_type=query_type,
            difficulty=difficulty,
            relevant_ids=relevant_ids,
        )

        # Start timing
        start_time = time.time()

        try:
            # Create initial state
            from agrag.core import create_initial_state

            initial_state = create_initial_state(query)

            # Run agent to completion (no HITL interrupts)
            final_state = self.graph.invoke(initial_state, config=self.config)

            # Extract results
            result.final_answer = final_state.get("final_answer", "")
            result.tool_call_count = final_state.get("tool_call_count", 0)
            result.model_call_count = final_state.get("model_call_count", 0)

            # Extract entity IDs from response
            result.retrieved_ids = extract_entity_ids(
                result.final_answer,
                prioritize_test_cases=True,
            )

            # Also extract from tool results for more complete coverage
            messages = final_state.get("messages", [])
            tool_ids = extract_from_tool_results(messages)

            # Merge IDs (response IDs first, then tool result IDs)
            seen = set(result.retrieved_ids)
            for tid in tool_ids:
                if tid not in seen:
                    result.retrieved_ids.append(tid)
                    seen.add(tid)

            # Track tool usage
            tool_stats = self.tool_tracker.start_query(query_id, query)
            self.tool_tracker.extract_tool_calls_from_messages(messages, tool_stats)
            tool_stats.model_call_count = result.model_call_count
            self.tool_tracker.record_query(tool_stats)

            result.tools_used = tool_stats.tools_used

            # Calculate metrics
            result.metrics = evaluate_retrieval(
                result.retrieved_ids,
                relevant_ids,
                k_values=self.k_values,
            )

            # Check success indicators
            result.found_any_relevant = any(rid in relevant_ids for rid in result.retrieved_ids)

            # Find rank of first relevant item
            for i, rid in enumerate(result.retrieved_ids, start=1):
                if rid in relevant_ids:
                    result.first_relevant_rank = i
                    break

        except Exception as e:
            logger.error(f"Error evaluating query '{query_id}': {e}")
            result.error = str(e)

        # Record execution time
        result.execution_time_ms = (time.time() - start_time) * 1000

        return result

    def evaluate_dataset(
        self,
        queries: List[Dict[str, Any]],
        verbose: bool = False,
    ) -> AgentEvaluationSummary:
        """
        Evaluate agent on a full dataset of queries.

        Args:
            queries: List of query dicts with keys:
                - query: Query text
                - relevant_ids: List of relevant entity IDs
                - id (optional): Query ID
                - query_type (optional): Query type
                - difficulty (optional): Difficulty level
            verbose: Print progress

        Returns:
            AgentEvaluationSummary with aggregate metrics
        """
        summary = AgentEvaluationSummary()
        summary.total_queries = len(queries)

        all_results: List[AgentEvaluationResult] = []

        for i, query_data in enumerate(queries, 1):
            query = query_data["query"]
            relevant = set(query_data.get("relevant_ids", []))
            query_id = query_data.get("id", f"Q_{i}")
            query_type = query_data.get("query_type", "unknown")
            difficulty = query_data.get("difficulty", "unknown")

            if verbose:
                logger.info(f"[{i}/{len(queries)}] ({difficulty}) {query[:50]}...")

            # Evaluate single query
            result = self.evaluate_query(
                query_id=query_id,
                query=query,
                relevant_ids=relevant,
                query_type=query_type,
                difficulty=difficulty,
            )

            all_results.append(result)

            if result.error is None:
                summary.successful_queries += 1

            if verbose:
                logger.info(
                    f"  Retrieved: {len(result.retrieved_ids)}, "
                    f"Tools: {result.tools_used}, "
                    f"RR: {result.metrics.get('reciprocal_rank', 0):.4f}"
                )

        # Calculate aggregate metrics
        summary.results = all_results

        # MAP and MRR
        aps = []
        rrs = []
        for r in all_results:
            if r.error is None:
                aps.append(average_precision(r.retrieved_ids, r.relevant_ids))
                rrs.append(reciprocal_rank(r.retrieved_ids, r.relevant_ids))

        if aps:
            summary.map_score = sum(aps) / len(aps)
        if rrs:
            summary.mrr_score = sum(rrs) / len(rrs)

        # Average P@k and R@k
        for k in self.k_values:
            p_scores = [r.metrics.get(f"precision@{k}", 0) for r in all_results if r.error is None]
            r_scores = [r.metrics.get(f"recall@{k}", 0) for r in all_results if r.error is None]

            if p_scores:
                summary.avg_precision_at_k[k] = sum(p_scores) / len(p_scores)
            if r_scores:
                summary.avg_recall_at_k[k] = sum(r_scores) / len(r_scores)

        # Tool usage statistics
        tool_summary = self.tool_tracker.get_summary()
        agg = tool_summary.get("aggregate", {})

        summary.total_tool_calls = agg.get("total_tool_calls", 0)
        summary.avg_tools_per_query = agg.get("avg_tools_per_query", 0)
        summary.tool_frequency = agg.get("tool_frequency", {})
        summary.tool_combinations = agg.get("tool_combinations", {})

        # Execution time
        exec_times = [r.execution_time_ms for r in all_results]
        if exec_times:
            summary.avg_execution_time_ms = sum(exec_times) / len(exec_times)

        return summary


def create_evaluation_graph():
    """
    Create an agent graph configured for evaluation (no HITL).

    Returns:
        Compiled StateGraph without HITL interrupts
    """
    from agrag.core import create_agent_graph

    # Create graph without checkpointer (no HITL)
    graph = create_agent_graph(checkpointer=None)

    return graph
