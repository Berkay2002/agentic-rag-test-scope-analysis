"""Agentic evaluation for full agent pipeline testing.

This module evaluates the complete ReAct agent loop on test scope queries,
measuring how well the agent dynamically selects retrieval strategies
compared to static baseline approaches.
"""

import asyncio
import hashlib
import logging
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from statistics import mean, pstdev

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
from agrag.evaluation.tool_tracker import ToolTracker
from agrag.evaluation.ragas_metrics import RagasEvaluator
from agrag.cli.utils import extract_message_content
from agrag.config import settings

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

    # Ragas and trial support
    ragas_metrics: Optional[Dict[str, float]] = None
    trial_number: int = 1
    contexts_used: List[str] = field(default_factory=list)

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
            "final_answer": self.final_answer,
            "error": self.error,
            "ragas_metrics": self.ragas_metrics,
            "trial_number": self.trial_number,
            "contexts_used": self.contexts_used,
        }


@dataclass
class AgentEvaluationSummary:
    """Summary of agent evaluation across all queries."""

    # Aggregate metrics
    map_score: float = 0.0
    mrr_score: float = 0.0
    avg_precision_at_k: Dict[int, float] = field(default_factory=dict)
    avg_recall_at_k: Dict[int, float] = field(default_factory=dict)
    avg_f1_at_k: Dict[int, float] = field(default_factory=dict)

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

    # Multi-trial and Ragas support
    trial_statistics: Optional[Dict[str, Any]] = None
    avg_ragas_metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "map": round(self.map_score, 4),
            "mrr": round(self.mrr_score, 4),
            "avg_precision_at_k": {str(k): round(v, 4) for k, v in self.avg_precision_at_k.items()},
            "avg_recall_at_k": {str(k): round(v, 4) for k, v in self.avg_recall_at_k.items()},
            "avg_f1_at_k": {str(k): round(v, 4) for k, v in self.avg_f1_at_k.items()},
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
            "trial_statistics": self.trial_statistics,
            "avg_ragas_metrics": self.avg_ragas_metrics,
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
        use_ragas: bool = False,
        num_trials: int = 1,
        enable_context_tracking: bool = True,
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
        self.use_ragas = use_ragas
        self.num_trials = max(1, num_trials)
        self.enable_context_tracking = enable_context_tracking
        self.ragas_evaluator = None
        if self.use_ragas:
            self.ragas_evaluator = RagasEvaluator(
                model_name=settings.ragas_model,
                max_retries=settings.ragas_max_retries,
            )

    def _build_initial_state(self, query: str) -> Dict[str, Any]:
        from agrag.core import create_initial_state

        initial_state = create_initial_state(query)
        initial_state["retrieved_contexts"] = []
        initial_state["enable_context_tracking"] = self.enable_context_tracking
        return initial_state

    def _run_async(self, coroutine: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        if loop.is_running():
            if asyncio.current_task(loop=loop) is not None:
                raise RuntimeError(
                    "Cannot run async evaluation inside a running event loop. "
                    "Use the async Ragas evaluator directly."
                )
            return asyncio.run_coroutine_threadsafe(coroutine, loop).result()
        return loop.run_until_complete(coroutine)

    @staticmethod
    def _format_contexts(retrieved_contexts: List[Dict[str, Any]]) -> List[str]:
        if not retrieved_contexts:
            return []

        seen_hashes = set()
        formatted: List[str] = []

        for context in retrieved_contexts:
            text: Optional[str] = None
            if isinstance(context, str):
                text = context
            elif isinstance(context, dict):
                text = (
                    context.get("chunk_text")
                    or context.get("content")
                    or context.get("text")
                    or context.get("chunk")
                )

            if not text:
                continue

            normalized = text.strip()
            if not normalized:
                continue

            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                continue

            seen_hashes.add(digest)
            formatted.append(normalized)

        return formatted

    def evaluate_query(
        self,
        query_id: str,
        query: str,
        relevant_ids: Set[str],
        query_type: str = "unknown",
        difficulty: str = "unknown",
        trial_number: int = 1,
        ground_truth_answer: Optional[str] = None,
        initial_state: Optional[Dict[str, Any]] = None,
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
            trial_number=trial_number,
        )

        # Start timing
        start_time = time.time()

        try:
            # Create initial state (new format for create_agent API)
            if initial_state is None:
                initial_state = self._build_initial_state(query)

            # Run agent to completion (no HITL interrupts)
            final_state = self.graph.invoke(initial_state, config=self.config)

            # Extract results from the new state format
            messages = final_state.get("messages", [])

            # Find final answer from the last AI message
            final_answer = ""
            tool_call_count = 0
            model_call_count = 0

            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_call_count += len(msg.tool_calls)
                if hasattr(msg, "type") and msg.type == "ai":
                    model_call_count += 1
                    # Check if this is a final response (no tool calls)
                    if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                        if hasattr(msg, "content") and msg.content:
                            final_answer = extract_message_content(msg.content)

            result.final_answer = final_answer
            result.tool_call_count = tool_call_count
            result.model_call_count = model_call_count

            # Extract entity IDs from response
            result.retrieved_ids = extract_entity_ids(
                result.final_answer,
                prioritize_test_cases=True,
            )

            # Also extract from tool results for more complete coverage
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

            contexts = []
            if self.enable_context_tracking:
                contexts = final_state.get("retrieved_contexts", []) or []

            formatted_contexts = self._format_contexts(contexts)
            result.contexts_used = formatted_contexts

            if self.ragas_evaluator is not None and formatted_contexts:
                result.ragas_metrics = self._run_async(
                    self.ragas_evaluator.evaluate_with_ragas(
                        query=query,
                        answer=result.final_answer,
                        contexts=formatted_contexts,
                        ground_truth=ground_truth_answer,
                    )
                )

        except Exception as e:
            logger.error(f"Error evaluating query '{query_id}': {e}")
            result.error = str(e)

        # Record execution time
        result.execution_time_ms = (time.time() - start_time) * 1000

        return result

    def evaluate_query_with_trials(
        self,
        query_id: str,
        query: str,
        relevant_ids: Set[str],
        query_type: str = "unknown",
        difficulty: str = "unknown",
        ground_truth_answer: Optional[str] = None,
    ) -> List[AgentEvaluationResult]:
        """Evaluate a query with multiple trials."""
        trial_results: List[AgentEvaluationResult] = []
        for trial_number in range(1, self.num_trials + 1):
            initial_state = self._build_initial_state(query)
            result = self.evaluate_query(
                query_id=query_id,
                query=query,
                relevant_ids=relevant_ids,
                query_type=query_type,
                difficulty=difficulty,
                trial_number=trial_number,
                ground_truth_answer=ground_truth_answer,
                initial_state=initial_state,
            )
            trial_results.append(result)
        return trial_results

    def aggregate_trial_statistics(
        self,
        trial_results: List[AgentEvaluationResult],
    ) -> Dict[str, Any]:
        """Aggregate statistics across multiple trials."""
        if not trial_results:
            return {
                "num_trials": 0,
                "success_rate": 0.0,
                "pass_at_1": 0.0,
                "pass_at_k": 0.0,
                "mean_metrics": {},
                "std_metrics": {},
                "min_metrics": {},
                "max_metrics": {},
                "stability_score": 0.0,
            }

        success_flags = [result.error is None for result in trial_results]
        num_trials = len(trial_results)
        success_rate = sum(success_flags) / max(1, num_trials)
        pass_at_1 = 1.0 if any(success_flags) else 0.0
        pass_at_k = 1.0 if all(success_flags) else 0.0

        metric_values: Dict[str, List[float]] = {}
        for result in trial_results:
            if result.error is not None:
                continue
            for key, value in result.metrics.items():
                metric_values.setdefault(key, []).append(value)

        mean_metrics = {key: mean(vals) for key, vals in metric_values.items() if vals}
        std_metrics = {key: pstdev(vals) for key, vals in metric_values.items() if len(vals) > 1}
        min_metrics = {key: min(vals) for key, vals in metric_values.items() if vals}
        max_metrics = {key: max(vals) for key, vals in metric_values.items() if vals}

        normalized_stds = []
        for key, std_value in std_metrics.items():
            avg = mean_metrics.get(key, 0.0)
            if avg > 0:
                normalized_stds.append(std_value / avg)

        stability_score = 1.0
        if normalized_stds:
            stability_score = max(0.0, 1.0 - mean(normalized_stds))

        return {
            "num_trials": num_trials,
            "success_rate": round(success_rate, 4),
            "pass_at_1": round(pass_at_1, 4),
            "pass_at_k": round(pass_at_k, 4),
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
            "min_metrics": min_metrics,
            "max_metrics": max_metrics,
            "stability_score": round(stability_score, 4),
        }

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

        per_query_trial_stats: List[Dict[str, Any]] = []

        for i, query_data in enumerate(queries, 1):
            query = query_data["query"]
            relevant = set(query_data.get("relevant_ids", []))
            query_id = query_data.get("id", f"Q_{i}")
            query_type = query_data.get("query_type", "unknown")
            difficulty = query_data.get("difficulty", "unknown")
            ground_truth_answer = query_data.get("reference_answer") or query_data.get(
                "ground_truth"
            )

            if verbose:
                logger.info(f"[{i}/{len(queries)}] ({difficulty}) {query[:50]}...")

            if self.num_trials > 1:
                trial_results = self.evaluate_query_with_trials(
                    query_id=query_id,
                    query=query,
                    relevant_ids=relevant,
                    query_type=query_type,
                    difficulty=difficulty,
                    ground_truth_answer=ground_truth_answer,
                )
                all_results.extend(trial_results)

                per_query_stats = self.aggregate_trial_statistics(trial_results)
                per_query_trial_stats.append(per_query_stats)

                if any(result.error is None for result in trial_results):
                    summary.successful_queries += 1

                if verbose and trial_results:
                    sample = trial_results[0]
                    logger.info(
                        f"  Trials: {len(trial_results)}, "
                        f"Retrieved: {len(sample.retrieved_ids)}, "
                        f"Tools: {sample.tools_used}, "
                        f"RR: {sample.metrics.get('reciprocal_rank', 0):.4f}"
                    )
            else:
                result = self.evaluate_query(
                    query_id=query_id,
                    query=query,
                    relevant_ids=relevant,
                    query_type=query_type,
                    difficulty=difficulty,
                    ground_truth_answer=ground_truth_answer,
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
            f1_scores = [r.metrics.get(f"f1@{k}", 0) for r in all_results if r.error is None]

            if p_scores:
                summary.avg_precision_at_k[k] = sum(p_scores) / len(p_scores)
            if r_scores:
                summary.avg_recall_at_k[k] = sum(r_scores) / len(r_scores)
            if f1_scores:
                summary.avg_f1_at_k[k] = sum(f1_scores) / len(f1_scores)

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

        if self.num_trials > 1:
            if per_query_trial_stats:
                aggregate_stats = self.aggregate_trial_statistics(all_results)
                summary.trial_statistics = {
                    "num_trials": self.num_trials,
                    "success_rate": round(
                        sum(1 for r in all_results if r.error is None)
                        / max(1, len(all_results)),
                        4,
                    ),
                    "pass_at_1": round(
                        sum(1 for stats in per_query_trial_stats if stats.get("pass_at_1") == 1.0)
                        / max(1, len(per_query_trial_stats)),
                        4,
                    ),
                    "pass_at_k": round(
                        sum(1 for stats in per_query_trial_stats if stats.get("pass_at_k") == 1.0)
                        / max(1, len(per_query_trial_stats)),
                        4,
                    ),
                    "mean_metrics": aggregate_stats.get("mean_metrics", {}),
                    "std_metrics": aggregate_stats.get("std_metrics", {}),
                    "min_metrics": aggregate_stats.get("min_metrics", {}),
                    "max_metrics": aggregate_stats.get("max_metrics", {}),
                    "stability_score": aggregate_stats.get("stability_score", 0.0),
                }

        ragas_values: Dict[str, List[float]] = {}
        for result in all_results:
            if result.ragas_metrics:
                for key, value in result.ragas_metrics.items():
                    ragas_values.setdefault(key, []).append(value)
        if ragas_values:
            summary.avg_ragas_metrics = {key: mean(vals) for key, vals in ragas_values.items()}

        return summary


def create_evaluation_graph():
    """
    Create an agent graph configured for evaluation (no HITL).

    Returns:
        Compiled agent graph without HITL interrupts
    """
    from agrag.core import create_agent_graph

    # Create graph without checkpointer and with HITL disabled
    graph = create_agent_graph(checkpointer=None, enable_hitl=False)

    return graph
