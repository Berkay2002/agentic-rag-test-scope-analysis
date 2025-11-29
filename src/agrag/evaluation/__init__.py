"""Evaluation metrics and utilities."""

from .metrics import (
    precision_at_k,
    recall_at_k,
    f1_score_at_k,
    average_precision,
    mean_average_precision,
    reciprocal_rank,
    mean_reciprocal_rank,
    evaluate_retrieval,
    log_metrics,
)

from .entity_extractor import (
    extract_entity_ids,
    extract_entity_ids_detailed,
    extract_from_tool_results,
    ExtractionResult,
)

from .tool_tracker import (
    ToolCall,
    ToolUsageStats,
    AggregateToolStats,
    ToolTracker,
)

from .agentic_evaluator import (
    AgentEvaluationResult,
    AgentEvaluationSummary,
    AgenticEvaluator,
    create_evaluation_graph,
)

__all__ = [
    # Metrics
    "precision_at_k",
    "recall_at_k",
    "f1_score_at_k",
    "average_precision",
    "mean_average_precision",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "evaluate_retrieval",
    "log_metrics",
    # Entity extraction
    "extract_entity_ids",
    "extract_entity_ids_detailed",
    "extract_from_tool_results",
    "ExtractionResult",
    # Tool tracking
    "ToolCall",
    "ToolUsageStats",
    "AggregateToolStats",
    "ToolTracker",
    # Agentic evaluation
    "AgentEvaluationResult",
    "AgentEvaluationSummary",
    "AgenticEvaluator",
    "create_evaluation_graph",
]
