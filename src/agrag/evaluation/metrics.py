"""Evaluation metrics for retrieval quality assessment."""

import logging
from typing import List, Set, Dict, Any

logger = logging.getLogger(__name__)


def precision_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int = 10,
) -> float:
    """
    Calculate Precision@k.

    Precision@k = (# relevant items in top-k) / k

    Args:
        retrieved: List of retrieved item IDs (in ranked order)
        relevant: Set of relevant item IDs (ground truth)
        k: Number of top results to consider

    Returns:
        Precision@k score (0.0 to 1.0)
    """
    if k <= 0:
        raise ValueError("k must be positive")

    # Take top k results
    top_k = retrieved[:k]

    # Count relevant items in top k
    relevant_in_top_k = sum(1 for item_id in top_k if item_id in relevant)

    # Calculate precision
    precision = relevant_in_top_k / k

    return precision


def recall_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int = 10,
) -> float:
    """
    Calculate Recall@k.

    Recall@k = (# relevant items in top-k) / (total # relevant items)

    Args:
        retrieved: List of retrieved item IDs (in ranked order)
        relevant: Set of relevant item IDs (ground truth)
        k: Number of top results to consider

    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if k <= 0:
        raise ValueError("k must be positive")

    if len(relevant) == 0:
        # No relevant items to retrieve
        return 0.0

    # Take top k results
    top_k = retrieved[:k]

    # Count relevant items in top k
    relevant_in_top_k = sum(1 for item_id in top_k if item_id in relevant)

    # Calculate recall
    recall = relevant_in_top_k / len(relevant)

    return recall


def average_precision(
    retrieved: List[str],
    relevant: Set[str],
) -> float:
    """
    Calculate Average Precision (AP).

    AP = (sum of P@k for each relevant item) / (total # relevant items)

    Args:
        retrieved: List of retrieved item IDs (in ranked order)
        relevant: Set of relevant item IDs (ground truth)

    Returns:
        Average Precision score (0.0 to 1.0)
    """
    if len(relevant) == 0:
        return 0.0

    precisions = []
    relevant_count = 0

    # Iterate through retrieved items
    for k, item_id in enumerate(retrieved, start=1):
        if item_id in relevant:
            # Found a relevant item
            relevant_count += 1
            precision_at_rank = relevant_count / k
            precisions.append(precision_at_rank)

    if len(precisions) == 0:
        return 0.0

    # Average of precisions at relevant positions
    ap = sum(precisions) / len(relevant)

    return ap


def mean_average_precision(
    results: List[Dict[str, Any]],
) -> float:
    """
    Calculate Mean Average Precision (MAP) across multiple queries.

    MAP = (sum of AP for all queries) / (# queries)

    Args:
        results: List of result dictionaries with keys:
            - 'retrieved': List of retrieved item IDs
            - 'relevant': Set of relevant item IDs

    Returns:
        MAP score (0.0 to 1.0)
    """
    if len(results) == 0:
        return 0.0

    # Calculate AP for each query
    aps = []
    for result in results:
        retrieved = result.get("retrieved", [])
        relevant = result.get("relevant", set())

        ap = average_precision(retrieved, relevant)
        aps.append(ap)

    # Mean of all APs
    map_score = sum(aps) / len(aps)

    return map_score


def f1_score_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int = 10,
) -> float:
    """
    Calculate F1 score at k.

    F1@k = 2 * (Precision@k * Recall@k) / (Precision@k + Recall@k)

    Args:
        retrieved: List of retrieved item IDs (in ranked order)
        relevant: Set of relevant item IDs (ground truth)
        k: Number of top results to consider

    Returns:
        F1@k score (0.0 to 1.0)
    """
    precision = precision_at_k(retrieved, relevant, k)
    recall = recall_at_k(retrieved, relevant, k)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def reciprocal_rank(
    retrieved: List[str],
    relevant: Set[str],
) -> float:
    """
    Calculate Reciprocal Rank (RR).

    RR = 1 / (rank of first relevant item)

    Args:
        retrieved: List of retrieved item IDs (in ranked order)
        relevant: Set of relevant item IDs (ground truth)

    Returns:
        Reciprocal Rank score (0.0 to 1.0)
    """
    for rank, item_id in enumerate(retrieved, start=1):
        if item_id in relevant:
            return 1.0 / rank

    # No relevant items found
    return 0.0


def mean_reciprocal_rank(
    results: List[Dict[str, Any]],
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) across multiple queries.

    MRR = (sum of RR for all queries) / (# queries)

    Args:
        results: List of result dictionaries with keys:
            - 'retrieved': List of retrieved item IDs
            - 'relevant': Set of relevant item IDs

    Returns:
        MRR score (0.0 to 1.0)
    """
    if len(results) == 0:
        return 0.0

    # Calculate RR for each query
    rrs = []
    for result in results:
        retrieved = result.get("retrieved", [])
        relevant = result.get("relevant", set())

        rr = reciprocal_rank(retrieved, relevant)
        rrs.append(rr)

    # Mean of all RRs
    mrr = sum(rrs) / len(rrs)

    return mrr


def evaluate_retrieval(
    retrieved: List[str],
    relevant: Set[str],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """
    Comprehensive retrieval evaluation at multiple k values.

    Args:
        retrieved: List of retrieved item IDs (in ranked order)
        relevant: Set of relevant item IDs (ground truth)
        k_values: List of k values to evaluate

    Returns:
        Dictionary of metric name to score
    """
    metrics = {}

    # Calculate metrics at each k
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        metrics[f"f1@{k}"] = f1_score_at_k(retrieved, relevant, k)

    # Calculate rank-based metrics
    metrics["average_precision"] = average_precision(retrieved, relevant)
    metrics["reciprocal_rank"] = reciprocal_rank(retrieved, relevant)

    return metrics


def log_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Log metrics in a formatted way.

    Args:
        metrics: Dictionary of metric name to score
        prefix: Optional prefix for metric names
    """
    logger.info(f"{prefix}Retrieval Metrics:")
    for metric_name, score in sorted(metrics.items()):
        logger.info(f"  {metric_name}: {score:.4f}")
