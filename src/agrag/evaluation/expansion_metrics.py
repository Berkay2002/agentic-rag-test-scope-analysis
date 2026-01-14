"""Evaluation metrics for query expansion effectiveness."""

import logging
from typing import List, Dict, Any
from agrag.tools.schemas import SearchResult

try:
    import Levenshtein
except ImportError:
    # Fallback if python-levenshtein is not installed
    Levenshtein = None

logger = logging.getLogger(__name__)


def calculate_expansion_metrics(
    original_query: str,
    expanded_queries: List[str],
    original_results: List[SearchResult],
    expanded_results: List[SearchResult],
) -> Dict[str, Any]:
    """
    Calculate metrics to evaluate query expansion effectiveness.

    This function evaluates how effective query expansion is by comparing
    results from the original query versus results from expanded queries.
    It measures unique results gained, recall improvement, diversity of
    expansions, and quality of newly discovered results.

    Args:
        original_query: The original user query
        expanded_queries: List of expanded query variations
        original_results: Results from the original query
        expanded_results: Results from expanded queries (combined)

    Returns:
        Dictionary containing expansion metrics:
            - num_expansions: Number of query expansions generated
            - unique_results_gained: Count of unique results from expansions
            - recall_improvement_ratio: Ratio of expanded to original relevant results
            - expansion_diversity: Average normalized Levenshtein distance between expansions
            - new_result_quality: Average score of newly discovered results
            - original_result_count: Number of results from original query
            - expanded_result_count: Total number of results from expansions
            - overlap_ratio: Ratio of overlapping results to total unique results

    Example:
        >>> metrics = calculate_expansion_metrics(
        ...     "handover tests",
        ...     ["handover test cases", "handover testing scenarios"],
        ...     original_results,
        ...     expanded_results
        ... )
        >>> print(f"Gained {metrics['unique_results_gained']} new results")
    """
    # Extract result IDs
    original_ids = {r.id for r in original_results}
    expanded_ids = {r.id for r in expanded_results}

    # Calculate unique results gained
    new_ids = expanded_ids - original_ids
    unique_results_gained = len(new_ids)

    # Calculate recall improvement ratio
    if len(original_ids) == 0:
        recall_improvement_ratio = 1.0 if len(expanded_ids) > 0 else 0.0
    else:
        recall_improvement_ratio = len(expanded_ids) / len(original_ids)

    # Calculate expansion diversity
    expansion_diversity = _calculate_expansion_diversity(expanded_queries)

    # Calculate new result quality (average score of new results)
    new_results = [r for r in expanded_results if r.id in new_ids]
    if new_results:
        new_result_quality = sum(r.score for r in new_results) / len(new_results)
    else:
        new_result_quality = 0.0

    # Calculate overlap ratio
    overlap_ids = original_ids & expanded_ids
    total_unique_ids = original_ids | expanded_ids
    overlap_ratio = len(overlap_ids) / len(total_unique_ids) if total_unique_ids else 0.0

    metrics = {
        "num_expansions": len(expanded_queries),
        "unique_results_gained": unique_results_gained,
        "recall_improvement_ratio": recall_improvement_ratio,
        "expansion_diversity": expansion_diversity,
        "new_result_quality": new_result_quality,
        "original_result_count": len(original_results),
        "expanded_result_count": len(expanded_results),
        "overlap_ratio": overlap_ratio,
    }

    logger.info(
        f"Query expansion metrics calculated for '{original_query}': "
        f"+{unique_results_gained} unique results, "
        f"{recall_improvement_ratio:.2f}x recall improvement"
    )

    return metrics


def _calculate_expansion_diversity(expanded_queries: List[str]) -> float:
    """
    Calculate diversity of expanded queries using Levenshtein distance.

    This helper function measures how diverse the expanded queries are by
    computing the average normalized edit distance between all query pairs.
    Higher values indicate more diverse expansions.

    Args:
        expanded_queries: List of expanded query strings

    Returns:
        Average normalized Levenshtein distance (0.0 to 1.0)
    """
    if len(expanded_queries) < 2:
        return 0.0

    if Levenshtein is None:
        logger.warning("python-levenshtein not installed, using simple character difference")
        return _simple_diversity(expanded_queries)

    distances = []

    # Calculate pairwise Levenshtein distances
    for i in range(len(expanded_queries)):
        for j in range(i + 1, len(expanded_queries)):
            query1 = expanded_queries[i]
            query2 = expanded_queries[j]

            # Calculate edit distance
            edit_distance = Levenshtein.distance(query1, query2)

            # Normalize by maximum possible distance (length of longer query)
            max_length = max(len(query1), len(query2))
            if max_length > 0:
                normalized_distance = edit_distance / max_length
                distances.append(normalized_distance)

    # Return average normalized distance
    return sum(distances) / len(distances) if distances else 0.0


def _simple_diversity(expanded_queries: List[str]) -> float:
    """
    Fallback diversity calculation when Levenshtein is not available.

    Uses a simple character-based difference metric as a proxy for edit distance.
    """
    if len(expanded_queries) < 2:
        return 0.0

    distances = []

    for i in range(len(expanded_queries)):
        for j in range(i + 1, len(expanded_queries)):
            query1 = expanded_queries[i]
            query2 = expanded_queries[j]

            # Simple difference metric based on unique characters
            set1 = set(query1.lower())
            set2 = set(query2.lower())

            # Jaccard distance
            intersection = len(set1 & set2)
            union = len(set1 | set2)

            if union > 0:
                distance = 1.0 - (intersection / union)
                distances.append(distance)

    return sum(distances) / len(distances) if distances else 0.0
