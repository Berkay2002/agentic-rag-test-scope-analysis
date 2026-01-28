"""Evaluation metrics for result diversification."""

from typing import List
import numpy as np
from agrag.tools.shared.schemas import SearchResult


def calculate_diversity_metrics(results: List[SearchResult]) -> dict:
    """
    Calculate diversity metrics for a result set.

    This function evaluates how diverse a set of search results is by computing:
    - Average pairwise similarity between results (lower is more diverse)
    - Number of unique entity types
    - Diversity score (inverse of average similarity)
    - Maximum and minimum pairwise similarities

    Args:
        results: List of search results

    Returns:
        Dictionary containing diversity metrics:
            - avg_pairwise_similarity: Mean cosine similarity between all result pairs
            - unique_entities: Count of unique entity types in results
            - diversity_score: 1 - avg_pairwise_similarity (higher is more diverse)
            - max_similarity: Maximum similarity between any two results
            - min_similarity: Minimum similarity between any two results

    Example:
        >>> results = [SearchResult(...), SearchResult(...), ...]
        >>> metrics = calculate_diversity_metrics(results)
        >>> print(f"Diversity score: {metrics['diversity_score']:.3f}")
    """
    if len(results) < 2:
        return {
            "avg_pairwise_similarity": 0.0,
            "unique_entities": len(results),
            "diversity_score": 1.0,
            "max_similarity": 0.0,
            "min_similarity": 0.0,
        }

    # Calculate pairwise similarities
    similarities = []
    features = _extract_features(results)

    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            sim = _cosine_similarity(features[i], features[j])
            similarities.append(sim)

    avg_similarity = np.mean(similarities) if similarities else 0.0

    # Count unique entities (by type)
    entity_types = [r.metadata.get("entity_type") for r in results]
    unique_entities = len(set(entity_types))

    # Diversity score (inverse of average similarity)
    diversity_score = 1.0 - avg_similarity

    return {
        "avg_pairwise_similarity": float(avg_similarity),
        "unique_entities": unique_entities,
        "diversity_score": float(diversity_score),
        "max_similarity": float(max(similarities)) if similarities else 0.0,
        "min_similarity": float(min(similarities)) if similarities else 0.0,
    }


def _extract_features(results: List[SearchResult]) -> np.ndarray:
    """
    Extract feature vectors for similarity calculation.

    Creates a feature vector for each search result containing:
    - Normalized score
    - Content length (normalized to 1000 words)
    - Binary flag for code content (1.0 if contains function/class definitions)

    Args:
        results: List of search results

    Returns:
        NumPy array of shape (n_results, 3) containing feature vectors
    """
    features = []
    for result in results:
        # Feature: [score, content_length, has_code]
        feature = [
            result.score,
            len(result.content.split()) / 1000.0,
            1.0 if "def " in result.content or "class " in result.content else 0.0
        ]
        features.append(feature)
    return np.array(features)


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Cosine similarity measures the cosine of the angle between two vectors,
    ranging from -1 (opposite) to 1 (identical), with 0 indicating orthogonality.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Cosine similarity score between -1 and 1

    Note:
        Returns 0.0 if either vector has zero magnitude to avoid division by zero
    """
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)