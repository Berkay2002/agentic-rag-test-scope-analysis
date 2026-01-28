"""Result diversification algorithms for retrieval augmentation."""

import logging
from typing import List, Optional, Union

import numpy as np

# Make sklearn imports optional
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Define dummy functions for when sklearn is not available
    def cosine_similarity(*args, **kwargs):
        raise ImportError("scikit-learn is required for diversification. Install with: pip install scikit-learn")
    KMeans = None

from agrag.tools.shared.schemas import SearchResult

logger = logging.getLogger(__name__)


class MaximalMarginalRelevance:
    """Maximal Marginal Relevance (MMR) for result diversification.

    MMR balances relevance and diversity by iteratively selecting results
    that are both relevant to the query and dissimilar to already selected results.
    """

    def __init__(self, lambda_param: float = 0.5):
        """Initialize MMR diversifier.

        Args:
            lambda_param: Trade-off parameter between relevance and diversity.
                         0 = maximum diversity, 1 = maximum relevance.
        """
        self.lambda_param = lambda_param

    def diversify(
        self,
        results: List[SearchResult],
        k: int,
        query_embedding: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Apply MMR diversification to search results.

        Args:
            results: List of search results to diversify
            k: Number of results to return
            query_embedding: Optional query embedding for relevance calculation
            **kwargs: Additional parameters (ignored)

        Returns:
            Diversified list of search results
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Returning original results.")
            return results[:k]

        if not results or k <= 0:
            return []

        if k >= len(results):
            return results

        # Extract embeddings if available
        embeddings = []
        for result in results:
            if hasattr(result, 'embedding') and result.embedding is not None:
                embeddings.append(result.embedding)
            else:
                # Use zero vector if embedding not available
                embeddings.append(np.zeros(768))  # Assuming 768-dim embeddings

        embeddings = np.array(embeddings)

        # Calculate relevance scores (cosine similarity to query if available)
        if query_embedding is not None:
            relevance_scores = cosine_similarity(
                query_embedding.reshape(1, -1), embeddings
            )[0]
        else:
            # Fallback to original scores
            relevance_scores = np.array([result.score for result in results])

        # Normalize relevance scores to [0, 1]
        if relevance_scores.max() > relevance_scores.min():
            relevance_scores = (relevance_scores - relevance_scores.min()) / (
                relevance_scores.max() - relevance_scores.min()
            )

        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(results)))

        while len(selected_indices) < k and remaining_indices:
            # Calculate MMR scores for remaining items
            mmr_scores = []

            for idx in remaining_indices:
                relevance = relevance_scores[idx]

                if not selected_indices:
                    # First item: only consider relevance
                    mmr_score = relevance
                else:
                    # Calculate max similarity to selected items
                    selected_embeddings = embeddings[selected_indices]
                    similarities = cosine_similarity(
                        embeddings[idx].reshape(1, -1), selected_embeddings
                    )[0]
                    max_similarity = similarities.max()

                    # MMR score
                    mmr_score = (
                        self.lambda_param * relevance -
                        (1 - self.lambda_param) * max_similarity
                    )

                mmr_scores.append(mmr_score)

            # Select item with highest MMR score
            best_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Return selected results
        return [results[i] for i in selected_indices]


class ClusteringDiversifier:
    """Cluster-based diversification using KMeans clustering.

    Groups results into clusters and selects representatives from each cluster
    to ensure diversity across different topics or themes.
    """

    def __init__(self, n_clusters: Optional[int] = None):
        """Initialize clustering diversifier.

        Args:
            n_clusters: Number of clusters. If None, uses sqrt(n) heuristic.
        """
        self.n_clusters = n_clusters

    def diversify(
        self,
        results: List[SearchResult],
        k: int,
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Apply cluster-based diversification to search results.

        Args:
            results: List of search results to diversify
            k: Number of results to return
            embeddings: Optional pre-computed embeddings
            **kwargs: Additional parameters (ignored)

        Returns:
            Diversified list of search results
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Returning original results.")
            return results[:k]

        if not results or k <= 0:
            return []

        if k >= len(results):
            return results

        # Get embeddings
        if embeddings is None:
            embeddings = []
            for result in results:
                if hasattr(result, 'embedding') and result.embedding is not None:
                    embeddings.append(result.embedding)
                else:
                    embeddings.append(np.zeros(768))
            embeddings = np.array(embeddings)

        # Determine number of clusters
        n_clusters = self.n_clusters or int(np.sqrt(len(results)))
        n_clusters = min(n_clusters, k, len(results))

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Select representatives from clusters
        selected_results = []

        # Group results by cluster
        clusters = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(cluster_labels):
            clusters[label].append((idx, results[idx]))

        # Sort clusters by best score (to prioritize better clusters)
        cluster_scores = []
        for cluster_id, cluster_items in clusters.items():
            best_score = max(item[1].score for item in cluster_items)
            cluster_scores.append((cluster_id, best_score))

        cluster_scores.sort(key=lambda x: x[1], reverse=True)

        # Select items round-robin from clusters
        while len(selected_results) < k and cluster_scores:
            for cluster_id, _ in cluster_scores[:]:
                cluster_items = clusters[cluster_id]
                if cluster_items:
                    # Select best remaining item from this cluster
                    cluster_items.sort(key=lambda x: x[1].score, reverse=True)
                    idx, result = cluster_items.pop(0)
                    selected_results.append(result)

                    if len(selected_results) >= k:
                        break
                else:
                    # Remove empty cluster
                    cluster_scores.remove((cluster_id, _))

            if not any(clusters[cid] for cid, _ in cluster_scores):
                break

        return selected_results


class DedupingDiversifier:
    """Deduplication-based diversification to remove near-duplicates.

    Removes results that are too similar to each other, keeping only
    the best representative from groups of similar items.
    """

    def __init__(self, similarity_threshold: float = 0.9):
        """Initialize deduping diversifier.

        Args:
            similarity_threshold: Similarity threshold above which items
                                are considered duplicates.
        """
        self.similarity_threshold = similarity_threshold

    def diversify(
        self,
        results: List[SearchResult],
        k: int,
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Apply deduplication to search results.

        Args:
            results: List of search results to deduplicate
            k: Number of results to return
            embeddings: Optional pre-computed embeddings
            **kwargs: Additional parameters (ignored)

        Returns:
            Deduplicated list of search results
        """
        if not results or k <= 0:
            return []

        if k >= len(results):
            k = len(results)

        # Get embeddings
        if embeddings is None:
            embeddings = []
            for result in results:
                if hasattr(result, 'embedding') and result.embedding is not None:
                    embeddings.append(result.embedding)
                else:
                    embeddings.append(None)
        else:
            embeddings = list(embeddings)

        # Sort by score (descending)
        sorted_items = sorted(
            zip(results, embeddings),
            key=lambda x: x[0].score,
            reverse=True
        )

        selected_results = []

        for result, embedding in sorted_items:
            is_duplicate = False

            # Check similarity against already selected items
            for selected_result in selected_results:
                similarity = self._calculate_similarity(result, selected_result, embedding)
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    logger.debug(
                        f"Filtered duplicate: '{result.content[:50]}...' "
                        f"(similarity: {similarity:.3f})"
                    )
                    break

            if not is_duplicate:
                selected_results.append(result)

                if len(selected_results) >= k:
                    break

        return selected_results

    def _calculate_similarity(
        self,
        result1: SearchResult,
        result2: SearchResult,
        embedding1: Optional[np.ndarray] = None
    ) -> float:
        """Calculate similarity between two results.

        Args:
            result1: First result
            result2: Second result
            embedding1: Optional embedding for first result

        Returns:
            Similarity score between 0 and 1
        """
        # Try embedding-based similarity first
        if embedding1 is not None and hasattr(result2, 'embedding') and result2.embedding is not None:
            similarity = cosine_similarity(
                embedding1.reshape(1, -1),
                result2.embedding.reshape(1, -1)
            )[0][0]
            return float(similarity)

        # Fallback to text-based similarity
        text1 = result1.content.lower().strip()
        text2 = result2.content.lower().strip()

        # Simple Jaccard similarity on words
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 and not words2:
            return 1.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0