"""Hybrid search tool combining vector and keyword search with RRF fusion.

Uses PostgreSQL's pgvector for semantic similarity and pg_search BM25 for keyword
ranking, providing cloud-persistent hybrid retrieval with Reciprocal Rank Fusion.

Uses the @tool decorator pattern from LangChain for cleaner tool definition.
"""

import time
from typing import Optional, List, Dict, Any
import logging
import numpy as np

from langchain_core.tools import tool

from agrag.tools.shared.schemas import HybridSearchInput, HybridSearchOutput, SearchResult
from agrag.tools.shared.base import (
    BaseToolWrapper,
    format_search_output,
    process_search_results,
)
from agrag.tools.enhancements.diversification import (
    MaximalMarginalRelevance,
    ClusteringDiversifier,
    DedupingDiversifier,
)
from agrag.tools.shared.search_utils import extract_signal_tokens
from agrag.storage import PostgresClient
from agrag.storage.retry_decorators import with_fallback
from agrag.models import get_embedding_service

logger = logging.getLogger(__name__)


def _merge_hybrid_results(
    all_results: List[Dict[str, Any]],
    k: int,
    rrf_k: int
) -> List[Dict[str, Any]]:
    """Merge hybrid search results from multiple query variants.

    Args:
        all_results: Combined results from all query variants
        k: Number of final results to return
        rrf_k: RRF constant for score normalization

    Returns:
        Merged and deduplicated results with boosted original query scores
    """
    if not all_results:
        return []

    # Group results by ID to handle duplicates
    result_groups = {}
    for result in all_results:
        metadata = result.get("metadata") or {}
        result_id = metadata.get("entity_id") or result.get("id") or result.get("chunk_id")
        if result_id is None:
            continue
        if result_id not in result_groups:
            result_groups[result_id] = []
        result_groups[result_id].append(result)

    # Merge duplicate results
    merged_results = []
    for result_id, group in result_groups.items():
        # Take the result with the highest RRF score
        best_result = max(group, key=lambda r: r.get("rrf_score", 0))

        # Boost score if original query found this result
        original_hits = [r for r in group if r.get("_is_original", False)]
        if original_hits:
            # Boost by 10% for being found by original query
            best_result["rrf_score"] = best_result.get("rrf_score", 0) * 1.1
            best_result["_boosted"] = True

        # Count how many variants found this result
        best_result["_variant_count"] = len(group)
        merged_results.append(best_result)

    # Sort by RRF score (descending)
    merged_results.sort(key=lambda r: r.get("rrf_score", 0), reverse=True)

    logger.info(f"Merged {len(all_results)} results into {len(merged_results)} unique results")

    return merged_results[:k * 5]  # Return more than needed for further processing


def _expand_queries(
    query: str,
    expansion_service: Optional[Any],
    enable_expansion: bool,
    expansion_methods: Optional[List[str]] = None,
    max_expansions: Optional[int] = None
) -> List[str]:
    """Expand query using expansion service if enabled.

    Args:
        query: Original search query
        expansion_service: Query expansion service instance
        enable_expansion: Whether to enable query expansion
        expansion_methods: List of expansion methods to use
        max_expansions: Maximum number of expansions to return

    Returns:
        List of queries (original + expansions if enabled)
    """
    if not enable_expansion or not expansion_service:
        return [query]

    try:
        logger.info(f"Expanding query: {query}")
        expanded_queries = expansion_service.expand(
            query=query,
            methods=expansion_methods,
            max_expansions=max_expansions
        )
        logger.info(f"Generated {len(expanded_queries)} query variants")
        return expanded_queries
    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        return [query]


def _format_hybrid_output(output: HybridSearchOutput) -> str:
    """Format HybridSearchOutput for agent consumption.

    Args:
        output: HybridSearchOutput object

    Returns:
        Formatted string
    """
    return format_search_output(
        output=output,
        search_type="Hybrid Search",
        score_label="RRF Score",
        additional_info=output.fusion_method,
        footer_note="Note: RRF combines pgvector similarity and pg_search BM25 ranking for optimal precision.",
    )


def _apply_diversification(
    results: List[SearchResult],
    enable: bool,
    method: str,
    diversity_factor: float,
    dedup_threshold: float,
    k: int,
    query_embedding: Optional[List[float]] = None,
) -> List[SearchResult]:
    """Apply diversification to search results if enabled.

    Args:
        results: List of search results to diversify
        enable: Whether to enable diversification
        method: Diversification method ("mmr", "clustering", "dedup")
        diversity_factor: Diversity vs relevance trade-off (0.0=max diversity, 1.0=max relevance)
        dedup_threshold: Similarity threshold for deduplication
        k: Number of results to return
        query_embedding: Optional query embedding for MMR relevance calculation

    Returns:
        Diversified results if enabled, otherwise original results
    """
    if not enable or not results:
        return results

    try:
        logger.info(f"Applying {method} diversification with diversity_factor={diversity_factor}")

        if method == "mmr":
            diversifier = MaximalMarginalRelevance(lambda_param=diversity_factor)
            return diversifier.diversify(
                results=results,
                k=k,
                query_embedding=np.array(query_embedding) if query_embedding else None
            )
        elif method == "clustering":
            diversifier = ClusteringDiversifier()
            return diversifier.diversify(results=results, k=k)
        elif method == "dedup":
            diversifier = DedupingDiversifier(similarity_threshold=dedup_threshold)
            return diversifier.diversify(results=results, k=k)
        else:
            logger.warning(f"Unknown diversification method: {method}. Skipping diversification.")
            return results
    except Exception as e:
        logger.error(f"Diversification failed with method '{method}': {e}")
        return results


def _boost_signal_token_matches(
    results: List[SearchResult],
    query: str,
    any_match_multiplier: float = 1.5,
    all_match_multiplier: float = 2.0,
) -> List[SearchResult]:
    """Boost results that contain high-signal query tokens in their content."""
    signal_tokens = extract_signal_tokens(query)
    if not signal_tokens:
        return results

    lower_tokens = [token.lower() for token in signal_tokens]
    for result in results:
        content_lower = (result.content or "").lower()
        matches = [token for token in lower_tokens if token in content_lower]
        if not matches:
            continue
        if len(matches) == len(lower_tokens):
            result.score *= all_match_multiplier
        else:
            result.score *= any_match_multiplier
    return results


def _ensure_signal_match_in_top_k(
    results: List[SearchResult],
    query: str,
    client: PostgresClient,
    k: int,
    metadata_filter: Optional[Dict[str, Any]],
) -> List[SearchResult]:
    """Ensure at least one top-k result matches all signal tokens using keyword fallback."""
    signal_tokens = extract_signal_tokens(query)
    if not signal_tokens or not results:
        return results

    lower_tokens = [token.lower() for token in signal_tokens]

    def has_all_tokens(result: SearchResult) -> bool:
        content_lower = (result.content or "").lower()
        return all(token in content_lower for token in lower_tokens)

    if any(has_all_tokens(result) for result in results[:k]):
        return results

    try:
        keyword_raw = client.keyword_search(
            query=query,
            k=k * 5,
            metadata_filter=metadata_filter if metadata_filter else None,
        )
        keyword_results = process_search_results(
            raw_results=keyword_raw,
            score_field="rank",
            source_name="keyword_fallback",
        )
    except Exception as exc:
        logger.warning(f"Keyword fallback failed for signal token boost: {exc}")
        return results

    for candidate in keyword_results:
        if has_all_tokens(candidate):
            # Promote candidate to top with a slight boost
            if results:
                candidate.score = max(candidate.score, results[0].score * 1.1)
            results = [result for result in results if result.id != candidate.id]
            results.insert(0, candidate)
            break

    return results


def _keyword_only_search(
    client: PostgresClient,
    query: str,
    query_embedding: List[float],  # Not used but kept for signature compatibility
    k: int,
    metadata_filter: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Fallback: keyword search only when vector search fails."""
    logging.warning("Hybrid search failed, falling back to keyword search only")
    return client.keyword_search(query=query, k=k, metadata_filter=metadata_filter)


def create_hybrid_search_tool(postgres_client: Optional[PostgresClient] = None):
    """Factory function to create a hybrid search tool with injected dependencies.

    Args:
        postgres_client: PostgreSQL client instance for both vector and keyword search

    Returns:
        Configured hybrid_search tool
    """
    client = postgres_client or PostgresClient()
    embedding_service = get_embedding_service()

    # Initialize expansion service if enabled
    expansion_service = None
    try:
        from agrag.config.settings import settings
        if hasattr(settings, 'enable_query_expansion') and settings.enable_query_expansion:
            from agrag.tools.enhancements.query_expansion import QueryExpansionService
            expansion_service = QueryExpansionService(
                llm_service=embedding_service.llm if hasattr(embedding_service, 'llm') else None,
                vector_service=embedding_service
            )
    except ImportError:
        logger.info("Query expansion service not available or not enabled")

    @tool("hybrid_search", args_schema=HybridSearchInput)
    def hybrid_search(
        query: str,
        k: int = 10,
        rrf_k: int = 60,
        entity_type: Optional[str] = None,
        enable_diversification: bool = False,
        diversification_method: str = "mmr",
        diversity_factor: float = 0.5,
        deduplication_threshold: float = 0.9,
        enable_query_expansion: bool = False,
        expansion_methods: List[str] = None,
        max_expansions: int = None,
    ) -> str:
        """Use this tool when you need both semantic understanding AND exact keyword matching.

        Combines vector similarity search (pgvector) with BM25 keyword search (pg_search) using RRF fusion.

        Best for:
        - Complex queries requiring both conceptual understanding and specific terms
        - Balancing semantic similarity with lexical precision
        - Queries that mix concepts with technical identifiers

        Examples: "tests for LTE signaling with timeout errors", "handover functions in network module"

        Args:
            query: Search query combining semantic and lexical requirements
            k: Number of results to return (1-50)
            rrf_k: Reciprocal Rank Fusion constant (default 60)
            entity_type: Filter by entity type
            enable_diversification: Enable result diversification to reduce redundancy
            diversification_method: Diversification method: mmr|clustering|dedup
            diversity_factor: Diversity vs relevance trade-off (0.0=max diversity, 1.0=max relevance)
            deduplication_threshold: Similarity threshold for deduplication (0.9=90% similar considered duplicate)
            enable_query_expansion: Enable query expansion to generate search variants
            expansion_methods: List of expansion methods to use (e.g., ["synonyms", "llm", "pseudo_relevance"])
            max_expansions: Maximum number of query expansions to generate (overrides default)
        """
        start_time = time.time()

        if embedding_service is None:
            return "Error: Embedding service not initialized"
        if client is None:
            return "Error: PostgreSQL client not initialized"

        try:
            # Expand queries if enabled
            expanded_queries = _expand_queries(
                query=query,
                expansion_service=expansion_service,
                enable_expansion=enable_query_expansion,
                expansion_methods=expansion_methods,
                max_expansions=max_expansions
            )

            # Build metadata filter if entity type provided
            metadata_filter = {}
            if entity_type:
                metadata_filter["entity_type"] = entity_type

            all_results = []
            original_query_embedding = None

            # Perform hybrid search for each expanded query
            for i, expanded_query in enumerate(expanded_queries):
                logger.info(f"Processing hybrid search variant {i+1}/{len(expanded_queries)}: {expanded_query}")

                # Generate embedding for this query variant
                query_embedding = embedding_service.embed_query(expanded_query)

                # Store original query embedding for diversification
                if expanded_query == query:
                    original_query_embedding = query_embedding

                # Perform hybrid search through core function with fallback
                try:
                    results = _hybrid_search_core(
                        client=client,
                        query=expanded_query,
                        query_embedding=query_embedding,
                        k=k * 5,  # Get more results to allow for merging/boosting
                        rrf_k=rrf_k,
                        metadata_filter=metadata_filter if metadata_filter else None,
                    )

                    # Add query variant info to results for tracking
                    for result in results:
                        result["_query_variant"] = expanded_query
                        result["_is_original"] = expanded_query == query

                    all_results.extend(results)

                except Exception as e:
                    logger.error(f"Hybrid search failed for query variant '{expanded_query}': {e}")
                    # Continue with other variants

            if not all_results:
                return "Error: No results found from any query variant"

            # Merge results from all variants
            merged_results = _merge_hybrid_results(
                all_results=all_results,
                k=k,
                rrf_k=rrf_k
            )

            # Format results using shared processing logic
            search_results = process_search_results(
                raw_results=merged_results,
                score_field="rrf_score",
                source_name="hybrid",
            )

            # Boost results that match high-signal query tokens (e.g., X2, GTP)
            boosted_results = _boost_signal_token_matches(search_results, query=query)
            boosted_results = _ensure_signal_match_in_top_k(
                results=boosted_results,
                query=query,
                client=client,
                k=k,
                metadata_filter=metadata_filter if metadata_filter else None,
            )
            boosted_results.sort(key=lambda result: result.score, reverse=True)

            # Use original query embedding for diversification if available
            diversification_embedding = original_query_embedding or embedding_service.embed_query(query)

            # Apply diversification if enabled
            diversified_results = _apply_diversification(
                results=boosted_results,
                enable=enable_diversification,
                method=diversification_method,
                diversity_factor=diversity_factor,
                dedup_threshold=deduplication_threshold,
                k=min(k, len(boosted_results)),
                query_embedding=diversification_embedding
            )

            retrieval_time_ms = (time.time() - start_time) * 1000

            output = HybridSearchOutput(
                results=diversified_results[:k],
                query=query,
                total_results=len(diversified_results[:k]),
                retrieval_time_ms=retrieval_time_ms,
                fusion_method="RRF (pgvector + pg_search BM25) with Query Expansion",
            )

            return _format_hybrid_output(output)

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return f"Error performing hybrid search: {str(e)}"

    return hybrid_search


@with_fallback(fallback_func=_keyword_only_search)
def _hybrid_search_core(
    client: PostgresClient,
    query: str,
    query_embedding: List[float],
    k: int,
    rrf_k: int,
    metadata_filter: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Core hybrid search logic wrapped with fallback decorator."""
    # Perform PostgreSQL hybrid search (pgvector + pg_search BM25 with RRF fusion)
    logger.info(f"Performing hybrid search (pgvector + pg_search BM25, RRF k={rrf_k})...")
    return client.hybrid_search(
        query=query,
        query_embedding=query_embedding,
        k=k,
        rrf_k=rrf_k,
        metadata_filter=metadata_filter if metadata_filter else None,
    )


# For backwards compatibility, provide a class-based wrapper
class HybridSearchTool(BaseToolWrapper):
    """Wrapper class for backwards compatibility.

    Use create_hybrid_search_tool() factory function for new code.
    """

    def __init__(self, postgres_client: Optional[PostgresClient] = None):
        """Initialize hybrid search tool.

        Args:
            postgres_client: PostgreSQL client instance for both vector and keyword search
        """
        tool = create_hybrid_search_tool(postgres_client)
        super().__init__(tool)
