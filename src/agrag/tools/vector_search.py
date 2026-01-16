"""Vector search tool for semantic retrieval using PostgreSQL pgvector.

Uses the @tool decorator pattern from LangChain for cleaner tool definition.
"""

import time
from typing import Optional
import logging

from langchain_core.tools import tool

from agrag.tools.schemas import VectorSearchInput, VectorSearchOutput, SearchResult
from agrag.tools.base import (
    BaseToolWrapper,
    format_search_output,
    process_search_results,
)
from agrag.tools.diversification import (
    MaximalMarginalRelevance,
    ClusteringDiversifier,
    DedupingDiversifier,
)
from agrag.tools.search_utils import extract_signal_tokens
from agrag.storage import PostgresClient
from agrag.models import get_embedding_service
from agrag.kg.registry import get_registry

logger = logging.getLogger(__name__)


def _format_vector_output(output: VectorSearchOutput) -> str:
    """Format VectorSearchOutput for agent consumption.

    Args:
        output: VectorSearchOutput object

    Returns:
        Formatted string
    """
    return format_search_output(
        output=output,
        search_type="Vector Search",
        score_label="Similarity",
    )


def _apply_diversification(
    results: list[SearchResult],
    enable: bool,
    method: str,
    diversity_factor: float,
    dedup_threshold: float,
    k: int,
) -> list[SearchResult]:
    """Apply diversification to search results.

    Args:
        results: List of search results
        enable: Whether to enable diversification
        method: Diversification method (mmr|clustering|dedup)
        diversity_factor: Diversity vs relevance trade-off (0.0-1.0)
        dedup_threshold: Similarity threshold for deduplication (0.0-1.0)
        k: Number of results to return

    Returns:
        Diversified results if enabled, original results otherwise
    """
    if not enable or not results:
        return results

    try:
        if method == "mmr":
            diversifier = MaximalMarginalRelevance(lambda_param=diversity_factor)
            # Note: MMR requires query_embedding for optimal performance
            # For now, we'll pass None and it will fall back to scores
            diversified = diversifier.diversify(
                results=results,
                k=k,
                query_embedding=None,
            )
        elif method == "clustering":
            diversifier = ClusteringDiversifier()
            diversified = diversifier.diversify(
                results=results,
                k=k,
            )
        elif method == "dedup":
            diversifier = DedupingDiversifier(similarity_threshold=dedup_threshold)
            diversified = diversifier.diversify(
                results=results,
                k=k,
            )
        else:
            logger.warning(f"Unknown diversification method: {method}. Skipping diversification.")
            return results

        logger.info(f"Applied {method} diversification: {len(results)} -> {len(diversified)} results")
        return diversified

    except Exception as e:
        logger.error(f"Diversification failed with method {method}: {e}")
        return results


def _ensure_signal_match_in_top_k(
    results: list[SearchResult],
    query: str,
    client,
    k: int,
    node_type: Optional[str],
) -> list[SearchResult]:
    """Ensure top-k includes at least one signal-token match via keyword fallback."""
    signal_tokens = extract_signal_tokens(query)
    if not signal_tokens or not results:
        return results

    lower_tokens = [token.lower() for token in signal_tokens]
    stopwords = {
        "which",
        "test",
        "case",
        "verify",
        "verifies",
        "using",
        "find",
        "handling",
    }
    import re

    topical_tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z]{4,}", query)
        if token.lower() not in stopwords
    ]

    def has_all_tokens(result: SearchResult) -> bool:
        content_lower = (result.content or "").lower()
        has_signal = all(token in content_lower for token in lower_tokens)
        has_topic = any(token in content_lower for token in topical_tokens) if topical_tokens else True
        return has_signal and has_topic

    if any(has_all_tokens(result) for result in results[:k]):
        return results

    metadata_filter = {"entity_type": node_type} if node_type else None
    try:
        keyword_raw = client.keyword_search(
            query=query,
            k=k * 5,
            metadata_filter=metadata_filter,
        )
        keyword_results = process_search_results(
            raw_results=keyword_raw,
            score_field="rank",
            source_name="keyword_fallback",
        )
    except Exception as exc:
        logger.warning(f"Keyword fallback failed for vector search: {exc}")
        return results

    for candidate in keyword_results:
        if has_all_tokens(candidate):
            if results:
                candidate.score = max(candidate.score, results[0].score * 1.1)
            results = [result for result in results if result.id != candidate.id]
            results.insert(0, candidate)
            break

    return results


def _expand_queries(
    query: str,
    expansion_service,
    enable_expansion: bool,
    expansion_methods: list,
    max_expansions: int = None
) -> list[str]:
    """Expand a query using expansion service.

    Args:
        query: Original query
        expansion_service: Query expansion service
        enable_expansion: Whether expansion is enabled
        expansion_methods: List of expansion methods to use
        max_expansions: Maximum number of expansions to generate

    Returns:
        List of expanded queries (including original)
    """
    if not enable_expansion or not expansion_service:
        return [query]

    try:
        # Use expansion service to generate query variants
        expanded_queries = expansion_service.expand(
            query=query,
            methods=expansion_methods,
            max_expansions=max_expansions
        )

        logger.info(f"Query expansion generated {len(expanded_queries)} variants")
        for i, variant in enumerate(expanded_queries):
            logger.debug(f"Expansion {i}: {variant}")

        return expanded_queries
    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        return [query]


def _execute_multi_query_search(
    queries: list[str],
    client,
    embedding_service,
    k: int,
    node_type: Optional[str],
    similarity_threshold: Optional[float]
) -> list:
    """Execute search for multiple query variants and merge results.

    Args:
        queries: List of query variants
        client: PostgreSQL client
        embedding_service: Embedding service
        k: Number of results to return
        node_type: Type of nodes to search
        similarity_threshold: Minimum similarity threshold

    Returns:
        Merged search results from all queries
    """
    all_results = []
    seen_ids = set()

    # Build metadata filter if node_type provided
    metadata_filter = None
    if node_type:
        metadata_filter = {"entity_type": node_type}

    # Execute search for each query variant
    for query in queries:
        try:
            # Generate embedding for this query variant
            logger.info(f"Searching with query variant: {query}")
            query_embedding = embedding_service.embed_query(query)

            # Perform vector search
            results = client.vector_search(
                query_embedding=query_embedding,
                k=k * 2,  # Get more results for merging
                metadata_filter=metadata_filter,
            )

            # Add results to collection, avoiding duplicates
            for result in results:
                result_id = result.get('id', result.get('node_id', str(hash(str(result)))))
                if result_id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result_id)

        except Exception as e:
            logger.error(f"Search failed for query variant '{query}': {e}")
            continue

    # Sort by similarity score and limit to k results
    all_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)

    return all_results[:k]


def create_vector_search_tool(postgres_client: Optional[PostgresClient] = None):
    """Factory function to create a vector search tool with injected dependencies.

    Args:
        postgres_client: PostgreSQL client instance (creates new if not provided)

    Returns:
        Configured vector_search tool
    """
    client = postgres_client or PostgresClient()
    embedding_service = get_embedding_service()

    # Initialize expansion service if enabled
    expansion_service = None
    try:
        from agrag.config.settings import settings
        if hasattr(settings, 'enable_query_expansion') and settings.enable_query_expansion:
            from agrag.tools.query_expansion import QueryExpansionService
            expansion_service = QueryExpansionService(
                llm_service=embedding_service.llm if hasattr(embedding_service, 'llm') else None,
                vector_service=embedding_service
            )
    except ImportError:
        logger.info("Query expansion service not available or not enabled")

    @tool("vector_search", args_schema=VectorSearchInput)
    def vector_search(
        query: str,
        k: int = 10,
        node_type: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        enable_diversification: bool = False,
        diversification_method: str = "mmr",
        diversity_factor: float = 0.5,
        deduplication_threshold: float = 0.9,
        enable_query_expansion: bool = False,
        expansion_methods: list = None,
        max_expansions: int = None,
    ) -> str:
        """Use this tool for semantic queries requiring conceptual understanding.

        Best for:
        - Finding semantically similar content
        - Queries about concepts, meanings, or intent
        - When you need to understand the "meaning" behind the query

        Examples: "tests related to handover failures", "authentication requirements"

        Args:
            query: Natural language query for semantic search
            k: Number of results to return (1-50)
            node_type: Type of nodes to search (e.g., TestCase, Requirement, Function)
            similarity_threshold: Minimum similarity threshold (0.0-1.0)
            enable_diversification: Enable result diversification to reduce redundancy
            diversification_method: Diversification method: mmr|clustering|dedup
            diversity_factor: Diversity vs relevance trade-off (0.0=max diversity, 1.0=max relevance)
            deduplication_threshold: Similarity threshold for deduplication (0.9=90% similar considered duplicate)
            enable_query_expansion: Enable query expansion to generate search variants
            expansion_methods: List of expansion methods to use (e.g., ["synonyms", "llm", "pseudo_relevance"])
            max_expansions: Maximum number of query expansions to generate (overrides default)
        """
        # Call the core function with all parameters
        return _vector_search_core(
            query=query,
            k=k,
            node_type=node_type,
            similarity_threshold=similarity_threshold,
            enable_diversification=enable_diversification,
            diversification_method=diversification_method,
            diversity_factor=diversity_factor,
            deduplication_threshold=deduplication_threshold,
            expansion_service=expansion_service,
            enable_query_expansion=enable_query_expansion,
            expansion_methods=expansion_methods,
            max_expansions=max_expansions,
            client=client,
            embedding_service=embedding_service
        )

    return vector_search


def _vector_search_core(
    query: str,
    k: int,
    node_type: Optional[str],
    similarity_threshold: Optional[float],
    enable_diversification: bool,
    diversification_method: str,
    diversity_factor: float,
    deduplication_threshold: float,
    expansion_service,
    enable_query_expansion: bool,
    expansion_methods: list,
    max_expansions: int,
    client,
    embedding_service
) -> str:
    """Core vector search logic that supports query expansion.

    Args:
        query: Search query
        k: Number of results
        node_type: Node type filter
        similarity_threshold: Similarity threshold
        enable_diversification: Enable diversification
        diversification_method: Diversification method
        diversity_factor: Diversity factor
        deduplication_threshold: Deduplication threshold
        expansion_service: Query expansion service
        enable_query_expansion: Enable query expansion
        expansion_methods: Expansion methods to use
        max_expansions: Max expansions
        client: PostgreSQL client
        embedding_service: Embedding service

    Returns:
        Formatted search results
    """
    start_time = time.time()

    if embedding_service is None:
        return "Error: Embedding service not initialized"
    if client is None:
        return "Error: PostgreSQL client not initialized"

    try:
        registry = get_registry()
        if node_type:
            original_node_type = node_type
            node_type = registry.normalize_label(node_type)
            if not node_type:
                return f"Error: Unknown node type '{original_node_type}'"
        # Expand query if enabled
        expanded_queries = _expand_queries(
            query=query,
            expansion_service=expansion_service,
            enable_expansion=enable_query_expansion,
            expansion_methods=expansion_methods,
            max_expansions=max_expansions
        )

        # Execute search (single or multi-query)
        if len(expanded_queries) == 1:
            # Original behavior - single query search
            logger.info(f"Generating embedding for query: {query}")
            query_embedding = embedding_service.embed_query(query)

            # Build metadata filter if node_type provided
            metadata_filter = None
            if node_type:
                metadata_filter = {"entity_type": node_type}

            # Perform vector search in PostgreSQL using pgvector
            logger.info(f"Performing pgvector search (entity_type={node_type})")
            raw_results = client.vector_search(
                query_embedding=query_embedding,
                k=k,
                metadata_filter=metadata_filter,
            )
        else:
            # Multi-query search with expanded queries
            logger.info(f"Executing multi-query search with {len(expanded_queries)} variants")
            raw_results = _execute_multi_query_search(
                queries=expanded_queries,
                client=client,
                embedding_service=embedding_service,
                k=k,
                node_type=node_type,
                similarity_threshold=similarity_threshold
            )

        # Format results using shared processing logic
        # Apply similarity threshold filter if provided
        score_filter_fn = None
        if similarity_threshold is not None:
            def threshold_filter(score: float) -> bool:
                return score >= similarity_threshold
            score_filter_fn = threshold_filter

        search_results = process_search_results(
            raw_results=raw_results,
            score_field="similarity",
            source_name="pgvector",
            score_filter_fn=score_filter_fn,
        )

        # Ensure signal-token matches are present in top-k
        boosted_results = _ensure_signal_match_in_top_k(
            results=search_results,
            query=query,
            client=client,
            k=k,
            node_type=node_type,
        )

        # Apply diversification if enabled
        diversified_results = _apply_diversification(
            results=boosted_results,
            enable=enable_diversification,
            method=diversification_method,
            diversity_factor=diversity_factor,
            dedup_threshold=deduplication_threshold,
            k=min(k, len(boosted_results))
        )

        retrieval_time_ms = (time.time() - start_time) * 1000

        output = VectorSearchOutput(
            results=diversified_results,
            query=query,
            total_results=len(diversified_results),
            retrieval_time_ms=retrieval_time_ms,
        )

        return _format_vector_output(output)

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return f"Error performing vector search: {str(e)}"


# For backwards compatibility, provide a class-based wrapper
class VectorSearchTool(BaseToolWrapper):
    """Wrapper class for backwards compatibility.

    Use create_vector_search_tool() factory function for new code.
    """

    def __init__(self, postgres_client: Optional[PostgresClient] = None):
        """Initialize vector search tool.

        Args:
            postgres_client: PostgreSQL client instance (creates new if not provided)
        """
        tool = create_vector_search_tool(postgres_client)
        super().__init__(tool)
