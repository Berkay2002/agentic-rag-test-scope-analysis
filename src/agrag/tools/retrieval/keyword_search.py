"""Keyword search tool for lexical retrieval using PostgreSQL pg_search BM25.

Uses ParadeDB's pg_search extension with true BM25 ranking algorithm for
cloud-persistent keyword search stored alongside vector embeddings.

Uses the @tool decorator pattern from LangChain for cleaner tool definition.
"""

import time
from typing import Optional, List
import logging

from langchain_core.tools import tool

from agrag.tools.shared.schemas import KeywordSearchInput, KeywordSearchOutput, SearchResult
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
from agrag.storage import PostgresClient
from agrag.models import get_embedding_service

logger = logging.getLogger(__name__)


def _format_keyword_output(output: KeywordSearchOutput, num_expansions: int = 0) -> str:
    """Format KeywordSearchOutput for agent consumption.

    Args:
        output: KeywordSearchOutput object
        num_expansions: Number of query expansions used (0 if none)

    Returns:
        Formatted string
    """
    footer_note = "Note: Uses pg_search extension with true BM25 ranking (ParadeDB)."
    if num_expansions > 0:
        footer_note += f" Query expanded to {num_expansions} variants."

    return format_search_output(
        output=output,
        search_type="Keyword Search",
        score_label="FTS Rank",
        footer_note=footer_note,
    )


def _apply_diversification(
    results: List[SearchResult],
    enable: bool,
    method: str,
    diversity_factor: float,
    dedup_threshold: float,
    k: int,
) -> List[SearchResult]:
    """Apply diversification to search results if enabled.

    Args:
        results: List of search results
        enable: Whether to enable diversification
        method: Diversification method ("mmr", "clustering", "dedup")
        diversity_factor: Diversity vs relevance trade-off (0.0=max diversity, 1.0=max relevance)
        dedup_threshold: Similarity threshold for deduplication
        k: Number of results to return

    Returns:
        Diversified results if enabled, original results otherwise
    """
    if not enable or not results:
        return results

    try:
        if method == "mmr":
            diversifier = MaximalMarginalRelevance(lambda_param=diversity_factor)
            return diversifier.diversify(results=results, k=k)
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


def _expand_queries(
    query: str,
    expansion_service,
    expansion_methods: list = None,
    max_expansions: int = None
) -> List[str]:
    """Expand query using the expansion service.

    Args:
        query: Original query
        expansion_service: QueryExpansionService instance
        expansion_methods: List of expansion methods to use
        max_expansions: Maximum number of expansions

    Returns:
        List of expanded queries (including original)
    """
    if not expansion_service:
        return [query]

    try:
        return expansion_service.expand(
            query=query,
            methods=expansion_methods,
            max_expansions=max_expansions
        )
    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        return [query]


def _execute_multi_query_search(
    queries: List[str],
    client,
    k: int,
    metadata_filter: dict = None
) -> List[SearchResult]:
    """Execute keyword search for multiple query variants and merge results.

    Args:
        queries: List of query variants
        client: PostgreSQL client
        k: Number of results to return
        metadata_filter: Optional metadata filter

    Returns:
        Merged and deduplicated search results
    """
    all_results = []
    original_query = queries[0] if queries else ""

    # Execute search for each query variant
    for i, query in enumerate(queries):
        try:
            logger.info(f"Performing pg_search BM25 keyword search (variant {i+1}/{len(queries)}): {query}")
            results = client.keyword_search(
                query=query,
                k=k * 2,  # Get more results for merging
                metadata_filter=metadata_filter if metadata_filter else None,
            )

            # Format results
            search_results = process_search_results(
                raw_results=results,
                score_field="rank",
                source_name="postgres_fts",
            )

            # Boost original query results slightly
            if query == original_query:
                for result in search_results:
                    result.score *= 1.1  # 10% boost for original query

            all_results.extend(search_results)

        except Exception as e:
            logger.error(f"Failed to execute search for query variant '{query}': {e}")
            continue

    if not all_results:
        return []

    # Deduplicate by result id, keeping highest score
    seen_ids = {}
    deduped_results = []

    for result in all_results:
        result_id = result.id
        if result_id not in seen_ids or result.score > seen_ids[result_id].score:
            seen_ids[result_id] = result

    deduped_results = list(seen_ids.values())

    # Sort by score (descending)
    deduped_results.sort(key=lambda x: x.score, reverse=True)

    # Return top k results
    return deduped_results[:k]


def create_keyword_search_tool(postgres_client: Optional[PostgresClient] = None):
    """Factory function to create a keyword search tool with injected dependencies.

    Args:
        postgres_client: PostgreSQL client instance (creates new if not provided)

    Returns:
        Configured keyword_search tool
    """
    client = postgres_client or PostgresClient()

    # Initialize expansion service if enabled
    expansion_service = None
    try:
        from agrag.config.settings import settings
        if hasattr(settings, 'enable_query_expansion') and settings.enable_query_expansion:
            from agrag.tools.enhancements.query_expansion import QueryExpansionService
            embedding_service = get_embedding_service()
            expansion_service = QueryExpansionService(
                llm_service=embedding_service.llm if hasattr(embedding_service, 'llm') else None,
                vector_service=embedding_service
            )
    except ImportError:
        logger.info("Query expansion service not available or not enabled")

    @tool("keyword_search", args_schema=KeywordSearchInput)
    def keyword_search(
        query: str,
        k: int = 10,
        entity_type: Optional[str] = None,
        enable_diversification: bool = False,
        diversification_method: str = "mmr",
        diversity_factor: float = 0.5,
        deduplication_threshold: float = 0.9,
        enable_query_expansion: bool = False,
        expansion_methods: list = None,
        max_expansions: int = None,
    ) -> str:
        """Use this tool for exact matches and lexical queries using BM25 ranking.

        Best for:
        - Specific identifiers (test IDs, function names, error codes)
        - Exact keyword matching with BM25 probabilistic ranking
        - When you know the specific terms that should appear

        Examples: "TestLoginTimeout", "error code E503", "initiate_handover"

        Args:
            query: Keyword query for exact/lexical matching
            k: Number of results to return (1-50)
            entity_type: Filter by entity type (e.g., 'TestCase', 'Function')
            enable_diversification: Enable result diversification to reduce redundancy
            diversification_method: Diversification method: mmr|clustering|dedup
            diversity_factor: Diversity vs relevance trade-off (0.0=max diversity, 1.0=max relevance)
            deduplication_threshold: Similarity threshold for deduplication (0.9=90% similar considered duplicate)
            enable_query_expansion: Enable query expansion to generate search variants
            expansion_methods: List of expansion methods to use (e.g., ["synonyms", "llm", "pseudo_relevance"])
            max_expansions: Maximum number of query expansions to generate (overrides default)
        """
        start_time = time.time()

        if client is None:
            return "Error: PostgreSQL client not initialized"

        try:
            # Build metadata filter if entity type provided
            metadata_filter = {}
            if entity_type:
                metadata_filter["entity_type"] = entity_type

            # Expand queries if enabled
            queries = [query]
            if enable_query_expansion and expansion_service:
                logger.info(f"Expanding query: {query}")
                queries = _expand_queries(
                    query=query,
                    expansion_service=expansion_service,
                    expansion_methods=expansion_methods,
                    max_expansions=max_expansions
                )
                logger.info(f"Generated {len(queries)} query variants")

            # Execute search
            if len(queries) == 1:
                # Single query - use original behavior
                logger.info(f"Performing pg_search BM25 keyword search: {query}")
                results = client.keyword_search(
                    query=query,
                    k=k,
                    metadata_filter=metadata_filter if metadata_filter else None,
                )

                # Format results using shared processing logic
                search_results = process_search_results(
                    raw_results=results,
                    score_field="rank",
                    source_name="postgres_fts",
                )
            else:
                # Multiple queries - use expanded search
                search_results = _execute_multi_query_search(
                    queries=queries,
                    client=client,
                    k=k,
                    metadata_filter=metadata_filter if metadata_filter else None,
                )

            # Apply diversification if enabled
            diversified_results = _apply_diversification(
                results=search_results,
                enable=enable_diversification,
                method=diversification_method,
                diversity_factor=diversity_factor,
                dedup_threshold=deduplication_threshold,
                k=min(k, len(search_results))
            )

            retrieval_time_ms = (time.time() - start_time) * 1000

            output = KeywordSearchOutput(
                results=diversified_results,
                query=query,
                total_results=len(diversified_results),
                retrieval_time_ms=retrieval_time_ms,
            )

            return _format_keyword_output(output, num_expansions=len(queries) if enable_query_expansion else 0)

        except Exception as e:
            logger.error(f"pg_search BM25 keyword search failed: {e}")
            return f"Error performing keyword search: {str(e)}"

    return keyword_search


# For backwards compatibility, provide a class-based wrapper
class KeywordSearchTool(BaseToolWrapper):
    """Wrapper class for backwards compatibility.

    Use create_keyword_search_tool() factory function for new code.
    """

    def __init__(self, postgres_client: Optional[PostgresClient] = None):
        """Initialize keyword search tool.

        Args:
            postgres_client: PostgreSQL client instance (creates new if not provided)
        """
        tool = create_keyword_search_tool(postgres_client)
        super().__init__(tool)
