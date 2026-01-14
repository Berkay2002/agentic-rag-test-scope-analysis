"""Keyword search tool for lexical retrieval using PostgreSQL pg_search BM25.

Uses ParadeDB's pg_search extension with true BM25 ranking algorithm for
cloud-persistent keyword search stored alongside vector embeddings.

Uses the @tool decorator pattern from LangChain for cleaner tool definition.
"""

import time
from typing import Optional
import logging

from langchain_core.tools import tool

from agrag.tools.schemas import KeywordSearchInput, KeywordSearchOutput, SearchResult
from agrag.tools.base import (
    BaseToolWrapper,
    format_search_results_header,
    format_search_result_item,
    format_search_results_footer,
    build_metadata_with_chunk_id,
    extract_score_or_default,
)
from agrag.storage import PostgresClient

logger = logging.getLogger(__name__)


def _format_keyword_output(output: KeywordSearchOutput) -> str:
    """Format KeywordSearchOutput for agent consumption.

    Args:
        output: KeywordSearchOutput object

    Returns:
        Formatted string
    """
    if not output.results:
        return f"No results found for query: '{output.query}'"

    # Build header
    header = format_search_results_header(
        query=output.query,
        total_results=output.total_results,
        retrieval_time_ms=output.retrieval_time_ms,
        search_type="Keyword Search",
    )

    # Format each result
    result_items = []
    for i, result in enumerate(output.results, 1):
        entity_type = result.metadata.get("entity_type", "Unknown") if result.metadata else None
        item = format_search_result_item(
            index=i,
            result_id=result.id,
            score=result.score,
            score_label="FTS Rank",
            content=result.content,
            entity_type=entity_type,
        )
        result_items.append(item)

    # Add footer note
    footer = format_search_results_footer(
        "Note: Uses pg_search extension with true BM25 ranking (ParadeDB)."
    )

    return header + "\n".join(result_items) + footer


def create_keyword_search_tool(postgres_client: Optional[PostgresClient] = None):
    """Factory function to create a keyword search tool with injected dependencies.

    Args:
        postgres_client: PostgreSQL client instance (creates new if not provided)

    Returns:
        Configured keyword_search tool
    """
    client = postgres_client or PostgresClient()

    @tool("keyword_search", args_schema=KeywordSearchInput)
    def keyword_search(
        query: str,
        k: int = 10,
        entity_type: Optional[str] = None,
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
        """
        start_time = time.time()

        if client is None:
            return "Error: PostgreSQL client not initialized"

        try:
            # Build metadata filter if entity type provided
            metadata_filter = {}
            if entity_type:
                metadata_filter["entity_type"] = entity_type

            # Perform pg_search BM25 keyword search
            logger.info(f"Performing pg_search BM25 keyword search: {query}")
            results = client.keyword_search(
                query=query,
                k=k,
                metadata_filter=metadata_filter if metadata_filter else None,
            )

            # Format results
            search_results = []
            for result in results:
                # Handle None score from paradedb.score() - use 0.0 as fallback
                score = extract_score_or_default(result.get("rank"))

                metadata = build_metadata_with_chunk_id(
                    result.get("metadata", {}), result.get("chunk_id")
                )

                search_result = SearchResult(
                    id=metadata.get("entity_id") or result.get("chunk_id", "unknown"),
                    content=result.get("content", ""),
                    score=score,
                    metadata=metadata,
                    source="postgres_fts",
                )
                search_results.append(search_result)

            retrieval_time_ms = (time.time() - start_time) * 1000

            output = KeywordSearchOutput(
                results=search_results,
                query=query,
                total_results=len(search_results),
                retrieval_time_ms=retrieval_time_ms,
            )

            return _format_keyword_output(output)

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
