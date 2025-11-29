"""Keyword search tool for lexical retrieval using PostgreSQL pg_search BM25.

Uses ParadeDB's pg_search extension with true BM25 ranking algorithm for
cloud-persistent keyword search stored alongside vector embeddings.
"""

import time
from typing import Optional
from langchain_core.tools import BaseTool
import logging

from agrag.tools.schemas import KeywordSearchInput, KeywordSearchOutput, SearchResult
from agrag.storage import PostgresClient

logger = logging.getLogger(__name__)


class KeywordSearchTool(BaseTool):
    """Tool for keyword-based lexical search using pg_search BM25.

    Uses ParadeDB's pg_search extension with true BM25 probabilistic ranking,
    providing cloud-persistent keyword search stored in the same database as
    vector embeddings.
    """

    name: str = "keyword_search"
    description: str = """Use this tool for exact matches and lexical queries using BM25 ranking.
    Best for:
    - Specific identifiers (test IDs, function names, error codes)
    - Exact keyword matching with BM25 probabilistic ranking
    - When you know the specific terms that should appear
    Examples: "TestLoginTimeout", "error code E503", "initiate_handover"
    """
    args_schema: type[KeywordSearchInput] = KeywordSearchInput  # type: ignore[assignment]

    postgres_client: Optional[PostgresClient] = None

    def __init__(self, postgres_client: Optional[PostgresClient] = None, **kwargs):
        """
        Initialize keyword search tool with PostgreSQL full-text search.

        Args:
            postgres_client: PostgreSQL client instance (creates new if not provided)
        """
        super().__init__(**kwargs)
        self.postgres_client = postgres_client or PostgresClient()

    def _run(
        self,
        query: str,
        k: int = 10,
        entity_type: Optional[str] = None,
    ) -> str:
        """
        Execute PostgreSQL full-text keyword search.

        Args:
            query: Keyword query string
            k: Number of results
            entity_type: Optional entity type filter

        Returns:
            Formatted search results as string
        """
        start_time = time.time()

        if self.postgres_client is None:
            return "Error: PostgreSQL client not initialized"

        try:
            # Build metadata filter if entity type provided
            metadata_filter = {}
            if entity_type:
                metadata_filter["entity_type"] = entity_type

            # Perform pg_search BM25 keyword search
            logger.info(f"Performing pg_search BM25 keyword search: {query}")
            results = self.postgres_client.keyword_search(
                query=query,
                k=k,
                metadata_filter=metadata_filter if metadata_filter else None,
            )

            # Format results
            search_results = []
            for result in results:
                search_result = SearchResult(
                    id=result.get("chunk_id", "unknown"),
                    content=result.get("content", ""),
                    score=float(result.get("rank", 0.0)),
                    metadata=result.get("metadata", {}),
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

            # Format for agent consumption
            return self._format_output(output)

        except Exception as e:
            logger.error(f"pg_search BM25 keyword search failed: {e}")
            return f"Error performing keyword search: {str(e)}"

    def _format_output(self, output: KeywordSearchOutput) -> str:
        """
        Format output for agent consumption.

        Args:
            output: KeywordSearchOutput object

        Returns:
            Formatted string
        """
        if not output.results:
            return f"No results found for query: '{output.query}'"

        lines = [
            f"Keyword Search Results (found {output.total_results} items in {output.retrieval_time_ms:.2f}ms):",
            f"Query: {output.query}",
            "",
        ]

        for i, result in enumerate(output.results, 1):
            lines.append(f"{i}. ID: {result.id} (FTS Rank: {result.score:.4f})")
            lines.append(f"   Content: {result.content[:200]}...")
            if result.metadata:
                entity_type = result.metadata.get("entity_type", "Unknown")
                lines.append(f"   Entity Type: {entity_type}")
            lines.append("")

        lines.append("Note: Uses pg_search extension with true BM25 ranking (ParadeDB).")

        return "\n".join(lines)
