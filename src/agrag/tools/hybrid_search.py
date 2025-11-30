"""Hybrid search tool combining vector and keyword search with RRF fusion.

Uses PostgreSQL's pgvector for semantic similarity and pg_search BM25 for keyword
ranking, providing cloud-persistent hybrid retrieval with Reciprocal Rank Fusion.

Uses the @tool decorator pattern from LangChain for cleaner tool definition.
"""

import time
from typing import Optional
import logging

from langchain.tools import tool

from agrag.tools.schemas import HybridSearchInput, HybridSearchOutput, SearchResult
from agrag.storage import PostgresClient
from agrag.models import get_embedding_service

logger = logging.getLogger(__name__)


def _format_hybrid_output(output: HybridSearchOutput) -> str:
    """Format HybridSearchOutput for agent consumption.

    Args:
        output: HybridSearchOutput object

    Returns:
        Formatted string
    """
    if not output.results:
        return f"No results found for query: '{output.query}'"

    lines = [
        f"Hybrid Search Results ({output.fusion_method}) - found {output.total_results} items in {output.retrieval_time_ms:.2f}ms:",
        f"Query: {output.query}",
        "",
    ]

    for i, result in enumerate(output.results, 1):
        lines.append(f"{i}. ID: {result.id} (RRF Score: {result.score:.4f})")
        lines.append(f"   Content: {result.content[:200]}...")
        if result.metadata:
            entity_type = result.metadata.get("entity_type", "Unknown")
            lines.append(f"   Entity Type: {entity_type}")
        lines.append("")

    lines.append(
        "Note: RRF combines pgvector similarity and pg_search BM25 ranking for optimal precision."
    )

    return "\n".join(lines)


def create_hybrid_search_tool(postgres_client: Optional[PostgresClient] = None):
    """Factory function to create a hybrid search tool with injected dependencies.

    Args:
        postgres_client: PostgreSQL client instance for both vector and keyword search

    Returns:
        Configured hybrid_search tool
    """
    client = postgres_client or PostgresClient()
    embedding_service = get_embedding_service()

    @tool("hybrid_search", args_schema=HybridSearchInput)
    def hybrid_search(
        query: str,
        k: int = 10,
        rrf_k: int = 60,
        entity_type: Optional[str] = None,
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
        """
        start_time = time.time()

        if embedding_service is None:
            return "Error: Embedding service not initialized"
        if client is None:
            return "Error: PostgreSQL client not initialized"

        try:
            # Generate query embedding for vector component
            logger.info(f"Generating embedding for hybrid search query: {query}")
            query_embedding = embedding_service.embed_query(query)

            # Build metadata filter if entity type provided
            metadata_filter = {}
            if entity_type:
                metadata_filter["entity_type"] = entity_type

            # Perform PostgreSQL hybrid search (pgvector + pg_search BM25 with RRF fusion)
            logger.info(f"Performing hybrid search (pgvector + pg_search BM25, RRF k={rrf_k})...")
            results = client.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                k=k,
                rrf_k=rrf_k,
                metadata_filter=metadata_filter if metadata_filter else None,
            )

            # Format results
            search_results = []
            for result in results:
                search_result = SearchResult(
                    id=result.get("chunk_id", "unknown"),
                    content=result.get("content", ""),
                    score=float(result.get("rrf_score", 0.0)),
                    metadata=result.get("metadata", {}),
                    source="hybrid",
                )
                search_results.append(search_result)

            retrieval_time_ms = (time.time() - start_time) * 1000

            output = HybridSearchOutput(
                results=search_results,
                query=query,
                total_results=len(search_results),
                retrieval_time_ms=retrieval_time_ms,
                fusion_method="RRF (pgvector + pg_search BM25)",
            )

            return _format_hybrid_output(output)

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return f"Error performing hybrid search: {str(e)}"

    return hybrid_search


# For backwards compatibility, provide a class-based wrapper
class HybridSearchTool:
    """Wrapper class for backwards compatibility.

    Use create_hybrid_search_tool() factory function for new code.
    """

    def __init__(self, postgres_client: Optional[PostgresClient] = None):
        """Initialize hybrid search tool.

        Args:
            postgres_client: PostgreSQL client instance for both vector and keyword search
        """
        self._tool = create_hybrid_search_tool(postgres_client)

    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def description(self) -> str:
        return self._tool.description

    def invoke(self, *args, **kwargs):
        return self._tool.invoke(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tool, name)
