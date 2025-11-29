"""Hybrid search tool combining vector and keyword search with RRF fusion."""

import time
from typing import Type, Optional, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel
import logging

from agrag.tools.schemas import HybridSearchInput, HybridSearchOutput, SearchResult
from agrag.storage import PostgresClient
from agrag.models import get_embedding_service

logger = logging.getLogger(__name__)


class HybridSearchTool(BaseTool):
    """Tool for hybrid search combining semantic and lexical retrieval."""

    name: str = "hybrid_search"
    description: str = """Use this tool when you need both semantic understanding AND exact keyword matching.
    Best for:
    - Complex queries requiring both conceptual understanding and specific terms
    - Balancing semantic similarity with lexical precision
    - Queries that mix concepts with technical identifiers
    Examples: "tests for LTE signaling with timeout errors", "handover functions in network module"
    """
    args_schema: Type[BaseModel] = HybridSearchInput

    postgres_client: Optional[PostgresClient] = None
    embedding_service: Optional[Any] = None

    def __init__(self, postgres_client: PostgresClient = None, **kwargs):
        """
        Initialize hybrid search tool.

        Args:
            postgres_client: PostgreSQL client instance (creates new if not provided)
        """
        super().__init__(**kwargs)
        self.postgres_client = postgres_client or PostgresClient()
        self.embedding_service = get_embedding_service()

    def _run(
        self,
        query: str,
        k: int = 10,
        rrf_k: int = 60,
        entity_type: str = None,
    ) -> str:
        """
        Execute hybrid search.

        Args:
            query: Search query string
            k: Number of results to return
            rrf_k: RRF constant (default 60)
            entity_type: Optional entity type filter

        Returns:
            Formatted search results as string
        """
        start_time = time.time()

        try:
            # Generate query embedding for vector component
            logger.info(f"Generating embedding for hybrid search query: {query}")
            query_embedding = self.embedding_service.embed_query(query)

            # Build metadata filter if entity type provided
            metadata_filter = {}
            if entity_type:
                metadata_filter["entity_type"] = entity_type

            # Perform hybrid search (RRF fusion)
            logger.info(f"Performing hybrid search with RRF (k={rrf_k})")
            results = self.postgres_client.hybrid_search(
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
                fusion_method="RRF",
            )

            # Format for agent consumption
            return self._format_output(output)

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return f"Error performing hybrid search: {str(e)}"

    def _format_output(self, output: HybridSearchOutput) -> str:
        """
        Format output for agent consumption.

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
            "Note: RRF (Reciprocal Rank Fusion) combines vector and keyword search rankings."
        )

        return "\n".join(lines)
