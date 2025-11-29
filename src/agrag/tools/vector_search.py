"""Vector search tool for semantic retrieval using PostgreSQL pgvector."""

import time
from typing import Optional, Any
from langchain_core.tools import BaseTool
import logging

from agrag.tools.schemas import VectorSearchInput, VectorSearchOutput, SearchResult
from agrag.storage import PostgresClient
from agrag.models import get_embedding_service

logger = logging.getLogger(__name__)


class VectorSearchTool(BaseTool):
    """Tool for semantic vector search using PostgreSQL pgvector."""

    name: str = "vector_search"
    description: str = """Use this tool for semantic queries requiring conceptual understanding.
    Best for:
    - Finding semantically similar content
    - Queries about concepts, meanings, or intent
    - When you need to understand the "meaning" behind the query
    Examples: "tests related to handover failures", "authentication requirements"
    """
    args_schema: type[VectorSearchInput] = VectorSearchInput  # type: ignore[assignment]

    postgres_client: Optional[PostgresClient] = None
    embedding_service: Optional[Any] = None

    def __init__(self, postgres_client: Optional[PostgresClient] = None, **kwargs):
        """
        Initialize vector search tool.

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
        node_type: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> str:
        """
        Execute vector search using pgvector.

        Args:
            query: Natural language query
            k: Number of results
            node_type: Optional entity type filter (e.g., "TestCase", "Requirement")
            similarity_threshold: Minimum similarity threshold (not used with pgvector distance)

        Returns:
            Formatted search results as string
        """
        start_time = time.time()

        if self.embedding_service is None:
            return "Error: Embedding service not initialized"
        if self.postgres_client is None:
            return "Error: PostgreSQL client not initialized"

        try:
            # Generate query embedding
            logger.info(f"Generating embedding for query: {query}")
            query_embedding = self.embedding_service.embed_query(query)

            # Build metadata filter if node_type provided
            metadata_filter = None
            if node_type:
                metadata_filter = {"entity_type": node_type}

            # Perform vector search in PostgreSQL using pgvector
            logger.info(f"Performing pgvector search (entity_type={node_type})")
            results = self.postgres_client.vector_search(
                query_embedding=query_embedding,
                k=k,
                metadata_filter=metadata_filter,
            )

            # Format results
            search_results = []
            for result in results:
                search_result = SearchResult(
                    id=result.get("chunk_id", "unknown"),
                    content=result.get("content", ""),
                    score=float(result.get("similarity", 0.0)),
                    metadata=result.get("metadata", {}),
                    source="pgvector",
                )
                search_results.append(search_result)

            retrieval_time_ms = (time.time() - start_time) * 1000

            output = VectorSearchOutput(
                results=search_results,
                query=query,
                total_results=len(search_results),
                retrieval_time_ms=retrieval_time_ms,
            )

            # Format for agent consumption
            return self._format_output(output)

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return f"Error performing vector search: {str(e)}"

    def _format_output(self, output: VectorSearchOutput) -> str:
        """
        Format output for agent consumption.

        Args:
            output: VectorSearchOutput object

        Returns:
            Formatted string
        """
        if not output.results:
            return f"No results found for query: '{output.query}'"

        lines = [
            f"Vector Search Results (found {output.total_results} items in {output.retrieval_time_ms:.2f}ms):",
            f"Query: {output.query}",
            "",
        ]

        for i, result in enumerate(output.results, 1):
            lines.append(f"{i}. ID: {result.id} (Similarity: {result.score:.4f})")
            lines.append(f"   Content: {result.content[:200]}...")
            if result.metadata:
                entity_type = result.metadata.get("entity_type", "Unknown")
                lines.append(f"   Entity Type: {entity_type}")
            lines.append("")

        return "\n".join(lines)
