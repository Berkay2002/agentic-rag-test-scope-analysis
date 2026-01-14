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
from agrag.storage import PostgresClient
from agrag.models import get_embedding_service
from agrag.kg.ontology import NodeLabel

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


def create_vector_search_tool(postgres_client: Optional[PostgresClient] = None):
    """Factory function to create a vector search tool with injected dependencies.

    Args:
        postgres_client: PostgreSQL client instance (creates new if not provided)

    Returns:
        Configured vector_search tool
    """
    client = postgres_client or PostgresClient()
    embedding_service = get_embedding_service()

    @tool("vector_search", args_schema=VectorSearchInput)
    def vector_search(
        query: str,
        k: int = 10,
        node_type: Optional[NodeLabel] = None,
        similarity_threshold: Optional[float] = None,
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
        """
        start_time = time.time()

        if embedding_service is None:
            return "Error: Embedding service not initialized"
        if client is None:
            return "Error: PostgreSQL client not initialized"

        try:
            # Generate query embedding
            logger.info(f"Generating embedding for query: {query}")
            query_embedding = embedding_service.embed_query(query)

            # Build metadata filter if node_type provided
            metadata_filter = None
            if node_type:
                metadata_filter = {"entity_type": node_type.value}

            # Perform vector search in PostgreSQL using pgvector
            logger.info(f"Performing pgvector search (entity_type={node_type})")
            results = client.vector_search(
                query_embedding=query_embedding,
                k=k,
                metadata_filter=metadata_filter,
            )

            # Format results using shared processing logic
            # Apply similarity threshold filter if provided
            score_filter_fn = None
            if similarity_threshold is not None:
                def threshold_filter(score: float) -> bool:
                    return score >= similarity_threshold
                score_filter_fn = threshold_filter

            search_results = process_search_results(
                raw_results=results,
                score_field="similarity",
                source_name="pgvector",
                score_filter_fn=score_filter_fn,
            )

            retrieval_time_ms = (time.time() - start_time) * 1000

            output = VectorSearchOutput(
                results=search_results,
                query=query,
                total_results=len(search_results),
                retrieval_time_ms=retrieval_time_ms,
            )

            return _format_vector_output(output)

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return f"Error performing vector search: {str(e)}"

    return vector_search


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
