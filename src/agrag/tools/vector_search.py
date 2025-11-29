"""Vector search tool for semantic retrieval using Neo4j vector indexes."""

import time
from typing import Type, Optional, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel
import logging

from agrag.tools.schemas import VectorSearchInput, VectorSearchOutput, SearchResult
from agrag.storage import Neo4jClient
from agrag.models import get_embedding_service

logger = logging.getLogger(__name__)


class VectorSearchTool(BaseTool):
    """Tool for semantic vector search using Neo4j vector indexes."""

    name: str = "vector_search"
    description: str = """Use this tool for semantic queries requiring conceptual understanding.
    Best for:
    - Finding semantically similar content
    - Queries about concepts, meanings, or intent
    - When you need to understand the "meaning" behind the query
    Examples: "tests related to handover failures", "authentication requirements"
    """
    args_schema: Type[BaseModel] = VectorSearchInput

    neo4j_client: Optional[Neo4jClient] = None
    embedding_service: Optional[Any] = None

    def __init__(self, neo4j_client: Neo4jClient = None, **kwargs):
        """
        Initialize vector search tool.

        Args:
            neo4j_client: Neo4j client instance (creates new if not provided)
        """
        super().__init__(**kwargs)
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.embedding_service = get_embedding_service()

    def _run(
        self,
        query: str,
        k: int = 10,
        node_type: str = "TestCase",
        similarity_threshold: float = None,
    ) -> str:
        """
        Execute vector search.

        Args:
            query: Natural language query
            k: Number of results
            node_type: Type of nodes to search
            similarity_threshold: Minimum similarity threshold

        Returns:
            Formatted search results as string
        """
        start_time = time.time()

        try:
            # Generate query embedding
            logger.info(f"Generating embedding for query: {query}")
            query_embedding = self.embedding_service.embed_query(query)

            # Map string to NodeLabel enum
            from agrag.kg.ontology import NodeLabel

            try:
                node_label = NodeLabel[node_type.upper().replace("TESTCASE", "TEST_CASE")]
            except KeyError:
                node_label = NodeLabel.TEST_CASE

            # Perform vector search in Neo4j
            logger.info(f"Performing vector search for {node_label.value}")
            results = self.neo4j_client.vector_search(
                query_embedding=query_embedding,
                node_label=node_label,
                k=k,
                similarity_threshold=similarity_threshold,
            )

            # Format results
            search_results = []
            for result in results:
                node = result["node"]
                search_result = SearchResult(
                    id=node.get("id", "unknown"),
                    content=node.get("description", node.get("name", "")),
                    score=float(result["score"]),
                    metadata={
                        "label": result["label"],
                        **{k: v for k, v in node.items() if k not in ["embedding", "id"]},
                    },
                    source="vector",
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
            lines.append(f"{i}. ID: {result.id} (Score: {result.score:.4f})")
            lines.append(f"   Content: {result.content[:200]}...")
            if result.metadata:
                meta_str = ", ".join(
                    [f"{k}: {v}" for k, v in result.metadata.items() if k != "label"]
                )
                if meta_str:
                    lines.append(f"   Metadata: {meta_str}")
            lines.append("")

        return "\n".join(lines)
