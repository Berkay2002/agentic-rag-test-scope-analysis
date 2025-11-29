"""Hybrid search tool combining vector and BM25 keyword search with RRF fusion."""

import time
from typing import Type, Optional, Any, Dict, List
from langchain_core.tools import BaseTool
from pydantic import BaseModel
import logging

from agrag.tools.schemas import HybridSearchInput, HybridSearchOutput, SearchResult
from agrag.storage import PostgresClient, BM25RetrieverManager
from agrag.models import get_embedding_service

logger = logging.getLogger(__name__)


class HybridSearchTool(BaseTool):
    """Tool for hybrid search combining semantic vector search and BM25 lexical retrieval."""

    name: str = "hybrid_search"
    description: str = """Use this tool when you need both semantic understanding AND exact keyword matching.
    Combines vector similarity search (HNSW) with BM25 probabilistic ranking using RRF fusion.
    Best for:
    - Complex queries requiring both conceptual understanding and specific terms
    - Balancing semantic similarity with lexical precision
    - Queries that mix concepts with technical identifiers
    Examples: "tests for LTE signaling with timeout errors", "handover functions in network module"
    """
    args_schema: Type[BaseModel] = HybridSearchInput

    postgres_client: Optional[PostgresClient] = None
    bm25_manager: Optional[BM25RetrieverManager] = None
    embedding_service: Optional[Any] = None

    def __init__(
        self,
        postgres_client: PostgresClient = None,
        bm25_manager: BM25RetrieverManager = None,
        **kwargs,
    ):
        """
        Initialize hybrid search tool.

        Args:
            postgres_client: PostgreSQL client instance for vector search
            bm25_manager: BM25 retriever manager for keyword search
        """
        super().__init__(**kwargs)
        self.postgres_client = postgres_client or PostgresClient()
        self.bm25_manager = bm25_manager or BM25RetrieverManager(k=10)
        self.embedding_service = get_embedding_service()

    def _run(
        self,
        query: str,
        k: int = 10,
        rrf_k: int = 60,
        entity_type: str = None,
    ) -> str:
        """
        Execute hybrid search with RRF fusion of vector and BM25 results.

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

            # Perform vector search
            logger.info("Performing vector search component...")
            vector_results = self.postgres_client.vector_search(
                query_embedding=query_embedding,
                k=k * 2,  # Get more candidates for fusion
                metadata_filter=metadata_filter if metadata_filter else None,
            )

            # Perform BM25 search
            logger.info("Performing BM25 keyword search component...")
            bm25_results = self.bm25_manager.search(
                query=query,
                k=k * 2,  # Get more candidates for fusion
                metadata_filter=metadata_filter if metadata_filter else None,
            )

            # Apply RRF fusion
            logger.info(f"Applying RRF fusion (k={rrf_k})...")
            fused_results = self._apply_rrf_fusion(
                vector_results=vector_results,
                bm25_results=bm25_results,
                k=k,
                rrf_k=rrf_k,
            )

            # Format results
            search_results = []
            for result in fused_results:
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
                fusion_method="RRF (Vector+BM25)",
            )

            # Format for agent consumption
            return self._format_output(output)

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return f"Error performing hybrid search: {str(e)}"

    def _apply_rrf_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        k: int,
        rrf_k: int,
    ) -> List[Dict]:
        """
        Apply Reciprocal Rank Fusion to combine vector and BM25 results.

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            k: Number of final results
            rrf_k: RRF constant

        Returns:
            Fused and ranked results
        """
        rrf_scores: Dict[str, float] = {}
        rrf_docs: Dict[str, Dict] = {}

        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result.get("chunk_id", "unknown")
            rrf_scores[chunk_id] = 1 / (rrf_k + rank)
            rrf_docs[chunk_id] = result

        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result.get("chunk_id", "unknown")
            if chunk_id in rrf_scores:
                rrf_scores[chunk_id] += 1 / (rrf_k + rank)
            else:
                rrf_scores[chunk_id] = 1 / (rrf_k + rank)
                rrf_docs[chunk_id] = result

        # Sort by RRF score and take top k
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        # Build final results
        final_results = []
        for chunk_id, score in sorted_results:
            doc = rrf_docs[chunk_id].copy()
            doc["rrf_score"] = score
            final_results.append(doc)

        return final_results

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
            "Note: RRF combines HNSW vector similarity and BM25 keyword ranking for optimal precision."
        )

        return "\n".join(lines)
