"""Keyword search tool for lexical retrieval using BM25 algorithm."""

import time
from typing import Type, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel
import logging

from agrag.tools.schemas import KeywordSearchInput, KeywordSearchOutput, SearchResult
from agrag.storage import BM25RetrieverManager

logger = logging.getLogger(__name__)


class KeywordSearchTool(BaseTool):
    """Tool for keyword-based lexical search using BM25 algorithm."""

    name: str = "keyword_search"
    description: str = """Use this tool for exact matches and lexical queries using BM25 ranking.
    Best for:
    - Specific identifiers (test IDs, function names, error codes)
    - Exact keyword matching with probabilistic ranking
    - When you know the specific terms that should appear
    Examples: "TestLoginTimeout", "error code E503", "initiate_handover"
    """
    args_schema: Type[BaseModel] = KeywordSearchInput

    bm25_manager: Optional[BM25RetrieverManager] = None

    def __init__(self, bm25_manager: BM25RetrieverManager = None, **kwargs):
        """
        Initialize keyword search tool with BM25.

        Args:
            bm25_manager: BM25 retriever manager instance (creates new if not provided)
        """
        super().__init__(**kwargs)
        self.bm25_manager = bm25_manager or BM25RetrieverManager(k=10)

    def _run(
        self,
        query: str,
        k: int = 10,
        entity_type: str = None,
    ) -> str:
        """
        Execute BM25 keyword search.

        Args:
            query: Keyword query string
            k: Number of results
            entity_type: Optional entity type filter

        Returns:
            Formatted search results as string
        """
        start_time = time.time()

        try:
            # Build metadata filter if entity type provided
            metadata_filter = {}
            if entity_type:
                metadata_filter["entity_type"] = entity_type

            # Perform BM25 search
            logger.info(f"Performing BM25 keyword search: {query}")
            results = self.bm25_manager.search(
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
                    score=float(result.get("score", 0.0)),
                    metadata=result.get("metadata", {}),
                    source="bm25",
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
            logger.error(f"BM25 keyword search failed: {e}")
            return f"Error performing BM25 keyword search: {str(e)}"

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
            f"BM25 Keyword Search Results (found {output.total_results} items in {output.retrieval_time_ms:.2f}ms):",
            f"Query: {output.query}",
            "",
        ]

        for i, result in enumerate(output.results, 1):
            lines.append(f"{i}. ID: {result.id} (BM25 Score: {result.score:.4f})")
            lines.append(f"   Content: {result.content[:200]}...")
            if result.metadata:
                entity_type = result.metadata.get("entity_type", "Unknown")
                lines.append(f"   Entity Type: {entity_type}")
            lines.append("")

        lines.append(
            "Note: BM25 (Best Matching 25) is a probabilistic ranking function for keyword search."
        )

        return "\n".join(lines)
