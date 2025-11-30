"""Retrieval tools for agentic RAG system.

Provides both the modern @tool decorator-based factory functions and
backwards-compatible class wrappers for tool creation.
"""

from .vector_search import VectorSearchTool, create_vector_search_tool
from .keyword_search import KeywordSearchTool, create_keyword_search_tool
from .graph_traverse import GraphTraverseTool, create_graph_traverse_tool
from .hybrid_search import HybridSearchTool, create_hybrid_search_tool
from .schemas import (
    VectorSearchInput,
    KeywordSearchInput,
    GraphTraverseInput,
    HybridSearchInput,
    SearchResult,
)

__all__ = [
    # Factory functions (recommended for new code)
    "create_vector_search_tool",
    "create_keyword_search_tool",
    "create_graph_traverse_tool",
    "create_hybrid_search_tool",
    # Backwards-compatible class wrappers
    "VectorSearchTool",
    "KeywordSearchTool",
    "GraphTraverseTool",
    "HybridSearchTool",
    # Input schemas
    "VectorSearchInput",
    "KeywordSearchInput",
    "GraphTraverseInput",
    "HybridSearchInput",
    "SearchResult",
]
