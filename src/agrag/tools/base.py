"""Base classes and utilities for retrieval tools.

Provides common functionality shared across search tools to reduce code duplication.
"""

from typing import Any, List, Optional


class BaseToolWrapper:
    """Base wrapper class for backwards compatibility with class-based tools.

    Modern code should use the factory functions (e.g., create_vector_search_tool)
    instead of these wrapper classes. This class exists only for backwards compatibility.

    Attributes:
        _tool: The underlying LangChain tool created by a factory function
    """

    def __init__(self, tool: Any):
        """Initialize the tool wrapper.

        Args:
            tool: The underlying LangChain tool instance
        """
        self._tool = tool

    @property
    def name(self) -> str:
        """Return the tool's name."""
        return self._tool.name

    @property
    def description(self) -> str:
        """Return the tool's description."""
        return self._tool.description

    def invoke(self, *args, **kwargs):
        """Invoke the underlying tool."""
        return self._tool.invoke(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate attribute access to the underlying tool."""
        return getattr(self._tool, name)


def format_search_results_header(
    query: str,
    total_results: int,
    retrieval_time_ms: float,
    search_type: str,
    additional_info: Optional[str] = None,
) -> str:
    """Format the header for search results.

    Args:
        query: The search query
        total_results: Number of results found
        retrieval_time_ms: Time taken for retrieval in milliseconds
        search_type: Type of search (e.g., "Vector Search", "Keyword Search")
        additional_info: Optional additional information (e.g., fusion method)

    Returns:
        Formatted header string
    """
    header_parts = [f"{search_type} Results"]
    if additional_info:
        header_parts[0] = f"{search_type} Results ({additional_info})"

    header_parts.append(
        f" - found {total_results} items in {retrieval_time_ms:.2f}ms:"
    )
    lines = ["".join(header_parts), f"Query: {query}", ""]
    return "\n".join(lines)


def format_search_result_item(
    index: int,
    result_id: str,
    score: float,
    score_label: str,
    content: str,
    entity_type: Optional[str] = None,
    content_preview_length: int = 200,
) -> str:
    """Format a single search result item.

    Args:
        index: Result index (1-based)
        result_id: Entity ID
        score: Relevance score
        score_label: Label for the score (e.g., "Similarity", "FTS Rank", "RRF Score")
        content: Result content
        entity_type: Optional entity type
        content_preview_length: Length of content preview

    Returns:
        Formatted result item string
    """
    lines = [
        f"{index}. Entity ID: {result_id} ({score_label}: {score:.4f})",
        f"   Snippet: {content[:content_preview_length]}...",
    ]
    if entity_type:
        lines.append(f"   Entity Type: {entity_type}")
    lines.append("")
    return "\n".join(lines)


def format_search_results_footer(footer_note: Optional[str] = None) -> str:
    """Format the footer for search results.

    Args:
        footer_note: Optional footer note to append

    Returns:
        Formatted footer string (empty if no note provided)
    """
    if footer_note:
        return f"\n{footer_note}"
    return ""


def build_metadata_with_chunk_id(
    metadata: Optional[dict], chunk_id: Optional[str]
) -> dict:
    """Build metadata dict including chunk_id if provided.

    Args:
        metadata: Existing metadata dict (may be None)
        chunk_id: Optional chunk ID to include

    Returns:
        Combined metadata dict
    """
    result_metadata = metadata or {}
    if chunk_id:
        result_metadata = {**result_metadata, "chunk_id": chunk_id}
    return result_metadata


def extract_score_or_default(
    score_value: Any, default: float = 0.0
) -> float:
    """Extract and convert score value to float, or return default.

    Args:
        score_value: Score value (may be None, float, or other type)
        default: Default value if score_value is None

    Returns:
        Float score value
    """
    return float(score_value) if score_value is not None else default
