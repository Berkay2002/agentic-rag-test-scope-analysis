"""Base classes and utilities for retrieval tools.

Provides common functionality shared across search tools to reduce code duplication.
"""

from typing import Any, List, Optional, Callable


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


def format_search_output(
    output: Any,
    search_type: str,
    score_label: str,
    additional_info: Optional[str] = None,
    footer_note: Optional[str] = None,
) -> str:
    """Format search output for agent consumption.

    Generic formatter that works with VectorSearchOutput, HybridSearchOutput,
    or any output object with results, query, total_results, and retrieval_time_ms attributes.

    Args:
        output: Search output object (must have results, query, total_results, retrieval_time_ms)
        search_type: Type of search (e.g., "Vector Search", "Hybrid Search")
        score_label: Label for the score (e.g., "Similarity", "RRF Score")
        additional_info: Optional additional information (e.g., fusion method)
        footer_note: Optional footer note to append

    Returns:
        Formatted string
    """
    if not output.results:
        return f"No results found for query: '{output.query}'"

    # Build header
    header = format_search_results_header(
        query=output.query,
        total_results=output.total_results,
        retrieval_time_ms=output.retrieval_time_ms,
        search_type=search_type,
        additional_info=additional_info,
    )

    # Format each result
    result_items = []
    for i, result in enumerate(output.results, 1):
        entity_type = result.metadata.get("entity_type", "Unknown") if result.metadata else None
        item = format_search_result_item(
            index=i,
            result_id=result.id,
            score=result.score,
            score_label=score_label,
            content=result.content,
            entity_type=entity_type,
        )
        result_items.append(item)

    # Add footer if provided
    footer = format_search_results_footer(footer_note)

    return header + "\n".join(result_items) + footer


def process_search_results(
    raw_results: List[dict],
    score_field: str,
    source_name: str,
    score_filter_fn: Optional[Callable[[float], bool]] = None,
) -> List["SearchResult"]:
    """Process raw search results into SearchResult objects.

    Args:
        raw_results: List of raw result dictionaries from the database
        score_field: Name of the score field in raw results (e.g., "similarity", "rrf_score")
        source_name: Name of the search source (e.g., "pgvector", "hybrid")
        score_filter_fn: Optional function that takes a score (float) and returns True to keep
            the result or False to filter it out. Example: lambda s: s >= 0.5

    Returns:
        List of SearchResult objects (filtered if score_filter_fn provided)
    """
    from agrag.tools.shared.schemas import SearchResult

    search_results = []
    for result in raw_results:
        # Extract score with fallback to 0.0
        score = extract_score_or_default(result.get(score_field))

        # Apply score filter if provided
        if score_filter_fn and not score_filter_fn(score):
            continue

        # Build metadata including chunk_id if present
        metadata = build_metadata_with_chunk_id(
            result.get("metadata", {}), result.get("chunk_id")
        )

        # Create SearchResult object
        search_result = SearchResult(
            id=metadata.get("entity_id") or result.get("chunk_id", "unknown"),
            content=result.get("content", ""),
            score=score,
            metadata=metadata,
            source=source_name,
        )
        search_results.append(search_result)

    return search_results

