"""Pydantic schemas for retrieval tool inputs and outputs."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from agrag.kg.ontology import NodeLabel, RelationshipType


# Input Schemas for Tools

class VectorSearchInput(BaseModel):
    """Input schema for vector search tool."""

    query: str = Field(
        ...,
        description="Natural language query for semantic search",
        examples=["tests related to handover failures", "authentication requirements"],
    )
    k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of results to return (1-50)",
    )
    node_type: NodeLabel = Field(
        default=NodeLabel.TEST_CASE,
        description="Type of nodes to search (e.g., TestCase, Requirement, Function)",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold (0.0-1.0)",
    )


class KeywordSearchInput(BaseModel):
    """Input schema for keyword search tool."""

    query: str = Field(
        ...,
        description="Keyword query for exact/lexical matching",
        examples=["TestLoginTimeout", "error code E503", "initiate_handover"],
    )
    k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of results to return (1-50)",
    )
    entity_type: Optional[str] = Field(
        default=None,
        description="Filter by entity type (e.g., 'TestCase', 'Function')",
    )


class GraphTraverseInput(BaseModel):
    """Input schema for graph traversal tool."""

    start_node_id: str = Field(
        ...,
        description="ID of the starting node for traversal",
        examples=["TC_HANDOVER_001", "REQ_AUTH_005", "FUNC_initiate_handover"],
    )
    start_node_label: NodeLabel = Field(
        ...,
        description="Label of the starting node",
    )
    relationship_types: Optional[List[RelationshipType]] = Field(
        default=None,
        description="Optional list of relationship types to follow. If None, follows all relationships.",
    )
    depth: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Maximum traversal depth (1-3)",
    )
    direction: str = Field(
        default="outgoing",
        description="Traversal direction: 'outgoing', 'incoming', or 'both'",
        pattern="^(outgoing|incoming|both)$",
    )


class HybridSearchInput(BaseModel):
    """Input schema for hybrid search tool."""

    query: str = Field(
        ...,
        description="Search query combining semantic and lexical requirements",
        examples=[
            "tests for LTE signaling with timeout errors",
            "handover functions in network module",
        ],
    )
    k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of results to return (1-50)",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        le=100,
        description="Reciprocal Rank Fusion constant (default 60)",
    )
    entity_type: Optional[str] = Field(
        default=None,
        description="Filter by entity type",
    )


# Output Schemas

class SearchResult(BaseModel):
    """Single search result."""

    id: str = Field(..., description="Entity ID")
    content: str = Field(..., description="Result content/description")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    source: str = Field(..., description="Source of result (e.g., 'vector', 'keyword', 'graph')")


class VectorSearchOutput(BaseModel):
    """Output schema for vector search tool."""

    results: List[SearchResult] = Field(
        default_factory=list,
        description="List of search results ordered by relevance",
    )
    query: str = Field(..., description="Original query")
    total_results: int = Field(..., description="Total number of results returned")
    retrieval_time_ms: float = Field(..., description="Retrieval time in milliseconds")


class KeywordSearchOutput(BaseModel):
    """Output schema for keyword search tool."""

    results: List[SearchResult] = Field(
        default_factory=list,
        description="List of search results ordered by relevance",
    )
    query: str = Field(..., description="Original query")
    total_results: int = Field(..., description="Total number of results returned")
    retrieval_time_ms: float = Field(..., description="Retrieval time in milliseconds")


class GraphNode(BaseModel):
    """Graph node in traversal result."""

    id: str = Field(..., description="Node ID")
    label: str = Field(..., description="Node label")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Node properties",
    )


class GraphPath(BaseModel):
    """Graph path in traversal result."""

    start_id: str = Field(..., description="Start node ID")
    end_id: str = Field(..., description="End node ID")
    depth: int = Field(..., description="Path depth")
    nodes: List[GraphNode] = Field(
        default_factory=list,
        description="Nodes in path",
    )


class GraphTraverseOutput(BaseModel):
    """Output schema for graph traversal tool."""

    paths: List[GraphPath] = Field(
        default_factory=list,
        description="List of traversal paths",
    )
    start_node_id: str = Field(..., description="Starting node ID")
    total_paths: int = Field(..., description="Total number of paths found")
    retrieval_time_ms: float = Field(..., description="Retrieval time in milliseconds")


class HybridSearchOutput(BaseModel):
    """Output schema for hybrid search tool."""

    results: List[SearchResult] = Field(
        default_factory=list,
        description="List of search results ordered by RRF score",
    )
    query: str = Field(..., description="Original query")
    total_results: int = Field(..., description="Total number of results returned")
    retrieval_time_ms: float = Field(..., description="Retrieval time in milliseconds")
    fusion_method: str = Field(
        default="RRF",
        description="Fusion method used (e.g., 'RRF')",
    )
