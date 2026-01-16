"""Graph traversal tool for structural queries using Neo4j graph patterns.

Uses the @tool decorator pattern from LangChain for cleaner tool definition.
"""

import time
from typing import List, Dict, Any, Optional
import logging

from langchain_core.tools import tool

from agrag.tools.schemas import (
    GraphTraverseInput,
    GraphTraverseOutput,
    GraphPath,
    GraphNode,
    GraphEdge,
)
from agrag.tools.base import BaseToolWrapper
from agrag.storage import Neo4jClient
from agrag.kg.registry import get_registry

logger = logging.getLogger(__name__)


def _parse_paths(results: List[Dict[str, Any]]) -> List[GraphPath]:
    """Parse Neo4j path results into GraphPath objects.

    Args:
        results: Raw path results from Neo4j

    Returns:
        List of GraphPath objects
    """
    paths = []

    def _extract_node_id(node_obj: Any) -> Optional[str]:
        if node_obj is None:
            return None
        try:
            node_dict = dict(node_obj)
            return node_dict.get("id")
        except Exception:
            return None

    for result in results:
        path_obj = result.get("path")
        if not path_obj:
            continue

        # Extract nodes from path
        nodes = []
        node_ids = []
        try:
            # Neo4j path object has nodes property
            for node in path_obj.nodes:
                node_dict = dict(node)
                node_labels = list(node.labels)

                graph_node = GraphNode(
                    id=node_dict.get("id", "unknown"),
                    label=node_labels[0] if node_labels else "Unknown",
                    properties={k: v for k, v in node_dict.items() if k not in ["embedding", "id"]},
                )
                nodes.append(graph_node)
                node_ids.append(graph_node.id)
        except Exception as e:
            logger.warning(f"Failed to parse path nodes: {e}")
            continue

        edges: List[GraphEdge] = []
        try:
            relationships = list(path_obj.relationships)
        except Exception as e:
            logger.warning(f"Failed to parse path relationships: {e}")
            relationships = []

        if relationships and nodes:
            for idx, rel in enumerate(relationships):
                rel_type = getattr(rel, "type", None)
                rel_type_str = str(rel_type) if rel_type else "UNKNOWN"

                direction = "->"
                if idx < len(node_ids) - 1:
                    start_id = getattr(rel, "start_node_id", None)
                    end_id = getattr(rel, "end_node_id", None)
                    if start_id is None or end_id is None:
                        start_id = _extract_node_id(getattr(rel, "start_node", None))
                        end_id = _extract_node_id(getattr(rel, "end_node", None))

                    if start_id and end_id:
                        if start_id == node_ids[idx] and end_id == node_ids[idx + 1]:
                            direction = "->"
                        elif start_id == node_ids[idx + 1] and end_id == node_ids[idx]:
                            direction = "<-"
                        else:
                            direction = "-"
                    else:
                        direction = "-"

                edges.append(GraphEdge(type=rel_type_str, direction=direction))

        if nodes:
            graph_path = GraphPath(
                start_id=result.get("start_id", "unknown"),
                end_id=result.get("end_id", "unknown"),
                depth=result.get("depth", 0),
                nodes=nodes,
                relationships=edges,
            )
            paths.append(graph_path)

    return paths


def _format_graph_output(output: GraphTraverseOutput) -> str:
    """Format GraphTraverseOutput for agent consumption.

    Args:
        output: GraphTraverseOutput object

    Returns:
        Formatted string
    """
    if not output.paths:
        return f"No paths found from node: '{output.start_node_id}'"

    lines = [
        f"Graph Traversal Results (found {output.total_paths} paths in {output.retrieval_time_ms:.2f}ms):",
        f"Start Node: {output.start_node_id}",
        "",
    ]

    for i, path in enumerate(output.paths, 1):
        lines.append(f"{i}. Path (depth {path.depth}): {path.start_id} → {path.end_id}")

        # Show node sequence with relationship types when available
        if path.relationships and len(path.relationships) == max(0, len(path.nodes) - 1):
            parts = []
            if path.nodes:
                parts.append(f"{path.nodes[0].label}:{path.nodes[0].id}")
            for edge, node in zip(path.relationships, path.nodes[1:]):
                if edge.direction == "<-":
                    parts.append(f"<-[{edge.type}]- {node.label}:{node.id}")
                elif edge.direction == "-":
                    parts.append(f"-[{edge.type}]- {node.label}:{node.id}")
                else:
                    parts.append(f"-[{edge.type}]-> {node.label}:{node.id}")
            lines.append(f"   Path: {' '.join(parts)}")
        else:
            node_sequence = " → ".join([f"{node.label}:{node.id}" for node in path.nodes])
            lines.append(f"   Sequence: {node_sequence}")

        # Show end node details if available
        if path.nodes:
            end_node = path.nodes[-1]
            if end_node.properties:
                # Show most relevant properties
                relevant_props = {
                    k: v
                    for k, v in end_node.properties.items()
                    if k in ["name", "description", "status", "priority", "test_type", "signature"]
                }
                if relevant_props:
                    prop_str = ", ".join([f"{k}: {v}" for k, v in relevant_props.items()])
                    lines.append(f"   End Node: {prop_str[:150]}")

        lines.append("")

    return "\n".join(lines)


def create_graph_traverse_tool(neo4j_client: Optional[Neo4jClient] = None):
    """Factory function to create a graph traversal tool with injected dependencies.

    Args:
        neo4j_client: Neo4j client instance (creates new if not provided)

    Returns:
        Configured graph_traverse tool
    """
    client = neo4j_client or Neo4jClient()

    @tool("graph_traverse", args_schema=GraphTraverseInput)
    def graph_traverse(
        start_node_id: str,
        start_node_label: str,
        relationship_types: Optional[List[str]] = None,
        depth: int = 2,
        direction: str = "outgoing",
    ) -> str:
        """Use this tool for structural and dependency queries.

        Best for:
        - Finding relationships between entities
        - Dependency analysis (what tests cover what requirements/functions)
        - Multi-hop structural queries (tests → functions → modules)
        - Tracing impact and coverage

        Examples: "tests that cover REQ_AUTH_005", "functions called by initiate_handover"

        Args:
            start_node_id: ID of the starting node for traversal
            start_node_label: Label of the starting node
            relationship_types: Optional list of relationship types to follow. If None, follows all.
            depth: Maximum traversal depth (1-3)
            direction: Traversal direction: 'outgoing', 'incoming', or 'both'
        """
        start_time = time.time()

        try:
            # Perform graph traversal in Neo4j
            logger.info(
                f"Performing graph traversal from {start_node_id} "
                f"(depth={depth}, direction={direction})"
            )
            registry = get_registry()
            normalized_label = registry.normalize_label(start_node_label)
            if not normalized_label:
                return f"Error: Unknown start_node_label '{start_node_label}'"

            normalized_relationships = None
            if relationship_types:
                normalized_relationships = []
                for rel in relationship_types:
                    rel_norm = registry.normalize_relationship(rel)
                    if not rel_norm:
                        return f"Error: Unknown relationship type '{rel}'"
                    normalized_relationships.append(rel_norm)

            results = client.graph_traverse(
                start_node_id=start_node_id,
                start_node_label=normalized_label,
                relationship_types=normalized_relationships,
                depth=depth,
                direction=direction,
            )

            # Parse Neo4j paths into structured format
            paths = _parse_paths(results)

            retrieval_time_ms = (time.time() - start_time) * 1000

            output = GraphTraverseOutput(
                paths=paths,
                start_node_id=start_node_id,
                total_paths=len(paths),
                retrieval_time_ms=retrieval_time_ms,
            )

            return _format_graph_output(output)

        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return f"Error performing graph traversal: {str(e)}"

    return graph_traverse


# For backwards compatibility, provide a class-based wrapper
class GraphTraverseTool(BaseToolWrapper):
    """Wrapper class for backwards compatibility.

    Use create_graph_traverse_tool() factory function for new code.
    """

    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        """Initialize graph traversal tool.

        Args:
            neo4j_client: Neo4j client instance (creates new if not provided)
        """
        tool = create_graph_traverse_tool(neo4j_client)
        super().__init__(tool)
