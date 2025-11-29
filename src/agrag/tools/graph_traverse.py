"""Graph traversal tool for structural queries using Neo4j graph patterns."""

import time
from typing import Type, List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel
import logging

from agrag.tools.schemas import (
    GraphTraverseInput,
    GraphTraverseOutput,
    GraphPath,
    GraphNode,
)
from agrag.storage import Neo4jClient
from agrag.kg.ontology import NodeLabel, RelationshipType

logger = logging.getLogger(__name__)


class GraphTraverseTool(BaseTool):
    """Tool for graph traversal to find structural relationships."""

    name: str = "graph_traverse"
    description: str = """Use this tool for structural and dependency queries.
    Best for:
    - Finding relationships between entities
    - Dependency analysis (what tests cover what requirements/functions)
    - Multi-hop structural queries (tests → functions → modules)
    - Tracing impact and coverage
    Examples: "tests that cover REQ_AUTH_005", "functions called by initiate_handover"
    """
    args_schema: Type[BaseModel] = GraphTraverseInput

    neo4j_client: Optional[Neo4jClient] = None

    def __init__(self, neo4j_client: Neo4jClient = None, **kwargs):
        """
        Initialize graph traversal tool.

        Args:
            neo4j_client: Neo4j client instance (creates new if not provided)
        """
        super().__init__(**kwargs)
        self.neo4j_client = neo4j_client or Neo4jClient()

    def _run(
        self,
        start_node_id: str,
        start_node_label: NodeLabel,
        relationship_types: List[RelationshipType] = None,
        depth: int = 2,
        direction: str = "outgoing",
    ) -> str:
        """
        Execute graph traversal.

        Args:
            start_node_id: ID of the starting node
            start_node_label: Label of the starting node
            relationship_types: Optional list of relationship types to follow
            depth: Maximum traversal depth (1-3)
            direction: Traversal direction ('outgoing', 'incoming', 'both')

        Returns:
            Formatted traversal results as string
        """
        start_time = time.time()

        try:
            # Perform graph traversal in Neo4j
            logger.info(
                f"Performing graph traversal from {start_node_id} "
                f"(depth={depth}, direction={direction})"
            )
            results = self.neo4j_client.graph_traverse(
                start_node_id=start_node_id,
                start_node_label=start_node_label,
                relationship_types=relationship_types,
                depth=depth,
                direction=direction,
            )

            # Parse Neo4j paths into structured format
            paths = self._parse_paths(results)

            retrieval_time_ms = (time.time() - start_time) * 1000

            output = GraphTraverseOutput(
                paths=paths,
                start_node_id=start_node_id,
                total_paths=len(paths),
                retrieval_time_ms=retrieval_time_ms,
            )

            # Format for agent consumption
            return self._format_output(output)

        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return f"Error performing graph traversal: {str(e)}"

    def _parse_paths(self, results: List[Dict[str, Any]]) -> List[GraphPath]:
        """
        Parse Neo4j path results into GraphPath objects.

        Args:
            results: Raw path results from Neo4j

        Returns:
            List of GraphPath objects
        """
        paths = []

        for result in results:
            path_obj = result.get("path")
            if not path_obj:
                continue

            # Extract nodes from path
            nodes = []
            try:
                # Neo4j path object has nodes property
                for node in path_obj.nodes:
                    node_dict = dict(node)
                    node_labels = list(node.labels)

                    graph_node = GraphNode(
                        id=node_dict.get("id", "unknown"),
                        label=node_labels[0] if node_labels else "Unknown",
                        properties={
                            k: v for k, v in node_dict.items() if k not in ["embedding", "id"]
                        },
                    )
                    nodes.append(graph_node)
            except Exception as e:
                logger.warning(f"Failed to parse path nodes: {e}")
                continue

            if nodes:
                graph_path = GraphPath(
                    start_id=result.get("start_id", "unknown"),
                    end_id=result.get("end_id", "unknown"),
                    depth=result.get("depth", 0),
                    nodes=nodes,
                )
                paths.append(graph_path)

        return paths

    def _format_output(self, output: GraphTraverseOutput) -> str:
        """
        Format output for agent consumption.

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

            # Show node sequence
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
                        if k
                        in ["name", "description", "status", "priority", "test_type", "signature"]
                    }
                    if relevant_props:
                        prop_str = ", ".join([f"{k}: {v}" for k, v in relevant_props.items()])
                        lines.append(f"   End Node: {prop_str[:150]}")

            lines.append("")

        return "\n".join(lines)
