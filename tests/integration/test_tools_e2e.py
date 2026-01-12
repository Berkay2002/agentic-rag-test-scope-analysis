from typing import Dict, Tuple

import pytest

from agrag.kg.ontology import NodeLabel, RelationshipType
from agrag.tools import (
    create_graph_traverse_tool,
    create_hybrid_search_tool,
    create_keyword_search_tool,
    create_vector_search_tool,
)


def _first_entity(dataset: Dict, prefix: str) -> Dict:
    for entity in dataset.get("entities", []):
        entity_id = entity.get("id", "")
        if entity_id.startswith(prefix):
            return entity
    raise AssertionError(f"No entity found with prefix {prefix}")


def _first_verifies_pair(dataset: Dict) -> Tuple[str, str]:
    for rel in dataset.get("relationships", []):
        if rel.get("relationship_type") == "VERIFIES":
            return rel.get("source_id", ""), rel.get("target_id", "")
    raise AssertionError("No VERIFIES relationship found in dataset")


@pytest.mark.usefixtures("postgres_client")
def test_keyword_search_e2e(postgres_client, dataset) -> None:
    requirement = _first_entity(dataset, "REQ_")
    tool = create_keyword_search_tool(postgres_client=postgres_client)

    output = tool.invoke(
        {
            "query": requirement["id"],
            "k": 5,
            "entity_type": "Requirement",
        }
    )

    assert "Keyword Search Results" in output
    assert requirement["id"] in output


@pytest.mark.usefixtures("embedding_available", "postgres_client")
def test_vector_search_e2e(postgres_client, dataset, embedding_available) -> None:
    requirement = _first_entity(dataset, "REQ_")
    tool = create_vector_search_tool(postgres_client=postgres_client)

    output = tool.invoke(
        {
            "query": requirement["description"],
            "k": 5,
            "node_type": NodeLabel.REQUIREMENT,
        }
    )

    assert "Vector Search Results" in output
    assert "No results found" not in output


@pytest.mark.usefixtures("embedding_available", "postgres_client")
def test_hybrid_search_e2e(postgres_client, dataset, embedding_available) -> None:
    requirement = _first_entity(dataset, "REQ_")
    tool = create_hybrid_search_tool(postgres_client=postgres_client)

    output = tool.invoke(
        {
            "query": requirement["id"],
            "k": 5,
            "entity_type": "Requirement",
        }
    )

    assert "Hybrid Search Results" in output
    assert "No results found" not in output


@pytest.mark.usefixtures("neo4j_client")
def test_graph_traverse_e2e(neo4j_client, dataset) -> None:
    _, requirement_id = _first_verifies_pair(dataset)
    tool = create_graph_traverse_tool(neo4j_client=neo4j_client)

    output = tool.invoke(
        {
            "start_node_id": requirement_id,
            "start_node_label": NodeLabel.REQUIREMENT,
            "relationship_types": [RelationshipType.VERIFIES],
            "depth": 1,
            "direction": "incoming",
        }
    )

    assert "Graph Traversal Results" in output
    assert "TC_" in output
