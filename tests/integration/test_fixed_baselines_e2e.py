from typing import Dict, Tuple

import pytest

from agrag.evaluation.baselines.fixed_baselines import run_fixed_graphrag, run_fixed_rag
from agrag.tools import create_graph_traverse_tool, create_hybrid_search_tool


def _first_verifies_pair(dataset: Dict) -> Tuple[str, str]:
    for rel in dataset.get("relationships", []):
        if rel.get("relationship_type") == "VERIFIES":
            return rel.get("target_id", ""), rel.get("source_id", "")
    raise AssertionError("No VERIFIES relationship found in dataset")


@pytest.mark.usefixtures("embedding_available", "postgres_client")
def test_fixed_rag_e2e(postgres_client, dataset) -> None:
    requirement_id, _ = _first_verifies_pair(dataset)
    tool = create_hybrid_search_tool(postgres_client=postgres_client)

    ids = run_fixed_rag(query=requirement_id, hybrid_tool=tool, k=10)

    assert requirement_id in ids


@pytest.mark.usefixtures("embedding_available", "postgres_client", "neo4j_client")
def test_fixed_graphrag_e2e(postgres_client, neo4j_client, dataset) -> None:
    requirement_id, test_case_id = _first_verifies_pair(dataset)
    hybrid_tool = create_hybrid_search_tool(postgres_client=postgres_client)
    graph_tool = create_graph_traverse_tool(neo4j_client=neo4j_client)

    ids = run_fixed_graphrag(
        query=requirement_id,
        hybrid_tool=hybrid_tool,
        graph_tool=graph_tool,
        k=10,
    )

    assert requirement_id in ids
    assert test_case_id in ids
