from __future__ import annotations

from typing import Any, Dict, List, Optional

from agrag.kg.ontology import NodeLabel
from agrag.tools.retrieval import vector_search as vector_search_module
from agrag.tools.retrieval.vector_search import create_vector_search_tool


class _StubEmbeddingService:
    def embed_query(self, text: str) -> List[float]:
        return [0.0] * 768


class _StubPostgresClient:
    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results
        self.last_metadata_filter: Optional[Dict[str, Any]] = None

    def vector_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        self.last_metadata_filter = metadata_filter
        return self.results[:k]


def test_vector_search_uses_node_label_value_and_threshold(monkeypatch) -> None:
    monkeypatch.setattr(
        vector_search_module,
        "get_embedding_service",
        lambda: _StubEmbeddingService(),
    )

    client = _StubPostgresClient(
        results=[
            {"chunk_id": "c1", "content": "low", "similarity": 0.4, "metadata": None},
            {
                "chunk_id": "c2",
                "content": "high",
                "similarity": 0.9,
                "metadata": {"entity_id": "REQ_1", "entity_type": "Requirement"},
            },
        ]
    )

    tool = create_vector_search_tool(postgres_client=client)
    output = tool.invoke(
        {
            "query": "anything",
            "k": 10,
            "node_type": NodeLabel.REQUIREMENT,
            "similarity_threshold": 0.5,
        }
    )

    assert client.last_metadata_filter == {"entity_type": NodeLabel.REQUIREMENT.value}
    assert "ID: REQ_1" in output
    assert "ID: c1" not in output
