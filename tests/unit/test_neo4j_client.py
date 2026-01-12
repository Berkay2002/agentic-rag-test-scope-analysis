import pytest

from agrag.storage.neo4j_client import Neo4jClient


def test_neo4j_setup_schema_fails_fast_on_no_connectivity(monkeypatch) -> None:
    client = Neo4jClient(uri="neo4j+s://example.invalid", username="neo4j", password="x")
    monkeypatch.setattr(client, "verify_connectivity", lambda: False)

    with pytest.raises(ConnectionError, match="connectivity check failed"):
        client.setup_schema()
