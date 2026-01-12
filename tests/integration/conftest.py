import json
from pathlib import Path

import pytest

from agrag.config import settings
from agrag.storage import Neo4jClient, PostgresClient

DATASET_PATH = Path("data/synthetic_dataset.json")


def _skip_if_missing_dataset() -> None:
    if not DATASET_PATH.exists():
        pytest.skip("Synthetic dataset not found at data/synthetic_dataset.json")


@pytest.fixture(scope="session")
def dataset() -> dict:
    _skip_if_missing_dataset()
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def postgres_client() -> PostgresClient:
    try:
        client = PostgresClient()
    except ValueError as exc:
        pytest.skip(f"PostgreSQL not configured: {exc}")
    if not client.verify_connectivity():
        pytest.skip("PostgreSQL not reachable")
    yield client
    client.close()


@pytest.fixture(scope="session")
def neo4j_client() -> Neo4jClient:
    try:
        client = Neo4jClient()
    except ValueError as exc:
        pytest.skip(f"Neo4j not configured: {exc}")
    if not client.verify_connectivity():
        pytest.skip("Neo4j not reachable")
    yield client
    client.close()


@pytest.fixture(scope="session")
def embedding_available() -> None:
    if not settings.google_api_key:
        pytest.skip("GOOGLE_API_KEY not configured for embeddings")
