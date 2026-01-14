"""Unit tests for storage_writers base functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from agrag.data.storage_writers import BaseWriter, GraphWriter, PostgresWriter, BM25Writer


class ConcreteWriter(BaseWriter):
    """Concrete implementation of BaseWriter for testing."""

    def __init__(self):
        super().__init__()
        self.write_calls = []

    def _write_entity_impl(self, entity: Dict[str, Any], entity_type: str) -> bool:
        """Store call for inspection."""
        self.write_calls.append((entity, entity_type))
        # Simulate success if entity has 'success' key set to True
        return entity.get("success", True)


class TestBaseWriter:
    """Tests for BaseWriter base class."""

    def test_init_initializes_stats(self):
        """Test that initialization creates stats dict."""
        writer = ConcreteWriter()
        assert "writes" in writer.stats
        assert "failures" in writer.stats
        assert writer.stats["writes"] == 0
        assert writer.stats["failures"] == 0

    def test_write_entity_success(self):
        """Test successful entity write."""
        writer = ConcreteWriter()
        entity = {"id": "TEST_001", "name": "Test", "success": True}

        result = writer.write_entity(entity, "TestCase")

        assert result is True
        assert len(writer.write_calls) == 1
        assert writer.write_calls[0] == (entity, "TestCase")
        assert writer.stats["failures"] == 0

    def test_write_entity_failure(self):
        """Test entity write with implementation returning False."""
        writer = ConcreteWriter()
        entity = {"id": "TEST_001", "name": "Test", "success": False}

        result = writer.write_entity(entity, "TestCase")

        assert result is False
        assert writer.stats["failures"] == 0  # Not an exception failure

    def test_write_entity_exception(self):
        """Test entity write with exception."""
        writer = ConcreteWriter()
        # Make _write_entity_impl raise an exception
        writer._write_entity_impl = Mock(side_effect=ValueError("Test error"))
        entity = {"id": "TEST_001", "name": "Test"}

        result = writer.write_entity(entity, "TestCase")

        assert result is False
        assert writer.stats["failures"] == 1

    def test_write_entities_batch(self):
        """Test batch write of multiple entities."""
        writer = ConcreteWriter()
        entities = [
            {"id": "TEST_001", "success": True},
            {"id": "TEST_002", "success": True},
            {"id": "TEST_003", "success": False},
            {"id": "TEST_004", "success": True},
        ]

        successes = writer.write_entities_batch(entities, "TestCase", batch_size=2)

        assert successes == 3  # 3 succeeded, 1 failed
        assert len(writer.write_calls) == 4

    def test_write_entities_batch_with_exceptions(self):
        """Test batch write with some exceptions."""
        writer = ConcreteWriter()
        entities = [
            {"id": "TEST_001", "success": True},
            {"id": "TEST_002", "success": True},
        ]

        # Make second write fail with exception
        original_impl = writer._write_entity_impl

        def selective_fail(entity, entity_type):
            if entity["id"] == "TEST_002":
                raise ValueError("Test error")
            return original_impl(entity, entity_type)

        writer._write_entity_impl = selective_fail

        successes = writer.write_entities_batch(entities, "TestCase")

        assert successes == 1  # Only first succeeded
        assert writer.stats["failures"] == 1


class TestGraphWriter:
    """Tests for GraphWriter."""

    @patch("agrag.data.storage_writers.Neo4jClient")
    def test_init_creates_client(self, mock_neo4j_class):
        """Test GraphWriter initializes with Neo4j client."""
        writer = GraphWriter()

        assert writer.neo4j_client is not None
        assert "writes" in writer.stats
        assert "failures" in writer.stats

    @patch("agrag.data.storage_writers.Neo4jClient")
    def test_write_entity_removes_embedding(self, mock_neo4j_class):
        """Test that embeddings are removed before writing to Neo4j."""
        mock_client = MagicMock()
        mock_client.execute_cypher.return_value = [{"id": "TEST_001"}]
        writer = GraphWriter(neo4j_client=mock_client)

        entity = {"id": "TEST_001", "name": "Test", "embedding": [0.1, 0.2, 0.3]}

        result = writer.write_entity(entity, "TestCase")

        assert result is True
        # Check that execute_cypher was called
        assert mock_client.execute_cypher.called


class TestPostgresWriter:
    """Tests for PostgresWriter."""

    @patch("agrag.data.storage_writers.PostgresClient")
    def test_init_creates_client(self, mock_postgres_class):
        """Test PostgresWriter initializes with Postgres client."""
        writer = PostgresWriter()

        assert writer.postgres_client is not None
        assert "writes" in writer.stats
        assert "skipped" in writer.stats
        assert "failures" in writer.stats

    @patch("agrag.data.storage_writers.PostgresClient")
    def test_write_entity_skips_without_embedding(self, mock_postgres_class):
        """Test that entities without embeddings are skipped."""
        mock_client = MagicMock()
        writer = PostgresWriter(postgres_client=mock_client)

        entity = {"id": "TEST_001", "name": "Test"}  # No embedding

        result = writer.write_entity(entity, "TestCase")

        assert result is False
        assert writer.stats["skipped"] == 1
        assert not mock_client.insert_document_chunk.called

    @patch("agrag.data.storage_writers.PostgresClient")
    def test_build_content(self, mock_postgres_class):
        """Test content building from entity fields."""
        writer = PostgresWriter()

        entity = {
            "id": "TEST_001",
            "title": "Test Title",
            "name": "Test Name",
            "description": "Test Description",
        }

        content = writer._build_content(entity)

        assert "TEST_001" in content
        assert "Test Title" in content
        assert "Test Name" in content
        assert "Test Description" in content

    @patch("agrag.data.storage_writers.PostgresClient")
    def test_build_metadata(self, mock_postgres_class):
        """Test metadata building from entity fields."""
        writer = PostgresWriter()

        entity = {
            "id": "TEST_001",
            "file_path": "/path/to/file.py",
            "priority": "high",
            "status": "active",
        }

        metadata = writer._build_metadata(entity, "TestCase")

        assert metadata["entity_type"] == "TestCase"
        assert metadata["entity_id"] == "TEST_001"
        assert metadata["file_path"] == "/path/to/file.py"
        assert metadata["priority"] == "high"
        assert metadata["status"] == "active"


class TestBM25Writer:
    """Tests for BM25Writer."""

    @patch("agrag.data.storage_writers.BM25RetrieverManager")
    def test_init_creates_manager(self, mock_bm25_class):
        """Test BM25Writer initializes with BM25 manager."""
        writer = BM25Writer()

        assert writer.bm25_manager is not None
        assert writer.index_path == "data/bm25_index.pkl"
        assert "writes" in writer.stats
        assert "failures" in writer.stats

    @patch("agrag.data.storage_writers.BM25RetrieverManager")
    def test_build_content_includes_tags(self, mock_bm25_class):
        """Test that content building includes tags."""
        writer = BM25Writer()

        entity = {
            "id": "TEST_001",
            "name": "Test Name",
            "tags": ["tag1", "tag2", "tag3"],
        }

        content = writer._build_content(entity)

        assert "TEST_001" in content
        assert "Test Name" in content
        assert "tag1" in content
        assert "tag2" in content
        assert "tag3" in content

    @patch("agrag.data.storage_writers.BM25RetrieverManager")
    def test_write_entity_skips_empty_content(self, mock_bm25_class):
        """Test that entities with empty content are skipped."""
        mock_manager = MagicMock()
        writer = BM25Writer(bm25_manager=mock_manager)

        entity = {}  # Empty entity

        result = writer.write_entity(entity, "TestCase")

        assert result is False
        assert not mock_manager.add_texts.called

    @patch("agrag.data.storage_writers.BM25RetrieverManager")
    def test_persist_index(self, mock_bm25_class):
        """Test index persistence."""
        mock_manager = MagicMock()
        writer = BM25Writer(bm25_manager=mock_manager)

        writer.persist_index()

        assert mock_manager.save.called
        mock_manager.save.assert_called_once_with("data/bm25_index.pkl")
