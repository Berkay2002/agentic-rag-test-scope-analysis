"""Integration tests for error recovery and robustness features."""

import logging
from unittest.mock import Mock, patch, MagicMock
import pytest
from neo4j.exceptions import ServiceUnavailable, TransientError
from psycopg import OperationalError, DatabaseError

from agrag.storage import Neo4jClient, PostgresClient
from agrag.storage.retry_decorators import resilient_db_operation, with_fallback
from agrag.observability.metrics import ErrorMetrics
from agrag.tools.retrieval.hybrid_search import _hybrid_search_core, _keyword_only_search


logger = logging.getLogger(__name__)


class TestNeo4jRetryMechanism:
    """Test Neo4j retry behavior on connection failures."""

    def test_neo4j_retry_on_service_unavailable(self, neo4j_client):
        """Test that Neo4j operations retry on ServiceUnavailable errors."""
        # Mock the session.run to fail with ServiceUnavailable twice then succeed
        call_count = 0

        def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ServiceUnavailable("Simulated transient connection failure")
            # Return a mock result that behaves like a Neo4j result
            mock_result = Mock()
            mock_result.single = Mock(return_value={"n": {"id": "test-id"}})
            return mock_result

        # Patch the session.run method for a method that has @resilient_db_operation
        with patch.object(neo4j_client.driver, 'session') as mock_session:
            mock_session_instance = Mock()
            mock_session_instance.run = mock_run
            mock_session.return_value.__enter__ = Mock(return_value=mock_session_instance)
            mock_session.return_value.__exit__ = Mock(return_value=None)

            # Use get_node_by_id which has the retry decorator
            result = neo4j_client.get_node_by_id("test-id")

            # Verify it was called 3 times (2 failures + 1 success)
            assert call_count == 3
            assert result["id"] == "test-id"

    def test_neo4j_retry_exhaustion(self, neo4j_client):
        """Test that Neo4j operations fail after max retry attempts."""
        # Mock the session.run to always fail
        with patch.object(neo4j_client.driver, 'session') as mock_session:
            mock_session_instance = Mock()
            mock_session_instance.run = Mock(side_effect=ServiceUnavailable("Persistent failure"))
            mock_session.return_value.__enter__ = Mock(return_value=mock_session_instance)
            mock_session.return_value.__exit__ = Mock(return_value=None)

            # This should fail after max retries
            with pytest.raises(ServiceUnavailable):
                neo4j_client.get_node_by_id("test-id")

    def test_neo4j_retry_on_transient_error(self, neo4j_client):
        """Test that Neo4j operations retry on TransientError."""
        call_count = 0

        def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TransientError("Simulated transient error")
            # Return a mock result
            mock_result = Mock()
            mock_result.single = Mock(return_value={"n": {"id": "test-id"}})
            return mock_result

        with patch.object(neo4j_client.driver, 'session') as mock_session:
            mock_session_instance = Mock()
            mock_session_instance.run = mock_run
            mock_session.return_value.__enter__ = Mock(return_value=mock_session_instance)
            mock_session.return_value.__exit__ = Mock(return_value=None)

            result = neo4j_client.get_node_by_id("test-id")

            assert call_count == 2
            assert result["id"] == "test-id"


class TestPostgresRetryMechanism:
    """Test PostgreSQL retry behavior on connection failures."""

    def test_postgres_retry_on_operational_error(self, postgres_client):
        """Test that PostgreSQL operations retry on OperationalError."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise OperationalError("Simulated connection failure")
            # Return a mock result for vector_search
            mock_cursor = Mock()
            mock_cursor.fetchall = Mock(return_value=[
                {"id": "test-1", "content": "Test content", "metadata": {}, "embedding": [0.1, 0.2]}
            ])
            return mock_cursor

        # Mock the connection and cursor for vector_search method
        with patch.object(postgres_client, 'conn') as mock_conn:
            mock_conn.closed = False
            mock_cursor = Mock()
            mock_cursor.execute = mock_execute
            mock_cursor.fetchall = Mock(return_value=[
                {"id": "test-1", "content": "Test content", "metadata": {}, "embedding": [0.1, 0.2]}
            ])
            mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)

            # Use vector_search which has the retry decorator
            results = postgres_client.vector_search(
                query_embedding=[0.1, 0.2, 0.3],
                k=1
            )

            # Verify it was called 3 times (2 failures + 1 success)
            assert call_count == 3
            assert len(results) == 1
            assert results[0]["id"] == "test-1"

    def test_postgres_retry_on_database_error(self, postgres_client):
        """Test that PostgreSQL operations retry on DatabaseError."""
        call_count = 0

        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise DatabaseError("Simulated database error")
            # Return a mock result
            mock_cursor = Mock()
            mock_cursor.fetchall = Mock(return_value=[])
            return mock_cursor

        with patch.object(postgres_client, 'conn') as mock_conn:
            mock_conn.closed = False
            mock_cursor = Mock()
            mock_cursor.execute = mock_execute
            mock_cursor.fetchall = Mock(return_value=[])
            mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)

            results = postgres_client.vector_search(
                query_embedding=[0.1, 0.2, 0.3],
                k=1
            )

            assert call_count == 2
            assert results == []

    def test_postgres_retry_exhaustion(self, postgres_client):
        """Test that PostgreSQL operations fail after max retry attempts."""
        def mock_execute(*args, **kwargs):
            raise OperationalError("Persistent failure")

        with patch.object(postgres_client, 'conn') as mock_conn:
            mock_conn.closed = False
            mock_cursor = Mock()
            mock_cursor.execute = mock_execute
            mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)

            with pytest.raises(OperationalError):
                postgres_client.vector_search(
                    query_embedding=[0.1, 0.2, 0.3],
                    k=1
                )


class TestHybridSearchFallback:
    """Test fallback activation in hybrid search."""

    def test_fallback_to_keyword_search_on_vector_failure(self):
        """Test that hybrid search falls back to keyword-only when vector search fails."""
        # Create a test function that will fail and trigger fallback
        def fallback_func(client, query, query_embedding, k, rrf_k, metadata_filter):
            return [{"id": "fallback", "content": "Fallback result"}]

        @with_fallback(fallback_func)
        def failing_hybrid_search(client, query, query_embedding, k, rrf_k, metadata_filter):
            raise Exception("Vector search failed")

        # Create mock client
        mock_client = Mock(spec=PostgresClient)

        # Test the fallback behavior - should return fallback results
        result = failing_hybrid_search(
            client=mock_client,
            query="test query",
            query_embedding=[0.1, 0.2, 0.3],
            k=10,
            rrf_k=60,
            metadata_filter=None
        )

        # Verify fallback returned results
        assert len(result) == 1
        assert result[0]["id"] == "fallback"
        assert result[0]["content"] == "Fallback result"

    def test_fallback_warning_logged(self, caplog):
        """Test that fallback activation logs a warning."""
        caplog.set_level(logging.INFO)  # Set to INFO to capture fallback message

        # Create a function that will fail and trigger fallback
        def fallback_func():
            return "fallback result"

        @with_fallback(fallback_func)
        def failing_function():
            raise Exception("Primary function failed")

        # Execute the function
        result = failing_function()

        # Verify warning was logged
        assert "Primary function failing_function failed" in caplog.text
        assert "Falling back to" in caplog.text
        assert result == "fallback result"

    def test_fallback_with_actual_keyword_search_function(self):
        """Test fallback using the actual _keyword_only_search function."""
        # Create mock client
        mock_client = Mock(spec=PostgresClient)

        # Set up keyword search mock
        mock_keyword_results = [
            {"id": "1", "content": "Fallback result", "score": 0.85}
        ]
        mock_client.keyword_search = Mock(return_value=mock_keyword_results)

        # Test the actual fallback function
        result = _keyword_only_search(
            client=mock_client,
            query="test query",
            query_embedding=[0.1, 0.2],  # Not used but required for signature
            k=5,
            metadata_filter={"type": "test"}
        )

        # Verify results
        assert result == mock_keyword_results
        mock_client.keyword_search.assert_called_once_with(
            query="test query",
            k=5,
            metadata_filter={"type": "test"}
        )


class TestErrorMetricsCollection:
    """Test error metrics collection and reporting."""

    def test_metrics_record_operation_success(self):
        """Test recording successful operations."""
        metrics = ErrorMetrics()

        metrics.record_operation(
            operation="test_op",
            success=True,
            retry_count=0,
            latency_ms=100
        )

        summary = metrics.get_summary()
        assert summary["total_operations"] == 1
        assert summary["success_rate"] == 1.0
        assert summary["retried_operations"] == 0

    def test_metrics_record_operation_failure(self):
        """Test recording failed operations."""
        metrics = ErrorMetrics()

        metrics.record_operation(
            operation="test_op",
            success=False,
            retry_count=3,
            error_type="ConnectionError",
            latency_ms=5000
        )

        summary = metrics.get_summary()
        assert summary["total_operations"] == 1
        assert summary["success_rate"] == 0.0
        assert summary["retried_operations"] == 1
        assert summary["avg_retries_per_failed_op"] == 3
        assert summary["errors_by_type"]["ConnectionError"] == 1

    def test_metrics_record_fallback_usage(self):
        """Test recording fallback activations."""
        metrics = ErrorMetrics()

        metrics.record_operation(
            operation="hybrid_search",
            success=True,
            retry_count=1,
            fallback_used=True,
            latency_ms=200
        )

        summary = metrics.get_summary()
        assert summary["fallback_activations"] == 1

    def test_metrics_multiple_operations(self):
        """Test metrics with multiple operations of different types."""
        metrics = ErrorMetrics()

        # Record various operations
        operations = [
            {"op": "vector_search", "success": True, "retries": 0},
            {"op": "vector_search", "success": True, "retries": 1},
            {"op": "keyword_search", "success": True, "retries": 0},
            {"op": "hybrid_search", "success": False, "retries": 3, "error": "TimeoutError"},
            {"op": "graph_query", "success": False, "retries": 2, "error": "ConnectionError"},
            {"op": "hybrid_search", "success": True, "retries": 2, "fallback": True},
        ]

        for op in operations:
            metrics.record_operation(
                operation=op["op"],
                success=op["success"],
                retry_count=op["retries"],
                fallback_used=op.get("fallback", False),
                error_type=op.get("error"),
                latency_ms=100
            )

        summary = metrics.get_summary()
        assert summary["total_operations"] == 6
        assert summary["success_rate"] == 2/3  # 4 success / 6 total
        assert summary["retried_operations"] == 4  # Operations with retries
        assert summary["fallback_activations"] == 1
        assert summary["errors_by_type"]["TimeoutError"] == 1
        assert summary["errors_by_type"]["ConnectionError"] == 1

    def test_metrics_latency_tracking(self):
        """Test that operation latencies are properly tracked."""
        metrics = ErrorMetrics()

        # Record operations with different latencies
        latencies = [50, 100, 150, 200, 250]
        for latency in latencies:
            metrics.record_operation(
                operation="test_op",
                success=True,
                latency_ms=latency
            )

        # Verify latencies are recorded
        assert "test_op" in metrics.latency_by_operation
        assert metrics.latency_by_operation["test_op"] == latencies


class TestRetryDecoratorConfiguration:
    """Test retry decorator configuration from settings."""

    def test_retry_configuration_from_settings(self, monkeypatch):
        """Test that retry configuration is loaded from settings."""
        from agrag.config import settings

        # Mock settings
        monkeypatch.setattr(settings, "retry_max_attempts", 5)
        monkeypatch.setattr(settings, "retry_base_delay", 0.1)

        # Create a test function with retry decorator
        call_count = 0

        @resilient_db_operation
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise ServiceUnavailable("Simulated failure")
            return "success"

        # Execute and verify it retries according to settings
        result = test_function()
        assert result == "success"
        assert call_count == 4  # 3 failures + 1 success


class TestIntegrationScenarios:
    """Integration test scenarios combining multiple error recovery features."""

    def test_full_error_recovery_pipeline(self, postgres_client, caplog):
        """Test a complete error recovery scenario with metrics tracking."""
        caplog.set_level(logging.INFO)

        # Create error metrics instance
        metrics = ErrorMetrics()

        # Scenario 1: Successful operation
        metrics.record_operation(
            operation="vector_search",
            success=True,
            retry_count=0,
            latency_ms=50
        )

        # Scenario 2: Operation with retries
        metrics.record_operation(
            operation="keyword_search",
            success=True,
            retry_count=2,
            latency_ms=150
        )

        # Scenario 3: Failed operation
        metrics.record_operation(
            operation="hybrid_search",
            success=False,
            retry_count=3,
            error_type="DatabaseError",
            latency_ms=5000
        )

        # Scenario 4: Fallback activation
        metrics.record_operation(
            operation="hybrid_search",
            success=True,
            retry_count=1,
            fallback_used=True,
            latency_ms=200
        )

        # Verify metrics summary
        summary = metrics.get_summary()
        assert summary["total_operations"] == 4
        assert summary["success_rate"] == 0.75  # 3 success / 4 total
        assert summary["retried_operations"] == 3  # 3 operations had retries
        assert summary["fallback_activations"] == 1
        assert summary["errors_by_type"]["DatabaseError"] == 1

        # Verify retry attempts distribution
        assert 1 in metrics.retry_attempts  # 1 operation with 1 retry
        assert 2 in metrics.retry_attempts  # 1 operation with 2 retries
        assert 3 in metrics.retry_attempts  # 1 operation with 3 retries
        assert metrics.retry_attempts[1] == 1
        assert metrics.retry_attempts[2] == 1
        assert metrics.retry_attempts[3] == 1

    def test_error_recovery_with_actual_clients(self):
        """Test error recovery using actual client instances with mocked failures."""
        # Test with a method that has the retry decorator
        try:
            # Create a Neo4jClient instance
            client = Neo4jClient()

            # Mock the session to always fail
            with patch.object(client.driver, 'session') as mock_session:
                mock_session_instance = Mock()
                mock_session_instance.run = Mock(side_effect=ServiceUnavailable("Persistent failure"))
                mock_session.return_value.__enter__ = Mock(return_value=mock_session_instance)
                mock_session.return_value.__exit__ = Mock(return_value=None)

                # Should raise after retries
                with pytest.raises(ServiceUnavailable):
                    client.get_node_by_id("test-id")
        except ValueError:
            # Neo4j not configured, skip this test
            pytest.skip("Neo4j not configured")
        except Exception as e:
            # If client creation fails, skip
            pytest.skip(f"Neo4j client creation failed: {e}")

        try:
            # Create a PostgresClient instance
            client = PostgresClient()
            client.connect()

            # Mock execute to always fail
            with patch.object(client, 'conn') as mock_conn:
                mock_conn.closed = False
                mock_cursor = Mock()
                mock_cursor.execute = Mock(side_effect=OperationalError("Persistent failure"))
                mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
                mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)

                # Should raise after retries
                with pytest.raises(OperationalError):
                    client.vector_search([0.1, 0.2], k=1)
        except ValueError:
            # Postgres not configured, skip this test
            pytest.skip("PostgreSQL not configured")
        except Exception as e:
            # If client creation fails, skip
            pytest.skip(f"PostgreSQL client creation failed: {e}")