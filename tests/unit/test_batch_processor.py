"""Unit tests for batch processing functionality."""

import json
import csv
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock

from agrag.cli.batch_processor import (
    BatchQueryProcessor,
    QueryResult,
    load_queries_from_file,
    save_results_to_file,
)


class TestLoadQueriesFromFile:
    """Test loading queries from various file formats."""

    def test_load_queries_json_array_of_objects(self):
        """Test loading queries from JSON file with array of objects."""
        queries = [
            {"query": "test query 1", "metadata": "value1"},
            {"query": "test query 2", "difficulty": "hard"},
            {"query": "test query 3"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(queries, f)
            f.flush()

            loaded = load_queries_from_file(f.name)

            assert len(loaded) == 3
            assert loaded[0]["query"] == "test query 1"
            assert loaded[0]["metadata"] == "value1"
            assert loaded[1]["query"] == "test query 2"
            assert loaded[1]["difficulty"] == "hard"
            assert loaded[2] == "test query 3"  # Simple string

            f.close()
            Path(f.name).unlink()

    def test_load_queries_json_array_of_strings(self):
        """Test loading queries from JSON file with array of strings."""
        queries = ["query 1", "query 2", "query 3"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(queries, f)
            f.flush()

            loaded = load_queries_from_file(f.name)

            assert len(loaded) == 3
            assert loaded == queries

            f.close()
            Path(f.name).unlink()

    def test_load_queries_json_invalid_format(self):
        """Test error handling for invalid JSON format."""
        data = {"not": "a list"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            with pytest.raises(ValueError, match="JSON file must contain a list of queries"):
                load_queries_from_file(f.name)

            f.close()
            Path(f.name).unlink()

    def test_load_queries_txt(self):
        """Test loading queries from text file."""
        content = "query1\nquery2\nquery3\n\nquery4\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            f.flush()

            loaded = load_queries_from_file(f.name)

            assert len(loaded) == 4
            assert loaded == ["query1", "query2", "query3", "query4"]

            f.close()
            Path(f.name).unlink()

    def test_load_queries_csv_with_query_column(self):
        """Test loading queries from CSV file with query column."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["query", "difficulty", "category"])
            writer.writerow(["test query 1", "easy", "auth"])
            writer.writerow(["test query 2", "hard", "handover"])
            writer.writerow(["", "", ""])  # Empty row
            f.flush()

            loaded = load_queries_from_file(f.name)

            assert len(loaded) == 2
            assert loaded[0]["query"] == "test query 1"
            assert loaded[0]["difficulty"] == "easy"
            assert loaded[0]["category"] == "auth"
            assert loaded[1]["query"] == "test query 2"
            assert loaded[1]["difficulty"] == "hard"

            f.close()
            Path(f.name).unlink()

    def test_load_queries_csv_without_query_column(self):
        """Test loading queries from CSV file without query column."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["question", "difficulty"])
            writer.writerow(["test question 1", "easy"])
            writer.writerow(["test question 2", "hard"])
            f.flush()

            loaded = load_queries_from_file(f.name)

            assert len(loaded) == 2
            # When no "query" column exists, it includes metadata for other columns
            assert loaded[0]["query"] == "test question 1"
            assert loaded[0]["difficulty"] == "easy"
            assert loaded[1]["query"] == "test question 2"
            assert loaded[1]["difficulty"] == "hard"

            f.close()
            Path(f.name).unlink()

    def test_load_queries_csv_empty_file(self):
        """Test error handling for empty CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("")
            f.flush()

            with pytest.raises(ValueError, match="CSV file is empty or has no columns"):
                load_queries_from_file(f.name)

            f.close()
            Path(f.name).unlink()

    def test_load_queries_unsupported_format(self):
        """Test error handling for unsupported file format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("test")
            f.flush()

            with pytest.raises(ValueError, match="Unsupported file format"):
                load_queries_from_file(f.name)

            f.close()
            Path(f.name).unlink()

    def test_load_queries_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError, match="Query file not found"):
            load_queries_from_file("/nonexistent/file.json")

    def test_load_queries_invalid_json_items(self):
        """Test handling of invalid items in JSON array."""
        queries = [
            "valid query 1",
            {"query": "valid query 2"},
            {"no_query_field": "invalid"},
            123,  # Invalid type
            "valid query 3",
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(queries, f)
            f.flush()

            loaded = load_queries_from_file(f.name)

            # Should only load valid items
            assert len(loaded) == 3
            assert loaded[0] == "valid query 1"
            assert loaded[1] == "valid query 2"  # Simple string since only has query field
            assert loaded[2] == "valid query 3"

            f.close()
            Path(f.name).unlink()


class TestSaveResultsToFile:
    """Test saving results to various file formats."""

    def test_save_results_json(self):
        """Test saving results in JSON format."""
        results = [
            QueryResult(
                query="test query 1",
                response="response 1",
                metadata={"param": "value1"},
                timestamp="2024-01-01T10:00:00",
                status="success",
                execution_time_ms=100.0,
            ),
            QueryResult(
                query="test query 2",
                response="response 2",
                metadata={"param": "value2"},
                timestamp="2024-01-01T10:01:00",
                status="error",
                error_message="Test error",
                execution_time_ms=50.0,
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_results_to_file(results, f.name, format="json")

            # Read back and verify
            with open(f.name, "r") as rf:
                data = json.load(rf)

            assert "metadata" in data
            assert data["metadata"]["total_queries"] == 2
            assert data["metadata"]["format"] == "json"
            assert "results" in data
            assert len(data["results"]) == 2

            # Check first result
            assert data["results"][0]["query"] == "test query 1"
            assert data["results"][0]["response"] == "response 1"
            assert data["results"][0]["status"] == "success"
            assert data["results"][0]["execution_time_ms"] == 100.0

            # Check second result
            assert data["results"][1]["query"] == "test query 2"
            assert data["results"][1]["status"] == "error"
            assert data["results"][1]["error_message"] == "Test error"

            f.close()
            Path(f.name).unlink()

    def test_save_results_jsonl(self):
        """Test saving results in JSONL format."""
        results = [
            QueryResult(
                query="test query 1",
                response="response 1",
                metadata={"param": "value1"},
                timestamp="2024-01-01T10:00:00",
                status="success",
            ),
            QueryResult(
                query="test query 2",
                response="response 2",
                metadata={"param": "value2"},
                timestamp="2024-01-01T10:01:00",
                status="error",
                error_message="Test error",
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            save_results_to_file(results, f.name, format="jsonl")

            # Read back and verify
            with open(f.name, "r") as rf:
                lines = rf.readlines()

            assert len(lines) == 2

            # Parse first line
            result1 = json.loads(lines[0])
            assert result1["query"] == "test query 1"
            assert result1["response"] == "response 1"
            assert result1["status"] == "success"

            # Parse second line
            result2 = json.loads(lines[1])
            assert result2["query"] == "test query 2"
            assert result2["status"] == "error"
            assert result2["error_message"] == "Test error"

            f.close()
            Path(f.name).unlink()

    def test_save_results_empty_list(self):
        """Test saving empty results list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_results_to_file([], f.name, format="json")

            # Read back and verify
            with open(f.name, "r") as rf:
                data = json.load(rf)

            assert data["metadata"]["total_queries"] == 0
            assert data["results"] == []

            f.close()
            Path(f.name).unlink()

    def test_save_results_unsupported_format(self):
        """Test error handling for unsupported format."""
        results = [
            QueryResult(
                query="test",
                response="response",
                metadata={},
                timestamp="2024-01-01T10:00:00",
                status="success",
            )
        ]

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            with pytest.raises(ValueError, match="Unsupported output format"):
                save_results_to_file(results, f.name, format="xml")


class TestBatchQueryProcessor:
    """Test the BatchQueryProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a test processor instance."""
        return BatchQueryProcessor(output_format="json", debug=False)

    @pytest.mark.asyncio
    async def test_process_queries_string_list(self, processor):
        """Test processing a list of string queries."""
        queries = ["query 1", "query 2", "query 3"]

        # Mock the _run_query_headless method
        with patch.object(processor, "_run_query_headless", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, "Test response")

            results = await processor.process_queries(queries)

            assert len(results) == 3
            assert all(r.status == "success" for r in results)
            assert all(r.query in queries for r in results)
            assert mock_run.call_count == 3

    @pytest.mark.asyncio
    async def test_process_queries_dict_list(self, processor):
        """Test processing a list of query dictionaries."""
        queries = [
            {"query": "query 1", "difficulty": "easy"},
            {"query": "query 2", "difficulty": "hard"},
        ]

        with patch.object(processor, "_run_query_headless", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, "Test response")

            results = await processor.process_queries(queries)

            assert len(results) == 2
            assert results[0].metadata["difficulty"] == "easy"
            assert results[1].metadata["difficulty"] == "hard"
            assert results[0].query == "query 1"
            assert results[1].query == "query 2"

    @pytest.mark.asyncio
    async def test_process_queries_with_shared_params(self, processor):
        """Test processing with shared parameters."""
        queries = ["query 1", "query 2"]
        shared_params = {"thread_id": "test-thread", "debug": True}

        with patch.object(processor, "_run_query_headless", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, "Test response")

            results = await processor.process_queries(queries, **shared_params)

            # Check that shared params are included in metadata
            assert all(r.metadata["thread_id"] == "test-thread" for r in results)
            assert all(r.metadata["debug"] is True for r in results)

    @pytest.mark.asyncio
    async def test_process_queries_with_query_specific_params(self, processor):
        """Test processing with query-specific parameters."""
        queries = [
            {"query": "query 1", "thread_id": "thread-1"},
            {"query": "query 2", "thread_id": "thread-2", "debug": True},
        ]
        shared_params = {"output_format": "json"}

        with patch.object(processor, "_run_query_headless", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, "Test response")

            results = await processor.process_queries(queries, **shared_params)

            # Check parameter merging
            assert results[0].metadata["thread_id"] == "thread-1"
            assert results[0].metadata["output_format"] == "json"
            assert "debug" not in results[0].metadata

            assert results[1].metadata["thread_id"] == "thread-2"
            assert results[1].metadata["debug"] is True
            assert results[1].metadata["output_format"] == "json"

    @pytest.mark.asyncio
    async def test_process_queries_error_handling(self, processor):
        """Test error handling during query processing."""
        queries = ["query 1", "query 2"]

        with patch.object(processor, "_run_query_headless", new_callable=AsyncMock) as mock_run:
            # First call succeeds, second fails
            mock_run.side_effect = [(0, "Success response"), (1, "Error message")]

            results = await processor.process_queries(queries)

            assert len(results) == 2
            assert results[0].status == "success"
            assert results[0].response == "Success response"
            assert results[1].status == "error"
            assert results[1].error_message == "Error message"

    @pytest.mark.asyncio
    async def test_process_queries_exception_handling(self, processor):
        """Test exception handling during query processing."""
        queries = ["query 1"]

        with patch.object(processor, "_run_query_headless", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = Exception("Unexpected error")

            results = await processor.process_queries(queries)

            assert len(results) == 1
            assert results[0].status == "error"
            assert "Unexpected error" in results[0].error_message

    def test_generate_report_empty_results(self, processor):
        """Test report generation with no results."""
        report = processor.generate_report()

        assert report.total_queries == 0
        assert report.successful_queries == 0
        assert report.failed_queries == 0
        assert report.skipped_queries == 0
        assert report.total_execution_time_ms == 0.0
        assert report.average_execution_time_ms == 0.0

    def test_generate_report_with_results(self, processor):
        """Test report generation with results."""
        processor.results = [
            QueryResult(
                query="query 1",
                response="response 1",
                metadata={},
                timestamp="2024-01-01T10:00:00",
                status="success",
                execution_time_ms=100.0,
            ),
            QueryResult(
                query="query 2",
                response="",
                metadata={},
                timestamp="2024-01-01T10:01:00",
                status="error",
                error_message="Test error",
                execution_time_ms=50.0,
            ),
            QueryResult(
                query="query 3",
                response="response 3",
                metadata={},
                timestamp="2024-01-01T10:02:00",
                status="success",
                execution_time_ms=200.0,
            ),
        ]

        report = processor.generate_report()

        assert report.total_queries == 3
        assert report.successful_queries == 2
        assert report.failed_queries == 1
        assert report.skipped_queries == 0
        assert report.total_execution_time_ms == 350.0
        assert report.average_execution_time_ms == 350.0 / 3  # Average over all queries
        assert report.start_time == "2024-01-01T10:00:00"
        assert report.end_time == "2024-01-01T10:02:00"

    @pytest.mark.asyncio
    async def test_run_query_headless_integration(self, processor):
        """Test the _run_query_headless method."""
        # Test successful execution
        exit_code, response = await processor._run_query_headless(
            "test query", {"output_format": "json", "debug": True}
        )

        assert exit_code == 0
        assert "Simulated response" in response
        assert "test query" in response

    def test_default_parameters(self):
        """Test that default parameters are properly set."""
        processor = BatchQueryProcessor(output_format="jsonl", thread_id="test-thread", debug=True)

        assert processor.output_format == "jsonl"
        assert processor.default_params["thread_id"] == "test-thread"
        assert processor.default_params["debug"] is True

    @pytest.mark.asyncio
    async def test_parameter_precedence(self, processor):
        """Test parameter precedence: default < shared < query-specific."""
        queries = [{"query": "test", "debug": False}]  # Query-specific

        processor.default_params["debug"] = True  # Default
        shared_params = {"thread_id": "shared"}  # Shared

        with patch.object(processor, "_run_query_headless", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, "Test response")

            results = await processor.process_queries(queries, **shared_params)

            # Query-specific should override
            assert results[0].metadata["debug"] is False
            # Shared should be included
            assert results[0].metadata["thread_id"] == "shared"


class TestBatchProcessingIntegration:
    """Integration tests for batch processing workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow_json_to_json(self):
        """Test complete workflow: JSON input to JSON output."""
        # Create input file
        input_queries = [
            {"query": "What tests cover REQ_AUTH_001?", "difficulty": "easy"},
            {"query": "Find functions related to handover", "difficulty": "medium"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as input_file:
            json.dump(input_queries, input_file)
            input_file.flush()

            # Load queries
            queries = load_queries_from_file(input_file.name)
            assert len(queries) == 2

            # Process queries
            processor = BatchQueryProcessor(output_format="json")

            with patch.object(processor, "_run_query_headless", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (0, "Test response")

                results = await processor.process_queries(queries)

                assert len(results) == 2
                assert all(r.status == "success" for r in results)

            # Generate report
            report = processor.generate_report()
            assert report.total_queries == 2
            assert report.successful_queries == 2

            # Save results
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as output_file:
                save_results_to_file(results, output_file.name, format="json")

                # Verify saved results
                with open(output_file.name, "r") as f:
                    data = json.load(f)

                assert data["metadata"]["total_queries"] == 2
                assert len(data["results"]) == 2

                output_file.close()
                Path(output_file.name).unlink()

            input_file.close()
            Path(input_file.name).unlink()

    @pytest.mark.asyncio
    async def test_full_workflow_txt_to_jsonl(self):
        """Test complete workflow: TXT input to JSONL output."""
        # Create input file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as input_file:
            input_file.write("Query 1\nQuery 2\nQuery 3\n")
            input_file.flush()

            # Load queries
            queries = load_queries_from_file(input_file.name)
            assert len(queries) == 3

            # Process queries
            processor = BatchQueryProcessor()

            with patch.object(processor, "_run_query_headless", new_callable=AsyncMock) as mock_run:
                # Make one query fail
                mock_run.side_effect = [(0, "Response 1"), (1, "Error message"), (0, "Response 3")]

                results = await processor.process_queries(queries)

                assert len(results) == 3
                assert results[0].status == "success"
                assert results[1].status == "error"
                assert results[2].status == "success"

            # Save results
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as output_file:
                save_results_to_file(results, output_file.name, format="jsonl")

                # Verify saved results
                with open(output_file.name, "r") as f:
                    lines = f.readlines()

                assert len(lines) == 3

                # Check individual results
                result_data = [json.loads(line) for line in lines]
                assert result_data[0]["status"] == "success"
                assert result_data[1]["status"] == "error"
                assert result_data[2]["status"] == "success"

                output_file.close()
                Path(output_file.name).unlink()

            input_file.close()
            Path(input_file.name).unlink()

    def test_csv_with_special_characters(self):
        """Test CSV loading with special characters and quotes."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["query", "description"])
            writer.writerow(["query with, comma", 'description with "quotes"'])
            writer.writerow(["query with\nnewline", "normal description"])
            f.flush()

            loaded = load_queries_from_file(f.name)

            assert len(loaded) == 2
            assert loaded[0]["query"] == "query with, comma"
            assert loaded[0]["description"] == 'description with "quotes"'
            assert loaded[1]["query"] == "query with\nnewline"

            f.close()
            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_large_batch_processing(self):
        """Test processing a large batch of queries."""
        # Create many queries
        queries = [f"Query {i}" for i in range(100)]

        processor = BatchQueryProcessor()

        with patch.object(processor, "_run_query_headless", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, "Bulk response")

            results = await processor.process_queries(queries)

            assert len(results) == 100
            assert mock_run.call_count == 100
            assert all(r.status == "success" for r in results)

        # Check report
        report = processor.generate_report()
        assert report.total_queries == 100
        assert report.successful_queries == 100
        assert report.failed_queries == 0
