"""Batch processing for multiple queries."""

import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result from processing a single query."""

    query: str
    response: str
    metadata: Dict[str, Any]
    timestamp: str
    status: str  # "success", "error", "skipped"
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None
    tool_calls: Optional[List[str]] = None


@dataclass
class BatchProcessingReport:
    """Report summarizing batch processing results."""

    total_queries: int
    successful_queries: int
    failed_queries: int
    skipped_queries: int
    total_execution_time_ms: float
    average_execution_time_ms: float
    start_time: str
    end_time: str
    parameters: Dict[str, Any]


class BatchQueryProcessor:
    """Processor for handling multiple queries in batch."""

    def __init__(self, output_format: str = "json", **default_params):
        """
        Initialize batch processor.

        Args:
            output_format: Output format ("json" or "jsonl")
            **default_params: Default parameters for all queries
        """
        self.output_format = output_format
        self.default_params = default_params
        self.results: List[QueryResult] = []

    async def process_queries(
        self, queries: List[Union[str, Dict[str, Any]]], **shared_params
    ) -> List[QueryResult]:
        """
        Process multiple queries.

        Args:
            queries: List of queries (strings or dicts with query field)
            **shared_params: Parameters shared across all queries

        Returns:
            List of query results
        """
        self.results = []

        # Merge default and shared parameters
        params = {**self.default_params, **shared_params}

        for i, query_item in enumerate(queries):
            # Extract query and query-specific parameters
            if isinstance(query_item, dict):
                query = query_item.get("query", "")
                # Merge query-specific params with shared params
                # Query-specific params take precedence over shared params
                query_params = {**params, **{k: v for k, v in query_item.items() if k != "query"}}
                # Don't include None values or default False values unless explicitly set
                query_params = {
                    k: v
                    for k, v in query_params.items()
                    if v is not None
                    and (k not in self.default_params or v != self.default_params[k])
                }
            else:
                query = query_item
                query_params = {k: v for k, v in params.items() if v is not None}

            logger.info(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")

            try:
                # Run the query
                query_start = datetime.now()

                # Use run_headless for processing
                exit_code, response = await self._run_query_headless(query, query_params)

                query_end = datetime.now()
                execution_time = (query_end - query_start).total_seconds() * 1000

                if exit_code == 0:
                    result = QueryResult(
                        query=query,
                        response=response,
                        metadata=query_params,
                        timestamp=query_start.isoformat(),
                        status="success",
                        execution_time_ms=execution_time,
                    )
                else:
                    result = QueryResult(
                        query=query,
                        response="",
                        metadata=query_params,
                        timestamp=query_start.isoformat(),
                        status="error",
                        error_message=response,
                        execution_time_ms=execution_time,
                    )

            except Exception as e:
                logger.error(f"Failed to process query '{query}': {e}")
                result = QueryResult(
                    query=query,
                    response="",
                    metadata=query_params,
                    timestamp=datetime.now().isoformat(),
                    status="error",
                    error_message=str(e),
                )

            self.results.append(result)

        return self.results

    async def _run_query_headless(self, query: str, params: Dict[str, Any]) -> tuple[int, str]:
        """
        Run a single query using headless mode.

        Args:
            query: The query to run
            params: Parameters for the query

        Returns:
            Tuple of (exit_code, response)
        """
        # Extract parameters for run_headless
        output_format = params.get("output_format", "text")
        thread_id = params.get("thread_id")
        debug = params.get("debug", False)

        # Run headless and capture output
        try:
            # Call the actual run_headless function
            exit_code = run_headless(
                prompt=query,
                output_format=output_format,
                thread_id=thread_id,
                debug=debug,
                params=params
            )

            # For now, return a simple response
            # In a real implementation, we'd capture stdout/stderr
            if exit_code == 0:
                response = f"Query processed successfully"
            else:
                response = f"Query failed with exit code: {exit_code}"

            return exit_code, response

        except Exception as e:
            logger.error(f"Error in run_headless: {e}")
            return 1, str(e)

    def generate_report(self) -> BatchProcessingReport:
        """Generate a summary report of the batch processing."""
        if not self.results:
            return BatchProcessingReport(
                total_queries=0,
                successful_queries=0,
                failed_queries=0,
                skipped_queries=0,
                total_execution_time_ms=0.0,
                average_execution_time_ms=0.0,
                start_time="",
                end_time="",
                parameters=self.default_params,
            )

        successful = sum(1 for r in self.results if r.status == "success")
        failed = sum(1 for r in self.results if r.status == "error")
        skipped = sum(1 for r in self.results if r.status == "skipped")

        execution_times = [
            r.execution_time_ms for r in self.results if r.execution_time_ms is not None
        ]
        total_time = sum(execution_times) if execution_times else 0.0
        avg_time = total_time / len(self.results) if self.results else 0.0

        timestamps = [r.timestamp for r in self.results if r.timestamp]
        start_time = min(timestamps) if timestamps else ""
        end_time = max(timestamps) if timestamps else ""

        return BatchProcessingReport(
            total_queries=len(self.results),
            successful_queries=successful,
            failed_queries=failed,
            skipped_queries=skipped,
            total_execution_time_ms=total_time,
            average_execution_time_ms=avg_time,
            start_time=start_time,
            end_time=end_time,
            parameters=self.default_params,
        )


def load_queries_from_file(file_path: Union[str, Path]) -> List[Union[str, Dict[str, Any]]]:
    """
    Load queries from a file.

    Supports:
    - JSON files with array of strings or objects
    - Text files with one query per line
    - CSV files with query column

    Args:
        file_path: Path to the file

    Returns:
        List of queries
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Query file not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            # Validate each item
            queries = []
            for item in data:
                if isinstance(item, str):
                    queries.append(item)
                elif isinstance(item, dict) and "query" in item:
                    # If dict only has query field, return as string
                    if len(item) == 1:
                        queries.append(item["query"])
                    else:
                        queries.append(item)
                else:
                    logger.warning(f"Skipping invalid query item: {item}")
            return queries
        else:
            raise ValueError("JSON file must contain a list of queries")

    elif suffix == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            # Read lines, strip whitespace, and filter out empty lines
            queries = [line.strip() for line in f if line.strip()]
        return queries

    elif suffix == ".csv":
        queries = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV file is empty or has no columns")

            if "query" not in reader.fieldnames:
                # Try to use first column if no "query" column
                fieldnames = list(reader.fieldnames)
                if fieldnames:
                    query_field = fieldnames[0]
                    logger.info(f"Using column '{query_field}' as query field")
                else:
                    raise ValueError("CSV file is empty or has no columns")
            else:
                query_field = "query"

            for row in reader:
                query_text = row.get(query_field, "").strip()
                if query_text:
                    # Include other columns as metadata
                    metadata = {k: v for k, v in row.items() if k != query_field and v}
                    if metadata:
                        queries.append({"query": query_text, **metadata})
                    else:
                        queries.append(query_text)
        return queries

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def save_results_to_file(
    results: List[QueryResult], output_path: Union[str, Path], format: str = "json"
) -> None:
    """
    Save query results to a file.

    Args:
        results: List of query results
        output_path: Output file path
        format: Output format ("json" or "jsonl")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        # Save as single JSON file
        data = {
            "metadata": {
                "total_queries": len(results),
                "generated_at": datetime.now().isoformat(),
                "format": "json",
            },
            "results": [asdict(result) for result in results],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    elif format == "jsonl":
        # Save as JSON Lines (one JSON object per line)
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                json.dump(asdict(result), f, ensure_ascii=False)
                f.write("\n")

    else:
        raise ValueError(f"Unsupported output format: {format}")

    logger.info(f"Results saved to {output_path} in {format} format")
