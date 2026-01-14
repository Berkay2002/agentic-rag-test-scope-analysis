"""
Batch processing module for efficient multi-query execution.

This module provides the BatchQueryProcessor class that enables processing multiple queries
in batch mode with support for various input/output formats, parallel execution, and
detailed reporting.
"""

import asyncio
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..cli.headless import run_headless

logger = logging.getLogger(__name__)


class BatchQueryProcessor:
    """
    Processor for executing multiple queries in batch mode.

    This class provides functionality to:
    - Load queries from various file formats (JSON, JSONL, CSV, TXT)
    - Process queries with shared and query-specific parameters
    - Execute queries with optional parallel processing
    - Save results in multiple formats
    - Generate detailed execution reports
    """

    def __init__(self, thread_id: Optional[str] = None):
        """
        Initialize the batch query processor.

        Args:
            thread_id: Optional thread ID for shared context. If not provided,
                      a unique ID will be generated based on timestamp.
        """
        self.thread_id = thread_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    async def process_queries(
        self,
        queries: List[Union[str, Dict[str, Any]]],
        shared_params: Optional[Dict[str, Any]] = None,
        enable_parallel: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process a list of queries.

        Args:
            queries: List of queries. Each query can be a string or a dictionary
                    containing 'query' key and optional parameters.
            shared_params: Parameters to apply to all queries (e.g., output_format,
                          yolo_mode, thread_id).
            enable_parallel: Whether to execute queries in parallel. Default is False
                           for sequential execution to avoid rate limits.

        Returns:
            List of results with query, status, and metadata.
        """
        self.start_time = datetime.now()
        self.results = []

        # Parse query items and merge parameters
        parsed_queries = []
        for query_item in queries:
            if isinstance(query_item, str):
                query_dict = {"query": query_item}
            else:
                query_dict = query_item.copy()

            # Merge shared params with query-specific params
            merged_params = (shared_params or {}).copy()
            if "params" in query_dict:
                merged_params.update(query_dict["params"])
                del query_dict["params"]
            query_dict["params"] = merged_params

            parsed_queries.append(query_dict)

        logger.info(f"Processing {len(parsed_queries)} queries with thread_id: {self.thread_id}")

        # Execute queries
        if enable_parallel:
            # Execute in parallel with semaphore to limit concurrency
            semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent executions

            async def process_with_semaphore(query_dict):
                async with semaphore:
                    return await self._execute_single_query(
                        query_dict["query"],
                        query_dict["params"]
                    )

            # Process all queries in parallel
            tasks = [process_with_semaphore(q) for q in parsed_queries]
            self.results = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert exceptions to error results
            for i, result in enumerate(self.results):
                if isinstance(result, Exception):
                    self.results[i] = {
                        "query": parsed_queries[i]["query"],
                        "status": "error",
                        "error": str(result),
                        "timestamp": datetime.now().isoformat()
                    }
        else:
            # Execute sequentially
            for query_dict in parsed_queries:
                try:
                    result = await self._execute_single_query(
                        query_dict["query"],
                        query_dict["params"]
                    )
                    self.results.append(result)
                except Exception as e:
                    logger.error(f"Error processing query: {query_dict['query']}")
                    self.results.append({
                        "query": query_dict["query"],
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })

        self.end_time = datetime.now()
        logger.info(f"Batch processing completed. Processed {len(self.results)} queries.")

        return self.results

    async def _execute_single_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a single query using headless mode.

        Args:
            query: The query string to execute.
            params: Optional parameters for the query execution.

        Returns:
            Result dictionary with query, response, and metadata.
        """
        params = params or {}

        # Ensure thread_id is set
        if "thread_id" not in params:
            params["thread_id"] = self.thread_id

        # Execute query using headless mode
        try:
            # Run headless function (it's synchronous)
            exit_code = run_headless(
                prompt=query,
                thread_id=params["thread_id"],
                output_format=params.get("output_format", "json"),
                debug=params.get("debug", False),
                params=params
            )

            # For now, return a simple success response
            # In a real implementation, we'd capture the actual output
            response_data = f"Query processed with exit code: {exit_code}"

            return {
                "query": query,
                "status": "success",
                "response": response_data,
                "timestamp": datetime.now().isoformat(),
                "params": params
            }

        except Exception as e:
            logger.error(f"Error executing query: {query}")
            return {
                "query": query,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "params": params
            }

    def save_results(
        self,
        output_path: Union[str, Path],
        format: str = "json"
    ) -> None:
        """
        Save batch processing results to a file.

        Args:
            output_path: Path to save the results.
            format: Output format - 'json', 'jsonl', or 'csv'.

        Raises:
            ValueError: If format is not supported.
        """
        output_path = Path(output_path)

        if format.lower() == "json":
            # Save as JSON array
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)

        elif format.lower() == "jsonl":
            # Save as JSON Lines
            with open(output_path, "w", encoding="utf-8") as f:
                for result in self.results:
                    json.dump(result, f, ensure_ascii=False)
                    f.write("\n")

        elif format.lower() == "csv":
            # Save as CSV
            if not self.results:
                logger.warning("No results to save")
                return

            # Flatten nested dictionaries for CSV
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                # Get all unique keys
                all_keys = set()
                for result in self.results:
                    all_keys.update(result.keys())

                # Ensure required columns
                fieldnames = ["query", "status", "timestamp"]
                for key in all_keys:
                    if key not in fieldnames:
                        fieldnames.append(key)

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for result in self.results:
                    # Convert nested structures to strings
                    row = {}
                    for field in fieldnames:
                        value = result.get(field, "")
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value, ensure_ascii=False)
                        row[field] = value
                    writer.writerow(row)

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json', 'jsonl', or 'csv'.")

        logger.info(f"Results saved to {output_path} in {format} format")

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a summary report of the batch processing.

        Returns:
            Dictionary containing statistics and metadata.
        """
        if not self.results:
            return {
                "total_queries": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0,
                "execution_time": None
            }

        total = len(self.results)
        successful = sum(1 for r in self.results if r.get("status") == "success")
        failed = total - successful
        success_rate = successful / total if total > 0 else 0.0

        execution_time = None
        if self.start_time and self.end_time:
            execution_time = (self.end_time - self.start_time).total_seconds()

        # Count errors by type
        error_counts = {}
        for result in self.results:
            if result.get("status") == "error":
                error_msg = result.get("error", "Unknown error")
                # Extract error type (first line or first 50 chars)
                error_type = error_msg.split("\n")[0][:50]
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

        report = {
            "total_queries": total,
            "successful": successful,
            "failed": failed,
            "success_rate": success_rate,
            "execution_time_seconds": execution_time,
            "thread_id": self.thread_id,
            "timestamp_start": self.start_time.isoformat() if self.start_time else None,
            "timestamp_end": self.end_time.isoformat() if self.end_time else None,
            "error_summary": error_counts
        }

        return report


def load_queries_from_file(input_path: Union[str, Path]) -> List[Union[str, Dict[str, Any]]]:
    """
    Load queries from various file formats.

    Supported formats:
    - JSON: Array of strings or objects
    - JSONL: One JSON object per line
    - CSV: Must have a 'query' column
    - TXT: One query per line

    Args:
        input_path: Path to the input file.

    Returns:
        List of queries (strings or dictionaries).

    Raises:
        ValueError: If file format is not supported or invalid.
        FileNotFoundError: If input file doesn't exist.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix == ".json":
        # Load JSON array
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain an array")

        # Validate items
        queries = []
        for item in data:
            if isinstance(item, str):
                queries.append(item)
            elif isinstance(item, dict) and "query" in item:
                queries.append(item)
            else:
                raise ValueError("JSON items must be strings or objects with 'query' key")

        return queries

    elif suffix == ".jsonl":
        # Load JSON Lines
        queries = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    if isinstance(item, str):
                        queries.append(item)
                    elif isinstance(item, dict) and "query" in item:
                        queries.append(item)
                    else:
                        raise ValueError(f"Line {line_num}: Must be string or object with 'query' key")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num}: {e}")

        return queries

    elif suffix == ".csv":
        # Load CSV with 'query' column
        queries = []
        with open(input_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "query" not in reader.fieldnames:
                raise ValueError("CSV file must have a 'query' column")

            for row_num, row in enumerate(reader, 1):
                query_text = row["query"].strip()
                if not query_text:
                    continue

                # Include other columns as parameters
                params = {k: v for k, v in row.items() if k != "query" and v}
                if params:
                    queries.append({
                        "query": query_text,
                        "params": params
                    })
                else:
                    queries.append(query_text)

        return queries

    elif suffix == ".txt":
        # Load plain text (one query per line)
        queries = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):  # Skip empty lines and comments
                    queries.append(line)
        return queries

    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .json, .jsonl, .csv, or .txt")