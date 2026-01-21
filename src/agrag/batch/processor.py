"""Batch query processor for processing multiple queries efficiently."""

import asyncio
import json
import csv
import time
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import logging

from agrag.core import create_agent_graph, create_initial_state
from agrag.core.checkpointing import initialize_checkpointer
from agrag.cli.utils import extract_message_content

logger = logging.getLogger(__name__)


class BatchQueryProcessor:
    """Processes multiple queries in batch mode with progress tracking and reporting."""

    def __init__(self, thread_id: Optional[str] = None):
        """Initialize the batch processor.

        Args:
            thread_id: Optional thread ID for shared context across queries
        """
        self.thread_id = thread_id
        self.results = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "tool_usage": {},
            "error_types": {},
            "execution_times": []
        }

    async def process_queries(
        self,
        queries: List[Dict[str, Any]] | List[str],
        shared_params: Optional[Dict[str, Any]] = None,
        parallel: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Process a list of queries.

        Args:
            queries: List of query dictionaries with 'query' key
            shared_params: Optional shared parameters for all queries
            parallel: Whether to process queries in parallel (future feature)
            progress_callback: Optional callback for progress updates

        Returns:
            List of results with query, answer, and metadata
        """
        self.stats["total_queries"] = len(queries)
        shared_params = shared_params or {}
        self.start_time = datetime.now()
        self.end_time = None

        try:
            # Initialize checkpointer if thread_id is provided
            checkpointer = None
            if self.thread_id:
                init_result = initialize_checkpointer(enable_hitl=False)
                checkpointer = init_result.checkpointer

            # Create agent graph (or mock for tests/offline runs)
            if os.getenv("AGRAG_BATCH_MOCK", "").lower() in {"1", "true", "yes"}:
                graph = self._create_mock_graph()
            else:
                graph = create_agent_graph(checkpointer=checkpointer, enable_hitl=False)

            # Process queries
            if parallel:
                # Process in parallel (future implementation)
                results = await self._process_parallel(
                    graph,
                    queries,
                    shared_params,
                    progress_callback,
                )
            else:
                # Process sequentially
                results = await self._process_sequential(
                    graph,
                    queries,
                    shared_params,
                    progress_callback,
                )
        finally:
            self.end_time = datetime.now()

        self.results = results
        return results

    @staticmethod
    def _create_mock_graph():
        """Create a mock graph for offline/test batch execution."""
        from langchain_core.messages import AIMessage

        class _MockGraph:
            def invoke(self, state, config=None):
                messages = state.get("messages", [])
                query = ""
                if messages:
                    last = messages[-1]
                    query = getattr(last, "content", "") or ""
                return {
                    "messages": [AIMessage(content=f"Mocked response for: {query}")]
                }

        return _MockGraph()

    async def _process_sequential(
        self,
        graph,
        queries: List[Dict[str, Any]],
        shared_params: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Process queries sequentially."""
        results = []

        for i, query_data in enumerate(queries):
            if isinstance(query_data, str):
                query_data = {"query": query_data}
            query = query_data.get("query", "")
            query_id = query_data.get("id", f"Q_{i+1}")

            logger.info(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")

            start_time = time.time()

            try:
                # Merge query-specific params with shared params
                query_params = {**shared_params}
                if "params" in query_data and isinstance(query_data["params"], dict):
                    query_params.update(query_data["params"])
                for key, value in query_data.items():
                    if key in {"query", "id", "params"}:
                        continue
                    if value is not None:
                        query_params[key] = value
                if self.thread_id:
                    query_params["thread_id"] = self.thread_id

                # Create initial state
                config = {}
                if self.thread_id:
                    config["configurable"] = {"thread_id": self.thread_id}

                initial_state = create_initial_state(query)

                # Execute query
                result = self._execute_single_query(graph, initial_state, config)

                execution_time = time.time() - start_time
                self.stats["execution_times"].append(execution_time)
                self.stats["successful_queries"] += 1

                # Extract tool usage
                if "tool_calls" in result:
                    for tool_name in result["tool_calls"]:
                        self.stats["tool_usage"][tool_name] = (
                            self.stats["tool_usage"].get(tool_name, 0) + 1
                        )

                response = result.get("answer", "")
                results.append({
                    "query_id": query_id,
                    "query": query,
                    "answer": response,
                    "response": response,
                    "tool_calls": result.get("tool_calls", []),
                    "execution_time_ms": execution_time * 1000,
                    "success": True,
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                    "params": query_params
                })

            except Exception as e:
                execution_time = time.time() - start_time
                self.stats["execution_times"].append(execution_time)
                self.stats["failed_queries"] += 1

                error_type = type(e).__name__
                self.stats["error_types"][error_type] = (
                    self.stats["error_types"].get(error_type, 0) + 1
                )

                logger.error(f"Query failed: {query[:50]}... Error: {str(e)}")

                results.append({
                    "query_id": query_id,
                    "query": query,
                    "answer": f"Error: {str(e)}",
                    "response": "",
                    "tool_calls": [],
                    "execution_time_ms": execution_time * 1000,
                    "success": False,
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "params": query_params,
                    "error": str(e),
                    "error_type": error_type
                })

            # Update progress
            if progress_callback:
                progress_callback()

        return results

    async def _process_parallel(
        self,
        graph,
        queries: List[Dict[str, Any]],
        shared_params: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Process queries in parallel (future implementation)."""
        # For now, fall back to sequential processing
        logger.warning("Parallel processing not yet implemented, falling back to sequential")
        return await self._process_sequential(graph, queries, shared_params, progress_callback)

    def _execute_single_query(self, graph, initial_state, config) -> Dict[str, Any]:
        """Execute a single query and extract results."""
        tool_calls = []
        final_answer = "No answer generated"

        # Run graph
        final_state = graph.invoke(initial_state, config=config)
        messages = final_state.get("messages", [])

        # Extract tool calls and final answer
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    tool_calls.append(tool_name)

            if (
                hasattr(msg, "type")
                and msg.type == "ai"
                and hasattr(msg, "content")
                and msg.content
            ):
                final_answer = extract_message_content(msg.content)

        return {
            "answer": final_answer,
            "tool_calls": tool_calls
        }

    def save_results(
        self,
        results: Optional[List[Dict[str, Any]]] = None,
        output: Optional[str] = None,
        format: str = "json",
    ) -> str:
        """Save results to file.

        Args:
            results: Optional results to save (uses self.results if None)
            output: Output file path
            format: Output format (json, jsonl, csv)

        Returns:
            Path to saved file
        """
        if isinstance(results, (str, Path)) and output is None:
            output = str(results)
            results = None
        if results is None:
            results = self.results

        if output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = f"batch_results_{timestamp}.{format}"

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

        elif format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')

        elif format == "csv":
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                if not results:
                    return str(output_path)

                fieldnames = ["query", "status", "timestamp"]
                for result in results:
                    for key in result.keys():
                        if key not in fieldnames:
                            fieldnames.append(key)

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for result in results:
                    row = {}
                    for field in fieldnames:
                        value = result.get(field, "")
                        if field == "status" and not value:
                            if "success" in result:
                                value = "success" if result.get("success") else "error"
                        if field == "response" and not value and result.get("answer"):
                            value = result["answer"]
                        if field == "answer" and not value and result.get("response"):
                            value = result["response"]
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value)
                        row[field] = value
                    writer.writerow(row)

        return str(output_path)

    def generate_report(self, results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Generate a summary report of the batch processing results.

        Args:
            results: Optional results to analyze (uses self.results if None)

        Returns:
            Dictionary containing summary statistics
        """
        if results is None:
            results = self.results

        if not results:
            return {
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "success_rate": 0.0,
                "avg_execution_time_ms": 0.0,
                "total_execution_time_s": 0.0,
                "tool_usage": {},
                "error_types": {},
                "successful": 0,
                "failed": 0,
                "execution_time_seconds": 0.0
            }

        def _is_success(result: Dict[str, Any]) -> bool:
            if "success" in result:
                return bool(result.get("success"))
            status = result.get("status")
            if status is not None:
                return status == "success"
            return False

        # Calculate statistics
        successful = sum(1 for r in results if _is_success(r))
        failed = len(results) - successful
        success_rate = successful / len(results) if results else 0

        execution_times = [
            r.get("execution_time_ms", 0)
            for r in results
            if r.get("execution_time_ms") is not None
        ]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
        total_execution_time = sum(execution_times) / 1000 if execution_times else 0.0

        # Aggregate tool usage
        tool_usage = {}
        for result in results:
            for tool in result.get("tool_calls", []):
                tool_name = tool.get("name") if isinstance(tool, dict) else tool
                if not tool_name:
                    tool_name = "unknown"
                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

        # Aggregate error types
        error_types = {}
        for result in results:
            if not _is_success(result):
                error_type = result.get("error_type", "Unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1

        execution_time_seconds = total_execution_time
        if self.start_time and self.end_time:
            execution_time_seconds = (self.end_time - self.start_time).total_seconds()

        return {
            "total_queries": len(results),
            "successful_queries": successful,
            "failed_queries": failed,
            "success_rate": success_rate,
            "avg_execution_time_ms": avg_execution_time,
            "total_execution_time_s": total_execution_time,
            "tool_usage": tool_usage,
            "error_types": error_types,
            "successful": successful,
            "failed": failed,
            "execution_time_seconds": execution_time_seconds
        }
