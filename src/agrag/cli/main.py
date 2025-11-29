"""CLI application for Agentic GraphRAG Test Scope Analysis."""

import click
import logging
import sys
import json
from typing import Optional
from pathlib import Path

from langchain_core.runnables.config import RunnableConfig

from agrag.config import setup_logging, settings
from agrag.storage import Neo4jClient, PostgresClient
from agrag.core import create_agent_graph, create_initial_state
from agrag.core.checkpointing import initialize_checkpointer, summarize_error

logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level",
)
@click.option(
    "--log-format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Log format",
)
def cli(log_level: str, log_format: str):
    """Agentic GraphRAG for Test Scope Analysis.

    A comprehensive retrieval-augmented generation system for analyzing
    test coverage and dependencies in telecommunications software systems.
    """
    setup_logging(level=log_level, format_type=log_format)
    logger.info("AgRAG CLI initialized")


@cli.command()
def init():
    """Initialize database schemas (Neo4j + PostgreSQL)."""
    click.echo("Initializing database schemas...")

    try:
        # Initialize Neo4j
        click.echo("\n[Neo4j] Connecting...")
        neo4j_client = Neo4jClient()
        neo4j_client.setup_schema()
        click.echo("[Neo4j] ✓ Schema initialized (constraints + vector indexes)")

        # Initialize PostgreSQL
        click.echo("\n[PostgreSQL] Connecting...")
        postgres_client = PostgresClient()
        postgres_client.setup_schema()
        click.echo("[PostgreSQL] ✓ Schema initialized (pgvector + FTS indexes)")

        click.echo("\n✓ All schemas initialized successfully!")
        click.echo("\nNext steps:")
        click.echo("  1. Load your data using the data ingestion module")
        click.echo('  2. Run queries with: agrag query "your question"')

    except Exception as e:
        click.echo(f"\n✗ Error initializing schemas: {e}", err=True)
        logger.exception("Schema initialization failed")
        sys.exit(1)


@cli.command()
@click.option(
    "--thread-id",
    type=str,
    default=None,
    help="Thread ID to resume a previous conversation (auto-generated if not provided)",
)
@click.option(
    "--yolo/--no-yolo",
    default=False,
    help="Disable approvals - agent executes autonomously (YOLO mode)",
)
def chat(
    thread_id: Optional[str],
    yolo: bool,
):
    """Start an interactive chat session with the agent.

    This provides a conversational interface similar to Claude Code, Copilot CLI, etc.
    You can ask questions naturally and the agent will respond using the appropriate tools.

    All conversations are automatically saved and can be resumed later using --thread-id.

    Safety: By default, you approve each tool execution (HITL mode).
      - Default: Agent asks for approval before each tool (safe, you control everything)
      - --yolo: Agent executes autonomously without asking (faster but uncontrolled)

    Examples:
      agrag chat                           # Safe mode - you approve each action
      agrag chat --thread-id my-session    # Resume previous chat (with approvals)
      agrag chat --yolo                    # YOLO mode - autonomous execution
    """
    from agrag.cli.interactive import start_interactive_chat

    try:
        start_interactive_chat(
            thread_id=thread_id,
            enable_hitl=not yolo,  # HITL is enabled unless YOLO mode
        )
    except Exception as e:
        click.echo(f"\n✗ Chat failed: {e}", err=True)
        logger.exception("Interactive chat failed")
        sys.exit(1)


@cli.command()
@click.argument("query_text")
@click.option(
    "--thread-id",
    type=str,
    default=None,
    help="Thread ID for conversation persistence (HITL)",
)
@click.option(
    "--checkpoint/--no-checkpoint",
    default=False,
    help="Enable checkpointing for HITL workflows",
)
@click.option(
    "--stream/--no-stream",
    default=True,
    help="Stream output as it's generated",
)
def query(
    query_text: str,
    thread_id: Optional[str],
    checkpoint: bool,
    stream: bool,
):
    """Run a single query against the GraphRAG system.

    Examples:
      agrag query "What tests cover requirement REQ_AUTH_005?"
      agrag query "Find all handover-related test cases"
      agrag query "Show me functions called by initiate_handover"
    """
    click.echo(f"\nQuery: {query_text}\n")

    try:
        # Create checkpointer if requested
        checkpointer = None
        config: RunnableConfig = {}

        if checkpoint:
            click.echo("[Checkpointer] Enabling HITL persistence...")
            init_result = initialize_checkpointer(enable_hitl=True)
            checkpointer = init_result.checkpointer

            if checkpointer:
                backend_label = (
                    "PostgreSQL"
                    if init_result.backend == "postgres"
                    else "in-memory (session only)"
                )
                click.echo(f"[Checkpointer] Using {backend_label} backend.")

                if init_result.backend == "memory" and init_result.error:
                    click.echo(
                        f"[Checkpointer] Postgres unavailable: {summarize_error(init_result.error)}"
                    )

                if thread_id:
                    config["configurable"] = {"thread_id": thread_id}
                    click.echo(f"[Thread] Using thread_id: {thread_id}")
            else:
                click.echo(
                    "[Checkpointer] Persistence unavailable; continuing without HITL approvals."
                )
                if thread_id:
                    click.echo("[Thread] thread_id ignored because no checkpointer is active.")

        # Create graph
        graph = create_agent_graph(checkpointer=checkpointer)

        # Create initial state
        initial_state = create_initial_state(query_text)

        # Variables for stats
        tool_call_count = 0
        model_call_count = 0

        # Run graph
        if stream:
            click.echo("--- Agent Execution ---\n")
            for event in graph.stream(initial_state, config=config):
                # Log each step
                for node_name, node_state in event.items():
                    click.echo(f"[{node_name}]")

                    if "messages" in node_state:
                        messages = node_state["messages"]
                        for msg in messages:
                            msg_type = msg.__class__.__name__
                            content_preview = str(msg.content)[:100]
                            click.echo(f"  {msg_type}: {content_preview}...")

            # Get final result - get_state returns StateSnapshot with .values attribute
            state_snapshot = graph.get_state(config)
            state_values = state_snapshot.values
            final_answer = state_values.get("final_answer", "No answer generated")
            tool_call_count = state_values.get("tool_call_count", 0)
            model_call_count = state_values.get("model_call_count", 0)

        else:
            # Non-streaming execution - invoke returns the state dict directly
            final_state = graph.invoke(initial_state, config=config)
            final_answer = final_state.get("final_answer", "No answer generated")
            tool_call_count = final_state.get("tool_call_count", 0)
            model_call_count = final_state.get("model_call_count", 0)

        # Display final answer
        click.echo("\n--- Final Answer ---\n")
        click.echo(final_answer)

        # Display stats
        click.echo("\n--- Statistics ---")
        click.echo(f"Tool calls: {tool_call_count}")
        click.echo(f"Model calls: {model_call_count}")

    except TimeoutError as e:
        click.echo(f"\n✗ Query timed out: {e}", err=True)
        logger.exception("Query execution timed out")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n✗ Query failed: {e}", err=True)
        logger.exception("Query execution failed")
        sys.exit(1)


@cli.command()
@click.option(
    "--dataset",
    type=click.Path(exists=True),
    required=True,
    help="Path to evaluation dataset (JSON)",
)
@click.option(
    "--output",
    type=click.Path(),
    default="evaluation_results.json",
    help="Output file for results",
)
@click.option(
    "--k-values",
    type=str,
    default="1,3,5,10",
    help="Comma-separated k values for P@k, R@k",
)
@click.option(
    "--strategy",
    type=click.Choice(["vector", "keyword", "hybrid", "graph", "agent", "all"]),
    default="all",
    help="Retrieval strategy to evaluate (default: all)",
)
@click.option(
    "--verbose/--no-verbose",
    default=False,
    help="Show per-query metrics",
)
@click.option(
    "--bm25-index",
    type=click.Path(),
    default="data/bm25_index.pkl",
    help="Path to BM25 index file (default: data/bm25_index.pkl)",
)
def evaluate(
    dataset: str, output: str, k_values: str, strategy: str, verbose: bool, bm25_index: str
):
    """Run evaluation on a dataset with actual retrieval.

    Executes retrieval using the specified strategy and calculates metrics
    against ground truth. Supports comparing all strategies at once.

    Available strategies:
      - vector: Neo4j vector similarity search
      - keyword: BM25 keyword search (requires populated index)
      - hybrid: Vector + BM25 fusion with RRF
      - graph: Graph traversal for structural queries
      - agent: Full LLM agent with dynamic tool selection (RQ2)
      - all: Compare all strategies side-by-side

    The dataset should be a JSON file with the following structure:
    {
        "queries": [
            {
                "query": "...",
                "relevant_ids": ["id1", "id2", ...]
            },
            ...
        ]
    }

    Or a flat list of query objects.

    Examples:
      agrag evaluate --dataset data/eval.json --strategy vector
      agrag evaluate --dataset data/eval.json --strategy all
      agrag evaluate --dataset data/eval.json --strategy hybrid --verbose
      agrag evaluate --dataset data/eval.json --strategy graph
      agrag evaluate --dataset data/eval.json --strategy agent  # Full agent eval
    """
    import json
    from agrag.evaluation import (
        evaluate_retrieval,
        mean_average_precision,
        mean_reciprocal_rank,
    )
    from agrag.tools import VectorSearchTool, KeywordSearchTool, HybridSearchTool, GraphTraverseTool
    from agrag.storage import Neo4jClient, PostgresClient, BM25RetrieverManager

    click.echo(f"\nEvaluating dataset: {dataset}")
    click.echo(f"Strategy: {strategy}")

    # Parse k values
    k_list = [int(k.strip()) for k in k_values.split(",")]
    click.echo(f"K values: {k_list}")

    try:
        # Load dataset
        with open(dataset, "r") as f:
            data = json.load(f)

        # Handle both nested and flat formats
        if isinstance(data, dict) and "queries" in data:
            queries = data["queries"]
        elif isinstance(data, list):
            queries = data
        else:
            raise ValueError("Dataset must be a list or have a 'queries' key")

        click.echo(f"Loaded {len(queries)} queries\n")

        # Handle agent strategy separately
        if strategy == "agent":
            _run_agent_evaluation(queries, k_list, output, verbose)
            return

        # Initialize database clients
        click.echo("Initializing retrieval tools...")
        neo4j_client = Neo4jClient()
        postgres_client = PostgresClient()
        bm25_manager = BM25RetrieverManager(k=max(k_list) * 2)

        # Load BM25 index from disk if available
        if Path(bm25_index).exists():
            click.echo(f"Loading BM25 index from {bm25_index}...")
            bm25_manager.load(bm25_index)
            click.echo(f"  Loaded {bm25_manager.get_document_count()} documents")
        else:
            click.echo(f"Warning: BM25 index not found at {bm25_index}")
            click.echo("  Keyword search may return 0 results.")
            click.echo("  Run 'agrag generate --ingest' to populate the index.")

        # Initialize retrieval tools
        tools = {}
        if strategy in ["vector", "all"]:
            tools["vector"] = VectorSearchTool(neo4j_client=neo4j_client)
        if strategy in ["keyword", "all"]:
            tools["keyword"] = KeywordSearchTool(bm25_manager=bm25_manager)
        if strategy in ["hybrid", "all"]:
            tools["hybrid"] = HybridSearchTool(
                postgres_client=postgres_client,
                bm25_manager=bm25_manager,
            )
        if strategy in ["graph", "all"]:
            tools["graph"] = GraphTraverseTool(neo4j_client=neo4j_client)

        click.echo(f"Initialized tools: {list(tools.keys())}\n")

        # Run evaluation for each strategy
        strategies_to_run = list(tools.keys())
        all_strategy_results = {}

        for strat_name in strategies_to_run:
            tool = tools[strat_name]
            click.echo(f"\n{'='*50}")
            click.echo(f"Evaluating strategy: {strat_name.upper()}")
            click.echo(f"{'='*50}\n")

            all_results = []

            for i, query_data in enumerate(queries, 1):
                query = query_data["query"]
                relevant = set(query_data["relevant_ids"])
                difficulty = query_data.get("difficulty", "unknown")

                if verbose:
                    click.echo(f"[{i}/{len(queries)}] ({difficulty}) {query[:60]}...")

                # Execute actual retrieval (pass query_data for graph traversal context)
                retrieved = _execute_retrieval(tool, strat_name, query, max(k_list), query_data)

                metrics = evaluate_retrieval(retrieved, relevant, k_values=k_list)
                all_results.append(
                    {
                        "query": query,
                        "query_id": query_data.get("id", f"Q_{i}"),
                        "difficulty": difficulty,
                        "retrieved": retrieved,
                        "relevant": list(relevant),
                        "metrics": metrics,
                    }
                )

                # Log metrics if verbose
                if verbose:
                    click.echo(f"  Retrieved: {len(retrieved)} items")
                    for metric_name, score in metrics.items():
                        if "precision" in metric_name or "recall" in metric_name:
                            click.echo(f"    {metric_name}: {score:.4f}")

            # Calculate aggregate metrics
            map_score = mean_average_precision(all_results)
            mrr_score = mean_reciprocal_rank(all_results)

            # Calculate average P@k and R@k
            avg_metrics = {}
            for k in k_list:
                p_scores = [r["metrics"][f"precision@{k}"] for r in all_results]
                r_scores = [r["metrics"][f"recall@{k}"] for r in all_results]
                avg_metrics[f"avg_precision@{k}"] = sum(p_scores) / len(p_scores)
                avg_metrics[f"avg_recall@{k}"] = sum(r_scores) / len(r_scores)

            # Store results for this strategy
            all_strategy_results[strat_name] = {
                "map": map_score,
                "mrr": mrr_score,
                **avg_metrics,
                "per_query_results": all_results,
            }

            # Display aggregate metrics
            click.echo(f"\n--- {strat_name.upper()} Aggregate Metrics ---")
            click.echo(f"MAP: {map_score:.4f}")
            click.echo(f"MRR: {mrr_score:.4f}")
            for k in k_list:
                click.echo(f"Avg P@{k}: {avg_metrics[f'avg_precision@{k}']:.4f}")
                click.echo(f"Avg R@{k}: {avg_metrics[f'avg_recall@{k}']:.4f}")

        # Print comparison table if multiple strategies
        if len(strategies_to_run) > 1:
            click.echo(f"\n{'='*70}")
            click.echo("STRATEGY COMPARISON")
            click.echo(f"{'='*70}")

            # Header
            header = f"{'Strategy':<12} | {'MAP':>8} | {'MRR':>8}"
            for k in k_list:
                header += f" | {'P@'+str(k):>8} | {'R@'+str(k):>8}"
            click.echo(header)
            click.echo("-" * len(header))

            # Rows
            for strat_name in strategies_to_run:
                results = all_strategy_results[strat_name]
                row = f"{strat_name:<12} | {results['map']:>8.4f} | {results['mrr']:>8.4f}"
                for k in k_list:
                    row += f" | {results[f'avg_precision@{k}']:>8.4f} | {results[f'avg_recall@{k}']:>8.4f}"
                click.echo(row)

            # Find best strategy
            best_map = max(strategies_to_run, key=lambda s: all_strategy_results[s]["map"])
            best_mrr = max(strategies_to_run, key=lambda s: all_strategy_results[s]["mrr"])

            click.echo(f"\nBest for MAP: {best_map} ({all_strategy_results[best_map]['map']:.4f})")
            click.echo(f"Best for MRR: {best_mrr} ({all_strategy_results[best_mrr]['mrr']:.4f})")

        # Save results
        output_data = {
            "dataset": dataset,
            "queries_count": len(queries),
            "k_values": k_list,
            "strategies_evaluated": strategies_to_run,
            "results": all_strategy_results,
        }

        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)

        click.echo(f"\n✓ Results saved to: {output}")

    except Exception as e:
        click.echo(f"\n✗ Evaluation failed: {e}", err=True)
        logger.exception("Evaluation failed")
        sys.exit(1)


def _execute_retrieval(
    tool, strategy_name: str, query: str, k: int, query_data: Optional[dict] = None
) -> list:
    """
    Execute retrieval using the specified tool and extract IDs from results.

    Args:
        tool: The retrieval tool instance
        strategy_name: Name of the strategy (vector, keyword, hybrid, graph)
        query: The search query
        k: Number of results to retrieve
        query_data: Optional query metadata for graph traversal

    Returns:
        List of retrieved entity IDs
    """
    try:
        if strategy_name == "vector":
            # Vector search returns formatted string, need to parse
            result_str = tool._run(query=query, k=k, node_type="TestCase")
            return _parse_result_ids(result_str)
        elif strategy_name == "keyword":
            result_str = tool._run(query=query, k=k)
            return _parse_result_ids(result_str)
        elif strategy_name == "hybrid":
            result_str = tool._run(query=query, k=k)
            return _parse_result_ids(result_str)
        elif strategy_name == "graph":
            # Graph traversal requires extracting entity ID from query
            return _execute_graph_traversal(tool, query, k, query_data)
        else:
            return []
    except Exception as e:
        logger.warning(f"Retrieval failed for query '{query[:50]}...': {e}")
        return []


def _execute_graph_traversal(tool, query: str, k: int, query_data: Optional[dict] = None) -> list:
    """
    Execute graph traversal for structural queries.

    Graph traversal works differently from text search - it requires:
    1. A start node ID extracted from the query
    2. Relationship types to traverse
    3. Direction of traversal

    For evaluation, we extract entity IDs from the query and traverse
    to find related test cases.

    Args:
        tool: GraphTraverseTool instance
        query: The search query (e.g., "tests covering REQ_AUTH_005")
        k: Max results to return
        query_data: Query metadata with query_type info

    Returns:
        List of retrieved entity IDs
    """
    import re
    from agrag.kg.ontology import NodeLabel, RelationshipType

    # Extract entity ID from query
    # Match patterns like REQ_AUTH_005, FUNC_initiate_handover, etc.
    id_patterns = [
        (r"REQ_[A-Z]+_\d+", NodeLabel.REQUIREMENT),
        (r"FUNC_[A-Za-z_]+", NodeLabel.FUNCTION),
        (r"CLASS_[A-Za-z_]+", NodeLabel.CLASS),
        (r"MOD_[A-Za-z_.]+", NodeLabel.MODULE),
        (r"TC_[A-Z]+_\d+", NodeLabel.TEST_CASE),
    ]

    start_node_id = None
    start_node_label = None

    for pattern, label in id_patterns:
        match = re.search(pattern, query)
        if match:
            start_node_id = match.group()
            start_node_label = label
            break

    if not start_node_id:
        # No entity ID found in query - graph traversal not applicable
        logger.debug(f"No entity ID found in query for graph traversal: {query[:50]}...")
        return []

    # Map entity types to appropriate traversal patterns
    if start_node_label == NodeLabel.REQUIREMENT:
        # Find tests that verify this requirement (TestCase -[VERIFIES]-> Requirement)
        relationship_types = [RelationshipType.VERIFIES]
        direction = "incoming"  # We want incoming VERIFIES relationships
    elif start_node_label == NodeLabel.FUNCTION:
        # Find tests that cover this function (TestCase -[COVERS]-> Function)
        relationship_types = [RelationshipType.COVERS]
        direction = "incoming"
    elif start_node_label == NodeLabel.CLASS:
        # Find tests that cover methods in this class
        relationship_types = [RelationshipType.COVERS, RelationshipType.DEFINED_IN]
        direction = "incoming"
    elif start_node_label == NodeLabel.MODULE:
        # Find tests for code in this module
        relationship_types = [RelationshipType.DEFINED_IN, RelationshipType.COVERS]
        direction = "incoming"
    else:
        # Default: find any connected test cases
        relationship_types = None
        direction = "both"

    try:
        result_str = tool._run(
            start_node_id=start_node_id,
            start_node_label=start_node_label,
            relationship_types=relationship_types,
            depth=2,
            direction=direction,
        )

        # Parse the graph traversal output
        return _parse_graph_result_ids(result_str)

    except Exception as e:
        logger.warning(f"Graph traversal failed for {start_node_id}: {e}")
        return []


def _parse_graph_result_ids(result_str: str) -> list:
    """
    Parse IDs from graph traversal output.

    Graph traversal returns paths like:
    "1. Path (depth 1): REQ_AUTH_005 → TC_AUTH_001"
    "   Sequence: Requirement:REQ_AUTH_005 → TestCase:TC_AUTH_001"

    We extract all entity IDs from the sequence, prioritizing test cases.

    Args:
        result_str: The formatted result string from graph traversal

    Returns:
        List of extracted IDs (test cases first, then other entities)
    """
    import re

    test_case_ids = []
    other_ids = []

    # Match patterns for different ID formats in graph output
    # Format: "Label:ID" in sequences
    patterns = [
        (r"TestCase:(TC_[A-Z]+_\d+)", True),  # Test cases (priority)
        (r"Requirement:(REQ_[A-Z]+_\d+)", False),  # Requirements
        (r"Function:(FUNC_[A-Za-z_]+)", False),  # Functions
        (r"Class:(CLASS_[A-Za-z_]+)", False),  # Classes
        (r"Module:(MOD_[A-Za-z_.]+)", False),  # Modules
    ]

    for pattern, is_test_case in patterns:
        matches = re.findall(pattern, result_str)
        for match in matches:
            if is_test_case:
                if match not in test_case_ids:
                    test_case_ids.append(match)
            else:
                if match not in other_ids and match not in test_case_ids:
                    other_ids.append(match)

    # Return test cases first, then other entities
    return test_case_ids + other_ids


def _parse_result_ids(result_str: str) -> list:
    """
    Parse IDs from the formatted tool output string.

    The tools return formatted strings like:
    "1. ID: TC_HANDOVER_001 (Score: 0.85)"
    or
    "1. ID: TestCase_TC_HANDOVER_001 (RRF Score: 0.016)"

    This function extracts the IDs and normalizes them.

    Args:
        result_str: The formatted result string from a tool

    Returns:
        List of extracted IDs
    """
    import re

    ids = []

    # Match patterns for different ID formats:
    # - TC_HANDOVER_001 (test cases)
    # - REQ_AUTH_005 (requirements)
    # - FUNC_something_001 (functions)
    # Also handle prefixed versions like TestCase_TC_HANDOVER_001
    patterns = [
        r"ID:\s*(?:TestCase_)?(TC_[A-Z]+_\d+)",  # Test cases
        r"ID:\s*(?:Requirement_)?(REQ_[A-Z]+_\d+)",  # Requirements
        r"ID:\s*(?:Function_)?(FUNC_[A-Za-z_]+)",  # Functions
        r"ID:\s*(?:Class_)?(CLASS_[A-Za-z_]+)",  # Classes
        r"ID:\s*(?:Module_)?(MOD_[A-Za-z_.]+)",  # Modules
    ]

    for pattern in patterns:
        matches = re.findall(pattern, result_str)
        for match in matches:
            if match not in ids:  # Avoid duplicates
                ids.append(match)

    return ids


def _run_agent_evaluation(
    queries: list,
    k_list: list,
    output: str,
    verbose: bool,
):
    """
    Run full agent evaluation on the dataset.

    This evaluates the complete ReAct agent loop, measuring how well
    the agent dynamically selects retrieval strategies compared to
    static baseline approaches.

    Args:
        queries: List of query dictionaries
        k_list: K values for P@k, R@k metrics
        output: Output file path
        verbose: Whether to show per-query progress
    """
    import json
    from agrag.evaluation.agentic_evaluator import (
        AgenticEvaluator,
        create_evaluation_graph,
    )

    click.echo("\n" + "=" * 60)
    click.echo("AGENTIC EVALUATION (Full LLM Agent)")
    click.echo("=" * 60)
    click.echo("\nThis runs the complete ReAct agent loop per query,")
    click.echo("allowing the LLM to dynamically select retrieval tools.")
    click.echo("This directly tests RQ2: agent vs static strategies.\n")

    # Create agent graph without HITL
    click.echo("Initializing agent graph (no HITL interrupts)...")
    graph = create_evaluation_graph()
    click.echo("✓ Agent graph ready\n")

    # Create evaluator
    evaluator = AgenticEvaluator(
        graph=graph,
        k_values=k_list,
    )

    # Run evaluation
    click.echo(f"Running agent on {len(queries)} queries...\n")

    summary = evaluator.evaluate_dataset(
        queries=queries,
        verbose=verbose,
    )

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo("AGENT EVALUATION RESULTS")
    click.echo("=" * 60)

    click.echo("\n--- Aggregate Metrics ---")
    click.echo(f"MAP: {summary.map_score:.4f}")
    click.echo(f"MRR: {summary.mrr_score:.4f}")

    for k in k_list:
        p = summary.avg_precision_at_k.get(k, 0)
        r = summary.avg_recall_at_k.get(k, 0)
        click.echo(f"Avg P@{k}: {p:.4f}")
        click.echo(f"Avg R@{k}: {r:.4f}")

    click.echo("\n--- Tool Usage Statistics ---")
    click.echo(f"Total tool calls: {summary.total_tool_calls}")
    click.echo(f"Avg tools per query: {summary.avg_tools_per_query:.2f}")

    if summary.tool_frequency:
        click.echo("\nTool frequency:")
        for tool, count in sorted(summary.tool_frequency.items(), key=lambda x: -x[1]):
            pct = 100 * count / summary.total_queries
            click.echo(f"  {tool}: {count} ({pct:.1f}%)")

    if summary.tool_combinations:
        click.echo("\nTool combinations (top 5):")
        sorted_combos = sorted(summary.tool_combinations.items(), key=lambda x: -x[1])[:5]
        for combo, count in sorted_combos:
            click.echo(f"  {combo}: {count}")

    click.echo("\n--- Execution Statistics ---")
    click.echo(f"Total queries: {summary.total_queries}")
    click.echo(f"Successful queries: {summary.successful_queries}")
    click.echo(f"Success rate: {summary.successful_queries / max(1, summary.total_queries):.1%}")
    click.echo(f"Avg execution time: {summary.avg_execution_time_ms:.0f}ms")

    # Save results
    output_data = {
        "strategy": "agent",
        "dataset": output.replace("_results.json", ".json"),
        "queries_count": len(queries),
        "k_values": k_list,
        "results": {
            "agent": summary.to_dict(),
        },
    }

    with open(output, "w") as f:
        json.dump(output_data, f, indent=2)

    click.echo(f"\n✓ Results saved to: {output}")


@cli.command()
def info():
    """Show system configuration and status."""
    click.echo("\n=== AgRAG System Configuration ===\n")

    # LLM config
    click.echo("[LLM]")
    click.echo(f"  Model: {settings.google_model}")
    click.echo(f"  Temperature: {settings.agent_temperature}")
    click.echo(f"  Max tool calls: {settings.max_tool_calls}")
    click.echo(f"  Max model calls: {settings.max_model_calls}")

    # Embedding config
    click.echo("\n[Embeddings]")
    click.echo(f"  Model: {settings.google_embedding_model}")
    click.echo(f"  Dimensions: {settings.embedding_dimensions}")

    # Neo4j config
    click.echo("\n[Neo4j]")
    click.echo(f"  URI: {settings.neo4j_uri}")
    click.echo(f"  Database: {settings.neo4j_database}")

    # PostgreSQL config
    click.echo("\n[PostgreSQL]")
    postgres_uri = settings.postgres_connection_string
    # Mask password in URI
    if "@" in postgres_uri:
        masked_uri = postgres_uri.split("@")[1]
        click.echo(f"  Connection: ***@{masked_uri}")
    else:
        click.echo(f"  Connection: {postgres_uri}")

    # LangSmith config
    click.echo("\n[LangSmith]")
    click.echo(f"  Tracing: {settings.langchain_tracing_v2}")
    click.echo(f"  Project: {settings.langchain_project}")

    click.echo("")


@cli.command()
@click.confirmation_option(prompt="Are you sure you want to delete ALL data from both databases?")
def reset():
    """Delete all data from Neo4j and PostgreSQL databases.

    WARNING: This will permanently delete all entities, relationships,
    and embeddings. This action cannot be undone.

    Use this before ingesting a new dataset to avoid duplicates.
    """
    click.echo("\n=== Resetting Databases ===\n")

    try:
        # Clear Neo4j
        click.echo("[Neo4j] Deleting all nodes and relationships...")
        neo4j_client = Neo4jClient()
        neo4j_count = neo4j_client.delete_all()
        click.echo(f"[Neo4j] ✓ Deleted {neo4j_count} nodes and their relationships")

        # Clear PostgreSQL
        click.echo("\n[PostgreSQL] Deleting all document chunks...")
        postgres_client = PostgresClient()
        postgres_count = postgres_client.delete_all_chunks()
        click.echo(f"[PostgreSQL] ✓ Deleted {postgres_count} document chunks")

        click.echo("\n✓ Databases reset successfully!")
        click.echo("\nNext step: agrag ingest <dataset.json>")

    except Exception as e:
        click.echo(f"\n✗ Reset failed: {e}", err=True)
        logger.exception("Database reset failed")
        sys.exit(1)


@cli.command()
@click.option("--requirements", default=50, help="Number of requirements to generate")
@click.option("--testcases", default=200, help="Number of test cases to generate")
@click.option("--output", default="data/synthetic_dataset.json", help="Output file path")
@click.option(
    "--with-eval/--no-with-eval",
    default=False,
    help="Also generate evaluation dataset",
)
@click.option(
    "--eval-output",
    default=None,
    help="Output file for evaluation dataset (default: <output>_eval.json)",
)
@click.option(
    "--ingest/--no-ingest",
    default=False,
    help="Reset databases and ingest generated data immediately",
)
def generate(
    requirements: int,
    testcases: int,
    output: str,
    with_eval: bool,
    eval_output: Optional[str],
    ingest: bool,
):
    """Generate synthetic telecommunications dataset.

    Creates realistic synthetic data including requirements, test cases,
    functions, classes, modules, and their relationships.

    Optionally generates an evaluation dataset with stratified query difficulties
    for benchmarking retrieval strategies (RQ2).

    Use --ingest to automatically reset databases and load the generated data.

    Examples:
      agrag generate
      agrag generate --requirements 30 --testcases 150
      agrag generate --output my_dataset.json
      agrag generate --with-eval --eval-output eval_queries.json
      agrag generate --ingest  # Reset DBs and ingest immediately
    """
    from agrag.data.generators import TelecomDataGenerator

    click.echo("\n=== Generating Synthetic Dataset ===\n")
    click.echo(f"Requirements: {requirements}")
    click.echo(f"Test cases: {testcases}")
    click.echo(f"Output: {output}")
    if with_eval:
        eval_out = eval_output or output.replace(".json", "_eval.json")
        click.echo(f"Evaluation dataset: {eval_out}")
    if ingest:
        click.echo("Ingest: Enabled (will reset databases)")
    click.echo()

    try:
        # Create generator
        generator = TelecomDataGenerator()

        # Generate dataset
        click.echo("Generating entities (this may take a few minutes)...")
        dataset = generator.generate_full_dataset(
            requirement_count=requirements, testcase_count=testcases
        )

        # Create output directory if needed
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to JSON
        with open(output, "w") as f:
            json.dump(dataset, f, indent=2)

        # Display summary
        metadata = dataset.get("metadata", {})
        click.echo("\n✓ Dataset generated successfully!")
        click.echo("\nSummary:")
        click.echo(f"  Requirements: {metadata.get('requirement_count', 0)}")
        click.echo(f"  Test cases: {metadata.get('testcase_count', 0)}")
        click.echo(f"  Functions: {metadata.get('function_count', 0)}")
        click.echo(f"  Classes: {metadata.get('class_count', 0)}")
        click.echo(f"  Modules: {metadata.get('module_count', 0)}")
        click.echo(f"  Relationships: {metadata.get('relationship_count', 0)}")
        click.echo(f"\nSaved to: {output}")

        # Generate evaluation dataset if requested
        if with_eval:
            click.echo("\n=== Generating Evaluation Dataset ===\n")

            # Extract entities from dataset
            entities = dataset.get("entities", [])
            relationships = dataset.get("relationships", [])

            # Separate by type
            test_cases = [e for e in entities if e.get("id", "").startswith("TC_")]
            requirements_list = [e for e in entities if e.get("id", "").startswith("REQ_")]
            functions_list = [e for e in entities if e.get("id", "").startswith("FUNC_")]

            # Generate evaluation queries
            eval_dataset = generator.generate_evaluation_dataset(
                test_cases=test_cases,
                requirements=requirements_list,
                functions=functions_list,
                relationships=relationships,
            )

            # Save evaluation dataset
            eval_out = eval_output or output.replace(".json", "_eval.json")
            with open(eval_out, "w") as f:
                json.dump(eval_dataset, f, indent=2)

            eval_meta = eval_dataset.get("metadata", {})
            click.echo("✓ Evaluation dataset generated!")
            click.echo(f"  Total queries: {eval_meta.get('total_queries', 0)}")
            click.echo(f"  Difficulty distribution: {eval_meta.get('difficulty_distribution', {})}")
            click.echo(f"\nSaved to: {eval_out}")

        # Ingest into databases if requested
        if ingest:
            click.echo("\n=== Resetting and Ingesting Data ===\n")

            # Reset databases
            click.echo("[Neo4j] Deleting all nodes and relationships...")
            neo4j_client = Neo4jClient()
            neo4j_count = neo4j_client.delete_all()
            click.echo(f"[Neo4j] ✓ Deleted {neo4j_count} nodes")

            click.echo("[PostgreSQL] Deleting all document chunks...")
            postgres_client = PostgresClient()
            postgres_count = postgres_client.delete_all_chunks()
            click.echo(f"[PostgreSQL] ✓ Deleted {postgres_count} chunks")

            # Ingest new data
            click.echo("\n[Ingestion] Loading data into databases...")
            from agrag.data.ingestion import DataIngestion

            ingestion = DataIngestion()
            results = ingestion.ingest_full_dataset(dataset)

            click.echo("\n✓ Ingestion complete!")
            click.echo(f"  Neo4j entities: {results['neo4j_entities']}")
            click.echo(f"  PostgreSQL entities: {results['postgres_entities']}")
            click.echo(f"  Relationships: {results['relationships']}")
        else:
            click.echo(f"\nNext step: agrag ingest {output}")

    except Exception as e:
        click.echo(f"\n✗ Generation failed: {e}", err=True)
        logger.exception("Dataset generation failed")
        sys.exit(1)


@cli.command("generate-eval")
@click.option(
    "--dataset",
    type=click.Path(exists=True),
    default="data/synthetic_dataset.json",
    help="Path to synthetic dataset (generated by 'generate' command)",
)
@click.option(
    "--output", default="data/eval_queries.json", help="Output file for evaluation dataset"
)
@click.option("--simple", default=40, help="Number of simple queries")
@click.option("--moderate", default=35, help="Number of moderate queries")
@click.option("--complex", default=25, help="Number of complex queries")
@click.option("--negative", default=8, help="Number of negative/out-of-scope queries")
@click.option(
    "--paraphrases/--no-paraphrases",
    default=True,
    help="Use query paraphrases for diversity",
)
def generate_eval(
    dataset: str,
    output: str,
    simple: int,
    moderate: int,
    complex: int,
    negative: int,
    paraphrases: bool,
):
    """Generate ground truth evaluation dataset with stratified queries.

    Creates an evaluation dataset from an existing synthetic dataset for
    benchmarking retrieval strategies (RQ2). Queries are stratified by
    difficulty level:

    - Simple (40%): Single entity lookup, attribute filters
    - Moderate (35%): Single-hop relationships, combined filters
    - Complex (25%): Multi-hop traversal, aggregations, coverage gaps
    - Negative: Out-of-scope queries for precision measurement

    Examples:
      agrag generate-eval
      agrag generate-eval --dataset my_dataset.json --output my_eval.json
      agrag generate-eval --simple 50 --moderate 30 --complex 20 --negative 10
    """
    from agrag.data.generators import TelecomDataGenerator

    click.echo("\n=== Generating Evaluation Dataset ===\n")
    click.echo(f"Source dataset: {dataset}")
    click.echo(f"Output: {output}")
    click.echo(
        f"Query distribution: simple={simple}, moderate={moderate}, complex={complex}, negative={negative}"
    )
    click.echo(f"Paraphrases: {'Enabled' if paraphrases else 'Disabled'}")
    click.echo()

    try:
        # Load dataset
        click.echo("Loading source dataset...")
        with open(dataset, "r") as f:
            source_data = json.load(f)

        entities = source_data.get("entities", [])
        relationships = source_data.get("relationships", [])

        click.echo(f"Loaded {len(entities)} entities and {len(relationships)} relationships\n")

        # Separate entities by type
        test_cases = [e for e in entities if e.get("id", "").startswith("TC_")]
        requirements_list = [e for e in entities if e.get("id", "").startswith("REQ_")]
        functions_list = [e for e in entities if e.get("id", "").startswith("FUNC_")]

        click.echo(
            f"Found {len(test_cases)} test cases, {len(requirements_list)} requirements, {len(functions_list)} functions\n"
        )

        # Create generator (no need to reinitialize embedding service for eval generation)
        generator = TelecomDataGenerator()

        # Generate evaluation queries
        click.echo("Generating evaluation queries...")
        eval_dataset = generator.generate_evaluation_dataset(
            test_cases=test_cases,
            requirements=requirements_list,
            functions=functions_list,
            relationships=relationships,
            num_simple=simple,
            num_moderate=moderate,
            num_complex=complex,
            num_negative=negative,
            use_paraphrases=paraphrases,
        )

        # Create output directory if needed
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to JSON
        with open(output, "w") as f:
            json.dump(eval_dataset, f, indent=2)

        # Display summary
        eval_meta = eval_dataset.get("metadata", {})
        click.echo("\n✓ Evaluation dataset generated successfully!")
        click.echo("\nSummary:")
        click.echo(f"  Total queries: {eval_meta.get('total_queries', 0)}")
        click.echo("  Difficulty distribution:")
        for difficulty, count in eval_meta.get("difficulty_distribution", {}).items():
            click.echo(f"    {difficulty}: {count}")
        click.echo(
            f"  Source dataset: {eval_meta.get('source_test_count', 0)} tests, "
            f"{eval_meta.get('source_requirement_count', 0)} requirements, "
            f"{eval_meta.get('source_function_count', 0)} functions"
        )
        click.echo(f"\nSaved to: {output}")
        click.echo(f"\nNext step: agrag evaluate --dataset {output}")

    except Exception as e:
        click.echo(f"\n✗ Evaluation generation failed: {e}", err=True)
        logger.exception("Evaluation dataset generation failed")
        sys.exit(1)


@cli.command()
@click.argument("dataset_path")
def ingest(dataset_path: str):
    """Ingest dataset into Neo4j and PostgreSQL.

    Loads a previously generated synthetic dataset into both databases.
    The dataset should be a JSON file created by the 'generate' command.

    Examples:
      agrag ingest data/synthetic_dataset.json
    """
    from agrag.data.ingestion import DataIngestion

    click.echo("\n=== Ingesting Dataset ===\n")
    click.echo(f"Dataset: {dataset_path}\n")

    try:
        # Load dataset
        click.echo("Loading dataset from file...")
        with open(dataset_path) as f:
            dataset = json.load(f)

        entities_count = len(dataset.get("entities", []))
        relationships_count = len(dataset.get("relationships", []))

        click.echo(f"Loaded {entities_count} entities and {relationships_count} relationships\n")

        # Ingest data
        click.echo("Starting ingestion (this may take a few minutes)...\n")
        ingestion = DataIngestion()
        results = ingestion.ingest_full_dataset(dataset)

        # Display results
        click.echo("\n✓ Ingestion complete!")
        click.echo("\nResults:")
        click.echo(f"  Neo4j entities: {results['neo4j_entities']}")
        click.echo(f"  PostgreSQL entities: {results['postgres_entities']}")
        click.echo(f"  Relationships: {results['relationships']}")

        click.echo('\nNext step: agrag query "What tests cover handover requirements?"')

    except Exception as e:
        click.echo(f"\n✗ Ingestion failed: {e}", err=True)
        logger.exception("Data ingestion failed")
        sys.exit(1)


@cli.group()
def load():
    """Load data from various sources (code repositories, documents, etc.)."""
    pass


@load.command("repo")
@click.argument("repo_path")
@click.option(
    "--languages",
    type=str,
    default="python",
    help="Comma-separated list of languages to process",
)
def load_repo(repo_path: str, languages: str):
    """Load and ingest code from a repository.

    Uses AST-based parsing to extract functions, classes, and modules
    from source code files while preserving structural relationships.

    Examples:
      agrag load repo /path/to/repo
      agrag load repo /path/to/repo --languages python,java
    """
    from agrag.data.ingestion import DataIngestion

    click.echo("\n=== Loading Code Repository ===\n")
    click.echo(f"Repository: {repo_path}")
    click.echo(f"Languages: {languages}\n")

    try:
        # Verify path exists
        repo_path_obj = Path(repo_path)
        if not repo_path_obj.exists():
            click.echo(f"✗ Repository path does not exist: {repo_path}", err=True)
            sys.exit(1)

        # Ingest repository
        click.echo("Analyzing code repository (this may take a few minutes)...\n")
        ingestion = DataIngestion()
        results = ingestion.ingest_from_code_repository(repo_path)

        # Display results
        click.echo("\n✓ Repository ingestion complete!")
        click.echo("\nResults:")
        click.echo(f"  Neo4j entities: {results.get('neo4j', 0)}")
        click.echo(f"  PostgreSQL entities: {results.get('postgres', 0)}")
        click.echo(f"  BM25 index entries: {results.get('bm25', 0)}")

        click.echo('\n\nNext step: agrag query "Find functions related to authentication"')

    except Exception as e:
        click.echo(f"\n✗ Repository loading failed: {e}", err=True)
        logger.exception("Repository loading failed")
        sys.exit(1)


@load.command("docs")
@click.argument("directory")
@click.option(
    "--formats",
    type=str,
    default="pdf,docx,markdown",
    help="Comma-separated list of formats (pdf,docx,xlsx,pptx,markdown,html,csv,images)",
)
@click.option(
    "--use-chunker/--no-chunker",
    default=True,
    help="Use Docling HybridChunker for semantic chunking",
)
@click.option(
    "--export-format",
    type=click.Choice(["markdown", "text", "json", "html", "doctags"]),
    default="markdown",
    help="Export format for documents",
)
@click.option(
    "--table-mode",
    type=click.Choice(["accurate", "fast"]),
    default="accurate",
    help="TableFormer mode: accurate (slower, better quality) or fast",
)
@click.option(
    "--max-pages",
    type=int,
    default=None,
    help="Maximum pages to process per document",
)
def load_docs(
    directory: str,
    formats: str,
    use_chunker: bool,
    export_format: str,
    table_mode: str,
    max_pages: Optional[int],
):
    """Load and ingest documentation/requirements using Docling.

    Docling provides production-grade parsing for PDF, DOCX, XLSX, PPTX,
    Markdown, HTML, CSV, images, and more. It uses AI models (DocLayNet,
    TableFormer) for accurate layout analysis and table extraction.

    Examples:
      agrag load docs /path/to/docs
      agrag load docs /path/to/docs --formats pdf,docx --use-chunker
      agrag load docs /path/to/docs --table-mode fast --max-pages 50
    """
    from agrag.data.ingestion import DataIngestion

    click.echo("\n=== Loading Documentation with Docling ===\n")
    click.echo(f"Directory: {directory}")
    click.echo(f"Formats: {formats}")
    click.echo(f"Chunker: {'Enabled (HybridChunker)' if use_chunker else 'Disabled'}")
    click.echo(f"Export format: {export_format}")
    click.echo(f"Table mode: {table_mode}")
    if max_pages:
        click.echo(f"Max pages: {max_pages}")
    click.echo()

    try:
        # Verify path exists
        dir_path = Path(directory)
        if not dir_path.exists():
            click.echo(f"✗ Directory does not exist: {directory}", err=True)
            sys.exit(1)

        # Ingest documents
        click.echo("Processing documents with Docling AI models...\n")
        click.echo(
            "Note: First run may download models (~500MB). Subsequent runs use cached models.\n"
        )

        ingestion = DataIngestion()
        results = ingestion.ingest_from_documents(
            directory,
            use_chunker=use_chunker,
            export_format=export_format,
            table_mode=table_mode,
            max_num_pages=max_pages,
        )

        # Display results
        click.echo("\n✓ Documentation ingestion complete!")
        click.echo("\nResults:")
        click.echo(f"  Neo4j entities: {results.get('neo4j', 0)}")
        click.echo(f"  PostgreSQL entities: {results.get('postgres', 0)}")
        click.echo(f"  BM25 index entries: {results.get('bm25', 0)}")

        click.echo('\n\nNext step: agrag query "What requirements cover authentication?"')

    except Exception as e:
        click.echo(f"\n✗ Documentation loading failed: {e}", err=True)
        logger.exception("Documentation loading failed")
        sys.exit(1)


@load.command("tgf")
@click.argument("csv_path")
@click.option(
    "--filter-results",
    type=str,
    default=None,
    help="Filter by test results (e.g., FAIL,ERROR)",
)
@click.option(
    "--show-stats/--no-show-stats",
    default=True,
    help="Display statistics about loaded data",
)
def load_tgf(csv_path: str, filter_results: Optional[str], show_stats: bool):
    """Load test execution data from Ericsson TGF CSV export.

    Loads test results from Ericsson's Test Governance Framework (TGF) CSV format,
    creating TestCase entities and linking them to Requirements and Functions.

    CSV Format (expected columns):
      test_id, test_suite, test_name, test_type, feature_area, sub_feature,
      requirement_ids (semicolon-separated), function_names (semicolon-separated),
      result, execution_time_ms, timestamp, failure_reason, test_file_path,
      code_coverage_pct, priority, tags (semicolon-separated)

    Examples:
      agrag load tgf /path/to/tgf_export.csv
      agrag load tgf tests.csv --filter-results FAIL,ERROR
      agrag load tgf tests.csv --no-show-stats
    """
    from agrag.data.loaders.tgf_loader import TGFCSVLoader
    from agrag.data.ingestion import DataIngestion

    click.echo("\n=== Loading TGF Test Data ===\n")
    click.echo(f"CSV file: {csv_path}")
    if filter_results:
        click.echo(f"Filtering results: {filter_results}")
    click.echo()

    try:
        # Verify path exists
        csv_path_obj = Path(csv_path)
        if not csv_path_obj.exists():
            click.echo(f"✗ CSV file does not exist: {csv_path}", err=True)
            sys.exit(1)

        # Parse filter results
        filter_list = None
        if filter_results:
            filter_list = [r.strip().upper() for r in filter_results.split(",")]

        # Load TGF data
        click.echo("Loading TGF CSV data...")
        loader = TGFCSVLoader(
            file_path=csv_path,
            filter_results=filter_list,
        )
        documents = loader.load()

        click.echo(f"✓ Loaded {len(documents)} test cases from TGF CSV\n")

        # Show statistics
        if show_stats:
            stats = loader.get_statistics()
            click.echo("=== TGF Data Statistics ===")
            click.echo(f"Total tests: {stats.get('total_tests', 0)}")
            click.echo("\nResult Distribution:")
            for result, count in stats.get("result_distribution", {}).items():
                click.echo(f"  {result}: {count}")
            click.echo(f"\nFeature Areas: {stats.get('unique_feature_areas', 0)}")
            click.echo(f"Test Types: {stats.get('unique_test_types', 0)}")
            click.echo(f"Avg Requirements/Test: {stats.get('avg_requirements_per_test', 0):.2f}")
            click.echo(f"Avg Functions/Test: {stats.get('avg_functions_per_test', 0):.2f}")
            click.echo()

        # Ingest into databases
        click.echo("Ingesting into Neo4j and PostgreSQL...")
        ingestion = DataIngestion()

        # Convert documents to entities format
        entities = []
        relationships = []

        for doc in documents:
            entity = doc.metadata.get("entity", {})
            entity_relationships = doc.metadata.get("relationships", [])

            entities.append(
                {
                    "type": doc.metadata.get("entity_type"),
                    "data": entity,
                }
            )

            for rel in entity_relationships:
                relationships.append(
                    {
                        "source_id": entity["id"],
                        "source_label": doc.metadata.get("entity_type"),
                        "target_id": rel["target_id"],
                        "target_label": rel["target_label"],
                        "relationship_type": rel["type"],
                        "properties": rel.get("properties", {}),
                    }
                )

        # Prepare dataset
        dataset = {
            "entities": entities,
            "relationships": relationships,
            "metadata": {
                "source": "TGF CSV",
                "file": csv_path,
                "total_tests": len(documents),
            },
        }

        results = ingestion.ingest_full_dataset(dataset)

        # Display results
        click.echo("\n✓ TGF ingestion complete!")
        click.echo("\nResults:")
        click.echo(f"  Neo4j entities: {results['neo4j_entities']}")
        click.echo(f"  PostgreSQL entities: {results['postgres_entities']}")
        click.echo(f"  Relationships: {results['relationships']}")

        click.echo('\n\nNext step: agrag query "What tests failed for handover feature?"')

    except Exception as e:
        click.echo(f"\n✗ TGF loading failed: {e}", err=True)
        logger.exception("TGF loading failed")
        sys.exit(1)


@load.command("stats")
def load_stats():
    """Show statistics about loaded data."""
    from agrag.storage import Neo4jClient

    click.echo("\n=== Data Statistics ===\n")

    try:
        neo4j_client = Neo4jClient()

        # Count entities by type
        query = """
        MATCH (n)
        RETURN labels(n)[0] AS type, count(*) AS count
        ORDER BY count DESC
        """

        results = neo4j_client.execute_cypher(query)

        if results:
            click.echo("Entity Counts:")
            for row in results:
                entity_type = row.get("type", "Unknown")
                count = row.get("count", 0)
                click.echo(f"  {entity_type}: {count}")
        else:
            click.echo("No data found in the database.")
            click.echo("\nLoad some data first:")
            click.echo("  agrag load repo /path/to/repo")
            click.echo("  agrag load docs /path/to/docs")
            click.echo("  agrag load tgf /path/to/tgf.csv")

    except Exception as e:
        click.echo(f"\n✗ Failed to retrieve stats: {e}", err=True)
        logger.exception("Stats retrieval failed")
        sys.exit(1)


if __name__ == "__main__":
    cli()
