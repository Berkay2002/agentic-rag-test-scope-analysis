"""CLI application for Agentic GraphRAG Test Scope Analysis."""

import click
import logging
import sys
from typing import Optional

from agrag.config import setup_logging, settings
from agrag.storage import Neo4jClient, PostgresClient
from agrag.core import create_agent_graph, create_initial_state
from agrag.kg.ontology import NEO4J_CONSTRAINTS, NEO4J_VECTOR_INDEXES, POSTGRESQL_SCHEMA

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
        click.echo("  2. Run queries with: agrag query \"your question\"")

    except Exception as e:
        click.echo(f"\n✗ Error initializing schemas: {e}", err=True)
        logger.exception("Schema initialization failed")
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
    """Run a query against the GraphRAG system.

    Examples:
      agrag query "What tests cover requirement REQ_AUTH_005?"
      agrag query "Find all handover-related test cases"
      agrag query "Show me functions called by initiate_handover"
    """
    click.echo(f"\nQuery: {query_text}\n")

    try:
        # Create checkpointer if requested
        checkpointer = None
        config = {}

        if checkpoint:
            from langgraph.checkpoint.postgres import PostgresSaver
            import psycopg

            click.echo("[Checkpointer] Enabling PostgresSaver for HITL...")
            conn = psycopg.connect(settings.postgres_connection_string)
            checkpointer = PostgresSaver(conn)
            checkpointer.setup()

            if thread_id:
                config["configurable"] = {"thread_id": thread_id}
                click.echo(f"[Thread] Using thread_id: {thread_id}")

        # Create graph
        graph = create_agent_graph(checkpointer=checkpointer)

        # Create initial state
        initial_state = create_initial_state(query_text)

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

            # Get final result
            final_state = graph.get_state(config)
            final_answer = final_state.values.get("final_answer", "No answer generated")

        else:
            # Non-streaming execution
            final_state = graph.invoke(initial_state, config=config)
            final_answer = final_state.get("final_answer", "No answer generated")

        # Display final answer
        click.echo("\n--- Final Answer ---\n")
        click.echo(final_answer)

        # Display stats
        click.echo("\n--- Statistics ---")
        click.echo(f"Tool calls: {final_state.get('tool_call_count', 0)}")
        click.echo(f"Model calls: {final_state.get('model_call_count', 0)}")

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
def evaluate(dataset: str, output: str, k_values: str):
    """Run evaluation on a dataset.

    The dataset should be a JSON file with the following structure:
    [
        {
            "query": "...",
            "relevant_ids": ["id1", "id2", ...]
        },
        ...
    ]
    """
    import json
    from agrag.evaluation import evaluate_retrieval, mean_average_precision

    click.echo(f"\nEvaluating dataset: {dataset}")

    # Parse k values
    k_list = [int(k.strip()) for k in k_values.split(",")]
    click.echo(f"K values: {k_list}")

    try:
        # Load dataset
        with open(dataset, "r") as f:
            queries = json.load(f)

        click.echo(f"Loaded {len(queries)} queries\n")

        # Run evaluation
        all_results = []

        for i, query_data in enumerate(queries, 1):
            query = query_data["query"]
            relevant = set(query_data["relevant_ids"])

            click.echo(f"[{i}/{len(queries)}] {query[:60]}...")

            # TODO: Implement retrieval execution
            # For now, placeholder
            retrieved = []  # Would run actual retrieval here

            metrics = evaluate_retrieval(retrieved, relevant, k_values=k_list)
            all_results.append({
                "retrieved": retrieved,
                "relevant": relevant,
            })

            # Log metrics
            for metric_name, score in metrics.items():
                click.echo(f"  {metric_name}: {score:.4f}")

        # Calculate aggregate metrics
        map_score = mean_average_precision(all_results)
        click.echo(f"\n--- Aggregate Metrics ---")
        click.echo(f"MAP: {map_score:.4f}")

        # Save results
        with open(output, "w") as f:
            json.dump({
                "queries": len(queries),
                "map": map_score,
                "per_query_results": all_results,
            }, f, indent=2)

        click.echo(f"\n✓ Results saved to: {output}")

    except Exception as e:
        click.echo(f"\n✗ Evaluation failed: {e}", err=True)
        logger.exception("Evaluation failed")
        sys.exit(1)


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


if __name__ == "__main__":
    cli()
