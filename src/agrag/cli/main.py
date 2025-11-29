"""CLI application for Agentic GraphRAG Test Scope Analysis."""

import click
import logging
import sys
import json
from typing import Optional
from pathlib import Path

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
        config = {}

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

            # Note: Retrieval execution placeholder for evaluation framework
            retrieved = []  # Would run actual retrieval here

            metrics = evaluate_retrieval(retrieved, relevant, k_values=k_list)
            all_results.append(
                {
                    "retrieved": retrieved,
                    "relevant": relevant,
                }
            )

            # Log metrics
            for metric_name, score in metrics.items():
                click.echo(f"  {metric_name}: {score:.4f}")

        # Calculate aggregate metrics
        map_score = mean_average_precision(all_results)
        click.echo("\n--- Aggregate Metrics ---")
        click.echo(f"MAP: {map_score:.4f}")

        # Save results
        with open(output, "w") as f:
            json.dump(
                {
                    "queries": len(queries),
                    "map": map_score,
                    "per_query_results": all_results,
                },
                f,
                indent=2,
            )

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
def generate(requirements: int, testcases: int, output: str):
    """Generate synthetic telecommunications dataset.

    Creates realistic synthetic data including requirements, test cases,
    functions, classes, modules, and their relationships.

    Examples:
      agrag generate
      agrag generate --requirements 30 --testcases 150
      agrag generate --output my_dataset.json
    """
    from agrag.data.generators import TelecomDataGenerator

    click.echo("\n=== Generating Synthetic Dataset ===\n")
    click.echo(f"Requirements: {requirements}")
    click.echo(f"Test cases: {testcases}")
    click.echo(f"Output: {output}\n")

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
        click.echo(f"\nNext step: agrag ingest {output}")

    except Exception as e:
        click.echo(f"\n✗ Generation failed: {e}", err=True)
        logger.exception("Dataset generation failed")
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


if __name__ == "__main__":
    cli()
