# AGENTS.md

## Project Overview

This is an **Agentic GraphRAG system** for test scope analysis in telecommunications software. It combines Knowledge Graphs (Neo4j), Vector Search (pgvector), and LangGraph-based agent orchestration to analyze test coverage and dependencies.

**Core Technologies:**
- **Language**: Python 3.11+
- **Agent Framework**: LangGraph (custom StateGraph) + LangChain
- **LLM**: Google Generative AI (Gemini 2.0 Flash)
- **Databases**: Neo4j (graph + vector) + PostgreSQL/Neon (pgvector + full-text search)
- **Observability**: LangSmith (full tracing)
- **Package Manager**: Poetry

**Architecture**: Custom ReAct agent with 4 retrieval tools (vector search, keyword search, graph traversal, hybrid search) operating on a dual-database architecture to answer test scope queries.

## Setup Commands

### Initial Setup

```bash
# Clone and navigate
git clone <repository-url>
cd agentic-rag-test-scope-analysis

# Install dependencies
poetry install

# Copy environment template
cp .env.example .env

# Edit .env with your credentials (see Configuration section below)
nano .env
```

### Database Initialization

```bash
# Initialize Neo4j and PostgreSQL schemas
poetry run agrag init
```

This creates:
- Neo4j constraints for entity uniqueness
- Neo4j vector indexes (768-dim, cosine similarity)
- PostgreSQL pgvector extension and tables
- PostgreSQL full-text search indexes

### Generate Synthetic Test Data

```bash
# Generate telecommunications dataset (requirements, test cases, functions, etc.)
poetry run agrag generate --requirements 50 --testcases 200

# Ingest generated data into databases
poetry run agrag ingest data/synthetic_dataset.json
```

### Configuration

Required environment variables in `.env`:

```bash
# Google AI (REQUIRED)
GOOGLE_API_KEY=your_key_here
GOOGLE_THINKING_LEVEL=low  # optional: low/high reasoning depth (Gemini 3)
GOOGLE_THINKING_BUDGET=256  # optional: token budget for Gemini 2.5

# Neo4j (REQUIRED - use Neo4j Aura or local instance)
NEO4J_URI=neo4j+s://your_instance.databases.neo4j.io
NEO4J_PASSWORD=your_password

# PostgreSQL/Neon (REQUIRED)
NEON_CONNECTION_STRING=postgresql://user:pass@host:5432/dbname?sslmode=require

# LangSmith (OPTIONAL - for observability)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=agrag-test-scope-analysis
```

## Development Workflow

### Project Structure

```
src/agrag/
├── cli/              # Click-based CLI application
├── config/           # Settings (Pydantic) + logging
├── core/             # StateGraph agent (nodes, state, graph)
├── tools/            # 4 retrieval tools + schemas
├── storage/          # Neo4j + PostgreSQL clients
├── models/           # LLM + embedding wrappers
├── kg/               # Knowledge graph ontology
├── data/             # Data generators + ingestion
├── evaluation/       # Metrics (Precision@k, MAP, MRR)
└── observability/    # LangSmith utilities
```

### Running the Agent

```bash
# Interactive chat mode (safe by default - you approve each action)
poetry run agrag chat

# Resume a previous conversation
poetry run agrag chat --thread-id my-session

# YOLO mode (autonomous execution without approvals)
poetry run agrag chat --yolo

# Single query (non-interactive, for scripting)
poetry run agrag query "What tests cover handover requirements?"

# Query with streaming output (default)
poetry run agrag query "Find authentication test cases" --stream

# Query with HITL checkpointing
poetry run agrag query "Show dependencies for initiate_handover" --checkpoint --thread-id session-123

# Show system configuration
poetry run agrag info
```

#### Interactive Chat Mode

The `agrag chat` command starts an interactive REPL session similar to Claude Code, Codex, or Copilot CLI.

**Features:**
- Natural conversation with the agent
- **Automatic conversation persistence** (all chats are saved by default)
- Real-time streaming responses
- Command shortcuts (`/help`, `/stats`, `/exit`, etc.)
- Session resumption via thread IDs
- **Safe by default** - you approve each tool execution

**Understanding Modes:**

- **Safe Mode (Default)**: Agent asks for approval before each tool execution
  - You see: "I want to run vector_search with query X"
  - You decide: approve, reject, or modify
  - ✅ Safer, you control everything
  - Best for: normal usage, learning, sensitive operations

- **YOLO Mode** (`--yolo`): Agent executes autonomously
  - Agent runs tools without asking
  - Faster workflow
  - ⚠️ Less control - agent decides everything
  - Best for: trusted workflows, demos, when you're confident

**Available Commands in Chat:**
- `/help` - Show help message
- `/clear` - Clear the screen
- `/history` - View conversation history
- `/stats` - Show session statistics
- `/reset` - Start a new conversation
- `/save` - Save conversation to file
- `/thinking [preset]` - Adjust Gemini thinking budget (`low`, `medium`, `high`, `dynamic`, or integer tokens)
- `/exit` or `/quit` - Exit chat

**Example Chat Session:**
```
🤖 AgRAG Interactive Chat

Session ID: chat-a1b2c3d4
Your conversation is automatically saved.
Resume with: agrag chat --thread-id chat-a1b2c3d4

🚦 Safe Mode (HITL)
The agent will ask for your approval before executing each tool.

You: What tests cover handover requirements?

🚦 Approval Required
The agent wants to execute: vector_search(query="handover requirements tests", k=10)

Approve? (yes/no/edit): yes
✓ Approved. Continuing...

🔧 Executing: vector_search
📝 Found 8 results...

🤖 Agent Response:
Based on the search results, here are the test cases covering handover requirements:
- TC_HANDOVER_001: X2 handover between eNodeBs
- TC_HANDOVER_003: S1 handover validation
...

Tool calls: 1 | Model calls: 1

You: /stats
Session Statistics:
- Messages: 1
- Tool calls: 1
- Mode: 🚦 Safe Mode (you approve each tool)
```

### Working with Data

```bash
# Reset databases (WARNING: deletes all data)
poetry run agrag reset

# Generate fresh dataset
poetry run agrag generate --requirements 30 --testcases 150 --output my_data.json

# Ingest custom dataset
poetry run agrag ingest my_data.json
```

## Testing Instructions

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/unit/test_vector_search.py

# Run with coverage
poetry run pytest --cov=agrag --cov-report=html

# Run only unit tests
poetry run pytest tests/unit/

# Run only integration tests
poetry run pytest tests/integration/
```

### Test Structure

- `tests/unit/` - Unit tests for individual modules
- `tests/integration/` - Integration tests with databases
- `tests/evaluation/` - Evaluation framework tests

**Test File Naming**: Use `test_*.py` pattern (pytest convention)

### Key Test Areas

1. **Tool Execution**: Test each retrieval tool with mock database clients
2. **Metric Calculations**: Verify Precision@k, Recall@k, MAP, MRR implementations
3. **Ontology Validation**: Ensure entity/relationship schemas are consistent
4. **Agent Workflow**: Test StateGraph execution with checkpointing
5. **Data Ingestion**: Validate dual-database writes

### Before Committing

```bash
# Format code
poetry run black src/ tests/

# Lint
poetry run ruff check src/ tests/

# Run full test suite
poetry run pytest
```

## Code Style Guidelines

### Python Style

- **Formatter**: Black (line length: 100)
- **Linter**: Ruff
- **Type Hints**: Use throughout (no strict mypy enforcement yet)
- **Docstrings**: Google-style docstrings for public APIs

```bash
# Auto-format code
poetry run black src/

# Check linting
poetry run ruff check src/
```

### Project Conventions

#### File Organization
- One class per file (except small dataclasses)
- Group related functions in modules
- Use `__init__.py` for public API exports

#### Import Order
```python
# Standard library
import json
from typing import Optional

# Third-party
from pydantic import BaseModel
from langchain_core.tools import BaseTool

# Local
from agrag.storage import Neo4jClient
from agrag.tools.schemas import VectorSearchInput
```

#### Naming Conventions
- **Classes**: PascalCase (`VectorSearchTool`, `Neo4jClient`)
- **Functions/Methods**: snake_case (`create_agent_graph`, `execute_tools`)
- **Constants**: UPPER_SNAKE_CASE (`NEO4J_CONSTRAINTS`, `MAX_TOOL_CALLS`)
- **Private**: Leading underscore (`_internal_helper`)

#### LangGraph State Management
- Use TypedDict for AgentState
- Use `Annotated[list, add_messages]` for message history
- Never mutate state directly; return updated values from node functions

#### Database Clients
- All database operations go through `Neo4jClient` or `PostgresClient`
- Use connection pooling (built into drivers)
- Always close connections in context managers or destructors
- Use parameterized queries (never string interpolation)

#### Tool Implementation
- Inherit from `BaseTool` (LangChain)
- Define Pydantic `args_schema` for inputs
- Return structured data (Pydantic models preferred)
- Include detailed descriptions for LLM tool selection
- Handle errors gracefully (return error messages, don't raise)

## Knowledge Graph Ontology

### Entity Types

**Core entities** (defined in `src/agrag/kg/ontology.py`):

1. **Requirement** - System requirements
   - Properties: `id`, `priority`, `status`, `description`, `embedding`
   
2. **TestCase** - Test cases
   - Properties: `id`, `test_type`, `file_path`, `expected_outcome`, `embedding`
   
3. **Function** - Code functions
   - Properties: `id`, `signature`, `code_snippet`, `file_path`, `line_number`, `embedding`
   
4. **Class** - Code classes
   - Properties: `id`, `name`, `methods`, `file_path`, `embedding`
   
5. **Module** - Code modules/packages
   - Properties: `id`, `name`, `architectural_component`, `embedding`

### Relationship Types

1. `VERIFIES` - TestCase → Requirement (test validates requirement)
2. `COVERS` - TestCase → Function/Class (test exercises code)
3. `CALLS` - Function → Function (function invocation)
4. `DEFINED_IN` - Function/Class → Module (code location)
5. `BELONGS_TO` - Class → Module (class membership)
6. `DEPENDS_ON` - Requirement → Requirement (requirement dependency)

### Working with the Ontology

When adding new entity types or relationships:

1. Update `NodeLabel` and `RelationshipType` enums in `ontology.py`
2. Add corresponding Neo4j constraint to `NEO4J_CONSTRAINTS`
3. Add vector index if entity has embeddings
4. Update PostgreSQL schema if needed for vector/FTS support
5. Update data generators to produce the new entities
6. Update retrieval tools if new query patterns are needed

## Retrieval Tools

The agent has access to 4 retrieval tools (defined in `src/agrag/tools/`):

### 1. Vector Search (`vector_search`)

**Use for**: Semantic queries, conceptual understanding

**Example queries**:
- "tests related to handover failures"
- "authentication requirements"
- "functions dealing with network timeouts"

**Implementation**: Neo4j vector index (HNSW, cosine similarity, 768-dim)

**Parameters**:
- `query`: Search text
- `k`: Number of results (default: 10)
- `node_type`: Filter by entity type (optional)
- `similarity_threshold`: Minimum cosine similarity (default: 0.7)

### 2. Keyword Search (`keyword_search`)

**Use for**: Exact matches, identifiers, error codes

**Example queries**:
- "TestLoginTimeout"
- "REQ_AUTH_005"
- "ERROR_E503"

**Implementation**: PostgreSQL full-text search (ts_rank_cd ranking)

**Parameters**:
- `query`: Search keywords
- `k`: Number of results (default: 10)

### 3. Graph Traversal (`graph_traverse`)

**Use for**: Structural dependencies, multi-hop relationships

**Example queries**:
- "all tests covering functions called by UserManager"
- "requirements depending on REQ_HANDOVER_001"
- "classes defined in network_module"

**Implementation**: Cypher pattern matching with depth limits

**Parameters**:
- `start_node_id`: Starting node ID
- `start_node_label`: Starting node type (e.g., "Requirement")
- `relationship_types`: List of relationship types to traverse
- `depth`: Maximum traversal depth (default: 2)
- `direction`: "outgoing", "incoming", or "both"

### 4. Hybrid Search (`hybrid_search`)

**Use for**: Complex queries needing semantic + lexical precision

**Example queries**:
- "tests for LTE signaling with timeout errors"
- "handover functions with retry logic"

**Implementation**: RRF fusion of vector + keyword results

**Parameters**:
- `query`: Search text
- `k`: Number of results (default: 10)
- `rrf_k`: RRF constant (default: 60)

## Evaluation Framework

Located in `src/agrag/evaluation/metrics.py`:

### Available Metrics

- `precision_at_k(retrieved, relevant, k)` - Precision@k
- `recall_at_k(retrieved, relevant, k)` - Recall@k
- `f1_score_at_k(retrieved, relevant, k)` - F1@k
- `average_precision(retrieved, relevant)` - Average Precision
- `mean_average_precision(results)` - MAP across queries
- `reciprocal_rank(retrieved, relevant)` - Reciprocal Rank
- `mean_reciprocal_rank(results)` - MRR across queries
- `evaluate_retrieval(retrieved, relevant, k_values)` - All metrics at once

### Running Evaluations

```bash
# Run evaluation on dataset
poetry run agrag evaluate --dataset data/eval_queries.json --output results.json --k-values "1,3,5,10"
```

**Evaluation dataset format** (`eval_queries.json`):
```json
[
  {
    "query": "tests for handover timeout",
    "relevant_ids": ["TC_HANDOVER_001", "TC_HANDOVER_003", "TC_TIMEOUT_012"]
  },
  {
    "query": "authentication requirements",
    "relevant_ids": ["REQ_AUTH_001", "REQ_AUTH_005"]
  }
]
```

### Creating Test Queries

When creating evaluation datasets:
1. Use diverse query types (semantic, structural, hybrid)
2. Include 3-10 relevant items per query
3. Cover different entity types (requirements, tests, functions)
4. Include edge cases (no results, ambiguous queries)
5. Balance easy and hard queries

## Build and Deployment

### Local Development

```bash
# Install in development mode
poetry install

# Activate virtual environment
poetry shell

# Run CLI directly
agrag query "your question here"
```

### Package Building

```bash
# Build wheel and sdist
poetry build

# Output in dist/
ls dist/
# agrag-0.1.0-py3-none-any.whl
# agrag-0.1.0.tar.gz
```

### Environment-Specific Configuration

The system uses Pydantic Settings with environment variable priority:

1. Environment variables (highest priority)
2. `.env` file
3. Default values in `settings.py`

**For production**:
- Set `LANGCHAIN_TRACING_V2=false` (unless using LangSmith)
- Use connection pooling for databases
- Set `MAX_TOOL_CALLS` and `MAX_MODEL_CALLS` for cost control
- Use `AGENT_TEMPERATURE=0.0` for deterministic behavior

### Database Setup

**Neo4j** (use Neo4j Aura for cloud):
1. Create database instance
2. Enable APOC plugin (for advanced operations)
3. Note connection URI and password
4. Run `agrag init` to create schema

**PostgreSQL** (use Neon for serverless):
1. Create database instance
2. Ensure PostgreSQL 15+ with pgvector extension
3. Note connection string
4. Run `agrag init` to create schema

## Human-in-the-Loop (HITL) Workflows

The agent uses HITL by default for safety - you approve each action before it executes.

### Understanding Modes

**Safe Mode (Default - HITL Enabled):**
```bash
poetry run agrag chat
```
- Agent proposes tool calls and pauses
- You review what the agent wants to do
- You approve, reject, or modify the action
- Agent executes only after your approval
- ✅ Best for: normal usage, learning, cost control, sensitive operations

**YOLO Mode (Autonomous):**
```bash
poetry run agrag chat --yolo
```
- Agent executes all tool calls autonomously
- No approval prompts - agent decides everything
- Faster workflow but less control
- ⚠️ Best for: trusted workflows, demos, when you're very confident

### How HITL Works

**Default behavior (Safe Mode):**
1. You ask a question
2. Agent analyzes and decides which tools to use
3. **Agent pauses** and shows: "I want to execute vector_search with parameters X"
4. You review the proposed action
5. You respond: approve (yes), reject (no), or edit parameters
6. Agent executes (if approved) or stops (if rejected)
7. Process repeats for each tool call

### Example HITL Session

```
You: Find tests for authentication timeout

🚦 Approval Required

The agent wants to execute the following tools:
- vector_search(query="authentication timeout tests", k=10)

Approve? (yes/no/edit): yes
✓ Approved. Continuing...

🔧 Executing: vector_search
📝 Found 8 results...

🚦 Approval Required

The agent wants to execute the following tools:
- graph_traverse(start_node="REQ_AUTH_005", relationship="VERIFIES")

Approve? (yes/no/edit): yes
✓ Approved. Continuing...

🤖 Agent Response: [complete answer]
```

### When to Use Each Mode

**Use Safe Mode (default) when:**
- You're exploring new queries
- Working with real/production data
- Learning how the agent works
- Controlling costs (LLM API calls)
- You want full visibility and control

**Use YOLO Mode (`--yolo`) when:**
- You trust the agent completely for this task
- Running demos or presentations
- Doing batch processing with known patterns
- Speed is more important than control

### Implementation Details

**Checkpointer**: PostgresSaver stores agent state in PostgreSQL
**Interrupts**: Configured with `interrupt_before=["execute_tools"]`
**State Recovery**: Can resume from any checkpoint using thread ID
**Persistence**: All conversation history and tool results saved

### Programmatic HITL

```python
from langgraph.checkpoint.postgres import PostgresSaver
from agrag.core import create_agent_graph, create_initial_state

# Create graph with checkpointing
checkpointer = PostgresSaver.from_conn_string(conn_string)
graph = create_agent_graph(checkpointer=checkpointer)

# Run with thread
config = {"configurable": {"thread_id": "session-123"}}
for event in graph.stream(initial_state, config):
    if "__interrupt__" in event:
        # Review proposed tool calls
        proposed_calls = event["execute_tools"]["messages"][-1].tool_calls
        
        # Options:
        # 1. Approve: graph.update_state(config, None)
        # 2. Edit: graph.update_state(config, modified_calls)
        # 3. Reject: graph.update_state(config, {"messages": [...]})
```

## Observability and Debugging

### LangSmith Integration

Automatic tracing when configured:
- Every LLM call (reasoning steps)
- Every tool execution (inputs/outputs)
- Graph state transitions
- Retrieval results with scores
- Error traces

**View traces**: https://smith.langchain.com/

### Logging

Structured logging configured in `src/agrag/config/logging_config.py`:

```bash
# Set log level
poetry run agrag --log-level DEBUG query "..."

# JSON format for log aggregation
poetry run agrag --log-format json query "..."
```

**Log locations**:
- Console: Formatted output (default: INFO)
- File: `logs/agrag.log` (if configured)

### Debugging Agent Behavior

1. **Enable streaming**: `--stream` flag shows real-time agent reasoning
2. **Check LangSmith**: View full execution traces
3. **Inspect state**: Use `--checkpoint` to save/inspect agent state
4. **Tool debugging**: Check tool inputs/outputs in logs
5. **Database debugging**: Query Neo4j/PostgreSQL directly to verify data

### Common Issues

**Issue**: Agent makes too many tool calls
- **Fix**: Reduce `MAX_TOOL_CALLS` in `.env`
- **Check**: LangSmith trace for reasoning loops

**Issue**: Poor retrieval results
- **Fix**: Check embedding quality, adjust similarity thresholds
- **Check**: Database has sufficient data, indexes are created

**Issue**: Database connection errors
- **Fix**: Verify connection strings in `.env`
- **Check**: Database instances are running and accessible

## Research Questions Alignment

This project addresses three research questions:

### RQ1: Knowledge Graph Ontology
**Implementation**: `src/agrag/kg/ontology.py`
- 5 entity types, 6 relationship types
- Optimized for software engineering domain
- Vector embeddings for all entities

### RQ2: Retrieval Strategy Comparison
**Implementation**: `src/agrag/tools/` + `src/agrag/evaluation/`
- 4 retrieval strategies (vector, keyword, graph, hybrid)
- Evaluation metrics (Precision@k, Recall@k, MAP, MRR)
- Baseline comparison framework

### RQ3: Human-in-the-Loop Workflows
**Implementation**: `src/agrag/core/graph.py` + LangGraph checkpointing
- PostgresSaver for state persistence
- Interrupt points for human approval
- Thread-based conversation management

## Development Tips

### Fast Iteration

```bash
# Use Python's `-m` flag for quick testing
poetry run python -m agrag.tools.vector_search

# Or enter Poetry shell
poetry shell
python -m agrag.tools.vector_search
```

### Database Inspection

**Neo4j Browser**: http://localhost:7474 (or Aura console)

```cypher
// Count all entities
MATCH (n) RETURN labels(n)[0] as type, count(*) as count

// Sample requirements
MATCH (r:Requirement) RETURN r LIMIT 10

// Find test coverage
MATCH (t:TestCase)-[:VERIFIES]->(r:Requirement)
RETURN r.id, collect(t.id) as tests
```

**PostgreSQL queries**:
```sql
-- Count chunks
SELECT COUNT(*) FROM document_chunks;

-- Sample embeddings
SELECT chunk_id, content, embedding <=> '[0.1, 0.2, ...]'::vector AS distance
FROM document_chunks
ORDER BY distance
LIMIT 10;
```

### Modifying the Agent

**To change system prompt**: Edit `src/agrag/core/graph.py`, function `create_agent_graph()`

**To add a tool**:
1. Create tool class in `src/agrag/tools/`
2. Define Pydantic schema in `src/agrag/tools/schemas.py`
3. Add to tool list in `create_agent_graph()`

**To modify state**:
1. Update `AgentState` in `src/agrag/core/state.py`
2. Update node functions in `src/agrag/core/nodes.py`

### Performance Optimization

- **Batch embedding generation**: Use `EmbeddingService.embed_batch()`
- **Connection pooling**: Clients reuse connections
- **Vector index**: HNSW provides O(log N) search
- **Cypher optimization**: Use EXPLAIN/PROFILE for query optimization

## Additional Resources

- **LangChain Docs**: https://python.langchain.com/
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Neo4j Cypher**: https://neo4j.com/docs/cypher-manual/
- **pgvector**: https://github.com/pgvector/pgvector
- **LangSmith**: https://smith.langchain.com/

## Contributing

When making changes:

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Run tests**: `poetry run pytest`
3. **Format code**: `poetry run black src/ tests/`
4. **Lint**: `poetry run ruff check src/ tests/`
5. **Update docs**: If adding features, update README.md and this file
6. **Commit**: Use clear commit messages
7. **Test end-to-end**: Run full query workflow before pushing

## Quick Reference

### Most Common Commands

```bash
# Setup
poetry install
cp .env.example .env
poetry run agrag init

# Data
poetry run agrag generate
poetry run agrag ingest data/synthetic_dataset.json

# Usage - Interactive Chat (Recommended)
poetry run agrag chat              # Safe mode (approve each tool)
poetry run agrag chat --yolo       # YOLO mode (autonomous)

# Usage - Single Queries
poetry run agrag query "your question"
poetry run agrag info

# Testing
poetry run pytest
poetry run black src/
poetry run ruff check src/

# Cleanup
poetry run agrag reset
```

### Key Files to Know

- `pyproject.toml` - Dependencies and build config
- `.env.example` - Environment variable template
- `src/agrag/config/settings.py` - Configuration management
- `src/agrag/kg/ontology.py` - Data model (entities + relationships)
- `src/agrag/core/graph.py` - Agent implementation
- `src/agrag/tools/` - Retrieval tools
- `src/agrag/cli/main.py` - CLI commands
