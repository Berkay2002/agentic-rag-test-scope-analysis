# AGENTS.md

## Project Overview

**Agentic GraphRAG system** for test scope analysis in telecommunications software.

**Stack:** Python 3.11+, Poetry | LangGraph + LangChain | Google Gemini LLM | Neo4j + PostgreSQL/Neon (pgvector HNSW + pg_search BM25) | LangSmith

**Architecture**: ReAct agent with 4 retrieval tools operating on dual-database architecture (PostgreSQL for retrieval, Neo4j for graph traversal).

## Setup

```bash
poetry install
cp .env.example .env  # Add API keys, DB creds
poetry run agrag init  # Creates DB schemas, indexes
```

**Required env vars:** `GOOGLE_API_KEY`, `NEO4J_URI`, `NEO4J_PASSWORD`, `NEON_CONNECTION_STRING`

**Generate data:**
```bash
poetry run agrag generate --requirements 50 --testcases 200
poetry run agrag ingest data/synthetic_dataset.json
```

## Project Structure

```
src/agrag/
├── cli/              # Click-based CLI
├── config/           # Pydantic settings, logging
├── core/             # StateGraph agent (state, graph)
├── tools/            # 4 retrieval tools + schemas
├── storage/          # Neo4j, PostgreSQL clients
├── models/           # LLM, embeddings
├── kg/               # Knowledge graph ontology
├── data/             # Data generators, ingestion
├── evaluation/       # P@k, MAP, MRR metrics
└── observability/    # LangSmith
```

**Codex skills**: `.codex/skills/` - Lightweight agent commands

## Running the Agent

```bash
# Interactive chat (safe mode - approves each tool)
poetry run agrag chat
poetry run agrag chat --thread-id my-session

# YOLO mode (autonomous execution)
poetry run agrag chat --yolo

# Headless (scripting)
poetry run agrag -p "query here" --output-format json
poetry run agrag query "your question" --stream
poetry run agrag info  # Show config
```

### Chat Commands
`/help`, `/clear`, `/history`, `/stats`, `/reset`, `/save`, `/export`, `/verbose`, `/thinking [preset]`, `/exit`

### HITL (Human-in-the-Loop)

**Safe mode** (default): Agent pauses before each tool. You approve, reject, or edit.

**YOLO mode** (`--yolo`): Agent executes autonomously.

**Example:**
```
You: What tests cover handover?
🚦 Approval Required
Agent wants: vector_search(query="handover tests", k=10)
Approve? (yes/no/edit): yes
✓ Approved. Executing...
Agent Response: [answer]
```

### Working with Data

```bash
poetry run agrag reset  # WARNING: deletes all data
poetry run agrag generate --requirements 30 --testcases 150
poetry run agrag ingest my_data.json
```

## Testing

```bash
poetry run pytest                           # All tests
poetry run pytest tests/unit/test_vector_search.py
poetry run pytest --cov=agrag --cov-report=html
poetry run black src/ tests/                # Format
poetry run ruff check src/ tests/           # Lint
poetry run ruff check --fix src/ tests/     # Fix auto-fixable
```

**Test structure:** `tests/unit/`, `tests/integration/`, `tests/evaluation/`

## Code Style

- **Black**: line length 100
- **Ruff**: linting
- **Imports**: stdlib → third-party → local
- **Naming**: PascalCase classes, snake_case functions, UPPER_SNAKE_CASE constants
- **LangGraph**: TypedDict state, never mutate directly
- **DB clients**: Parameterized queries, context managers
- **Tools**: Inherit BaseTool, Pydantic args_schema, handle errors gracefully

## Knowledge Graph Ontology

**Entities** (`src/agrag/kg/ontology.py`):
- ChangeRequest, File, Component
- Requirement (priority, status)
- TestCase (test_type, file_path)
- Function (signature, file_path, line_number)
- Class, Module (legacy optional)

**Relationships:**
- `TOUCHES`: ChangeRequest → File
- `DEFINED_IN`: Function → File
- `PART_OF`: File → Component
- `COVERS`: TestCase → Function
- `VERIFIES`: TestCase → Requirement

**Adding entities:** Update enums → Add Neo4j constraint → Add index → Update PostgreSQL → Update generators → Update tools

## Retrieval Tools

### Vector Search
- **For**: Semantic queries
- **Impl**: PostgreSQL pgvector (HNSW, cosine, 768-dim)
- **Params**: query, k (10 default), node_type (filter)

### Keyword Search
- **For**: Exact matches, identifiers
- **Impl**: PostgreSQL pg_search BM25
- **Params**: query, k, entity_type

### Graph Traversal
- **For**: Dependencies, multi-hop relationships
- **Impl**: Cypher pattern matching
- **Params**: start_node_id, start_node_label, relationship_types, depth (2), direction

### Hybrid Search
- **For**: Complex queries (semantic + lexical)
- **Impl**: PostgreSQL-native RRF fusion
- **Params**: query, k, rrf_k (60), entity_type

## Evaluation

**Metrics** (`src/agrag/evaluation/metrics.py`):
- Precision@k, Recall@k, F1@k
- Average Precision, MAP
- Reciprocal Rank, MRR

**Run eval:**
```bash
poetry run agrag evaluate \
  --dataset data/eval_queries.json \
  --output results.json \
  --k-values "1,3,5,10"
```

**Dataset format:**
```json
[{"query": "tests for handover", "relevant_ids": ["TC_001", "TC_003"]}]
```

## Observability

**LangSmith**: Full tracing of LLM calls, tool execution, state transitions, errors

**Logging** (`src/agrag/config/logging_config.py`):
```bash
poetry run agrag --log-level DEBUG query "..."
poetry run agrag --log-format json query "..."
```

**Common issues:**
- Too many tool calls → Reduce `MAX_TOOL_CALLS` in `.env`
- Poor retrieval → Check embeddings, indexes, data volume
- DB connection errors → Verify `.env` connection strings

## Development Tips

**Database inspection:**
```cypher
// Neo4j
MATCH (n) RETURN labels(n)[0] as type, count(*) as count
MATCH (t:TestCase)-[:VERIFIES]->(r:Requirement)
RETURN r.id, collect(t.id) as tests

-- PostgreSQL
SELECT COUNT(*) FROM document_chunks;
SELECT chunk_id, content, embedding <=> '[...]'::vector AS distance FROM document_chunks ORDER BY distance LIMIT 10;
SELECT chunk_id, content, paradedb.score(id) FROM document_chunks WHERE content @@@ 'handover' ORDER BY paradedb.score(id) DESC LIMIT 10;
```

**Modifying the agent:**
- Change system prompt → Edit `src/agrag/core/graph.py`
- Add a tool → Create in `src/agrag/tools/` → Define schema in `schemas.py` → Add to tool list in `create_agent_graph()`
- Modify state → Update `AgentState` in `src/agrag/core/state.py` → Update nodes in `nodes.py`

## Quick Reference

**Most common commands:**
```bash
poetry install && cp .env.example .env && poetry run agrag init
poetry run agrag generate && poetry run agrag ingest data/synthetic_dataset.json
poetry run agrag chat              # Safe mode
poetry run agrag chat --yolo       # YOLO mode
poetry run agrag query "your question"
poetry run pytest && poetry run black src/ && poetry run ruff check src/
```

**Key files:**
- `pyproject.toml` - Dependencies
- `.env.example` - Env vars
- `src/agrag/config/settings.py` - Config
- `src/agrag/kg/ontology.py` - Data model
- `src/agrag/core/graph.py` - Agent implementation
- `src/agrag/tools/` - Retrieval tools
- `src/agrag/cli/main.py` - CLI commands
