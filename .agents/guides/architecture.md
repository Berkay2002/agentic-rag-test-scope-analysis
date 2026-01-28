# Architecture

## Core Agent
- LangGraph StateGraph with a custom ReAct loop
- Graph construction: `src/agrag/core/graph.py`
- State schema: `src/agrag/core/state.py`
- Nodes: `src/agrag/core/nodes.py`
- HITL checkpointer utilities: `src/agrag/core/checkpointing.py`

## Retrieval Tools
- Factory functions in `src/agrag/tools/retrieval/`
- Shared schemas/helpers in `src/agrag/tools/shared/`
- Optional enhancements in `src/agrag/tools/enhancements/`

## Storage Layer
- Neo4j client: `src/agrag/storage/neo4j_client.py`
- PostgreSQL client: `src/agrag/storage/postgres_client.py`
- BM25/pgvector utilities under `src/agrag/storage/`

## Data Pipeline
- Generators: `src/agrag/data/generators/`
- Loaders (Docling + TGF): `src/agrag/data/loaders/`
- Ingestion: `src/agrag/data/ingestion.py`
- Storage writers: `src/agrag/data/storage_writers.py`

## Models
- Core abstractions: `src/agrag/models/core/`
- Provider implementations: `src/agrag/models/providers/`
- Service wrappers: `src/agrag/models/services/`

## Evaluation
- Baselines: `src/agrag/evaluation/baselines/`
- Evaluators: `src/agrag/evaluation/evaluators/`
- Metrics: `src/agrag/evaluation/metrics/`
- Utilities: `src/agrag/evaluation/utils/`

## CLI
- Entry: `src/agrag/cli/app/main.py`
- Commands: `src/agrag/cli/commands/`
- Interactive UI: `src/agrag/cli/interactive/`
- Console display: `src/agrag/cli/ui/`
- Helpers: `src/agrag/cli/utils/`
