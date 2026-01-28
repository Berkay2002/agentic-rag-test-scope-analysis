# Project Overview

Agentic GraphRAG system for test scope analysis in telecommunications software.

## Stack
- Python 3.11+, Poetry
- LangGraph + LangChain
- LLM providers: Google Gemini or OpenAI-compatible (configurable)
- Storage: Neo4j + PostgreSQL/Neon (pgvector + BM25)
- Observability: LangSmith (optional)

## Architecture Summary
- ReAct-style agent built with a LangGraph StateGraph
- Four retrieval tools over a dual-database setup (PostgreSQL retrieval + Neo4j traversal)
- HITL approvals via LangGraph interrupts with optional Postgres checkpointing

## Recent Repo Layout (Jan 28, 2026)
- CLI split into `src/agrag/cli/app/`, `src/agrag/cli/commands/`, `src/agrag/cli/interactive/`, `src/agrag/cli/ui/`
- Tools split into `src/agrag/tools/retrieval/`, `src/agrag/tools/enhancements/`, `src/agrag/tools/shared/`
- Models split into `src/agrag/models/core/`, `src/agrag/models/providers/`, `src/agrag/models/services/`
- Evaluation split into `src/agrag/evaluation/baselines/`, `src/agrag/evaluation/evaluators/`,
  `src/agrag/evaluation/metrics/`, `src/agrag/evaluation/utils/`

## Where to Look
- Agent graph: `src/agrag/core/graph.py`
- State schema: `src/agrag/core/state.py`
- CLI entrypoint: `src/agrag/cli/app/main.py`
- Ontology: `src/agrag/kg/ontology.py`
- Retrieval tools: `src/agrag/tools/retrieval/`
