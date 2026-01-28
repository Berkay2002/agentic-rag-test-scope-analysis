# AGENTS.md

Agentic GraphRAG system for test scope analysis in telecommunications software.

## Quick Reference
- Package manager: Poetry
- Python: 3.11+
- Setup: `poetry install && cp .env.example .env && poetry run agrag init`
- Run: `poetry run agrag chat` | `poetry run agrag query "..."` | `poetry run agrag -p "..." --output-format json`
- Tests: `poetry run pytest`
- Format/Lint: `poetry run black src/ tests/` and `poetry run ruff check src/ tests/`

## Required Env (choose by provider)
- `LLM_PROVIDER`, `EMBEDDINGS_PROVIDER`
- `GOOGLE_API_KEY` or `OPENAI_API_KEY` (+ `OPENAI_EMBEDDING_API_KEY` if needed)
- `NEO4J_URI`, `NEO4J_PASSWORD`
- `NEON_CONNECTION_STRING` or `POSTGRES_HOST`/`POSTGRES_USER`/`POSTGRES_PASSWORD`

## Detailed Guides
- [Overview](.agents/guides/overview.md)
- [Commands](.agents/guides/commands.md)
- [Architecture](.agents/guides/architecture.md)
- [Retrieval + Ontology](.agents/guides/retrieval-ontology.md)
- [Testing + Style](.agents/guides/testing-style.md)
- [Operations + Troubleshooting](.agents/guides/operations.md)
