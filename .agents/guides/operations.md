# Operations and Troubleshooting

## Required Environment
Choose providers in `.env`:
- `LLM_PROVIDER` and `EMBEDDINGS_PROVIDER` (google | openai)
- If google: `GOOGLE_API_KEY`
- If openai: `OPENAI_API_KEY` (plus `OPENAI_EMBEDDING_API_KEY` if embeddings use a different key)
- Neo4j: `NEO4J_URI`, `NEO4J_PASSWORD`
- PostgreSQL: `NEON_CONNECTION_STRING` or `POSTGRES_HOST`/`POSTGRES_USER`/`POSTGRES_PASSWORD`

## HITL + Checkpointing
- Safe mode requires a checkpointer for approvals.
- Postgres checkpointer is preferred; falls back to in-memory if unavailable.
- If `--thread-id` is provided without a checkpointer, the CLI will ignore it.

## Common Issues
- Too many tool calls: lower `MAX_TOOL_CALLS` in `.env`.
- Retrieval quality: verify embeddings, indexes, and data volume.
- DB errors: confirm `.env` credentials and Neo4j/Postgres availability.

## Observability
- LangSmith tracing is optional; enable with `LANGCHAIN_TRACING_V2` and `LANGCHAIN_API_KEY`.
- `LANGSMITH_*` aliases are also supported.
