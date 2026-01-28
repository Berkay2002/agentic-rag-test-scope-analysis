# Testing and Code Style

## Tests
```bash
poetry run pytest
poetry run pytest tests/unit/test_vector_search_tool.py
poetry run pytest --cov=agrag --cov-report=html
```

## Formatting and Linting
```bash
poetry run black src/ tests/
poetry run ruff check src/ tests/
poetry run ruff check --fix src/ tests/
```

## Conventions
- Imports: stdlib → third-party → local
- Naming: PascalCase classes, snake_case functions, UPPER_SNAKE_CASE constants
- LangGraph: `AgentState` is a TypedDict; do not mutate state directly
- DB access: parameterized queries + context managers
- Tools: inherit `BaseTool` or use `@tool` with Pydantic `args_schema`; handle errors gracefully
