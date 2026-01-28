# AgRAG CLI

This CLI supports interactive chat, headless automation, and task-specific commands
like `query`, `evaluate`, and `load`. Use `poetry run agrag ...` if you have not
installed the package into your environment.

## Interactive Chat

```bash
poetry run agrag chat
poetry run agrag chat --thread-id my-session
poetry run agrag chat --yolo
poetry run agrag chat --verbose
```

Chat commands:
- `/help` - Show help
- `/clear` - Clear the screen
- `/history` - View conversation history
- `/stats` - Show session statistics
- `/reset` - Start a new conversation
- `/save` - Save conversation to file
- `/export [filename] [--verbose]` - Export transcript (verbose includes tool args/results)
- `/verbose` - Toggle tool call arguments in output
- `/thinking [level|preset|tokens]` - Set thinking level or legacy budget
- `/exit` or `/quit` - Exit chat

Notes:
- `/export` defaults to a clean transcript without tool arguments/results.
- Use `/export --verbose` (or `/export debug`) for a debug transcript.

## Headless Mode

Headless mode runs without the interactive UI and is intended for scripting,
automation, and CI.

```bash
# Direct prompt
poetry run agrag -p "What tests cover handover requirements?"

# Stdin
echo "Summarize this" | poetry run agrag

# Combine prompt with stdin
cat README.md | poetry run agrag -p "Summarize this documentation"
```

Output formats:
- `text` (default): plain text response
- `json`: structured response with stats
- `stream-json`: JSONL events for live progress

```bash
poetry run agrag -p "List test cases" --output-format json
poetry run agrag -p "Analyze dependencies" --output-format stream-json
```

Streaming event types:
`init`, `message`, `tool_use`, `tool_result`, `error`, `result`.

Persistent headless sessions:
```bash
poetry run agrag -p "List handover requirements" --thread-id eval-001
poetry run agrag -p "Now show tests that verify those" --thread-id eval-001
```

When `--thread-id` is provided, the CLI attempts to use the Postgres checkpointer
to resume state between runs (falls back to in-memory if unavailable).

## Other Commands (Quick Reference)

```bash
poetry run agrag query "Find authentication test cases"
poetry run agrag init
poetry run agrag generate --requirements 50 --testcases 200
poetry run agrag ingest data/mock/synthetic_dataset.json
poetry run agrag evaluate --dataset data/mock/eval_queries.json --strategy all
poetry run agrag evaluate --suite synthetic-capability
poetry run agrag info
poetry run agrag load docs /path/to/docs --use-chunker
```

Notes:
- `--prompt/-p` is only valid without subcommands.
- Use `--debug` with headless runs to keep logging enabled.
