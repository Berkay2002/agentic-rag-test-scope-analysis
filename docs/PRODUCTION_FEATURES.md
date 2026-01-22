# Production Polish Features

This document describes the three "production polish" features that enhance the CLI agent's usability and robustness. These features translate web UI patterns into terminal UX patterns.

## 1. Reasoning Display (Separate "Thinking" from "Answer")

### Overview
Modern LLMs (especially Gemini with thinking mode) can produce structured outputs with multiple content blocks:
- **Reasoning blocks**: Internal deliberation, thinking process
- **Text blocks**: Final answer content
- **Tool calls**: Function invocations

The CLI now displays reasoning blocks separately from final answers, making the agent's thought process transparent.

### Features
- **Automatic extraction**: Reasoning blocks are automatically detected and extracted
- **Visual distinction**: Reasoning displayed with 🧠 icon and different styling
- **Collapsed by default**: Shows summary line to avoid flooding the terminal
- **Verbose mode**: Expands full reasoning when `/verbose` is enabled
- **No sensitive data leaks**: Reasoning can be hidden by default to avoid accidental exposure

### Usage

#### View reasoning in collapsed mode (default):
```bash
You: What tests cover authentication?
🧠 Thinking  12:34:56
┌─────────────────────────────────────────────┐
│ 1 reasoning block(s), ~245 chars           │
│ (use /verbose to expand)                   │
└─────────────────────────────────────────────┘

● Assistant  12:34:56
┌─────────────────────────────────────────────┐
│ Here are the tests covering authentication: │
│ ...                                         │
└─────────────────────────────────────────────┘
```

#### View reasoning expanded:
```bash
agrag chat --verbose

# Or toggle in-session:
You: /verbose on
✓ Verbose mode is now on

You: What tests cover authentication?
🧠 Thinking  12:34:56
┌─────────────────────────────────────────────┐
│ Internal Deliberation                       │
│ ─────────────────────────────────────────── │
│ Let me analyze this query...                │
│ I should use keyword_search since the user  │
│ mentioned "authentication" explicitly...    │
└─────────────────────────────────────────────┘
```

### Implementation Details
- **Function**: `extract_reasoning_and_answer()` in `src/agrag/cli/utils.py`
- **Display**: `print_reasoning()` in `src/agrag/cli/display.py`
- **Integration**: Modified `_process_event()` in `src/agrag/cli/interactive.py`

### Configuration
Reasoning display is controlled by:
- `/verbose` command: Toggle expanded reasoning
- `GOOGLE_THINKING_LEVEL` env var: Control LLM thinking depth (low/medium/high/minimal)
- `GOOGLE_THINKING_BUDGET` env var: Control thinking token budget

---

## 2. Resumable Streaming (Reconnect After Interruptions)

### Overview
Production CLI applications must handle interruptions gracefully:
- Terminal closes
- SSH drops
- Process crashes
- User hits Ctrl+C

The CLI now supports durable session identity and checkpoint-based resumption.

### Features
- **Durable thread IDs**: Conversations persist across sessions
- **Checkpoint history**: View all saved conversation states
- **Seamless resumption**: Continue from where you left off
- **PostgreSQL backend**: Durable storage (or in-memory fallback)

### Usage

#### Resume a previous session:
```bash
# Start new session (auto-generated thread ID)
agrag chat
# Thread ID: chat-a1b2c3d4

# Later, reconnect to same session
agrag chat --thread-id chat-a1b2c3d4
```

#### View checkpoint history:
```bash
You: /history

# or
You: /threads

┌────┬──────────────────────┬──────────┬─────────────────┐
│ #  │ Checkpoint ID        │ Messages │ Namespace       │
├────┼──────────────────────┼──────────┼─────────────────┤
│ 1  │ checkpoint-abc123... │ 5        │ default         │
│ 2  │ checkpoint-def456... │ 3        │ default         │
│ 3  │ checkpoint-ghi789... │ 1        │ default         │
└────┴──────────────────────┴──────────┴─────────────────┘
```

#### List available threads:
```bash
# Coming soon: List all threads across all sessions
agrag chat --list-threads
```

### Implementation Details
- **Checkpointer**: PostgreSQL-based checkpoint saver (LangGraph)
- **Thread ID**: Stored in `thread_id` field, persisted across sessions
- **Configuration**: `NEON_CONNECTION_STRING` for PostgreSQL

### Fallback Behavior
- If PostgreSQL is unavailable, falls back to in-memory checkpointer
- Session persists within single process only
- Warning displayed on startup

---

## 3. Branching Conversations (Time-Travel and Forks)

### Overview
Conversations are not linear—users may want to:
- Try different phrasings of a question
- Explore alternative agent paths
- Compare different reasoning approaches
- Fork from a past checkpoint

The CLI now supports git-like branching for conversation threads.

### Features
- **Checkpoint navigation**: View conversation history
- **Branch creation**: Fork from any checkpoint
- **Branch listing**: See all branches
- **Explicit branching**: No silent history overwriting

### Usage

#### View conversation checkpoints:
```bash
You: /history

┌────┬──────────────────────┬──────────┬─────────────────┐
│ #  │ Checkpoint ID        │ Messages │ Namespace       │
├────┼──────────────────────┼──────────┼─────────────────┤
│ 1  │ checkpoint-001       │ 2        │ default         │
│ 2  │ checkpoint-002       │ 4        │ default         │
│ 3  │ checkpoint-003       │ 6        │ default         │
└────┴──────────────────────┴──────────┴─────────────────┘
```

#### Create a branch from checkpoint:
```bash
You: /fork checkpoint-002 experiment-1
✓ Created branch from checkpoint checkpoint-002
New thread ID: chat-a1b2c3d4_experiment-1

To switch to this branch, restart with:
agrag chat --thread-id chat-a1b2c3d4_experiment-1
```

#### Auto-generate branch name:
```bash
You: /fork checkpoint-002
✓ Created branch from checkpoint checkpoint-002
New thread ID: chat-a1b2c3d4_fork_20240115_143022
```

#### List branches:
```bash
You: /branches

┌────────────┬──────────────────────┬──────────┬─────────────────────┐
│ Branch     │ Checkpoint           │ Messages │ Created             │
├────────────┼──────────────────────┼──────────┼─────────────────────┤
│ main       │ latest               │ 0        │ 2024-01-15 14:30:22 │
└────────────┴──────────────────────┴──────────┴─────────────────────┘
```

### Git-Like Workflow

```bash
# 1. Start conversation
agrag chat --thread-id my-analysis

# 2. Have a conversation (multiple turns)
You: What tests cover authentication?
Assistant: [answer]

You: What about authorization?
Assistant: [answer]

# 3. View checkpoints
You: /history
# See checkpoint IDs

# 4. Fork from an earlier point
You: /fork checkpoint-002 try-different-approach

# 5. Exit and switch to branch
exit
agrag chat --thread-id my-analysis_try-different-approach

# 6. Continue from forked point
You: Actually, show me handover tests instead
Assistant: [different path]
```

### Implementation Details
- **Module**: `src/agrag/cli/branching.py`
- **Class**: `BranchManager` handles checkpoint navigation
- **Storage**: Branches stored as thread IDs with parent metadata
- **Commands**: `/history`, `/branches`, `/fork`

### Limitations
- Branch switching requires restarting the chat
- No in-session branch switching (yet)
- Branch comparison is manual (view both thread logs)
- No branch merging or deletion (yet)

---

## Cross-Cutting Concerns

### Terminal Rendering
- **Interactive mode**: Rich formatting with panels and colors
- **Non-interactive mode**: Plain text for pipes/redirects
- **Logging separation**: User-facing output vs diagnostic logs

### Security Considerations
- **Reasoning visibility**: Default to collapsed to avoid data leaks
- **PII detection**: Not yet implemented (future enhancement)
- **Tool argument redaction**: Sensitive args should be masked

### Failure Semantics
- **Mid-stream failures**: Checkpoint saved before error
- **Recovery**: Resume from last checkpoint
- **Clear error messages**: "What failed" + "How to recover"

---

## Configuration Reference

### Environment Variables

```bash
# LLM Thinking
GOOGLE_THINKING_LEVEL=medium          # low|medium|high|minimal
GOOGLE_THINKING_BUDGET=1024           # Token budget (or -1 for dynamic)

# Persistence
NEON_CONNECTION_STRING=postgresql://user:pass@host/db  # PostgreSQL for checkpoints

# Tracing
LANGCHAIN_TRACING_V2=true             # Enable LangSmith tracing
LANGCHAIN_PROJECT=agrag-dev           # Project name
```

### CLI Flags

```bash
# Chat mode
agrag chat                            # Start new session (safe mode)
agrag chat --yolo                     # Autonomous mode (no approvals)
agrag chat --verbose                  # Show tool details and expanded reasoning
agrag chat --thread-id <id>           # Resume/fork specific thread

# Query mode
agrag query "question" --thread-id <id>     # Single query with persistence
agrag query "question" --checkpoint         # Enable HITL checkpointing
```

### In-Session Commands

```bash
/help                     # Show all commands
/verbose [on|off]         # Toggle reasoning expansion and tool details
/thinking [level|budget]  # Configure LLM thinking
/history                  # View conversation checkpoints
/threads                  # Alias for /history
/branches                 # List conversation branches
/fork <checkpoint> [name] # Create branch from checkpoint
/stats                    # Show session statistics
/export [file] [--verbose]# Export conversation transcript
/reset                    # Start fresh (new thread ID)
/exit                     # Exit chat
```

---

## Testing

### Unit Tests
```bash
# Reasoning extraction
pytest tests/unit/test_reasoning_extraction.py -v

# Branching logic
pytest tests/unit/test_branching.py -v
```

### Integration Tests
```bash
# Chat commands
pytest tests/integration/test_chat_commands.py -v
```

### Manual Testing Workflow
```bash
# 1. Test reasoning display
agrag chat --verbose
You: What tests exist?
# Verify: Reasoning blocks shown with 🧠 icon

# 2. Test collapsing
You: /verbose off
You: What requirements exist?
# Verify: Reasoning collapsed to summary

# 3. Test checkpointing
You: /history
# Verify: Checkpoints listed with message counts

# 4. Test branching
You: /fork checkpoint-001 test-branch
# Verify: Branch created message shown

# 5. Test resumption
exit
agrag chat --thread-id chat-XXXXX_test-branch
# Verify: Continues from checkpoint
```

---

## Troubleshooting

### Reasoning blocks not showing
- Check that `GOOGLE_THINKING_LEVEL` or `GOOGLE_THINKING_BUDGET` is set
- Verify Gemini model supports thinking (e.g., `gemini-2.0-flash-thinking-exp-01-21`)
- Try `/verbose on` to see if reasoning is present but collapsed

### Checkpoints not saving
- Verify PostgreSQL connection: `NEON_CONNECTION_STRING`
- Check for error messages on startup
- Fallback to in-memory if PostgreSQL unavailable (warning shown)

### Branch creation fails
- Ensure checkpoint ID is valid (use `/history` to list)
- Check PostgreSQL connection
- Verify thread ID is correct

### Thread ID lost
- Thread IDs are shown in welcome message
- Use `agrag chat --thread-id <id>` to resume
- For PostgreSQL backend, threads persist indefinitely
- For in-memory backend, threads lost on process exit

---

## Future Enhancements

### Planned Features
- [ ] In-session branch switching (no restart required)
- [ ] Branch comparison view (side-by-side diff)
- [ ] Branch deletion and pruning
- [ ] Branch merging (combine insights from multiple paths)
- [ ] PII detection and redaction in reasoning
- [ ] Configurable reasoning display (stderr channel option)
- [ ] Event cursor-based stream resumption
- [ ] Thread listing across all sessions
- [ ] Thread search and filtering

### Performance Optimizations
- [ ] Lazy loading of checkpoint history
- [ ] Checkpoint compression for storage
- [ ] Configurable checkpoint retention policies

---

## References

- **LangGraph Checkpointing**: https://langchain-ai.github.io/langgraph/concepts/persistence/
- **Gemini Thinking Mode**: https://ai.google.dev/gemini-api/docs/thinking-mode
- **Problem Statement**: Original problem statement describing the three features
- **AGENTS.md**: Developer guide with architecture details
- **CLAUDE.md**: Claude-specific guidance for this repository
