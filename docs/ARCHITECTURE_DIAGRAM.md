# Production Features Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Terminal                            │
│  • Rich formatting (colors, panels, tables)                     │
│  • Interactive prompts                                           │
│  • Command completion                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Interactive Chat Session                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ InteractiveChat                                           │  │
│  │  • Manages session state (thread_id, counters)           │  │
│  │  • Coordinates streaming and HITL                        │  │
│  │  • Processes events from agent graph                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│              ┌───────────────┼───────────────┐                 │
│              ▼               ▼               ▼                  │
│  ┌────────────────┐ ┌─────────────┐ ┌──────────────────┐      │
│  │ Display Module │ │  Commands   │ │ Branch Manager   │      │
│  │                │ │             │ │                  │      │
│  │ • print_       │ │ • /history  │ │ • list_         │      │
│  │   reasoning()  │ │ • /branches │ │   checkpoints() │      │
│  │ • print_       │ │ • /fork     │ │ • create_       │      │
│  │   agent_       │ │ • /verbose  │ │   branch()      │      │
│  │   response()   │ │ • /stats    │ │ • list_         │      │
│  │                │ │             │ │   branches()    │      │
│  └────────────────┘ └─────────────┘ └──────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Agent Graph (LangGraph)                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ StateGraph                                                │  │
│  │  ┌────────────────┐    ┌──────────────────┐            │  │
│  │  │ call_model     │───▶│ execute_tools    │            │  │
│  │  │ • Get LLM      │    │ • Run tool calls │            │  │
│  │  │   response     │    │ • Append results │            │  │
│  │  │ • Tool calls   │◀───│ • HITL approval  │            │  │
│  │  └────────────────┘    └──────────────────┘            │  │
│  │         │                                                │  │
│  │         ▼                                                │  │
│  │  ┌────────────────────────────────────────┐            │  │
│  │  │ AI Message Content                      │            │  │
│  │  │  [                                      │            │  │
│  │  │    {"type": "thinking", "text": "..."},│            │  │
│  │  │    {"type": "text", "text": "..."}     │            │  │
│  │  │  ]                                      │            │  │
│  │  └────────────────────────────────────────┘            │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               PostgreSQL Checkpointer (LangGraph)                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ PostgresSaver                                             │  │
│  │  • Stores conversation state after each turn            │  │
│  │  • Enables checkpoint listing/navigation                │  │
│  │  • Supports branching via thread IDs                    │  │
│  │  • Provides HITL interrupt mechanism                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Database Schema:                                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ checkpoints                                               │  │
│  │  - thread_id (PK, indexed)                               │  │
│  │  - checkpoint_id (PK)                                    │  │
│  │  - parent_checkpoint_id                                  │  │
│  │  - checkpoint_ns                                         │  │
│  │  - channel_values (JSONB) ─▶ contains messages          │  │
│  │  - metadata (JSONB)                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Feature 1: Reasoning Display Flow

```
User Query
    │
    ▼
Agent Graph (call_model)
    │
    ▼
Gemini LLM Response
    │
    └─▶ Content Blocks:
        [
          {"type": "thinking", "text": "Let me analyze..."},
          {"type": "text", "text": "Final answer..."}
        ]
    │
    ▼
extract_reasoning_and_answer()
    │
    ├─▶ reasoning_blocks = ["Let me analyze..."]
    └─▶ answer_text = "Final answer..."
    │
    ▼
Display Logic (interactive.py)
    │
    ├─▶ print_reasoning(reasoning_blocks, collapsed=!verbose)
    │   │
    │   ├─▶ Collapsed:  "🧠 1 block, ~50 chars (use /verbose)"
    │   └─▶ Expanded:   "🧠 Full reasoning in panel"
    │
    └─▶ print_agent_response(answer_text)
        └─▶ "● Assistant: Final answer..."
```

---

## Feature 2: Resumable Streaming Flow

```
Session Start
    │
    ▼
Check thread_id Parameter
    │
    ├─▶ Provided: Load existing thread
    │   │
    │   ├─▶ PostgresSaver.list(thread_id)
    │   │   └─▶ Returns all checkpoints
    │   │
    │   └─▶ Resume from last checkpoint
    │       └─▶ State includes all previous messages
    │
    └─▶ Not provided: Generate new thread_id
        └─▶ thread_id = "chat-{uuid}"
    │
    ▼
During Conversation
    │
    ├─▶ After each AI response
    │   └─▶ PostgresSaver.put(state, config)
    │       └─▶ Creates new checkpoint
    │
    └─▶ User command: /history
        └─▶ BranchManager.list_checkpoints()
            └─▶ PostgresSaver.list(config)
                └─▶ Returns checkpoint history
    │
    ▼
Interruption (terminal close, Ctrl+C, crash)
    │
    └─▶ State saved in PostgreSQL
        └─▶ Survives process termination
    │
    ▼
Resume Later
    │
    └─▶ agrag chat --thread-id chat-abc123
        └─▶ Loads state from last checkpoint
            └─▶ Continues conversation seamlessly
```

---

## Feature 3: Branching Flow

```
Active Conversation (thread_id = "chat-abc123")
    │
    ├─▶ Checkpoint 1 (2 messages)
    ├─▶ Checkpoint 2 (4 messages) ◀── Fork point
    └─▶ Checkpoint 3 (6 messages)
    │
    ▼
User: /fork checkpoint-002 experiment
    │
    ▼
BranchManager.create_branch()
    │
    ├─▶ Generate new thread_id:
    │   └─▶ "chat-abc123_experiment"
    │
    ├─▶ Set parent_checkpoint:
    │   └─▶ metadata["parent"] = "checkpoint-002"
    │
    └─▶ Return new_thread_id
    │
    ▼
User exits and restarts:
agrag chat --thread-id chat-abc123_experiment
    │
    ▼
Load state from checkpoint-002
    │
    └─▶ Conversation continues from fork point
        │
        ├─▶ Original branch: checkpoint-003 (6 msgs)
        └─▶ New branch: checkpoint-003' (different path)
    │
    ▼
Conversation Tree:
    │
    main: chat-abc123
    ├─▶ checkpoint-001 (2 msgs)
    ├─▶ checkpoint-002 (4 msgs) ◀── Fork point
    │   │
    │   ├─▶ main branch continues
    │   │   └─▶ checkpoint-003 (6 msgs)
    │   │
    │   └─▶ experiment branch forks
    │       └─▶ checkpoint-003' (4 msgs + new path)
    │
    └─▶ Both branches coexist independently
```

---

## Command Flow Examples

### /history Command

```
User: /history
    │
    ▼
CommandHandler.handle("/history")
    │
    ▼
BranchManager.list_checkpoints()
    │
    └─▶ checkpointer.list({"configurable": {"thread_id": ...}})
        │
        └─▶ Returns checkpoint tuples:
            [
              (checkpoint, metadata),
              (checkpoint, metadata),
              ...
            ]
    │
    ▼
Parse and format:
    │
    ├─▶ Extract checkpoint_id, message_count, namespace
    └─▶ Build list of checkpoint dicts
    │
    ▼
print_checkpoints(console, checkpoints)
    │
    └─▶ Display Rich table:
        ┌────┬─────────────┬──────────┬───────────┐
        │ #  │ Checkpoint  │ Messages │ Namespace │
        └────┴─────────────┴──────────┴───────────┘
```

### /verbose Command

```
User: /verbose [on|off]
    │
    ▼
CommandHandler.handle("/verbose ...")
    │
    ├─▶ No argument: Toggle
    │   └─▶ session.verbose = !session.verbose
    │
    └─▶ With argument: Set explicit
        ├─▶ "on": session.verbose = True
        └─▶ "off": session.verbose = False
    │
    ▼
session.set_verbose(enabled)
    │
    └─▶ Adjusts logging level:
        ├─▶ Verbose: Show tool details + expanded reasoning
        └─▶ Non-verbose: Hide tool details + collapsed reasoning
    │
    ▼
Affects subsequent interactions:
    │
    └─▶ _process_event() checks session.verbose
        └─▶ print_reasoning(collapsed=!verbose)
```

---

## Data Flow: Complete Query Lifecycle

```
1. User Input
   "What tests cover authentication?"
        │
        ▼
2. Interactive Chat
   create_initial_state(query)
        │
        ▼
3. Agent Graph Stream
   graph.stream(state, config)
        │
        ├─▶ call_model node
        │   └─▶ LLM generates response with thinking
        │
        ├─▶ execute_tools node (if tool calls)
        │   ├─▶ HITL approval (if enabled)
        │   └─▶ Tool execution
        │
        └─▶ Back to call_model (loop)
        │
        ▼
4. Event Processing
   _process_event(event, status)
        │
        ├─▶ AIMessage with content blocks
        │   │
        │   ├─▶ extract_reasoning_and_answer()
        │   │   ├─▶ reasoning: ["Let me search..."]
        │   │   └─▶ answer: "Here are the tests..."
        │   │
        │   ├─▶ print_reasoning() [if reasoning exists]
        │   │   └─▶ Display with 🧠 icon
        │   │
        │   └─▶ Store answer for final display
        │
        └─▶ ToolMessage (tool results)
        │
        ▼
5. Checkpoint Saving
   PostgresSaver.put(state, config)
        │
        └─▶ State saved after each turn
        │
        ▼
6. Display Final Answer
   print_agent_response(answer)
   print_query_stats(tool_calls, model_calls)
        │
        ▼
7. Ready for Next Query
   session.prompt("You: ")
```

---

## Error Handling & Edge Cases

### Checkpointer Unavailable

```
initialize_checkpointer()
    │
    ├─▶ Try PostgreSQL connection
    │   │
    │   ├─▶ Success: Use PostgresSaver
    │   │   └─▶ persistent=True, backend="postgres"
    │   │
    │   └─▶ Failure: Fall back
    │       └─▶ MemorySaver()
    │           └─▶ persistent=False, backend="memory"
    │           └─▶ Display warning to user
    │
    └─▶ User sees:
        "[yellow]Warning: PostgreSQL unavailable[/yellow]"
        "[yellow]Using in-memory (session only)[/yellow]"
```

### Reasoning Blocks Not Present

```
extract_reasoning_and_answer(content)
    │
    ├─▶ No thinking blocks found
    │   └─▶ reasoning = []
    │       answer = full content
    │
    └─▶ print_reasoning(reasoning)
        └─▶ Early return if len(reasoning) == 0
            └─▶ No reasoning display, only answer shown
```

### Branch Creation Failure

```
User: /fork checkpoint-999 my-branch
    │
    ▼
BranchManager.create_branch("checkpoint-999", "my-branch")
    │
    └─▶ try:
        │   new_thread_id = generate_thread_id()
        │   log_branch_creation()
        │   return new_thread_id
        │
        └─▶ except Exception as e:
            └─▶ Display error: "[red]✗ Failed to create branch: {e}[/red]"
```

---

## Security & Performance Considerations

### Reasoning Display Security

```
Reasoning may contain:
├─▶ Internal deliberations (safe)
├─▶ Tool arguments (may contain PII)
├─▶ Retrieved content (may contain sensitive data)
└─▶ Error messages (may expose system internals)

Mitigation:
├─▶ Collapsed by default (requires opt-in to expand)
├─▶ Verbose mode clearly labeled
└─▶ Future: PII detection & redaction
```

### Checkpoint Storage Growth

```
Each message creates a checkpoint:
├─▶ Storage grows linearly with conversation length
├─▶ Branches multiply storage requirements
└─▶ No automatic cleanup

Mitigation:
├─▶ Document retention policies in user guide
├─▶ Future: Configurable checkpoint pruning
└─▶ Future: Checkpoint compression
```

### Performance Optimization

```
Checkpoint listing:
├─▶ LIMIT 20 in display (don't show all)
├─▶ Lazy loading (fetch on demand)
└─▶ Indexed queries (thread_id indexed in PostgreSQL)

Reasoning extraction:
├─▶ O(n) where n = number of content blocks
├─▶ Typically n < 5, so performance is negligible
└─▶ No caching needed
```

---

## Technology Stack

```
┌─────────────────────────────────────────┐
│ Terminal UI                              │
│  • Rich (formatting, panels, tables)    │
│  • prompt_toolkit (interactive input)   │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ CLI Application                          │
│  • Click (command framework)             │
│  • Python 3.11+                          │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ Agent Framework                          │
│  • LangGraph (StateGraph)                │
│  • LangChain (tools, messages)           │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ LLM Provider                             │
│  • Google Gemini (with thinking mode)   │
│  • langchain-google-genai integration   │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ Persistence Layer                        │
│  • PostgreSQL (Neon)                     │
│  • langgraph-checkpoint-postgres         │
└─────────────────────────────────────────┘
```

