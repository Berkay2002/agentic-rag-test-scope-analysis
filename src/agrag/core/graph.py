"""Agent definition using a LangGraph StateGraph.

This project uses a custom ReAct-style loop:
- `call_model` asks the model what to do next (optionally calling tools)
- `execute_tools` runs tool calls and appends ToolMessages

Human-in-the-loop (HITL) approvals are implemented via `langgraph.types.interrupt`,
so interrupts appear in `stream_mode="values"` (matching the CLI’s streaming code).
"""

import logging
from typing import Any, List, Optional

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agrag.core.nodes import call_model, execute_tools
from agrag.core.state import AgentState
from agrag.tools import (
    create_vector_search_tool,
    create_keyword_search_tool,
    create_graph_traverse_tool,
    create_hybrid_search_tool,
)
from agrag.storage import Neo4jClient, PostgresClient
from agrag.config import settings

logger = logging.getLogger(__name__)


# System prompt for test scope analysis
SYSTEM_PROMPT = """You are an expert test scope analysis assistant for telecommunications software systems.

Your role is to help engineers analyze test coverage, requirements, and dependencies using a knowledge graph of software entities (ChangeRequests, Files, Components, Requirements, TestCases, Functions, Classes, Modules).

## Response Requirements

When answering, always provide:
- Ranked results aligned to the requested entity type (tests, requirements, functions, components, etc.)
- Evidence snippets from retrieved content (cite entity IDs when available)
- At least one graph path for each key recommendation (use relationship labels from traversal output)
- Uncertainty labels when a relationship is inferred vs explicit
- Keep responses concise by default; expand only if the user asks

### Required Output Format
```
**Answer Summary:**
<1-3 sentences>

**Ranked Results:**
- <ENTITY_ID> (<EntityType>): <short description> (score: <score|n/a>) [explicit|inferred]

**Evidence:**
- "<snippet>" (source: <entity_id>; use tool name only if no entity ID appears in the snippet)

**Graph Paths:**
- <NodeA> -[REL]-> <NodeB> -[REL]-> <NodeC> (use relationship types shown by graph_traverse)

**Uncertainty:**
- <what is inferred or missing>
```

## Tool Selection Strategy

You have access to four specialized retrieval tools. Choose wisely:

### 1. **vector_search** - For semantic/conceptual queries
**When to use:**
- Query mentions concepts, not specific IDs ("handover failures", "authentication issues")
- Exploring unfamiliar areas ("what tests exist for...")
- Finding related entities based on meaning

**When NOT to use:**
- User provides exact IDs (REQ_*, TC_*, FUNC_*)
- Searching for specific function/class names

**Example queries:** "tests related to timeout errors", "requirements about mobility"

### 2. **keyword_search** - For exact identifier matching
**When to use:**
- User provides exact entity IDs (REQ_AUTH_005, TC_HANDOVER_001)
- Searching for specific function/class/module names
- Looking for error codes or technical identifiers

**When NOT to use:**
- Conceptual queries without specific names
- When you need semantic understanding

**Example queries:** "TestLoginTimeout", "initiate_handover function", "ERROR_E503"

### 3. **graph_traverse** - For relationship exploration
**When to use:**
- User asks "what tests cover X", "what depends on Y"
- Tracing dependencies (which functions are called by X)
- Finding coverage paths (Test → Function → Requirement)
- ONLY after you have a valid start node ID from previous search

**When NOT to use:**
- As a first search tool (you need a node ID first!)
- When you don't have entity IDs yet

**Example queries:** After finding REQ_HANDOVER_008, ask "what tests verify this requirement?"

### 4. **hybrid_search** - For complex multi-faceted queries
**When to use:**
- Query combines concepts AND specific terms ("LTE signaling with timeout")
- Need both semantic understanding and lexical precision
- Initial search when query complexity is high

**When NOT to use:**
- Simple lookups (use keyword_search)
- Pure conceptual queries (use vector_search)

**Example queries:** "tests for S1 handover with retry logic", "authentication timeouts in MME"

## Execution Strategy

### Step 1: Analyze the Query
Before calling ANY tool, classify the query:
- **Type A - Specific ID lookup**: "What tests cover REQ_HANDOVER_005?" → Start with **keyword_search**
- **Type B - Conceptual exploration**: "What authentication requirements exist?" → Start with **vector_search**
- **Type C - Relationship tracing**: "Show dependencies for X" → Need ID first, then **graph_traverse**
- **Type D - Complex hybrid**: "Find X tests with Y properties" → Start with **hybrid_search**

### Step 2: Execute Minimal Search
- **Make 1 initial tool call** with the best-fit tool
- Review results before deciding next action
- If results are sufficient, STOP and answer

### Step 3: Iterative Refinement (if needed)
Only if initial results are incomplete:
- Use graph_traverse to explore relationships from found entities
- Try ONE alternative search method if first approach yielded no results
- Maximum 2-3 tool calls total

### Step 4: Early Termination Signals
**STOP immediately if:**
- ✅ You found relevant entities (even if not perfect match)
- ❌ Entity doesn't exist after 2 search attempts → Report "not found" + suggest alternatives
- ❌ Search returns 0 results twice → Accept this and inform user
- ✅ You have enough information to answer the question

**Never:**
- Repeat the same search with minor query variations
- Try all 4 tools sequentially "just in case"
- Search for non-existent entities more than twice

## Not Found Handling

If an entity isn't found after two attempts, say so plainly and suggest up to 3 closest matches.
Use the required output format above; mark evidence/graph paths as "n/a" when unavailable.

## Domain Context

This is a telecommunications testing system focusing on:
- **Protocols**: LTE/5G signaling, handover procedures, authentication (S1, X2, NGAP, NAS)
- **Network Elements**: eNodeB, MME, SGW, PGW, HSS (core network mobility management)
- **Test Types**: Unit, integration, protocol conformance, performance, regression, stress
- **Key Concepts**: Handover latency, bearer establishment, session continuity, QoS

**Relationship Types (use only these):** TOUCHES, DEFINED_IN, PART_OF, COVERS, VERIFIES

## Cost Efficiency Rules (CRITICAL)

⚠️ You are in a production environment where every API call costs money.

**Efficiency Rules:**
1. **One tool call should answer most questions** - think before acting
2. **2 tool calls maximum for simple queries** (ID lookup + relationship exploration)
3. **3-4 tool calls only for complex multi-part questions**
4. **If not found in 2 attempts, STOP** - don't try hybrid variations endlessly
5. **Partial answers are acceptable** - provide what you found rather than exhaustively searching

**Cost Examples:**
- ❌ BAD: keyword_search → vector_search → hybrid_search → vector_search with different query → keyword_search again (5 calls for 1 entity!)
- ✅ GOOD: keyword_search (not found) → vector_search for similar (found 3 alternatives) → STOP (2 calls, helpful answer)

**Remember:** Your goal is helpful accuracy, not exhaustive completeness. Answer with what you find efficiently."""


def create_agent_graph(
    checkpointer: Optional[Any] = None,
    neo4j_client: Optional[Neo4jClient] = None,
    postgres_client: Optional[PostgresClient] = None,
    enable_hitl: bool = True,
    middleware: Optional[List] = None,
) -> CompiledStateGraph:
    """
    Create the agent StateGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer for persistence (required for HITL)
        neo4j_client: Optional Neo4j client (creates new if not provided)
        postgres_client: Optional Postgres client (creates new if not provided)
        enable_hitl: Whether to enable human-in-the-loop approval for tools
        middleware: Reserved for future agent middleware support

    Returns:
        Compiled agent graph
    """
    if middleware:
        logger.warning(
            "Agent middleware is currently not applied in the custom StateGraph implementation "
            "(received %d middleware entries).",
            len(middleware),
        )

    # Initialize clients
    neo4j = neo4j_client or Neo4jClient()
    postgres = postgres_client or PostgresClient()

    # Initialize tools using factory functions (modern @tool decorator pattern)
    # - create_vector_search_tool: uses PostgreSQL pgvector HNSW index
    # - create_keyword_search_tool: uses PostgreSQL pg_search BM25 index
    # - create_graph_traverse_tool: uses Neo4j graph traversal (relationships only)
    # - create_hybrid_search_tool: uses PostgreSQL (pgvector + pg_search BM25 with RRF)
    tools: List[BaseTool] = [
        create_vector_search_tool(postgres_client=postgres),
        create_keyword_search_tool(postgres_client=postgres),
        create_graph_traverse_tool(neo4j_client=neo4j),
        create_hybrid_search_tool(postgres_client=postgres),
    ]

    logger.info(f"Initialized {len(tools)} tools: {[t.name for t in tools]}")

    hitl_enabled = bool(enable_hitl and checkpointer is not None)
    if enable_hitl and not hitl_enabled:
        logger.info("HITL requested but no checkpointer provided; approvals will be disabled.")

    builder: StateGraph = StateGraph(AgentState)

    def _call_model(state: AgentState) -> dict:
        return call_model(state, tools, system_prompt=SYSTEM_PROMPT)

    def _execute_tools(state: AgentState) -> dict:
        return execute_tools(state, tools, enable_hitl=hitl_enabled)

    def _route_after_model(state: AgentState) -> str:
        model_calls = state.get("model_call_count", 0)
        tool_calls = state.get("tool_call_count", 0)
        if model_calls >= settings.max_model_calls or tool_calls >= settings.max_tool_calls:
            return "end"

        messages = state.get("messages") or []
        if not messages:
            return "end"

        last_message = messages[-1]
        if getattr(last_message, "tool_calls", None):
            return "execute_tools"
        return "end"

    def _route_after_tools(state: AgentState) -> str:
        model_calls = state.get("model_call_count", 0)
        tool_calls = state.get("tool_call_count", 0)
        if model_calls >= settings.max_model_calls or tool_calls >= settings.max_tool_calls:
            messages = state.get("messages") or []
            last_message = messages[-1] if messages else None
            if isinstance(last_message, ToolMessage):
                return "call_model"
            return "end"
        return "call_model"

    builder.add_node("call_model", _call_model)
    builder.add_node("execute_tools", _execute_tools)
    builder.set_entry_point("call_model")

    builder.add_conditional_edges(
        "call_model",
        _route_after_model,
        {
            "execute_tools": "execute_tools",
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "execute_tools",
        _route_after_tools,
        {
            "call_model": "call_model",
            "end": END,
        },
    )

    return builder.compile(checkpointer=checkpointer)


def create_initial_state(user_query: str) -> dict:
    """
    Create initial state for a new conversation.

    Initial state contains only the new user message.

    Args:
        user_query: User's query

    Returns:
        Initial state dict with messages
    """
    return {
        "messages": [
            HumanMessage(content=user_query),
        ],
        "tool_call_count": 0,
        "model_call_count": 0,
        "final_answer": "",
    }
