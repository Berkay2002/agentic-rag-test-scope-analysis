"""Agent definition using LangChain's create_agent API."""

import logging
from typing import List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres import PostgresSaver

from agrag.tools import (
    create_vector_search_tool,
    create_keyword_search_tool,
    create_graph_traverse_tool,
    create_hybrid_search_tool,
)
from agrag.storage import Neo4jClient, PostgresClient
from agrag.models import get_llm
from agrag.config import settings

logger = logging.getLogger(__name__)


# System prompt for test scope analysis
SYSTEM_PROMPT = """You are an expert test scope analysis assistant for telecommunications software systems.

Your role is to help engineers analyze test coverage, requirements, and dependencies using a knowledge graph of software entities (Requirements, TestCases, Functions, Classes, Modules).

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

## Response Quality Guidelines

### Good Answer Format:
```
Based on [tool_name] search, I found:

**Handover Requirements:**
- REQ_HANDOVER_008: LTE handover between MME cells (<50ms latency)
- REQ_HANDOVER_009: NGAP handover between SGW cells (<50ms latency)

**Test Coverage:**
Using graph_traverse from REQ_HANDOVER_008:
- TC_HANDOVER_MME_001 verifies this requirement
- TC_LATENCY_003 validates timing constraints

Total: 2 requirements, 2 test cases
```

### When Entity Not Found:
```
I searched for "REQ_HANDOVER_005" using keyword_search and vector_search.

**Result:** This requirement ID does not exist in the database.

**Similar entities found:**
- REQ_HANDOVER_008: LTE handover between MME cells
- REQ_HANDOVER_009: NGAP handover between SGW cells
- REQ_HANDOVER_010: X2 handover between MME cells

Would you like details on any of these?
```

## Domain Context

This is a telecommunications testing system focusing on:
- **Protocols**: LTE/5G signaling, handover procedures, authentication (S1, X2, NGAP, NAS)
- **Network Elements**: eNodeB, MME, SGW, PGW, HSS (core network mobility management)
- **Test Types**: Unit, integration, protocol conformance, performance, regression, stress
- **Key Concepts**: Handover latency, bearer establishment, session continuity, QoS

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
    checkpointer: Optional[PostgresSaver] = None,
    neo4j_client: Optional[Neo4jClient] = None,
    postgres_client: Optional[PostgresClient] = None,
    enable_hitl: bool = True,
    middleware: Optional[List] = None,
) -> CompiledStateGraph:
    """
    Create the agent using LangChain's create_agent API.

    Args:
        checkpointer: Optional PostgresSaver for persistence (HITL)
        neo4j_client: Optional Neo4j client (creates new if not provided)
        postgres_client: Optional Postgres client (creates new if not provided)
        enable_hitl: Whether to enable human-in-the-loop approval for tools
        middleware: Optional list of additional middleware to apply

    Returns:
        Compiled agent graph
    """
    logger.info("Creating agent with create_agent API...")

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

    # Get LLM instance
    llm = get_llm()

    # Build middleware list
    agent_middleware = middleware or []

    # Add HITL middleware if enabled and checkpointer is available
    if enable_hitl and checkpointer:
        logger.info("Adding HumanInTheLoopMiddleware for tool approval")
        agent_middleware.append(
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "vector_search": True,  # All decisions allowed (approve, edit, reject)
                    "keyword_search": True,
                    "graph_traverse": True,
                    "hybrid_search": True,
                },
                description_prefix="Tool execution requires approval",
            )
        )

    # Add limit middleware to prevent runaway execution
    agent_middleware.extend(
        [
            ModelCallLimitMiddleware(
                run_limit=settings.max_model_calls,
                exit_behavior="end",
            ),
            ToolCallLimitMiddleware(
                run_limit=settings.max_tool_calls,
                exit_behavior="continue",
            ),
        ]
    )

    logger.info(f"Configured {len(agent_middleware)} middleware components")

    # Create agent using the new API
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=agent_middleware,
        checkpointer=checkpointer,
    )

    logger.info("Agent created successfully with create_agent API")

    return agent


def create_initial_state(user_query: str) -> dict:
    """
    Create initial state for a new conversation.

    The create_agent API uses a simpler state format with just messages.

    Args:
        user_query: User's query

    Returns:
        Initial state dict with messages
    """
    return {
        "messages": [
            {"role": "user", "content": user_query},
        ],
    }
