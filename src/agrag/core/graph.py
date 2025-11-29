"""StateGraph definition for agentic RAG system."""

import logging
from typing import List, Literal
from functools import partial

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

from agrag.core.state import AgentState
from agrag.core.nodes import call_model, execute_tools, finalize_answer
from agrag.tools import (
    VectorSearchTool,
    KeywordSearchTool,
    GraphTraverseTool,
    HybridSearchTool,
)
from agrag.storage import Neo4jClient, PostgresClient
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


def route_after_model(
    state: AgentState,
) -> Literal["execute_tools", "finalize"]:
    """
    Route after model call based on whether tools were called.

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    last_message = state["messages"][-1]

    # Check if model wants to use tools
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("Routing to execute_tools")
        return "execute_tools"

    # No tools needed, finalize answer
    logger.info("Routing to finalize")
    return "finalize"


def route_after_tools(
    state: AgentState,
) -> Literal["call_model", "finalize", END]:
    """
    Route after tool execution based on safety limits.

    Args:
        state: Current agent state

    Returns:
        Next node name or END
    """
    tool_count = state.get("tool_call_count", 0)
    model_count = state.get("model_call_count", 0)

    # Check safety limits
    if tool_count >= settings.max_tool_calls:
        logger.warning(f"Tool call limit reached ({tool_count}), finalizing")
        return "finalize"

    if model_count >= settings.max_model_calls:
        logger.warning(f"Model call limit reached ({model_count}), finalizing")
        return "finalize"

    # Continue the loop
    logger.info(
        f"Continuing loop (tools: {tool_count}/{settings.max_tool_calls}, "
        f"models: {model_count}/{settings.max_model_calls})"
    )
    return "call_model"


def create_agent_graph(
    checkpointer: PostgresSaver = None,
    neo4j_client: Neo4jClient = None,
    postgres_client: PostgresClient = None,
) -> StateGraph:
    """
    Create the StateGraph for the agentic RAG system.

    Args:
        checkpointer: Optional PostgresSaver for persistence (HITL)
        neo4j_client: Optional Neo4j client (creates new if not provided)
        postgres_client: Optional Postgres client (creates new if not provided)

    Returns:
        Compiled StateGraph
    """
    logger.info("Creating agent graph...")

    # Initialize clients
    neo4j = neo4j_client or Neo4jClient()
    postgres = postgres_client or PostgresClient()

    # Initialize tools
    tools: List[BaseTool] = [
        VectorSearchTool(neo4j_client=neo4j),
        KeywordSearchTool(postgres_client=postgres),
        GraphTraverseTool(neo4j_client=neo4j),
        HybridSearchTool(postgres_client=postgres),
    ]

    logger.info(f"Initialized {len(tools)} tools: {[t.name for t in tools]}")

    # Create graph
    graph = StateGraph(AgentState)

    # Add nodes (wrap with partial to pass tools)
    graph.add_node("call_model", partial(call_model, tools=tools))
    graph.add_node("execute_tools", partial(execute_tools, tools=tools))
    graph.add_node("finalize", finalize_answer)

    # Set entry point
    graph.set_entry_point("call_model")

    # Add conditional edges
    graph.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "execute_tools": "execute_tools",
            "finalize": "finalize",
        },
    )

    graph.add_conditional_edges(
        "execute_tools",
        route_after_tools,
        {
            "call_model": "call_model",
            "finalize": "finalize",
            END: END,
        },
    )

    # Finalize node always ends
    graph.add_edge("finalize", END)

    # Compile graph with optional checkpointer
    if checkpointer:
        logger.info("Compiling graph with PostgresSaver checkpointer (HITL enabled)")
        compiled = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["execute_tools"],  # HITL: pause before tool execution
        )
    else:
        logger.info("Compiling graph without checkpointer")
        compiled = graph.compile()

    logger.info("Agent graph created successfully")

    return compiled


def create_initial_state(user_query: str) -> AgentState:
    """
    Create initial state for a new conversation.

    Args:
        user_query: User's query

    Returns:
        Initial AgentState
    """
    return {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            ("user", user_query),
        ],
        "tool_call_count": 0,
        "model_call_count": 0,
        "final_answer": "",
    }
