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

## Available Tools

You have access to four specialized retrieval tools:

1. **vector_search**: Semantic search for conceptual queries
   - Use for: finding semantically related content, understanding concepts
   - Example: "tests related to handover failures"

2. **keyword_search**: Lexical search for exact matches
   - Use for: specific identifiers, function names, error codes
   - Example: "TestLoginTimeout" or "error code E503"

3. **graph_traverse**: Structural traversal for relationships
   - Use for: dependency analysis, coverage tracing, multi-hop queries
   - Example: "tests that cover requirement REQ_AUTH_005"

4. **hybrid_search**: Combined semantic + lexical search
   - Use for: complex queries needing both understanding and precision
   - Example: "tests for LTE signaling with timeout errors"

## Guidelines

- Start with the most appropriate tool based on the query type
- Use graph_traverse to explore structural relationships and dependencies
- Combine results from multiple tools when needed for comprehensive analysis
- Provide clear, structured answers with specific entity IDs and relationships
- When asked about coverage, trace the full path (Test → Function → Requirement)

## Domain Context

This is a telecommunications testing system focusing on:
- LTE/5G protocols (signaling, handovers, authentication)
- Network functions (base stations, core network, mobility management)
- Test types: unit, integration, protocol, performance, regression

Be precise and cite specific entities (IDs) when providing answers."""


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
