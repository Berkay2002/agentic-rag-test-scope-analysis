# Agentic GraphRAG System - Implementation Plan

## Overview

Production-grade Python implementation of an Agentic GraphRAG system for test scope analysis, combining custom LangGraph StateGraph with dual-database architecture (Neo4j + Neon PostgreSQL) and full LangSmith observability.

## Technology Stack

- **Language**: Python 3.11+
- **LLM**: Google Generative AI (`ChatGoogleGenerativeAI`)
- **Embeddings**: Google Generative AI embeddings (768-dim)
- **Graph DB**: Neo4j Cloud (Aura) with Cypher
- **Vector DB**: Neon PostgreSQL + pgvector extension
- **Framework**: LangChain + LangGraph (custom StateGraph)
- **Observability**: LangSmith (full tracing)
- **Persistence**: PostgresSaver with Neon PostgreSQL

## Project Structure

```
agentic-rag-test-scope-analysis/
├── src/agrag/
│   ├── cli/             # CLI application (Click-based)
│   ├── config/          # Settings, logging configuration
│   ├── core/            # StateGraph implementation
│   ├── tools/           # 4 retrieval tools (vector, keyword, graph, hybrid)
│   ├── storage/         # Neo4j + PostgreSQL clients
│   ├── models/          # LLM + Embeddings wrappers
│   ├── data/            # Synthetic generation + ingestion pipeline
│   ├── kg/              # Knowledge Graph ontology & schema
│   ├── middleware/      # Guardrails, PII detection, retry logic
│   ├── evaluation/      # Metrics (Precision@k, Recall@k, MAP)
│   └── observability/   # LangSmith integration utilities
├── tests/               # Unit + Integration tests
├── scripts/             # Setup scripts, data generation
├── notebooks/           # Analysis notebooks for thesis
├── pyproject.toml       # Poetry dependencies
└── README.md
```

## Implementation Phases (12 Weeks)

### Phase 0: Foundation & Infrastructure (Week 1-2)

**Objectives**:
- Initialize Python project with Poetry
- Configure cloud databases (Neo4j Aura, Neon PostgreSQL)
- Set up LangSmith workspace
- Establish development environment

**Deliverables**:
- Project scaffolding with Poetry
- Neo4j cloud instance with constraints & vector indexes
- Neon PostgreSQL with pgvector extension enabled
- LangSmith API keys configured
- Development environment tested

### Phase 1: Knowledge Graph Ontology (Week 3-4) - RQ1

**Objectives**: Design software engineering ontology for test scope analysis

**Entity Types**:
- `Requirement` (id, priority, status, description, embedding)
- `TestCase` (id, test_type, file_path, expected_outcome)
- `Function` (id, signature, code_snippet, file_path, line_number)
- `Class` (id, name, methods, file_path)
- `Module` (id, name, architectural_component)

**Relationships**:
- `(:Requirement)-[:VERIFIES]->(:TestCase)`
- `(:TestCase)-[:COVERS]->(:Function)`
- `(:Function)-[:CALLS]->(:Function)`
- `(:Function)-[:DEFINED_IN]->(:Class)`
- `(:Class)-[:INHERITS_FROM]->(:Class)`

**Neo4j Schema Setup**:
```cypher
CREATE CONSTRAINT requirement_id FOR (r:Requirement) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT testcase_id FOR (t:TestCase) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT function_id FOR (f:Function) REQUIRE f.id IS UNIQUE;
CREATE CONSTRAINT class_id FOR (c:Class) REQUIRE c.id IS UNIQUE;

CREATE VECTOR INDEX requirement_embeddings
FOR (r:Requirement) ON (r.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}};
```

**PostgreSQL Schema**:
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) UNIQUE,
    content TEXT,
    metadata JSONB,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON document_chunks USING gin(to_tsvector('english', content));
```

**Deliverables**:
- `/src/agrag/kg/ontology.py` - Entity and relationship definitions
- Database schemas deployed to Neo4j and PostgreSQL
- Sample data validation scripts

### Phase 2: Synthetic Data Generation (Week 5)

**Objectives**: Generate realistic software artifacts for testing

**Generators**:
1. **RequirementGenerator**: 20-30 telecommunications domain requirements (network protocols, handovers, signaling, base station management)
2. **CodeGenerator**: 50-100 functions, 20-30 classes (Python/Java style) for telecom scenarios
3. **TestCaseGenerator**: 100-150 test cases with coverage links (protocol tests, integration tests, performance tests)
4. **TraceLinker**: Establish relationships (80% coverage, 10% intentional gaps)

**Telecommunications Domain Examples**:
- Requirements: "The system SHALL support LTE handover between adjacent cells with latency < 50ms"
- Functions: `initiate_handover()`, `verify_signal_strength()`, `update_cell_configuration()`
- Test Cases: "TC_HANDOVER_001: Verify successful X2 handover between eNodeB cells"

**Strategy**: Use Google Generative AI with structured prompts to generate:
- Syntax-valid telecom-focused code
- Semantically aligned requirements ↔ implementations (aligned with Ericsson domain)
- Realistic test case descriptions for network protocols
- Multi-hop dependencies for graph traversal testing

**Deliverables**:
- `/src/agrag/data/generators/` - All generator classes
- `/scripts/generate_synthetic_data.py` - Main generation script
- Generated dataset (200+ artifacts) in JSON format

### Phase 3: Data Ingestion Pipeline (Week 6-7)

**Pipeline Flow**:
```
Raw Artifacts → Loaders → AST Chunking → Embedding → Dual Storage (Neo4j + PostgreSQL)
```

**Key Components**:

1. **ASTCodeSplitter**: Tree-sitter based code splitting preserving function boundaries
2. **EmbeddingService**: Google Generative AI with batch processing and caching
3. **DualStorageWriter**: Parallel writes to Neo4j (nodes/edges) and PostgreSQL (vectors)

**Critical Design Decision**: Never split mid-function; always preserve parent context (class, module hierarchy)

**Deliverables**:
- `/src/agrag/data/loaders/` - Document loaders
- `/src/agrag/data/chunking.py` - AST-based splitter
- `/src/agrag/storage/dual_writer.py` - Dual storage orchestrator
- End-to-end ingestion script with progress tracking

### Phase 4: Retrieval Tools (Week 8) - RQ2 Foundation

**Tool 1: Vector Search (Dense Semantic)**
```python
class VectorSearchTool(BaseTool):
    name = "vector_search"
    description = "Use for semantic queries requiring conceptual understanding"
    args_schema = VectorSearchInput

    def _run(self, query: str, k: int = 10) -> list:
        # 1. Generate query embedding
        # 2. Neo4j vector index search (HNSW)
        # 3. Return top-k with metadata
```

**Tool 2: Keyword Search (Sparse Lexical)**
```python
class KeywordSearchTool(BaseTool):
    name = "keyword_search"
    description = "Use for exact matches: error codes, function names, identifiers"
    args_schema = KeywordSearchInput

    def _run(self, query: str, k: int = 10) -> list:
        # PostgreSQL full-text search with BM25-style ranking
        # ts_rank_cd for relevance scoring
```

**Tool 3: Graph Traversal (Structural)**
```python
class GraphTraverseTool(BaseTool):
    name = "graph_traverse"
    description = "Use for structural dependency analysis via multi-hop traversal"
    args_schema = GraphTraverseInput

    def _run(self, start_node: str, relation: str, depth: int = 2) -> list:
        # Cypher pattern matching with depth limit
        # MATCH path=(start)-[rel*1..depth]->(end)
```

**Tool 4: Hybrid Search (RRF Fusion)**
```python
class HybridSearchTool(BaseTool):
    name = "hybrid_search"
    description = "Use for complex queries needing both semantic and lexical precision"
    args_schema = HybridSearchInput

    def _run(self, query: str, k: int = 10) -> list:
        # 1. Parallel execution: vector_search + keyword_search
        # 2. Reciprocal Rank Fusion: score = Σ 1/(k + rank)
        # 3. Return merged, ranked results
```

**Deliverables**:
- `/src/agrag/tools/vector_search.py`
- `/src/agrag/tools/keyword_search.py`
- `/src/agrag/tools/graph_traverse.py`
- `/src/agrag/tools/hybrid_search.py`
- `/src/agrag/tools/schemas.py` - Pydantic input schemas

### Phase 5: Custom StateGraph Agent (Week 9)

**Agent State Definition**:
```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration: int
    tool_calls_count: int
    model_calls_count: int
    retrieved_documents: list
    final_answer: str | None
```

**Graph Nodes**:
1. **call_model**: LLM reasoning (generates Thought + selects Action)
2. **execute_tools**: Tool execution (produces Observation)
3. **finalize_answer**: Synthesize final recommendation

**Conditional Routing**:
```python
def should_continue(state: AgentState) -> Literal["tools", "finalize", "end"]:
    last_message = state["messages"][-1]

    # Check limits
    if state["tool_calls_count"] >= 10 or state["model_calls_count"] >= 20:
        return "finalize"

    # Check for tool calls
    if last_message.tool_calls:
        return "tools"

    # Check for final answer
    if state.get("final_answer"):
        return "end"

    return "finalize"
```

**Graph Construction**:
```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres import PostgresSaver

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("call_model", nodes.call_model)
workflow.add_node("execute_tools", nodes.execute_tools)
workflow.add_node("finalize_answer", nodes.finalize_answer)

# Add edges
workflow.set_entry_point("call_model")
workflow.add_conditional_edges("call_model", should_continue)
workflow.add_edge("execute_tools", "call_model")  # ReAct loop
workflow.add_edge("finalize_answer", END)

# Compile with checkpointer and HITL interrupts
checkpointer = PostgresSaver.from_conn_string(NEON_CONNECTION_STRING)
graph = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute_tools"]  # Human-in-the-loop
)
```

**System Prompt**:
```
You are an expert Test Scope Analysis Agent for large-scale software systems.

Your goal: Given a change request or requirement, identify relevant test cases that should be executed.

Strategy:
1. Analyze query intent (semantic? structural? hybrid?)
2. Select appropriate retrieval tool(s)
3. Execute tool calls and analyze results
4. Cross-reference findings across multiple sources
5. Provide explainable recommendations with rationale

Available Tools:
- vector_search: For semantic queries ("tests related to authentication")
- keyword_search: For exact matches ("TestLoginTimeout", error code "E503")
- graph_traverse: For dependency analysis ("all tests covering functions called by UserManager")
- hybrid_search: For complex queries requiring both semantic and lexical precision

Always explain your reasoning and cite specific evidence from retrieval results.
```

**Deliverables**:
- `/src/agrag/core/graph.py` - StateGraph definition
- `/src/agrag/core/nodes.py` - Node implementations
- `/src/agrag/core/state.py` - State definition
- Working agent with LangSmith tracing

### Phase 6: Middleware & Guardrails (Week 10)

**Components**:

1. **PIIDetector**: Deterministic PII detection and redaction
   - Patterns: emails, SSNs, credit cards, phone numbers
   - Strategies: redact, mask, block

2. **ContextCompactor**: Summarization middleware
   - Trigger: When context exceeds token threshold
   - Action: Summarize old conversation history
   - Preserve: Recent messages, critical state

3. **RetryHandler**: Tool execution retry logic
   - Exponential backoff (1s, 2s, 4s)
   - Max retries: 3
   - Handles transient failures (network, timeouts)

4. **OutputValidator**: Structured output validation
   - Pydantic schema enforcement
   - Hallucination checks (verify cited test IDs exist)

5. **CallLimiters**: Operational limits
   - Max tool calls per thread: 10
   - Max model calls per thread: 20
   - Prevent runaway agents

**HITL Implementation**:
```python
# Interrupt before tool execution
config = {"configurable": {"thread_id": "user-123"}}
for event in graph.stream({"messages": [user_query]}, config, stream_mode="updates"):
    if event.get("__interrupt__"):
        # Human reviews proposed tool calls
        # Options: approve, edit parameters, reject
        user_decision = get_human_feedback(event)

        if user_decision == "approve":
            graph.update_state(config, None)  # Resume
        elif user_decision == "edit":
            graph.update_state(config, edited_tool_calls)
        else:  # reject
            graph.update_state(config, {"messages": [AIMessage("Action rejected")]})
```

**Deliverables**:
- `/src/agrag/middleware/pii_detector.py`
- `/src/agrag/middleware/compactor.py`
- `/src/agrag/middleware/retry_handler.py`
- `/src/agrag/middleware/validators.py`
- `/src/agrag/middleware/limiters.py`

### Phase 7: LangSmith Integration (Throughout)

**Configuration**:
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "<your-api-key>"
os.environ["LANGCHAIN_PROJECT"] = "agrag-test-scope-analysis"
```

**Traced Components**:
- Every LLM call (reasoning traces)
- Every tool execution (input/output)
- Graph state transitions (checkpoints)
- Retrieval results (documents, scores)
- Human feedback (HITL decisions)

**Evaluation Datasets**:
- Create test query datasets in LangSmith
- Tag runs by retrieval strategy
- Track metrics: precision@k, latency, tool selection accuracy
- Compare baseline approaches

**Deliverables**:
- LangSmith project configured
- Custom tags for different query types
- Evaluation datasets loaded
- Observability dashboard setup

### Phase 8: Evaluation Framework (Week 11) - RQ2

**Metrics Implementation**:
```python
def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / k if k > 0 else 0.0

def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / len(relevant) if len(relevant) > 0 else 0.0

def average_precision(retrieved: list, relevant: set) -> float:
    precisions = []
    num_relevant_found = 0

    for i, item in enumerate(retrieved, 1):
        if item in relevant:
            num_relevant_found += 1
            precisions.append(num_relevant_found / i)

    return sum(precisions) / len(relevant) if len(relevant) > 0 else 0.0

def mean_average_precision(queries_results: list) -> float:
    aps = [average_precision(r["retrieved"], r["relevant"]) for r in queries_results]
    return sum(aps) / len(aps) if aps else 0.0
```

**Baseline Approaches**:
1. **Keyword-Only**: Pure BM25 (keyword_search tool only)
2. **Vector-Only**: Pure HNSW (vector_search tool only)
3. **Agentic GraphRAG**: Full system with dynamic tool selection

**Evaluation Dataset**: 30-40 test queries with ground truth test case IDs

**Query Types Distribution**:
- 10 semantic queries ("tests related to handover failures")
- 10 structural queries ("all tests covering functions called by CellManager")
- 10 hybrid queries ("tests for LTE signaling with timeout errors")
- 5-10 edge cases (ambiguous queries, missing coverage scenarios)

**Experimental Design**:
- For each query, run all 3 approaches
- Measure: Precision@5, @10, @20; Recall@5, @10, @20; MAP
- Statistical significance testing (paired t-test)

**Deliverables**:
- `/src/agrag/evaluation/metrics.py` - Metric implementations
- `/src/agrag/evaluation/baselines.py` - Baseline runners
- `/scripts/run_evaluation.py` - Evaluation orchestrator
- Results analysis notebooks
- Statistical comparison report

### Phase 9: Testing Strategy (Week 10-11)

**Unit Tests**:
- Tool execution (mocked database clients)
- Metric calculations
- Pydantic schema validation
- Ontology consistency

**Integration Tests**:
- Full agent workflow with test databases
- Multi-hop graph traversal correctness
- RRF fusion algorithm validation
- Checkpointer persistence and recovery
- HITL interrupt/resume flow

**Evaluation Tests**:
- Baseline comparison accuracy
- Ground truth alignment
- Metric calculation verification

**Test Coverage Target**: >80%

**Deliverables**:
- `/tests/unit/` - Unit test suite
- `/tests/integration/` - Integration test suite
- `/tests/evaluation/` - Evaluation test suite
- CI configuration (GitHub Actions)

### Phase 10: CLI Application & Documentation (Week 12)

**CLI Application** (`src/agrag/cli/`):

**Commands**:
```bash
# Initialize database schemas
agrag init --neo4j-uri <uri> --postgres-uri <uri>

# Generate synthetic data
agrag generate --domain telecom --num-requirements 30

# Ingest data into databases
agrag ingest --data-dir ./data/synthetic

# Run interactive query session with HITL
agrag query --interactive

# Run query with specific strategy
agrag query "tests for handover timeout" --strategy hybrid

# Run evaluation
agrag evaluate --queries ./data/eval_queries.json --output ./results/

# Export results for thesis
agrag export --format latex --output ./thesis_results/
```

**CLI Features**:
- **Interactive Mode**: REPL-style interface for iterative querying with HITL approval/rejection
- **Streaming Output**: Real-time display of agent reasoning and tool calls
- **Result Formatting**: Pretty-print results with syntax highlighting
- **History**: Save query history to SQLite for analysis
- **Logging**: Structured logging to file with configurable verbosity

**Implementation**:
- Use `click` for command-line interface
- Use `rich` for colored terminal output and progress bars
- Use `prompt_toolkit` for interactive REPL with autocomplete

**Deliverables**:
- `/src/agrag/cli/main.py` - Main CLI entry point
- `/src/agrag/cli/commands/` - Individual command implementations
- `/src/agrag/cli/interactive.py` - Interactive REPL mode
- `/src/agrag/cli/formatters.py` - Output formatting utilities

### Phase 11: Documentation (Week 12)

**Documentation Deliverables**:

1. **README.md**: Installation, quickstart, usage examples
2. **docs/architecture.md**: System diagrams, data flow, component interactions
3. **docs/setup_guide.md**: Detailed database setup, environment configuration
4. **docs/api_reference.md**: Tool schemas, agent invocation patterns
5. **docs/evaluation.md**: Metrics explanation, baseline descriptions, results interpretation

**Deployment Guide**:
- Local development setup
- Cloud database configuration
- Environment variables
- LangSmith integration
- Production considerations (rate limiting, caching, monitoring)

**Thesis Integration**:
- Code snippets for thesis chapters
- Architecture diagrams
- Evaluation results formatting
- RQ1, RQ2, RQ3 evidence mapping

## Critical Implementation Files (Priority Order)

1. **`/src/agrag/kg/ontology.py`**
   - Defines core data model for entire system
   - All components depend on these entity/relationship definitions
   - Must be completed before any other implementation

2. **`/src/agrag/core/graph.py`**
   - Custom StateGraph implementation
   - Heart of the agent architecture
   - Demonstrates thesis requirement of full control over ReAct loop

3. **`/src/agrag/tools/schemas.py`**
   - Pydantic schemas for all 4 retrieval tools
   - Establishes tool interface contract
   - Required for LLM function calling

4. **`/src/agrag/storage/neo4j_client.py`**
   - Neo4j integration for graph operations
   - Provides graph traversal and vector search capabilities
   - Critical for multi-hop reasoning

5. **`/src/agrag/evaluation/metrics.py`**
   - Evaluation metrics (Precision@k, Recall@k, MAP)
   - Essential for RQ2 quantitative results
   - Required for thesis validation

## Key Architectural Decisions & Rationale

1. **Custom StateGraph vs createReactAgent**
   - **Decision**: Build custom graph from scratch
   - **Rationale**: Full control over ReAct loop, explicit HITL interrupts, thesis demonstration requirement, better for explaining architecture in academic context

2. **Dual Storage Architecture**
   - **Decision**: Neo4j (structural) + PostgreSQL (vector/BM25)
   - **Rationale**: Enables true hybrid retrieval, comparative analysis of approaches, aligns with PRD requirement for both semantic and structural reasoning

3. **Google Generative AI**
   - **Decision**: Use Google's models for LLM and embeddings
   - **Rationale**: State-of-the-art performance, unified API, cost-effective, native tool calling support

4. **LangSmith from Day 1**
   - **Decision**: Integrate observability from the start
   - **Rationale**: Essential for debugging complex agent behavior, evaluation tracking, industry best practice, supports thesis data collection

5. **PostgresSaver Checkpointer**
   - **Decision**: Use database-backed persistence
   - **Rationale**: Production-grade durability, HITL support with state preservation, fault tolerance, aligns with thesis requirement for reliable execution

6. **Synthetic Data Generation**
   - **Decision**: Generate demo data with LLM
   - **Rationale**: No access to real Ericsson data, allows controlled experiments, reproducible results, demonstrates system capabilities

## Dependencies (Poetry)

```toml
[tool.poetry.dependencies]
python = "^3.11"
langchain = "^0.3.0"
langchain-google-genai = "^2.0.0"
langgraph = "^0.2.0"
langgraph-checkpoint-postgres = "^2.0.0"
langsmith = "^0.2.0"
neo4j = "^5.20.0"
psycopg = {extras = ["binary"], version = "^3.2.0"}
pgvector = "^0.3.0"
pydantic = "^2.8.0"
tree-sitter = "^0.21.0"
numpy = "^1.26.0"
scipy = "^1.13.0"
click = "^8.1.0"
rich = "^13.7.0"
prompt-toolkit = "^3.0.0"

[tool.poetry.dev-dependencies]
pytest = "^8.0.0"
pytest-asyncio = "^0.23.0"
black = "^24.0.0"
ruff = "^0.5.0"
jupyter = "^1.0.0"
matplotlib = "^3.8.0"
pandas = "^2.2.0"

[tool.poetry.scripts]
agrag = "agrag.cli.main:cli"
```

## Timeline Summary

| Week | Phase | Key Deliverables |
|------|-------|------------------|
| 1-2 | Infrastructure | Databases configured, LangSmith set up |
| 3-4 | Ontology (RQ1) | Neo4j schema, ontology definitions |
| 5 | Synthetic Data | 200+ telecom artifacts generated |
| 6-7 | Ingestion | End-to-end pipeline working |
| 8 | Tools | All 4 retrieval tools functional |
| 9 | StateGraph | Working agent with full tracing |
| 10 | Middleware | Production guardrails implemented |
| 11 | Evaluation (RQ2) | 30-40 test queries, quantitative results |
| 12 | CLI + Documentation | Interactive CLI, thesis-ready system |

## Success Criteria

- ✓ Custom StateGraph with complete ReAct loop implementation
- ✓ All 4 retrieval tools working (vector, keyword, graph, hybrid)
- ✓ Dual database architecture operational (Neo4j + PostgreSQL)
- ✓ Full LangSmith tracing and evaluation
- ✓ HITL interrupts functional with PostgresSaver
- ✓ Quantitative metrics demonstrating improvement over baselines (RQ2)
- ✓ Synthetic telecommunications dataset with realistic network protocol artifacts
- ✓ 30-40 evaluation queries covering semantic, structural, and hybrid scenarios
- ✓ Interactive CLI application for demonstrations and HITL workflows
- ✓ >80% test coverage
- ✓ Comprehensive documentation for thesis integration

---

**Status**: Ready for implementation. This plan provides a clear, executable path to building a production-grade Agentic GraphRAG system that fulfills all thesis requirements (RQ1, RQ2, RQ3) while demonstrating industry best practices.

