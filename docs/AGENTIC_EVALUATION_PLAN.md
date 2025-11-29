# Agentic Evaluation Implementation Plan

## Executive Summary

The current evaluation tests each retrieval strategy (vector, keyword, hybrid, graph) **statically** on all queries. This fundamentally misaligns with the thesis hypothesis that an **agent-orchestrated dynamic approach** outperforms static pipelines.

This plan outlines the implementation of a proper agentic evaluation that directly tests RQ2:

> **RQ2**: To what extent does an agent-orchestrated GraphRAG approach improve retrieval accuracy (measured by precision@k and recall@k) compared to standard keyword-based and vector-only RAG methods?

---

## Problem Analysis

### Current Evaluation Issues

| Issue | Impact |
|-------|--------|
| Graph strategy evaluated on ALL 55 queries | 41 queries return empty (no entity ID to start traversal) |
| Each strategy tested in isolation | No measurement of dynamic tool selection |
| Binary graph results (0.0 or 1.0 RR) | Misleading 0.24 average hides 93% success on applicable queries |
| No multi-tool query support | Complex queries requiring 2+ tools not properly evaluated |

### Current Results (Misleading)

```
Strategy    | MAP    | MRR    | P@1    | R@10
------------|--------|--------|--------|-------
vector      | 0.1303 | 0.3888 | 0.2545 | 0.2305
keyword     | 0.1542 | 0.2633 | 0.2364 | 0.1702
hybrid      | 0.1652 | 0.4445 | 0.3273 | 0.2189
graph       | 0.2364 | 0.2364 | 0.2364 | 0.2364  ← All same (suspicious!)
```

### Oracle Analysis (Ceiling Performance)

If we always picked the best strategy per query:
- **Oracle MRR: 0.6147** (38% improvement over best static strategy)
- This proves dynamic selection has significant potential value

---

## Implementation Plan

### Phase 1: Agentic Evaluation Mode (Critical - RQ2)

**Goal**: Add `--strategy agent` to the evaluation CLI that runs the full LLM agent per query.

#### 1.1 Create Agentic Evaluator Module

**File**: `src/agrag/evaluation/agentic_evaluator.py`

```python
class AgenticEvaluator:
    """
    Evaluates the full agent pipeline on test scope queries.
    
    Unlike static strategy evaluation, this:
    1. Runs the complete ReAct agent loop per query
    2. Lets the LLM decide which tool(s) to use
    3. Extracts entity IDs from the agent's final response
    4. Logs tool selection patterns for analysis
    """
    
    def __init__(self, graph: CompiledStateGraph, config: RunnableConfig):
        self.graph = graph
        self.config = config
    
    def evaluate_query(self, query: str, relevant_ids: Set[str]) -> EvaluationResult:
        """Run agent on query and extract retrieved IDs."""
        # 1. Create initial state with query
        # 2. Run agent graph to completion (no HITL interrupts)
        # 3. Parse final response to extract entity IDs
        # 4. Calculate metrics against ground truth
        # 5. Log which tools were used
        pass
```

#### 1.2 Entity ID Extraction from Agent Response

The agent returns natural language responses. We need to extract entity IDs:

```python
def extract_entity_ids(response: str) -> List[str]:
    """
    Extract entity IDs from agent's natural language response.
    
    Patterns to match:
    - TC_SIGNALING_126, TC_HANDOVER_200 (test cases)
    - REQ_AUTH_005, REQ_HANDOVER_001 (requirements)  
    - FUNC_initiate_handover (functions)
    - CLASS_HandoverManager (classes)
    - MOD_authentication (modules)
    """
    patterns = [
        r'TC_[A-Z]+_\d+',           # Test cases
        r'REQ_[A-Z]+_\d+',          # Requirements
        r'FUNC_[A-Za-z_]+',         # Functions
        r'CLASS_[A-Za-z_]+',        # Classes
        r'MOD_[A-Za-z_.]+',         # Modules
    ]
    # Extract all matches, deduplicate, preserve order
```

#### 1.3 Tool Usage Tracking

Track which tools the agent selects for each query type:

```python
@dataclass
class AgentEvaluationResult:
    query_id: str
    query: str
    query_type: str
    difficulty: str
    
    # Retrieval results
    retrieved_ids: List[str]
    relevant_ids: Set[str]
    metrics: Dict[str, float]
    
    # Agent behavior analysis
    tools_used: List[str]           # e.g., ["vector_search", "graph_traverse"]
    tool_call_count: int
    model_call_count: int
    total_tokens: int
    execution_time_ms: float
    
    # Success indicators
    found_any_relevant: bool
    first_relevant_rank: Optional[int]
```

#### 1.4 CLI Integration

**File**: `src/agrag/cli/main.py`

Add `agent` to the strategy choices:

```python
@click.option(
    "--strategy",
    type=click.Choice(["vector", "keyword", "hybrid", "graph", "agent", "all"]),
    default="all",
    help="Retrieval strategy to evaluate (agent = full LLM agent)"
)
```

When `strategy == "agent"`:
1. Initialize the full agent graph (with tools)
2. Run agent per query (no HITL interrupts for eval)
3. Extract IDs from response
4. Calculate metrics

---

### Phase 2: Fair Graph Evaluation (Comparison Integrity)

**Goal**: Report "applicable-only" metrics so graph isn't unfairly penalized.

#### 2.1 Query Applicability Detection

```python
def is_graph_applicable(query: str, query_data: dict) -> bool:
    """
    Determine if graph traversal is applicable for this query.
    
    Graph needs a starting node ID to work. Applicable when:
    - Query contains explicit entity ID (REQ_*, TC_*, FUNC_*)
    - Query type is 'requirement_coverage', 'entity_lookup', 'multi_hop_traversal'
    """
    # Check for entity IDs in query text
    has_entity_id = bool(re.search(r'REQ_|TC_|FUNC_|CLASS_|MOD_', query))
    
    # Check query type metadata
    structural_types = {'requirement_coverage', 'entity_lookup', 'multi_hop_traversal'}
    is_structural = query_data.get('query_type', '') in structural_types
    
    return has_entity_id or is_structural
```

#### 2.2 Dual Metrics Reporting

Output both overall and applicable-only metrics:

```
GRAPH Strategy Results:
-----------------------
Overall (55 queries):
  MAP: 0.2364, MRR: 0.2364, P@1: 0.2364

Applicable Only (14 queries with entity IDs):
  MAP: 0.9286, MRR: 0.9286, P@1: 0.9286

Note: Graph traversal requires a starting node ID. 
      41/55 queries had no entity ID → returned empty.
```

---

### Phase 3: Enhanced Evaluation Dataset

**Goal**: Ensure evaluation dataset properly tests all strategies.

#### 3.1 Query Type Balance

Current distribution:
```
entity_lookup:        5 queries  (graph-friendly)
requirement_coverage: 8 queries  (graph-friendly after ID extraction)
function_coverage:    6 queries  (could be graph-friendly)
feature_filter:       4 queries  (vector/hybrid)
...
```

#### 3.2 Add Multi-Step Queries

Create queries that **require** the agent to use multiple tools:

```json
{
  "id": "Q_MULTI_001",
  "query": "What tests cover functions in the HandoverManager class?",
  "relevant_ids": ["TC_HANDOVER_001", "TC_HANDOVER_015", ...],
  "difficulty": "complex",
  "query_type": "multi_hop_traversal",
  "expected_tool_sequence": ["keyword_search", "graph_traverse"],
  "reasoning": "First find CLASS_HandoverManager, then traverse to find covering tests"
}
```

#### 3.3 Query Categories for Analysis

Tag queries with expected best strategy:

```json
{
  "id": "Q_056",
  "query": "Tests for requirement REQ_AUTH_005",
  "relevant_ids": [...],
  "expected_best_strategy": "graph",
  "applicable_strategies": ["graph", "keyword", "hybrid"],
  "rationale": "Has explicit REQ ID, graph traversal is optimal"
}
```

---

### Phase 4: Results Analysis & Visualization

#### 4.1 Strategy Selection Analysis

Track how often agent picks each strategy vs. "optimal":

```python
def analyze_tool_selection(results: List[AgentEvaluationResult]) -> Dict:
    """
    Analyze agent's tool selection patterns.
    
    Returns:
    - Tool usage frequency
    - Tool selection by query type
    - Correlation between tool choice and success
    - Cases where agent outperformed/underperformed oracle
    """
```

Output:
```
Tool Selection Analysis
=======================

By Query Type:
  entity_lookup:       keyword_search (80%), vector_search (20%)
  requirement_coverage: graph_traverse (60%), keyword_search (40%)
  feature_filter:      vector_search (70%), hybrid_search (30%)
  
Agent vs Oracle:
  Agent matched oracle:    35/55 (64%)
  Agent beat single-best:  8/55 (15%)  ← Multi-tool advantage
  Agent underperformed:    12/55 (22%)
```

#### 4.2 Results JSON Schema

Enhanced output format:

```json
{
  "evaluation_metadata": {
    "dataset": "data/synthetic_dataset_eval.json",
    "timestamp": "2025-11-29T10:30:00Z",
    "queries_count": 55,
    "k_values": [1, 3, 5, 10]
  },
  
  "strategy_results": {
    "vector": { "map": 0.13, "mrr": 0.39, ... },
    "keyword": { "map": 0.15, "mrr": 0.26, ... },
    "hybrid": { "map": 0.17, "mrr": 0.44, ... },
    "graph": {
      "overall": { "map": 0.24, "mrr": 0.24, ... },
      "applicable_only": { "map": 0.93, "mrr": 0.93, "query_count": 14 }
    },
    "agent": {
      "map": 0.52,
      "mrr": 0.58,
      "tool_usage": {
        "vector_search": 28,
        "keyword_search": 15,
        "graph_traverse": 12,
        "hybrid_search": 8
      },
      "avg_tools_per_query": 1.4,
      "avg_execution_time_ms": 1250
    }
  },
  
  "comparative_analysis": {
    "agent_vs_best_static": {
      "mrr_improvement": "+31%",
      "map_improvement": "+44%"
    },
    "oracle_ceiling": {
      "mrr": 0.61,
      "agent_achieves": "95% of oracle"
    }
  },
  
  "per_query_results": [...]
}
```

---

## Implementation Timeline

| Phase | Task | Effort | Priority |
|-------|------|--------|----------|
| **1.1** | Create `AgenticEvaluator` class | 2-3 hours | Critical |
| **1.2** | Implement entity ID extraction | 1 hour | Critical |
| **1.3** | Add tool usage tracking | 1 hour | Critical |
| **1.4** | CLI integration (`--strategy agent`) | 1 hour | Critical |
| **2.1** | Query applicability detection | 30 min | High |
| **2.2** | Dual metrics reporting | 1 hour | High |
| **3.1** | Analyze query type balance | 30 min | Medium |
| **3.2** | Add multi-step queries | 1 hour | Medium |
| **4.1** | Strategy selection analysis | 1-2 hours | Medium |
| **4.2** | Enhanced JSON output | 1 hour | Medium |

**Total Estimated Effort**: 10-12 hours

---

## Success Criteria

### Primary (RQ2)

| Metric | Target | Rationale |
|--------|--------|-----------|
| Agent MRR > Hybrid MRR (0.44) | Required | Proves dynamic selection helps |
| Agent MRR > 0.55 | Desired | Significant improvement |
| Agent achieves >85% of Oracle | Stretch | Near-optimal tool selection |

### Secondary (Analysis)

| Metric | Target |
|--------|--------|
| Agent uses 1-2 tools on average per query | Efficient |
| Agent selects graph for structural queries >70% | Correct tool mapping |
| Agent execution time <3s per query | Practical |

---

## Technical Considerations

### 1. Running Agent Without HITL

For evaluation, agent must run autonomously:

```python
# In create_agent_graph(), conditionally disable interrupts
def create_agent_graph(
    checkpointer: Optional[PostgresSaver] = None,
    enable_hitl: bool = True,  # NEW: disable for eval
) -> CompiledStateGraph:
    ...
    if enable_hitl:
        graph = builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["execute_tools"]
        )
    else:
        graph = builder.compile(checkpointer=checkpointer)
```

### 2. Cost Control

Agent evaluation = many LLM calls. Add safeguards:

```python
# Limit tool calls per query
MAX_TOOL_CALLS_EVAL = 5

# Use smaller model for eval (optional)
EVAL_MODEL = "gemini-2.0-flash"  # vs gemini-2.5-pro

# Batch queries to amortize overhead
BATCH_SIZE = 10
```

### 3. Reproducibility

For consistent evaluation:

```python
# Set temperature to 0 for deterministic outputs
EVAL_TEMPERATURE = 0.0

# Use fixed random seed if any sampling
RANDOM_SEED = 42

# Log full LangSmith traces for each eval run
LANGSMITH_PROJECT = "agrag-eval-{timestamp}"
```

---

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `src/agrag/evaluation/agentic_evaluator.py` | Main agentic evaluation logic |
| `src/agrag/evaluation/entity_extractor.py` | Extract entity IDs from responses |
| `src/agrag/evaluation/tool_tracker.py` | Track tool usage patterns |

### Modified Files

| File | Changes |
|------|---------|
| `src/agrag/cli/main.py` | Add `--strategy agent`, integrate evaluator |
| `src/agrag/core/graph.py` | Add `enable_hitl` parameter |
| `src/agrag/evaluation/__init__.py` | Export new classes |
| `data/synthetic_dataset_eval.json` | Add multi-step queries (Phase 3) |

---

## Validation Plan

After implementation:

1. **Smoke Test**: Run `agrag evaluate --strategy agent --dataset data/synthetic_dataset_eval.json`
2. **Verify Tool Tracking**: Check that tool usage is logged correctly
3. **Compare Results**: Agent MRR should be > Hybrid MRR (0.44)
4. **Review LangSmith Traces**: Verify agent reasoning is sound
5. **Run Full Comparison**: `agrag evaluate --strategy all` with new agent strategy

---

## Appendix: Current Codebase References

### Agent Graph Definition
- Location: `src/agrag/core/graph.py`
- Key function: `create_agent_graph()`
- System prompt: Defines tool selection strategy (lines 27-150)

### Evaluation CLI
- Location: `src/agrag/cli/main.py` 
- Function: `evaluate()` (lines 280-490)
- Helper: `_execute_retrieval()` (lines 490-520)

### Metrics
- Location: `src/agrag/evaluation/metrics.py`
- Functions: `precision_at_k`, `recall_at_k`, `mean_average_precision`, `mean_reciprocal_rank`

### Tools
- Vector: `src/agrag/tools/vector_search.py`
- Keyword: `src/agrag/tools/keyword_search.py`  
- Hybrid: `src/agrag/tools/hybrid_search.py`
- Graph: `src/agrag/tools/graph_traverse.py`
