# LangSmith + Ragas Integration Plan

**Project:** Agentic RAG Test Scope Analysis  
**Date:** January 16, 2026  
**Status:** Planning  
**Authors:** Development Team

## Executive Summary

This plan outlines the integration of LangSmith's evaluation experiments and Ragas RAG-specific metrics into the existing evaluation infrastructure. The integration will extend current entity-matching evaluations with semantic quality grading, multi-trial stability metrics, and experiment tracking, while maintaining backward compatibility with local evaluation workflows.

### Key Objectives

1. **Add semantic quality metrics** using Ragas (faithfulness, answer relevancy, context recall/precision, answer correctness)
2. **Enable LangSmith experiment tracking** for visualization, collaboration, and regression detection
3. **Implement multi-trial evaluation** with stability metrics (pass@k, variance, std deviation)
4. **Improve context tracking** for robust RAG evaluation
5. **Preserve backward compatibility** with existing CLI commands and evaluation workflows

### Success Criteria

- ✅ Ragas metrics run on existing datasets without errors
- ✅ LangSmith experiments show traces and metrics in UI
- ✅ Multi-trial evaluation produces stability statistics
- ✅ Backward compatible with existing `agrag evaluate` command
- ✅ Performance acceptable for batch evaluation (100+ queries)
- ✅ Clear documentation of metric definitions and usage

---

## Current State Analysis

### Existing Evaluation Infrastructure

**Strengths:**
- ✅ Robust entity ID extraction (`TC_*`, `REQ_*`, `FUNC_*`)
- ✅ Traditional IR metrics (P@k, R@k, F1@k, MAP, MRR)
- ✅ Tool usage tracking and analysis
- ✅ LangSmith tracing already configured
- ✅ Batch processing with async execution
- ✅ Multiple dataset formats (JSON, JSONL, CSV)

**Gaps:**
- ❌ No semantic similarity metrics
- ❌ No context relevance metrics
- ❌ No hallucination detection
- ❌ No LLM-as-judge evaluation
- ❌ No multi-trial support
- ❌ Limited context tracking for RAG evaluation
- ❌ No LangSmith experiment features (only tracing)

### Key Files and Components

**Evaluation:**
- `src/agrag/evaluation/agentic_evaluator.py` - Main evaluator (AgenticEvaluator class)
- `src/agrag/evaluation/metrics.py` - IR metrics implementation
- `src/agrag/evaluation/entity_extractor.py` - Entity ID extraction

**Agent:**
- `src/agrag/core/graph.py` - Agent graph creation
- `src/agrag/core/state.py` - AgentState TypedDict
- `src/agrag/core/nodes.py` - Graph nodes (call_model, execute_tools)

**CLI:**
- `src/agrag/cli/commands.py` - CLI commands including `evaluate`
- `src/agrag/cli/display.py` - Output formatting

**Configuration:**
- `src/agrag/config/settings.py` - Settings with LangSmith config
- `.env.example` - Environment variables

**Data:**
- `data/synthetic_dataset.json` - Full knowledge graph
- `data/test_dataset.json` - Test queries
- `src/agrag/data/synthetic_generator.py` - Dataset generator

---

## Implementation Plan

### Phase 1: Foundation - Ragas Dependencies & Metrics Module

**Timeline:** Days 1-2  
**Files:** `pyproject.toml`, `src/agrag/evaluation/ragas_metrics.py`

#### 1.1 Add Dependencies

Update `pyproject.toml`:

```toml
[tool.poetry.dependencies]
ragas = "^0.2.0"
datasets = "^2.19.0"
# Ensure langsmith is up to date
langsmith = "^0.6.2"
```

Run:
```bash
poetry lock
poetry install
```

#### 1.2 Create Ragas Metrics Module

Create `src/agrag/evaluation/ragas_metrics.py`:

**Key Components:**

```python
class RagasEvaluator:
    """Evaluator for RAG-specific metrics using Ragas with Gemini."""
    
    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        max_retries: int = 3,
        api_key: Optional[str] = None
    ):
        """Initialize with Gemini model matching agent configuration."""
        pass
    
    async def evaluate_with_ragas(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate using Ragas metrics.
        
        Returns:
            {
                'faithfulness': float,
                'answer_relevancy': float,
                'context_recall': float,
                'context_precision': float,
                'answer_correctness': float  # if ground_truth provided
            }
        """
        pass
    
    def format_contexts_for_ragas(
        self,
        retrieved_contexts: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Convert state contexts to Ragas format (list of strings).
        
        Deduplicates by content hash to reduce noise.
        Handles missing fields gracefully.
        """
        pass

def retry_with_backoff(max_retries: int = 3):
    """Decorator for exponential backoff on transient API failures."""
    pass
```

**Metrics Implemented:**
1. **Faithfulness** - Are answers grounded in retrieved context?
2. **Answer Relevancy** - Is the answer relevant to the query?
3. **Context Recall** - Did we retrieve relevant context for the ground truth?
4. **Context Precision** - Are retrieved chunks ranked well?
5. **Answer Correctness** - Combines semantic + factual correctness (requires ground truth)

**Configuration:**
- Use same Gemini model as agent (`gemini-3-flash-preview`)
- Default temperature: 0.0 for consistency
- Max retries: 3 with exponential backoff
- Deduplicate contexts by content hash

---

### Phase 2: Context Tracking Enhancement

**Timeline:** Days 2-3  
**Files:** `src/agrag/core/state.py`, `src/agrag/core/nodes.py`

#### 2.1 Extend AgentState

Update `src/agrag/core/state.py`:

```python
import operator
from typing import Annotated, List, Dict, Any

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    tool_call_count: int
    model_call_count: int
    final_answer: str
    
    # NEW: Context tracking for evaluation
    retrieved_contexts: Annotated[List[Dict[str, Any]], operator.add]
    enable_context_tracking: bool
```

**Context Structure:**
```python
{
    'chunk_text': str,        # Retrieved text
    'source': str,            # entity_id or file_path
    'entity_type': str,       # TestCase, Requirement, Function, etc.
    'score': float,           # Similarity/relevance score
    'tool_name': str,         # vector_search, keyword_search, etc.
    'timestamp': float        # When retrieved
}
```

#### 2.2 Modify Tool Execution Node

Update `src/agrag/core/nodes.py` in `execute_tools()`:

```python
def execute_tools(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Execute tool calls and optionally track contexts."""
    
    # ... existing tool execution logic ...
    
    # NEW: Extract and store contexts if tracking enabled
    if state.get('enable_context_tracking', False):
        contexts = []
        for tool_call, tool_result in zip(tool_calls, tool_results):
            # Parse tool result for retrieved entities/chunks
            extracted_contexts = extract_contexts_from_tool_result(
                tool_name=tool_call.name,
                tool_result=tool_result
            )
            contexts.extend(extracted_contexts)
        
        return {
            'messages': tool_result_messages,
            'tool_call_count': state['tool_call_count'] + len(tool_calls),
            'retrieved_contexts': contexts
        }
    
    # Standard return without contexts
    return {
        'messages': tool_result_messages,
        'tool_call_count': state['tool_call_count'] + len(tool_calls)
    }
```

**Helper Function:**
```python
def extract_contexts_from_tool_result(
    tool_name: str,
    tool_result: ToolMessage
) -> List[Dict[str, Any]]:
    """
    Extract structured contexts from tool result.
    
    Handles:
    - vector_search: List[Dict] with entity_id, content, score
    - keyword_search: List[Dict] with entity_id, content, score
    - graph_traverse: List[Dict] with entity_id, properties
    - hybrid_search: List[Dict] with entity_id, content, fused_score
    """
    pass
```

---

### Phase 3: Multi-Trial Evaluation

**Timeline:** Days 3-5  
**Files:** `src/agrag/evaluation/agentic_evaluator.py`

#### 3.1 Extend Data Models

Update `AgentEvaluationResult` dataclass:

```python
@dataclass
class AgentEvaluationResult:
    query_id: str
    query: str
    retrieved_ids: List[str]
    relevant_ids: Set[str]
    metrics: Dict[str, float]  # P@k, R@k, F1@k, AP, RR
    tools_used: List[str]
    tool_call_count: int
    model_call_count: int
    execution_time_ms: float
    final_answer: str
    error: Optional[str]
    
    # NEW: Ragas and trial support
    ragas_metrics: Optional[Dict[str, float]] = None
    trial_number: int = 1
    contexts_used: List[str] = field(default_factory=list)
```

Update `AgentEvaluationSummary` dataclass:

```python
@dataclass
class AgentEvaluationSummary:
    map_score: float
    mrr_score: float
    avg_precision_at_k: Dict[int, float]
    avg_recall_at_k: Dict[int, float]
    avg_f1_at_k: Dict[int, float]
    total_queries: int
    successful_queries: int
    total_tool_calls: int
    avg_tools_per_query: float
    tool_frequency: Dict[str, int]
    tool_combinations: Dict[str, int]
    results: List[AgentEvaluationResult]
    
    # NEW: Multi-trial and Ragas support
    trial_statistics: Optional[Dict[str, Any]] = None
    avg_ragas_metrics: Optional[Dict[str, float]] = None
```

#### 3.2 Add Multi-Trial Evaluation

Update `AgenticEvaluator.__init__()`:

```python
class AgenticEvaluator:
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        postgres_client: PostgresClient,
        models: ModelProvider,
        use_ragas: bool = False,
        num_trials: int = 1,
        enable_context_tracking: bool = True,
        k_values: Optional[List[int]] = None,
    ):
        """Initialize evaluator with optional Ragas and multi-trial support."""
        self.use_ragas = use_ragas
        self.num_trials = num_trials
        self.enable_context_tracking = enable_context_tracking
        
        if use_ragas:
            self.ragas_evaluator = RagasEvaluator(
                model_name=settings.ragas_model,
                max_retries=settings.ragas_max_retries
            )
```

**New Method:**

```python
async def evaluate_query_with_trials(
    self,
    query: str,
    relevant_ids: List[str],
    ground_truth_answer: Optional[str] = None,
    query_id: Optional[str] = None
) -> List[AgentEvaluationResult]:
    """
    Evaluate a query with multiple trials.
    
    Returns:
        List of results, one per trial
    """
    trial_results = []
    
    for trial_num in range(1, self.num_trials + 1):
        # Create initial state with context tracking enabled
        initial_state = {
            'messages': [HumanMessage(content=query)],
            'tool_call_count': 0,
            'model_call_count': 0,
            'final_answer': '',
            'retrieved_contexts': [],
            'enable_context_tracking': self.enable_context_tracking
        }
        
        # Run agent
        result = await self.evaluate_query(
            query=query,
            relevant_ids=relevant_ids,
            ground_truth_answer=ground_truth_answer,
            query_id=query_id,
            trial_number=trial_num,
            initial_state=initial_state
        )
        
        trial_results.append(result)
    
    return trial_results
```

**Update `evaluate_query()` Method:**

```python
async def evaluate_query(
    self,
    query: str,
    relevant_ids: List[str],
    ground_truth_answer: Optional[str] = None,
    query_id: Optional[str] = None,
    trial_number: int = 1,
    initial_state: Optional[Dict] = None
) -> AgentEvaluationResult:
    """Evaluate a single query (single trial)."""
    
    # ... existing agent invocation ...
    
    # Extract contexts if tracking enabled
    contexts = []
    if self.enable_context_tracking:
        contexts = self.ragas_evaluator.format_contexts_for_ragas(
            final_state.get('retrieved_contexts', [])
        )
    
    # Compute Ragas metrics if enabled
    ragas_metrics = None
    if self.use_ragas and contexts:
        ragas_metrics = await self.ragas_evaluator.evaluate_with_ragas(
            query=query,
            answer=final_answer,
            contexts=contexts,
            ground_truth=ground_truth_answer
        )
    
    return AgentEvaluationResult(
        # ... existing fields ...
        ragas_metrics=ragas_metrics,
        trial_number=trial_number,
        contexts_used=contexts
    )
```

#### 3.3 Aggregate Multi-Trial Statistics

**New Method:**

```python
def aggregate_trial_statistics(
    self,
    trial_results: List[AgentEvaluationResult]
) -> Dict[str, Any]:
    """
    Aggregate statistics across multiple trials.
    
    Returns:
        {
            'num_trials': int,
            'success_rate': float,
            'pass_at_1': float,  # At least 1 trial passed
            'pass_at_k': float,  # All trials passed
            'mean_metrics': {...},
            'std_metrics': {...},
            'min_metrics': {...},
            'max_metrics': {...},
            'stability_score': float  # 1 - normalized std
        }
    """
    pass
```

---

### Phase 4: LangSmith Integration

**Timeline:** Days 5-7  
**Files:** `src/agrag/evaluation/langsmith_evaluator.py`

#### 4.1 Create LangSmith Evaluator Module

Create `src/agrag/evaluation/langsmith_evaluator.py`:

```python
from langsmith import Client
from langsmith.evaluation import evaluate
from typing import Dict, List, Any, Optional
import json

class LangSmithEvaluator:
    """
    Evaluator that uploads datasets and runs experiments in LangSmith.
    """
    
    def __init__(
        self,
        project_name: str = "agrag-test-scope-analysis",
        use_ragas: bool = False,
        num_trials: int = 1
    ):
        """Initialize LangSmith client and configuration."""
        self.client = Client()
        self.project_name = project_name
        self.use_ragas = use_ragas
        self.num_trials = num_trials
    
    def upload_eval_dataset(
        self,
        dataset_name: str,
        queries: List[Dict[str, Any]],
        description: Optional[str] = None,
        version: Optional[str] = None
    ) -> str:
        """
        Upload evaluation queries to LangSmith dataset.
        
        Converts from local format:
        {
            "query": "What tests cover handover?",
            "relevant_ids": ["TC_001", "TC_002"],
            "reference_answer": "TC_001 and TC_002 test handover..."
        }
        
        To LangSmith format:
        {
            "inputs": {"query": "What tests cover handover?"},
            "outputs": {
                "expected_ids": ["TC_001", "TC_002"],
                "reference_answer": "TC_001 and TC_002 test handover..."
            },
            "metadata": {"query_type": "test_coverage", "difficulty": "medium"}
        }
        
        Returns:
            Dataset name (versioned if specified)
        """
        pass
    
    def run_experiment(
        self,
        dataset_name: str,
        agent_function: Callable,
        experiment_name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        max_concurrency: int = 10
    ) -> Dict[str, Any]:
        """
        Run LangSmith experiment with multi-trial support.
        
        Returns:
            {
                'experiment_url': str,
                'summary': {...},
                'results': [...]
            }
        """
        pass
    
    def create_evaluators(self) -> List[Callable]:
        """
        Create LangSmith-compatible evaluators.
        
        Returns list of evaluator functions for:
        1. Entity matching (P@k, MAP, MRR)
        2. Ragas metrics (if enabled)
        3. Tool usage validation
        4. Error detection
        """
        pass
```

#### 4.2 Custom Evaluators

**Entity Matching Evaluator:**

```python
def entity_matching_evaluator(run, example):
    """
    LangSmith evaluator for entity ID matching.
    
    Computes P@5, R@5, F1@5, MAP, MRR.
    """
    retrieved_ids = extract_entity_ids_from_answer(
        run.outputs.get('final_answer', '')
    )
    expected_ids = set(example.outputs.get('expected_ids', []))
    
    # Compute metrics
    metrics_result = evaluate_retrieval(
        retrieved_ids=retrieved_ids,
        relevant_ids=expected_ids,
        k_values=[5]
    )
    
    return {
        'key': 'entity_matching',
        'score': metrics_result['f1_at_5'],
        'comment': f"P@5={metrics_result['precision_at_5']:.3f}, MAP={metrics_result['map']:.3f}"
    }
```

**Ragas Bridge Evaluator:**

```python
async def ragas_evaluator(run, example):
    """
    LangSmith evaluator bridge to Ragas metrics.
    """
    if not self.use_ragas:
        return None
    
    # Extract contexts from run
    contexts = extract_contexts_from_run(run)
    
    ragas_eval = RagasEvaluator()
    metrics = await ragas_eval.evaluate_with_ragas(
        query=example.inputs['query'],
        answer=run.outputs.get('final_answer', ''),
        contexts=contexts,
        ground_truth=example.outputs.get('reference_answer')
    )
    
    return {
        'key': 'ragas_composite',
        'score': np.mean(list(metrics.values())),
        'comment': f"Faithfulness={metrics['faithfulness']:.3f}, Relevancy={metrics['answer_relevancy']:.3f}"
    }
```

**Tool Usage Evaluator:**

```python
def tool_usage_evaluator(run, example):
    """
    Validate tool selection appropriateness.
    """
    tools_used = extract_tools_from_messages(run.outputs.get('messages', []))
    query_type = example.metadata.get('query_type', 'unknown')
    
    # Expected tool patterns by query type
    expected_tools = {
        'test_coverage': ['vector_search', 'graph_traverse'],
        'impact_analysis': ['graph_traverse'],
        'semantic_search': ['vector_search', 'hybrid_search'],
        'exact_match': ['keyword_search']
    }
    
    appropriate = any(
        tool in tools_used 
        for tool in expected_tools.get(query_type, [])
    )
    
    return {
        'key': 'tool_appropriateness',
        'score': 1.0 if appropriate else 0.0,
        'comment': f"Used: {', '.join(tools_used)}"
    }
```

---

### Phase 5: CLI Enhancement & Configuration

**Timeline:** Days 7-8  
**Files:** `src/agrag/cli/commands.py`, `src/agrag/config/settings.py`

#### 5.1 Update Settings

Add to `src/agrag/config/settings.py`:

```python
class Settings(BaseSettings):
    # ... existing settings ...
    
    # Ragas Configuration
    ragas_model: str = "gemini-3-flash-preview"
    ragas_max_retries: int = 3
    ragas_enabled: bool = False
    
    # LangSmith Experiments
    langsmith_experiments_enabled: bool = False
    langsmith_dataset_version: str = "v1"
    langsmith_max_concurrency: int = 10
```

Add to `.env.example`:

```bash
# Ragas Configuration
RAGAS_MODEL=gemini-3-flash-preview
RAGAS_MAX_RETRIES=3
RAGAS_ENABLED=false

# LangSmith Experiments
LANGSMITH_EXPERIMENTS_ENABLED=false
LANGSMITH_DATASET_VERSION=v1
LANGSMITH_MAX_CONCURRENCY=10
```

#### 5.2 Extend Evaluate Command

Update `src/agrag/cli/commands.py`:

```python
@cli.command()
@click.option('--dataset', required=True, help='Path to evaluation dataset JSON')
@click.option('--strategy', 
              type=click.Choice(['vector', 'keyword', 'graph', 'hybrid', 'agent']),
              default='agent',
              help='Evaluation strategy')
@click.option('--output', help='Output JSON file path')
@click.option('--k-values', default='1,3,5,10', help='Comma-separated k values')
@click.option('--use-ragas', is_flag=True, help='Enable Ragas metrics')
@click.option('--use-langsmith', is_flag=True, help='Run as LangSmith experiment')
@click.option('--num-trials', default=1, type=int, help='Number of trials per query')
@click.option('--summary-only', is_flag=True, help='Output aggregate statistics only')
@click.option('--experiment-name', help='LangSmith experiment name')
def evaluate(
    dataset: str,
    strategy: str,
    output: Optional[str],
    k_values: str,
    use_ragas: bool,
    use_langsmith: bool,
    num_trials: int,
    summary_only: bool,
    experiment_name: Optional[str]
):
    """
    Evaluate agent on a dataset with optional Ragas and LangSmith.
    
    Examples:
        # Basic evaluation
        agrag evaluate --dataset data/eval.json
        
        # With Ragas metrics
        agrag evaluate --dataset data/eval.json --use-ragas
        
        # Multi-trial with summary
        agrag evaluate --dataset data/eval.json --num-trials 5 --summary-only
        
        # LangSmith experiment
        agrag evaluate --dataset data/eval.json --use-langsmith --use-ragas
    """
    
    # Load dataset
    with open(dataset) as f:
        eval_data = json.load(f)
    
    # Parse k values
    k_vals = [int(k) for k in k_values.split(',')]
    
    if use_langsmith:
        # LangSmith experiment path
        run_langsmith_experiment(
            dataset_path=dataset,
            use_ragas=use_ragas,
            num_trials=num_trials,
            experiment_name=experiment_name,
            output_path=output
        )
    else:
        # Local evaluation path
        run_local_evaluation(
            dataset=eval_data,
            strategy=strategy,
            k_values=k_vals,
            use_ragas=use_ragas,
            num_trials=num_trials,
            summary_only=summary_only,
            output_path=output
        )
```

#### 5.3 Add Experiment Command

New command:

```python
@cli.command()
@click.option('--dataset', required=True, help='Evaluation dataset name or path')
@click.option('--name', help='Experiment name')
@click.option('--num-trials', default=5, type=int, help='Trials per query')
@click.option('--use-ragas', is_flag=True, help='Enable Ragas metrics')
@click.option('--upload', is_flag=True, help='Upload dataset if not exists')
@click.option('--compare-to', help='Compare to existing experiment')
def experiment(
    dataset: str,
    name: Optional[str],
    num_trials: int,
    use_ragas: bool,
    upload: bool,
    compare_to: Optional[str]
):
    """
    Run LangSmith experiment with visualization.
    
    Examples:
        # Run new experiment
        agrag experiment --dataset agrag-eval-v1 --num-trials 5
        
        # Upload local dataset and run
        agrag experiment --dataset data/eval.json --upload --use-ragas
        
        # Compare with baseline
        agrag experiment --dataset agrag-eval-v1 --compare-to baseline-v1
    """
    pass
```

#### 5.4 Summary Output Formatter

Create helper in `src/agrag/cli/display.py`:

```python
def format_summary_table(
    summary: AgentEvaluationSummary,
    include_ragas: bool = False,
    include_trials: bool = False
) -> str:
    """
    Format evaluation summary as ASCII table.
    
    Example output:
    
    ┌─────────────────────────┬────────────┬───────────┐
    │ Metric                  │ Mean       │ Std Dev   │
    ├─────────────────────────┼────────────┼───────────┤
    │ MAP                     │ 0.847      │ 0.023     │
    │ MRR                     │ 0.912      │ 0.018     │
    │ Precision@5             │ 0.823      │ 0.034     │
    │ Recall@5                │ 0.891      │ 0.027     │
    │ F1@5                    │ 0.856      │ 0.029     │
    ├─────────────────────────┼────────────┼───────────┤
    │ Faithfulness            │ 0.934      │ 0.012     │
    │ Answer Relevancy        │ 0.878      │ 0.021     │
    │ Context Recall          │ 0.845      │ 0.031     │
    └─────────────────────────┴────────────┴───────────┘
    
    Success Rate: 98.5% (197/200 queries)
    Avg Execution Time: 3.2s
    Pass@5: 94.0%
    Stability Score: 0.89
    """
    pass
```

---

### Phase 6: Reference Answers & Testing

**Timeline:** Days 8-10  
**Files:** `src/agrag/data/synthetic_generator.py`, `tests/evaluation/`

#### 6.1 Generate Reference Answers

Update `src/agrag/data/synthetic_generator.py`:

```python
class TelecomDataGenerator:
    
    def generate_reference_answer(
        self,
        query: str,
        query_type: str,
        relevant_entities: List[Dict[str, Any]]
    ) -> str:
        """
        Generate reference answer for evaluation query.
        
        Format varies by query type:
        - test_coverage: "Tests X, Y cover requirement Z because..."
        - impact_analysis: "Changes to X affect Y and Z..."
        - failure_triage: "Failure likely caused by..."
        """
        
        if query_type == 'test_coverage':
            test_ids = [e['id'] for e in relevant_entities if e['type'] == 'TestCase']
            return f"The following tests cover this requirement: {', '.join(test_ids)}. These tests verify..."
        
        elif query_type == 'impact_analysis':
            # ... generate impact description ...
            pass
        
        # ... other query types ...
    
    def generate_evaluation_dataset(
        self,
        test_cases: List[Dict],
        requirements: List[Dict],
        functions: List[Dict],
        relationships: List[Dict],
        num_queries: int = 50
    ) -> Dict[str, Any]:
        """Generate evaluation queries WITH reference answers."""
        
        queries = []
        
        for i in range(num_queries):
            query_type = random.choice([
                'test_coverage', 
                'impact_analysis', 
                'coverage_by_component',
                'failure_triage'
            ])
            
            # ... generate query and find relevant entities ...
            
            # NEW: Generate reference answer
            reference_answer = self.generate_reference_answer(
                query=query_text,
                query_type=query_type,
                relevant_entities=relevant_entities
            )
            
            queries.append({
                'id': f'Q_{i+1:03d}',
                'query': query_text,
                'relevant_ids': [e['id'] for e in relevant_entities],
                'reference_answer': reference_answer,  # NEW
                'query_type': query_type,
                'difficulty': difficulty
            })
        
        return {'queries': queries}
```

#### 6.2 Create Integration Tests

**Test Ragas Integration:**

Create `tests/evaluation/test_ragas_integration.py`:

```python
import pytest
from src.agrag.evaluation.ragas_metrics import RagasEvaluator

class TestRagasIntegration:
    
    @pytest.fixture
    def ragas_evaluator(self):
        return RagasEvaluator(
            model_name="gemini-3-flash-preview",
            max_retries=3
        )
    
    def test_ragas_with_contexts(self, ragas_evaluator):
        """Test Ragas evaluation with valid contexts."""
        query = "What tests cover handover requirements?"
        answer = "TC_HANDOVER_001 and TC_HANDOVER_003 cover handover scenarios."
        contexts = [
            "TestCase TC_HANDOVER_001 verifies basic handover functionality...",
            "TestCase TC_HANDOVER_003 tests edge case scenarios for handover..."
        ]
        ground_truth = "TC_HANDOVER_001 and TC_HANDOVER_003 test handover."
        
        result = ragas_evaluator.evaluate_with_ragas(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth
        )
        
        assert 'faithfulness' in result
        assert 'answer_relevancy' in result
        assert 'context_recall' in result
        assert 'context_precision' in result
        assert 'answer_correctness' in result
        
        assert 0.0 <= result['faithfulness'] <= 1.0
        assert result['answer_relevancy'] > 0.5  # Should be relevant
    
    def test_ragas_without_ground_truth(self, ragas_evaluator):
        """Test Ragas without reference answer."""
        result = ragas_evaluator.evaluate_with_ragas(
            query="test query",
            answer="test answer",
            contexts=["context 1"],
            ground_truth=None
        )
        
        assert 'answer_correctness' not in result
        assert 'faithfulness' in result
    
    def test_context_deduplication(self, ragas_evaluator):
        """Test that duplicate contexts are deduplicated."""
        contexts = [
            {'chunk_text': 'Same content', 'source': 'TC_001'},
            {'chunk_text': 'Same content', 'source': 'TC_002'},
            {'chunk_text': 'Different content', 'source': 'TC_003'}
        ]
        
        formatted = ragas_evaluator.format_contexts_for_ragas(contexts)
        
        assert len(formatted) == 2  # Deduplicated
        assert 'Same content' in formatted
        assert 'Different content' in formatted
    
    def test_retry_on_api_failure(self, ragas_evaluator, monkeypatch):
        """Test exponential backoff on transient failures."""
        call_count = 0
        
        def mock_api_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Transient API error")
            return {'faithfulness': 0.9}
        
        monkeypatch.setattr(ragas_evaluator, '_call_ragas_api', mock_api_call)
        
        result = ragas_evaluator.evaluate_with_ragas(
            query="test",
            answer="answer",
            contexts=["context"]
        )
        
        assert call_count == 3  # Retried twice before success
        assert result['faithfulness'] == 0.9
```

**Test Multi-Trial Support:**

Create `tests/evaluation/test_multi_trial.py`:

```python
import pytest
from src.agrag.evaluation.agentic_evaluator import AgenticEvaluator

class TestMultiTrialEvaluation:
    
    @pytest.fixture
    def evaluator(self, postgres_client, neo4j_client):
        return AgenticEvaluator(
            neo4j_client=neo4j_client,
            postgres_client=postgres_client,
            models=ModelProvider(),
            num_trials=5,
            enable_context_tracking=True
        )
    
    async def test_multiple_trials_per_query(self, evaluator):
        """Test that multiple trials are executed per query."""
        results = await evaluator.evaluate_query_with_trials(
            query="What tests cover handover?",
            relevant_ids=['TC_001', 'TC_002']
        )
        
        assert len(results) == 5
        assert all(r.trial_number in range(1, 6) for r in results)
    
    async def test_trial_statistics_aggregation(self, evaluator):
        """Test that trial statistics are correctly aggregated."""
        results = await evaluator.evaluate_query_with_trials(
            query="Test query",
            relevant_ids=['TC_001']
        )
        
        stats = evaluator.aggregate_trial_statistics(results)
        
        assert 'num_trials' in stats
        assert 'success_rate' in stats
        assert 'pass_at_1' in stats
        assert 'pass_at_k' in stats
        assert 'mean_metrics' in stats
        assert 'std_metrics' in stats
        assert 'stability_score' in stats
        
        assert stats['num_trials'] == 5
        assert 0.0 <= stats['stability_score'] <= 1.0
    
    async def test_context_tracking_enabled(self, evaluator):
        """Test that contexts are tracked when enabled."""
        results = await evaluator.evaluate_query_with_trials(
            query="What tests cover handover?",
            relevant_ids=['TC_001']
        )
        
        for result in results:
            assert result.contexts_used is not None
            assert len(result.contexts_used) > 0
    
    async def test_ragas_metrics_per_trial(self, evaluator):
        """Test that Ragas metrics are computed per trial."""
        evaluator.use_ragas = True
        
        results = await evaluator.evaluate_query_with_trials(
            query="What tests cover handover?",
            relevant_ids=['TC_001'],
            ground_truth_answer="TC_001 tests handover"
        )
        
        for result in results:
            assert result.ragas_metrics is not None
            assert 'faithfulness' in result.ragas_metrics
```

**Test LangSmith Integration:**

Create `tests/evaluation/test_langsmith_integration.py`:

```python
import pytest
from src.agrag.evaluation.langsmith_evaluator import LangSmithEvaluator

@pytest.mark.skipif(
    not os.getenv('LANGSMITH_API_KEY'),
    reason="LangSmith API key not configured"
)
class TestLangSmithIntegration:
    
    @pytest.fixture
    def evaluator(self):
        return LangSmithEvaluator(
            project_name="agrag-test",
            use_ragas=True,
            num_trials=3
        )
    
    def test_dataset_upload(self, evaluator):
        """Test dataset upload to LangSmith."""
        queries = [
            {
                'query': 'Test query 1',
                'relevant_ids': ['TC_001'],
                'reference_answer': 'Answer 1'
            },
            {
                'query': 'Test query 2',
                'relevant_ids': ['TC_002', 'TC_003'],
                'reference_answer': 'Answer 2'
            }
        ]
        
        dataset_name = evaluator.upload_eval_dataset(
            dataset_name="test-dataset",
            queries=queries,
            version="test-v1"
        )
        
        assert dataset_name == "test-dataset-test-v1"
        
        # Verify dataset exists in LangSmith
        client = evaluator.client
        dataset = client.read_dataset(dataset_name=dataset_name)
        assert dataset is not None
    
    async def test_experiment_execution(self, evaluator):
        """Test running an experiment in LangSmith."""
        # ... test experiment run ...
        pass
```

---

## Implementation Timeline

### Week 1: Foundation (Days 1-5)
- **Day 1-2**: Add dependencies, create Ragas metrics module
- **Day 2-3**: Implement context tracking in agent state
- **Day 3-5**: Add multi-trial evaluation to AgenticEvaluator

### Week 2: Integration (Days 6-10)
- **Day 5-7**: Create LangSmith evaluator module
- **Day 7-8**: Update CLI commands and configuration
- **Day 8-10**: Add reference answers and integration tests

### Week 3: Testing & Refinement (Days 11-15)
- **Day 11-12**: Run full evaluation suite, fix bugs
- **Day 13-14**: Performance optimization, documentation
- **Day 15**: Final validation and deployment

---

## Testing Strategy

### Unit Tests
- ✅ Ragas metric wrappers
- ✅ Context extraction and formatting
- ✅ Trial aggregation logic
- ✅ Dataset conversion utilities

### Integration Tests
- ✅ Ragas evaluation end-to-end
- ✅ Multi-trial evaluation
- ✅ LangSmith dataset upload
- ✅ LangSmith experiment execution
- ✅ Context tracking in agent

### End-to-End Tests
- ✅ Full evaluation pipeline with Ragas
- ✅ LangSmith experiment with all features
- ✅ Backward compatibility with existing commands
- ✅ Performance benchmarks (100+ queries)

---

## Configuration Examples

### Local Evaluation with Ragas

```bash
# Enable Ragas in .env
RAGAS_ENABLED=true
RAGAS_MODEL=gemini-3-flash-preview

# Run evaluation
poetry run agrag evaluate \
  --dataset data/eval.json \
  --use-ragas \
  --num-trials 5 \
  --summary-only
```

### LangSmith Experiment

```bash
# Enable LangSmith in .env
LANGSMITH_EXPERIMENTS_ENABLED=true
LANGSMITH_API_KEY=lsv2_...

# Upload dataset and run experiment
poetry run agrag experiment \
  --dataset data/eval.json \
  --upload \
    --name "gemini-3-flash-preview" \
  --num-trials 5 \
  --use-ragas
```

### Compare Experiments

```bash
# Run new experiment and compare
poetry run agrag experiment \
  --dataset agrag-eval-v1 \
  --name "improved-prompt" \
  --compare-to "baseline" \
  --use-ragas
```

---

## Migration Path

### Phase 1: Opt-In (Week 1-2)
- Ragas and LangSmith features disabled by default
- Enable via flags: `--use-ragas`, `--use-langsmith`
- All existing commands work unchanged

### Phase 2: Validation (Week 3)
- Run parallel evaluations (old vs new)
- Validate metric consistency
- Gather performance data

### Phase 3: Documentation (Week 4)
- Update README with new features
- Add evaluation guide
- Create example notebooks

### Phase 4: Default Enable (Week 5+)
- Consider enabling Ragas by default
- Keep LangSmith opt-in for privacy
- Deprecate old evaluation format (if needed)

---

## Performance Considerations

### Expected Performance

**Single Query:**
- Agent execution: ~3-5s
- Ragas evaluation: ~2-3s
- Total per query: ~5-8s

**100 Query Dataset:**
- Sequential: ~8-13 minutes
- Parallel (10 workers): ~1-2 minutes

### Optimization Strategies

1. **Batch Ragas calls** - Group multiple queries
2. **Cache embeddings** - Reuse for similar contexts
3. **Parallel execution** - Use async/await
4. **LangSmith concurrency** - Max 10 concurrent
5. **Context pruning** - Limit to top-k contexts

---

## Documentation Deliverables

### User Documentation
1. **Evaluation Guide** - How to run evaluations with new features
2. **Metrics Reference** - Definition of all metrics
3. **CLI Reference** - Updated command documentation
4. **LangSmith Setup** - How to configure experiments

### Developer Documentation
1. **Architecture Overview** - System design
2. **API Reference** - New classes and methods
3. **Extension Guide** - How to add new metrics
4. **Troubleshooting** - Common issues and solutions

---

## Risk Mitigation

### Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Ragas API failures | High | Retry logic with exponential backoff |
| LangSmith quota limits | Medium | Rate limiting, graceful degradation |
| Performance degradation | Medium | Async execution, caching, profiling |
| Breaking changes | High | Backward compatibility, feature flags |
| Incorrect metric calculations | High | Extensive testing, validation against known results |

---

## Success Metrics

### Technical Metrics
- ✅ 100% backward compatibility with existing commands
- ✅ <10s per query evaluation (including Ragas)
- ✅ >95% test coverage for new code
- ✅ Zero regression in existing metrics

### Research Metrics
- ✅ Ragas metrics provide new insights beyond entity matching
- ✅ Multi-trial evaluation reveals stability issues
- ✅ LangSmith visualization aids debugging
- ✅ Clear correlation between Ragas metrics and retrieval quality

### User Experience
- ✅ Clear, actionable documentation
- ✅ Intuitive CLI commands
- ✅ Meaningful error messages
- ✅ Fast iteration cycle (minutes, not hours)

---

## Future Enhancements

### Post-MVP Features
1. **Custom Ragas metrics** - Domain-specific evaluation
2. **A/B testing framework** - Compare model versions
3. **Continuous evaluation** - Run on every commit
4. **Regression alerts** - Notify on metric drops
5. **Interactive dashboards** - Real-time monitoring
6. **Human calibration** - RLHF for evaluation quality

### Research Directions
1. **Adversarial evaluation** - Stress test agent
2. **Failure analysis** - Categorize error types
3. **Cost-quality tradeoffs** - Optimize for budget
4. **Prompt engineering** - Use evals to improve prompts
5. **Tool selection analysis** - Understand tool usage patterns

---

## Appendix

### A. Metric Definitions

**Traditional IR Metrics:**
- **Precision@k**: Fraction of retrieved items in top-k that are relevant
- **Recall@k**: Fraction of relevant items retrieved in top-k
- **F1@k**: Harmonic mean of Precision@k and Recall@k
- **MAP**: Mean Average Precision across all queries
- **MRR**: Mean Reciprocal Rank of first relevant item

**Ragas Metrics:**
- **Faithfulness**: Are claims in answer supported by contexts?
- **Answer Relevancy**: Is answer relevant to the query?
- **Context Recall**: Did we retrieve context needed for ground truth?
- **Context Precision**: Are retrieved contexts ranked by relevance?
- **Answer Correctness**: Semantic + factual correctness vs ground truth

**Multi-Trial Metrics:**
- **Pass@k**: At least one trial passed (success rate)
- **Pass^k**: All trials passed (consistency)
- **Stability Score**: 1 - normalized standard deviation
- **Variance**: Spread of metric values across trials

### B. Example Outputs

**Summary-Only Output:**

```
┌─────────────────────────┬────────────┬───────────┬───────────┐
│ Metric                  │ Mean       │ Std Dev   │ Pass@5    │
├─────────────────────────┼────────────┼───────────┼───────────┤
│ MAP                     │ 0.847      │ 0.023     │ 0.940     │
│ MRR                     │ 0.912      │ 0.018     │ 0.965     │
│ Precision@5             │ 0.823      │ 0.034     │ 0.920     │
│ Recall@5                │ 0.891      │ 0.027     │ 0.935     │
│ F1@5                    │ 0.856      │ 0.029     │ 0.928     │
├─────────────────────────┼────────────┼───────────┼───────────┤
│ Faithfulness            │ 0.934      │ 0.012     │ 0.980     │
│ Answer Relevancy        │ 0.878      │ 0.021     │ 0.945     │
│ Context Recall          │ 0.845      │ 0.031     │ 0.915     │
│ Context Precision       │ 0.889      │ 0.024     │ 0.950     │
│ Answer Correctness      │ 0.812      │ 0.038     │ 0.895     │
└─────────────────────────┴────────────┴───────────┴───────────┘

Dataset: data/eval.json (100 queries)
Trials per query: 5
Success Rate: 98.5% (493/500 trials)
Avg Execution Time: 6.2s per query
Total Time: 10m 20s
Stability Score: 0.89

LangSmith Experiment: https://smith.langchain.com/o/.../experiments/...
```

**Full JSON Output:**

```json
{
  "summary": {
    "map_score": 0.847,
    "mrr_score": 0.912,
    "avg_precision_at_k": {"5": 0.823, "10": 0.789},
    "avg_recall_at_k": {"5": 0.891, "10": 0.923},
    "avg_f1_at_k": {"5": 0.856, "10": 0.851},
    "total_queries": 100,
    "successful_queries": 98,
    "trial_statistics": {
      "num_trials": 5,
      "success_rate": 0.985,
      "pass_at_1": 0.980,
      "pass_at_k": 0.940,
      "mean_metrics": {...},
      "std_metrics": {...},
      "stability_score": 0.89
    },
    "avg_ragas_metrics": {
      "faithfulness": 0.934,
      "answer_relevancy": 0.878,
      "context_recall": 0.845,
      "context_precision": 0.889,
      "answer_correctness": 0.812
    }
  },
  "results": [
    {
      "query_id": "Q_001",
      "query": "What tests cover handover?",
      "trials": [
        {
          "trial_number": 1,
          "retrieved_ids": ["TC_001", "TC_003"],
          "relevant_ids": ["TC_001", "TC_003"],
          "metrics": {
            "precision_at_5": 1.0,
            "recall_at_5": 1.0,
            "map": 1.0
          },
          "ragas_metrics": {
            "faithfulness": 0.95,
            "answer_relevancy": 0.92
          },
          "contexts_used": ["TestCase TC_001...", "TestCase TC_003..."],
          "execution_time_ms": 5200
        }
        // ... 4 more trials ...
      ]
    }
    // ... 99 more queries ...
  ]
}
```

### C. Dependencies Checklist

```toml
# Required new dependencies
ragas = "^0.2.0"
datasets = "^2.19.0"

# Ensure up-to-date
langsmith = "^0.6.2"
langchain = "^1.2.3"
langgraph = "^1.0.5"

# Already installed
neo4j = "^6.0.0"
psycopg = {extras = ["binary"], version = "^3.3.2"}
pydantic = "^2.8.0"
```

### D. Key Decision Log

| Decision | Rationale | Date |
|----------|-----------|------|
| Use Gemini for Ragas | Consistency with agent model | Jan 16 |
| Store contexts only during eval | Reduce memory overhead | Jan 16 |
| Deduplicate contexts | Reduce Ragas input noise | Jan 16 |
| Max 3 retries for Ragas | Balance reliability vs speed | Jan 16 |
| Summary-only mode | Large multi-trial outputs | Jan 16 |
| Backward compatibility required | Don't break existing workflows | Jan 16 |
| LangSmith opt-in | Privacy considerations | Jan 16 |
| Dataset versioning scheme | Track changes over time | Jan 16 |

---

**END OF PLAN**

*This plan is a living document and should be updated as implementation progresses and new insights emerge.*
