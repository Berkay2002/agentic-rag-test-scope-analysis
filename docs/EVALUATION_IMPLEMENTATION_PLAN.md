# Evaluation Framework Implementation Plan

## Overview

This document outlines the implementation plan for generating TGF-compatible synthetic data and a ground truth evaluation dataset with stratified query difficulties for RQ2 (Retrieval Strategy Comparison).

## Goals

1. **Modify `synthetic.py`** to generate test cases matching TGF CSV schema
2. **Create ground truth evaluation dataset** with stratified query difficulties
3. **Include negative/out-of-scope queries** for precision measurement
4. **Add CLI command** for evaluation dataset generation

---

## Part 1: TGF-Compatible Synthetic Data Generation

### 1.1 New Fields to Add

Extend `generate_test_cases()` in `src/agrag/data/generators/synthetic.py`:

| Field | Type | Generation Strategy |
|-------|------|---------------------|
| `test_suite` | String | Map category to suite name |
| `feature_area` | String | Primary feature (Handover, Authentication, etc.) |
| `sub_feature` | String | Sub-component (X2, S1, UserAuth, etc.) |
| `result` | String | Weighted distribution: 60% PASS, 25% FAIL, 10% ERROR, 5% SKIP |
| `execution_time_ms` | Integer | Type-dependent: unit (10-100), integration (100-500), system (500-5000), performance (5000-30000) |
| `timestamp` | String | ISO 8601, random within last 30 days |
| `failure_reason` | String | Generated for FAIL/ERROR results only |
| `code_coverage_pct` | Float | 40-98% for PASS/FAIL, 0% for SKIP |
| `priority` | String | Weighted: 10% critical, 30% high, 40% medium, 20% low |
| `tags` | List[str] | Derived from category, scenario, result |
| `requirement_ids` | List[str] | Inline list (in addition to relationships) |
| `function_names` | List[str] | Inline list (in addition to relationships) |

### 1.2 Result Distribution Decision

**Choice: 60/25/10/5 (PASS/FAIL/ERROR/SKIP)**

Rationale:
- More failures than real-world (~75% PASS) ensures sufficient data for:
  - "Find failed tests in X" queries
  - "Debug test TC_XXX" queries  
  - Error analysis queries
- Still realistic enough for meaningful evaluation
- Creates ~50 failed tests from 200 total (vs ~30 with 75% PASS)

### 1.3 Test Suite Mapping

```python
SUITE_MAPPING = {
    "handover": "Handover Tests",
    "authentication": "Authentication Tests", 
    "signaling": "RRC Signaling Tests",
    "protocol": "Protocol Conformance Tests",
    "integration": "Integration Tests",
    "performance": "Performance Tests",
    "security": "Security Tests",
    "mobility": "Mobility Management Tests",
    "data": "Data Session Tests",
    "regression": "Regression Suite",
}
```

### 1.4 Feature Area Hierarchy

```python
FEATURE_HIERARCHY = {
    "handover": {
        "area": "Handover",
        "sub_features": ["X2", "S1", "Inter-RAT", "Intra-Freq", "Inter-Freq"]
    },
    "authentication": {
        "area": "Authentication", 
        "sub_features": ["UserAuth", "TokenRefresh", "SSO", "MFA", "CertValidation"]
    },
    "signaling": {
        "area": "RRC",
        "sub_features": ["Connection", "Release", "Reconfiguration", "Measurement"]
    },
    "performance": {
        "area": "Performance",
        "sub_features": ["Throughput", "Latency", "Capacity", "Stress"]
    },
    "security": {
        "area": "Security",
        "sub_features": ["InputValidation", "OutputEncoding", "AccessControl", "Encryption"]
    },
    "mobility": {
        "area": "Mobility",
        "sub_features": ["CellReselection", "Paging", "TAU", "Idle"]
    },
    "data": {
        "area": "DataSession",
        "sub_features": ["PDN", "Bearer", "QoS", "APN"]
    },
}
```

### 1.5 Failure Reason Templates

```python
FAILURE_TEMPLATES = {
    "handover": [
        "Handover timeout after {time}ms",
        "Target cell not found: {cell_id}",
        "X2 connection refused by target eNodeB",
        "UE context transfer failed: {error_code}",
    ],
    "authentication": [
        "Token validation failed: expired",
        "Invalid credentials for user {user_id}",
        "MFA challenge timeout",
        "Certificate chain validation error",
    ],
    "signaling": [
        "RRC message parsing error at offset {offset}",
        "Invalid state transition: {from_state} -> {to_state}",
        "Measurement report timeout",
    ],
    "performance": [
        "Throughput below threshold: {actual}Mbps < {expected}Mbps",
        "Latency exceeded: {actual}ms > {threshold}ms",
        "Connection dropped under load",
    ],
    "security": [
        "SQL injection vulnerability detected",
        "XSS payload not sanitized: {payload}",
        "Unauthorized access to {resource}",
    ],
}
```

---

## Part 2: Ground Truth Evaluation Dataset

### 2.1 Query Difficulty Stratification

| Difficulty | Percentage | Description | Example |
|------------|------------|-------------|---------|
| **Simple** | 40% | Single entity lookup or direct attribute filter | "Find test TC_HANDOVER_001" |
| **Moderate** | 35% | Single-hop relationship traversal | "Tests verifying REQ_AUTH_001" |
| **Complex** | 25% | Multi-hop traversal, filters, aggregation | "Failed security tests covering authenticate_user" |

### 2.2 Query Type Taxonomy

#### Simple Queries (40%)
1. **Entity lookup by ID**: "Find test case TC_HANDOVER_001"
2. **Feature area filter**: "List all Handover tests"
3. **Result filter**: "Show all failed tests"
4. **Priority filter**: "Find critical priority tests"
5. **Test type filter**: "List protocol tests"

#### Moderate Queries (35%)
1. **Requirement coverage**: "What tests verify REQ_AUTH_001?"
2. **Function coverage**: "Tests covering initiate_handover function"
3. **Suite membership**: "Tests in Handover Tests suite"
4. **Combined filters**: "Failed tests in Authentication feature"
5. **Tag-based**: "Tests tagged with 'regression'"

#### Complex Queries (25%)
1. **Multi-hop traversal**: "Tests for functions called by HandoverManager class"
2. **Aggregation**: "Feature areas with the most failing tests"
3. **Coverage gaps**: "Requirements with no test coverage"
4. **Cross-entity**: "Functions covered by failed authentication tests"
5. **Temporal+filter**: "Critical tests that failed in the last sprint"

### 2.3 Query Template Diversity

**Decision: Yes, use 3-5 natural language variations per query type**

Each query template has paraphrases to test semantic robustness:

```python
QUERY_PARAPHRASES = {
    "find_tests_for_requirement": [
        "What tests verify {req_id}?",
        "Find test cases that cover requirement {req_id}",
        "Which tests are linked to {req_id}?",
        "Show tests verifying {req_id}",
        "Tests for requirement {req_id}",
    ],
    "find_failed_tests": [
        "What tests failed?",
        "Show me failing test cases",
        "Which tests have failures?",
        "List tests with FAIL result",
        "Find failed tests",
    ],
    "find_tests_for_feature": [
        "Tests for {feature} feature",
        "What tests cover {feature}?",
        "Find test cases related to {feature}",
        "Show {feature} tests",
        "{feature} test coverage",
    ],
}
```

### 2.4 Negative/Out-of-Scope Queries

**Decision: Include 5-10 queries with empty `relevant_ids`**

Purpose: Measure precision on null cases (system should return empty or "no results found")

```python
NEGATIVE_QUERIES = [
    {
        "query": "Tests for quantum encryption feature",
        "relevant_ids": [],
        "difficulty": "negative",
        "query_type": "nonexistent_feature",
        "rationale": "Quantum encryption doesn't exist in telecom domain"
    },
    {
        "query": "Find test case TC_BLOCKCHAIN_001",
        "relevant_ids": [],
        "difficulty": "negative", 
        "query_type": "nonexistent_entity",
        "rationale": "No blockchain-related tests in dataset"
    },
    {
        "query": "Tests verifying REQ_AI_ML_001",
        "relevant_ids": [],
        "difficulty": "negative",
        "query_type": "nonexistent_requirement",
        "rationale": "No AI/ML requirements in telecom ontology"
    },
    {
        "query": "Functions covered by GUI tests",
        "relevant_ids": [],
        "difficulty": "negative",
        "query_type": "nonexistent_test_type",
        "rationale": "No GUI tests in telecom system tests"
    },
    {
        "query": "Tests for IPv7 protocol support",
        "relevant_ids": [],
        "difficulty": "negative",
        "query_type": "fictional_protocol",
        "rationale": "IPv7 doesn't exist"
    },
    {
        "query": "Critical tests that all passed with 100% coverage",
        "relevant_ids": [],
        "difficulty": "negative",
        "query_type": "impossible_condition",
        "rationale": "Synthetic data doesn't generate 100% coverage"
    },
]
```

---

## Part 3: Implementation Details

### 3.1 File Changes

| File | Changes |
|------|---------|
| `src/agrag/data/generators/synthetic.py` | Add TGF fields, `generate_evaluation_dataset()` method |
| `src/agrag/data/generators/__init__.py` | Export new classes/functions |
| `src/agrag/cli/main.py` | Add `generate-eval` CLI command |
| `src/agrag/evaluation/__init__.py` | Export evaluation utilities |

### 3.2 New Method: `generate_evaluation_dataset()`

```python
def generate_evaluation_dataset(
    self,
    test_cases: List[Dict],
    requirements: List[Dict], 
    functions: List[Dict],
    num_simple: int = 40,
    num_moderate: int = 35,
    num_complex: int = 25,
    num_negative: int = 8,
    use_paraphrases: bool = True,
) -> Dict[str, Any]:
    """
    Generate ground truth evaluation dataset with stratified difficulties.
    
    Returns:
        {
            "queries": [...],
            "metadata": {
                "total_queries": int,
                "difficulty_distribution": {...},
                "generated_at": str,
            }
        }
    """
```

### 3.3 Evaluation Dataset Schema

```json
{
  "queries": [
    {
      "id": "Q_001",
      "query": "What tests verify REQ_HANDOVER_001?",
      "relevant_ids": ["TC_HANDOVER_001", "TC_HANDOVER_003", "TC_REGRESSION_001"],
      "difficulty": "moderate",
      "query_type": "requirement_coverage",
      "expected_relationship": "VERIFIES",
      "notes": "Tests linked via VERIFIES relationship"
    }
  ],
  "metadata": {
    "total_queries": 108,
    "difficulty_distribution": {
      "simple": 40,
      "moderate": 35,
      "complex": 25,
      "negative": 8
    },
    "generated_at": "2025-11-29T12:00:00Z",
    "source_dataset": "synthetic_200_tests"
  }
}
```

### 3.4 CLI Command: `generate-eval`

```bash
# Generate evaluation dataset from existing synthetic data
poetry run agrag generate-eval --output data/eval_queries.json

# Generate with custom distribution
poetry run agrag generate-eval \
  --simple 50 \
  --moderate 30 \
  --complex 20 \
  --negative 10 \
  --output data/eval_queries.json

# Generate fresh synthetic data + evaluation dataset
poetry run agrag generate --requirements 50 --testcases 200 --with-eval
```

---

## Part 4: Ontology Decision

**Decision: Keep TGF fields in `metadata` dict, not as formal model fields**

Rationale:
- Flexibility: TGF schema may vary between Ericsson teams
- Backward compatibility: Existing code using `TestCase` model unaffected
- Validation: TGF loader already validates fields during ingestion
- Documentation: Schema documented in `TGF_INTEGRATION.md`

The `TestCase` Pydantic model in `ontology.py` remains unchanged. TGF-specific fields live in `metadata`:

```python
# Current model (unchanged)
class TestCase(BaseModel):
    id: str
    name: str
    description: str
    test_type: TestType
    file_path: Optional[str]
    expected_outcome: Optional[str]
    preconditions: Optional[str]
    steps: Optional[List[str]]
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]  # TGF fields go here
```

---

## Part 5: Implementation Order

### Phase 1: Synthetic Generator Enhancement (Day 1)
1. Add constants: `SUITE_MAPPING`, `FEATURE_HIERARCHY`, `FAILURE_TEMPLATES`
2. Add helper methods: `_generate_timestamp()`, `_generate_failure_reason()`, `_generate_execution_time()`, `_generate_tags()`
3. Update `generate_test_cases()` to populate TGF fields in metadata
4. Update `generate_relationships()` to backfill `requirement_ids`/`function_names` on entities

### Phase 2: Evaluation Dataset Generator (Day 2)
1. Add `QueryDifficulty` enum
2. Add `QUERY_PARAPHRASES` and `NEGATIVE_QUERIES` constants
3. Implement `generate_evaluation_dataset()` method
4. Add query generation helpers for each difficulty level

### Phase 3: CLI Integration (Day 3)
1. Add `generate-eval` command to `main.py`
2. Add `--with-eval` flag to existing `generate` command
3. Test end-to-end: generate → ingest → evaluate

### Phase 4: Validation (Day 4)
1. Generate 200-test dataset with evaluation queries
2. Verify ground truth by manual spot-checking
3. Run evaluation command to confirm metrics calculation
4. Document in `TGF_INTEGRATION.md`

---

## Part 6: Success Criteria

| Criterion | Target |
|-----------|--------|
| Synthetic tests generated | 200+ |
| TGF field coverage | 100% (all 16 fields populated) |
| Evaluation queries generated | 100+ |
| Query difficulty balance | 40/35/25 ± 5% |
| Negative queries included | 5-10 |
| Paraphrase coverage | 3+ per query type |
| Ground truth accuracy | 100% (auto-generated from relationships) |

---

## References

- [AGENTS.md](../AGENTS.md) - Project overview and evaluation framework
- [TGF_INTEGRATION.md](TGF_INTEGRATION.md) - TGF CSV schema documentation
- [synthetic.py](../src/agrag/data/generators/synthetic.py) - Current generator implementation
- [metrics.py](../src/agrag/evaluation/metrics.py) - Evaluation metrics
