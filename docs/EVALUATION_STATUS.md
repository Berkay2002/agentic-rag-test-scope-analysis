# Evaluation Framework Implementation Status

## Overview

This document tracks the implementation progress of the evaluation framework for RQ2 (Retrieval Strategy Comparison).

---

## ✅ Completed

### 1. TGF-Compatible Synthetic Data Generator

**File:** `src/agrag/data/generators/synthetic.py`

Enhanced the synthetic data generator with all 16 TGF-compatible fields:

| Field | Status | Description |
|-------|--------|-------------|
| `test_suite` | ✅ | Mapped from category (e.g., "Handover Tests") |
| `feature_area` | ✅ | Primary feature (Handover, Authentication, etc.) |
| `sub_feature` | ✅ | Sub-component (X2, S1, UserAuth, etc.) |
| `result` | ✅ | Weighted: 60% PASS, 25% FAIL, 10% ERROR, 5% SKIP |
| `execution_time_ms` | ✅ | Type-dependent ranges |
| `timestamp` | ✅ | ISO 8601, random within last 30 days |
| `failure_reason` | ✅ | Category-specific templates for FAIL/ERROR |
| `code_coverage_pct` | ✅ | 40-98% for PASS/FAIL, 0% for SKIP |
| `priority` | ✅ | Weighted: 10% critical, 30% high, 40% medium, 20% low |
| `tags` | ✅ | Dynamic tags based on category, scenario, result |
| `requirement_ids` | ✅ | Backfilled from VERIFIES relationships |
| `function_names` | ✅ | Backfilled from COVERS relationships |

### 2. Ground Truth Evaluation Dataset Generator

**File:** `src/agrag/data/generators/synthetic.py` - `generate_evaluation_dataset()` method

Generates stratified queries with ground truth:

| Difficulty | Target % | Actual | Query Types |
|------------|----------|--------|-------------|
| Simple | 40% | 17 | Entity lookup, feature/result/priority filters |
| Moderate | 35% | 22 | Requirement coverage, function coverage, suite membership |
| Complex | 25% | 8 | Multi-hop traversal, aggregations, coverage gaps |
| Negative | 5-10 | 8 | Out-of-scope queries for precision measurement |

**Features:**
- Query paraphrases (3-5 variations per query type)
- Automatic ground truth from relationships
- Negative queries for precision testing

### 3. CLI Commands

**File:** `src/agrag/cli/main.py`

| Command | Status | Description |
|---------|--------|-------------|
| `agrag generate --with-eval` | ✅ | Generate synthetic data + evaluation dataset |
| `agrag generate --ingest` | ✅ | Reset DBs and ingest immediately |
| `agrag generate-eval` | ✅ | Generate evaluation dataset from existing data |
| `agrag evaluate --dataset` | ✅ | Run retrieval and calculate metrics |
| `agrag evaluate --strategy` | ✅ | Choose strategy: vector, keyword, hybrid, all |
| `agrag evaluate --verbose` | ✅ | Show per-query metrics |

### 4. Metrics Framework

**File:** `src/agrag/evaluation/metrics.py`

All metrics implemented:
- `precision_at_k(retrieved, relevant, k)`
- `recall_at_k(retrieved, relevant, k)`
- `f1_score_at_k(retrieved, relevant, k)`
- `average_precision(retrieved, relevant)`
- `mean_average_precision(results)`
- `reciprocal_rank(retrieved, relevant)`
- `mean_reciprocal_rank(results)`
- `evaluate_retrieval(retrieved, relevant, k_values)`

### 5. Data Ingestion (Fixed)

- ✅ Synthetic data generated (288 entities, 565 relationships)
- ✅ Neo4j populated with entities + relationships + **embeddings**
- ✅ PostgreSQL/Neon populated with embeddings
- ✅ Database reset functionality

**Note:** Fixed ingestion to store embeddings in Neo4j for vector search to work.

### 6. Retrieval Execution in Evaluate Command ✅

**File:** `src/agrag/cli/main.py`

Implemented actual retrieval execution with:
- `--strategy vector`: Neo4j vector search
- `--strategy hybrid`: PostgreSQL vector + BM25 fusion  
- `--strategy keyword`: BM25 keyword search (now with persisted index)
- `--strategy graph`: Graph traversal for structural queries
- `--strategy all`: Compare all strategies side-by-side

**Features:**
- Initializes retrieval tools (VectorSearchTool, KeywordSearchTool, HybridSearchTool, GraphTraverseTool)
- Parses retrieved IDs from tool output
- Calculates per-query and aggregate metrics
- Comparison table for multiple strategies
- `--bm25-index` option to specify BM25 index path

### 7. BM25 Index Population ✅

**File:** `src/agrag/data/ingestion.py`

BM25 index is now populated during data ingestion:
- Added `_entities_to_bm25_documents()` to convert entities to LangChain Documents
- Added `_persist_bm25_index()` to save index to disk
- Index saved to `data/bm25_index.pkl` by default
- Evaluate command auto-loads persisted BM25 index

### 8. Graph Traversal Evaluation ✅

**File:** `src/agrag/cli/main.py`

Graph traversal now integrated into evaluation:
- `_execute_graph_traversal()` extracts entity IDs from queries
- Determines appropriate relationship types (VERIFIES, COVERS, etc.)
- Traverses graph to find related test cases
- `_parse_graph_result_ids()` extracts IDs from traversal output

---

## 📊 Sample Evaluation Output

```bash
$ agrag evaluate --dataset data/synthetic_dataset_eval.json --strategy all

======================================================================
STRATEGY COMPARISON
======================================================================
Strategy     |      MAP |      MRR |      P@1 |      R@1 |      P@3 |      R@3 |      P@5 |      R@5 |     P@10 |     R@10
--------------------------------------------------------------------------------------------------------------------------
vector       |   0.0865 |   0.3265 |   0.2364 |   0.0106 |   0.2182 |   0.0536 |   0.2109 |   0.0765 |   0.2127 |   0.1271
keyword      |   0.0523 |   0.2841 |   0.1818 |   0.0089 |   0.1697 |   0.0412 |   0.1673 |   0.0621 |   0.1655 |   0.1043
hybrid       |   0.0841 |   0.2972 |   0.2000 |   0.0100 |   0.2121 |   0.0505 |   0.2145 |   0.0762 |   0.2091 |   0.1226
graph        |   0.0412 |   0.1923 |   0.1273 |   0.0067 |   0.1152 |   0.0298 |   0.1091 |   0.0423 |   0.0982 |   0.0689

Best for MAP: vector (0.0865)
Best for MRR: vector (0.3265)
```

---

## 🎯 RQ2 Answer Requirements

### ✅ Completed
- [x] Vector search metrics
- [x] Hybrid search metrics (via PostgreSQL pgvector)
- [x] Keyword/BM25 search metrics (with persisted index)
- [x] Graph traversal metrics (for structural queries)
- [x] Strategy comparison table
- [x] Aggregate metrics (MAP, MRR, P@k, R@k)

### ⏳ TODO
- [ ] Per-difficulty breakdown
- [ ] Statistical analysis (mean, std dev)
- [ ] Visualization (bar charts, PR curves)

---

## 🚀 Quick Start Commands

```bash
# 1. Generate data and ingest (creates BM25 index + embeddings in Neo4j)
poetry run agrag generate --requirements 50 --testcases 200 --ingest --with-eval

# 2. Run evaluation with different strategies
poetry run agrag evaluate --dataset data/synthetic_dataset_eval.json --strategy vector
poetry run agrag evaluate --dataset data/synthetic_dataset_eval.json --strategy keyword
poetry run agrag evaluate --dataset data/synthetic_dataset_eval.json --strategy hybrid
poetry run agrag evaluate --dataset data/synthetic_dataset_eval.json --strategy graph
poetry run agrag evaluate --dataset data/synthetic_dataset_eval.json --strategy all

# 3. Verbose mode for per-query details
poetry run agrag evaluate --dataset data/synthetic_dataset_eval.json --strategy vector --verbose

# 4. Custom BM25 index path
poetry run agrag evaluate --dataset data/synthetic_dataset_eval.json --bm25-index path/to/index.pkl

# 5. Results saved to evaluation_results.json
```

---

## Summary

| Component | Status |
|-----------|--------|
| Synthetic data generation | ✅ Complete |
| TGF-compatible fields | ✅ Complete |
| Evaluation dataset generation | ✅ Complete |
| Ground truth queries | ✅ 55 queries generated |
| Metrics framework | ✅ Complete |
| Database ingestion | ✅ Complete (with embeddings) |
| CLI commands | ✅ Complete |
| **Retrieval execution in evaluate** | ✅ Complete |
| **Strategy comparison** | ✅ Complete |
| **BM25 index population** | ✅ Complete |
| **Graph traversal evaluation** | ✅ Complete |
| Results visualization | ⏳ TODO |

**Current State:** All four retrieval strategies (vector, keyword, hybrid, graph) can now be evaluated. BM25 index is persisted during ingestion and auto-loaded during evaluation.
