# Agentic GraphRAG for Test‑Scope Analysis (Thesis Plan)

## Introduction and Problem Context

Efficient test‑scope analysis is critical when introducing new features, change requests, or defect fixes. Test engineers must quickly identify legacy tests relevant to the change and decide whether existing coverage is adequate. Traditional keyword search in test management systems is brittle: naming styles vary, descriptions are incomplete, and domain terms are ambiguous.

Retrieval‑Augmented Generation (RAG) improves accuracy by grounding LLM outputs in retrieved evidence. However, vector‑only RAG often retrieves isolated text fragments and struggles with multi‑hop reasoning because it does not explicitly encode relationships between entities. GraphRAG addresses this by retrieving not only similar text, but also connected context via a knowledge graph (neighborhood/path traversals), which improves explainability and multi‑hop reasoning.

This thesis project builds an **agentic GraphRAG** system for test‑scope analysis using:

- **PostgreSQL** for retrieval (semantic search via **pgvector** and lexical search via **pg_search BM25**)
- **Neo4j** for traceability and multi‑hop reasoning (graph traversal)
- A **tool‑calling agent** (ReAct‑style) that selects retrieval tools and produces an explainable answer


## Decided Thesis Plan (v1)

This section is the “single source of truth” for the intended MVP and thesis experiments.

### Core research contributions (selected)

1. **GraphRAG vs RAG**: measure retrieval and answer quality improvements from graph traversal vs retrieval‑only.
2. **Fixed pipeline vs agentic planning**: compare a predetermined retrieval plan to an agent that chooses tools.
3. **Explainability**: provide explicit evidence and graph paths; evaluate usefulness with humans and LLM‑as‑judge.

### Query workloads the synthetic dataset must support

- **Change request → relevant tests**
- **Requirement → coverage summary (grouped by component)**
- **Impact analysis** (e.g., function/module change → impacted tests)
- **Failure triage** (e.g., error codes/symptoms → likely tests/components)

### Node types (v1)

The graph is intentionally minimal but supports multi‑hop reasoning:

- `ChangeRequest`
- `File`
- `Component`
- `Function`
- `TestCase`
- `Requirement`

Optional later (not required for v1 experiments): `Class`, `Module`, `BugReport`, `CI_Run`, `CoverageMetric`.

### Relationship types (v1)

These edges are sufficient to demonstrate GraphRAG value and produce explainable paths:

- `TOUCHES`: `ChangeRequest → File`
- `DEFINED_IN`: `Function → File`
- `PART_OF`: `File → Component`
- `COVERS`: `TestCase → Function`
- `VERIFIES`: `TestCase → Requirement`

Optional later: `CALLS` (`Function → Function`) for deeper impact analysis; `DEPENDS_ON` for component/module dependencies.

### Retrieval tools (current implementation)

The agent uses exactly four retrieval tools, matching the current code:

- `vector_search`: semantic retrieval in PostgreSQL via **pgvector**
- `keyword_search`: lexical retrieval in PostgreSQL via **pg_search BM25 (ParadeDB)**
- `hybrid_search`: **RRF fusion** of pgvector + pg_search BM25
- `graph_traverse`: Neo4j traversal for multi‑hop context

### Keyword search decision

- **Use `pg_search` BM25 only** (no alternative keyword system is treated as authoritative).
- Implementation note: the repository still contains a local `BM25RetrieverManager` (rank_bm25) used for offline ingestion/evaluation workflows; it is not part of the agent’s authoritative retrieval path.

### Chunking decision

- Start **entity‑level indexing** (one record per entity with rich metadata).
- Defer “token chunking overhaul” to later (can be a follow‑up ablation if needed).

### What we explicitly defer

- Real integrations (Jira/Git/CI)
- Learning‑to‑rank / ML reranking
- Large token‑based chunking overhaul


## System Architecture (as implemented + v1 expansion)

The system is designed to be end‑to‑end demoable with synthetic data and comparable baselines.

### PostgreSQL (retrieval store)

**Current schema (as implemented)** uses a single table for retrieval:

- Table: `document_chunks`
  - `chunk_id` (unique)
  - `content` (text)
  - `metadata` (JSONB)
  - `embedding vector(768)`

**Indexes (as implemented):**

- **pgvector HNSW** index on `embedding` for semantic search
- **pg_search BM25** index for keyword search over `content` (true BM25 ranking)
- GIN index on `metadata` + an expression index on `metadata->>'entity_type'`

**Design intent:** PostgreSQL is the authoritative retrieval layer. The `metadata` field stores structured attributes like `entity_type`, `entity_id`, `file_path`, etc., enabling filtering and evaluation.

### Neo4j (traceability graph)

**Current codebase** already supports Neo4j nodes and relationships for a small ontology (Requirement/TestCase/Function/Class/Module) and graph traversal.

**Thesis v1 expands the graph** to include `ChangeRequest`, `File`, and `Component` to enable clear multi‑hop explanations:

- Change‑oriented retrieval explanation:
  - `ChangeRequest → File → Function → TestCase → Requirement`
- Requirement coverage explanation:
  - `Requirement ← VERIFIES — TestCase — COVERS → Function → File → Component`

**Edge provenance and confidence:**

- v1 synthetic ground truth edges are generated deterministically by the dataset generator.
- LLM extraction can be evaluated as a separate “noisy/weak traceability” setting later, but the v1 experiments require a deterministic ground truth graph.


## Ingestion and Indexing (synthetic‑first)

Because real organizational data is not available in this thesis environment, the system is designed to ingest **synthetic telecom‑like datasets**.

**Entity‑level ingestion strategy (v1):**

- Generate entities: `ChangeRequest`, `Requirement`, `TestCase`, `Function`, `File`, `Component`.
- Generate edges using deterministic rules consistent with the generator’s scenario.
- Insert entity text and embeddings into PostgreSQL (`document_chunks`) with rich metadata.
- Insert nodes and edges into Neo4j.

**LLM extraction (planned, not v1 default):**

- Use LLMs to infer edges from unstructured descriptions (e.g., change request text → files/functions) and attach confidence.
- Evaluate robustness by comparing retrieval/explanations using inferred edges vs ground truth edges.


## Query‑Time Agent Loop (ReAct‑style)

The agent uses tool calls to retrieve evidence and then synthesizes an answer.

**Tools available (implemented):** `vector_search`, `keyword_search`, `hybrid_search`, `graph_traverse`.

**Deep agent, but bounded:**

For fair comparisons and cost/latency control, experiments enforce strict budgets:

- Limit number of tool calls per query
- Limit model calls per query
- Early stopping rules (don’t repeatedly try all tools)

**Typical change request query:**

1. Retrieve candidate entities with `hybrid_search` or `keyword_search` (depending on whether the query contains IDs/keywords).
2. Identify a start node ID (e.g., a `ChangeRequest`, `File`, or `Function`).
3. Use `graph_traverse` to expand context via v1 edges.
4. Synthesize answer with:
   - ranked tests
   - evidence snippets
   - graph paths
   - confidence/uncertainty labels


## Explainability Strategy (v1)

Each answer should be explainable and reproducible.

**Always show:**

- Ranked tests with scores (retrieval scores and/or fusion score)
- At least one evidence snippet per key recommendation
- At least one graph path per key recommendation
- Uncertainty labels when relationships are inferred (e.g., from LLM extraction)


## Implementation Plan (v1)

This plan intentionally focuses on v1 scope and the two primary ablations (RAG vs GraphRAG, fixed vs agentic). Items explicitly deferred (real integrations, learning-to-rank, chunking overhaul) are not part of this plan.

1. **Dataset + ground truth (synthetic)**
   - Extend the synthetic generator to emit v1 entities: `ChangeRequest`, `File`, `Component`, `Function`, `TestCase`, `Requirement`.
   - Emit v1 edges (`TOUCHES`, `DEFINED_IN`, `PART_OF`, `COVERS`, `VERIFIES`) and per-query ground truth relevant IDs.

2. **Ingestion to both stores**
   - Ingest entity text + embeddings into PostgreSQL `document_chunks` with metadata (`entity_type`, `entity_id`, `file_path`, `component`, etc.).
   - Ingest nodes + edges into Neo4j for traversal.

3. **Baselines (fixed)**
   - Implement a fixed retrieval-only baseline (RAG): retrieval → answer.
   - Implement a fixed GraphRAG baseline: retrieval → fixed-depth `graph_traverse` → answer.

4. **Agentic GraphRAG (bounded deep agent)**
   - Use the existing tool-calling agent with strict budgets (tool/model call limits + early stopping).
   - Ensure the agent produces the agreed explanation template (ranked tests, snippets, paths, uncertainty labels).

5. **Evaluation + reporting**
   - Run experiments across the two ablations:
     - RAG vs GraphRAG
     - Fixed vs agentic
   - Track retrieval metrics, latency/cost, and LLM-as-judge scores for explanation usefulness.

## Evaluation Plan

### Baselines (aligned with thesis contributions)

- **RAG baseline (retrieval‑only):** use PostgreSQL retrieval results without graph traversal.
- **Fixed GraphRAG baseline:** a deterministic plan (e.g., retrieval → fixed traversal depth → answer).
- **Agentic GraphRAG:** the agent chooses retrieval tools + decides whether to traverse.

### Metrics

- Retrieval metrics: **Precision@k**, **Recall@k**, **MRR**, and optionally **nDCG**
- **Latency/cost**: tool calls per query, model calls per query, wall‑clock time
- Explainability usefulness:
  - human‑readability checks (demo usability)
  - **LLM‑as‑judge** scores (secondary; used consistently across conditions)

### Ground truth

- Primary: **generator truth** (the synthetic generator emits relevant IDs and graph edges).
- Secondary: **LLM‑as‑judge** for explanation quality and perceived usefulness.

### Ablations (decided)

- **RAG vs GraphRAG**: retrieval‑only vs retrieval + traversal
- **Fixed vs agentic**: fixed plan vs tool‑planning agent

(Other ablations are explicitly out of scope for the next milestone.)


## Concrete Artifacts (implementation‑accurate)

### PostgreSQL

- Storage: `document_chunks(chunk_id, content, metadata, embedding)`
- Retrieval:
  - pgvector cosine similarity (`<=>`) for semantic search
  - pg_search BM25 (`@@@`, `paradedb.score`) for keyword search
  - RRF fusion for hybrid search

### Neo4j

- v1 nodes: `ChangeRequest`, `File`, `Component`, `Function`, `TestCase`, `Requirement`
- v1 edges: `TOUCHES`, `DEFINED_IN`, `PART_OF`, `COVERS`, `VERIFIES`

### Agent tools

- `vector_search(query, k, node_type, similarity_threshold)`
- `keyword_search(query, k, entity_type)`
- `hybrid_search(query, k, rrf_k, entity_type)`
- `graph_traverse(start_node_id, start_node_label, relationship_types, depth, direction)`


## References

- What Is GraphRAG? https://neo4j.com/blog/genai/what-is-graphrag/
- RAG vs. GraphRAG: A Systematic Evaluation and Key Insights: https://arxiv.org/html/2502.11371v1
- Neo4j Python driver performance recommendations: https://neo4j.com/docs/python-manual/current/performance/
- Unstructured text → knowledge graph with LLMs: https://neo4j.com/blog/developer/unstructured-text-to-knowledge-graph/
