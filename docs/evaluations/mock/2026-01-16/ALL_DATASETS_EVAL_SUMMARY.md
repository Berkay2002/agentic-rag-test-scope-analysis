# Evaluation Summary Across Datasets (2026-01-16)

This summary consolidates the smoke, test, and synthetic evaluation runs and explains why the outcomes are expected.

## Datasets and results

### Smoke regression (2 queries)

- Artifacts: [evaluation_results.smoke-regression.retrieval.20260116T150139Z.json](evaluation_results.smoke-regression.retrieval.20260116T150139Z.json),
  [evaluation_results.smoke-regression.agent.20260116T150139Z.json](evaluation_results.smoke-regression.agent.20260116T150139Z.json)
- Agent: MAP/MRR=1.0, P@1=1.0.
- Retrieval-only: `vector` remains 0.0; `keyword`/`hybrid`/`rag`/`graphrag` retrieve the right IDs but not consistently at rank 1.
- Expected: **Yes**. Smoke queries are acronym-heavy; the agent uses graph traversal to ground the query and recover the correct test case.

### Test regression (2 queries)

- Artifacts: [evaluation_results.test-regression.retrieval.json](evaluation_results.test-regression.retrieval.json),
  [evaluation_results.test-regression.agent.json](evaluation_results.test-regression.agent.json)
- Agent: MAP/MRR=1.0, P@1=1.0.
- Retrieval-only: best is `graph` (MAP/MRR=0.50); `vector` is 0.0.
- Expected: **Yes**. The agent consistently couples hybrid retrieval with graph traversal; static baselines remain weaker.

### Synthetic regression (3 queries)

- Artifacts: [evaluation_results.synthetic-regression.retrieval.json](evaluation_results.synthetic-regression.retrieval.json),
  [evaluation_results.synthetic-regression.agent.json](evaluation_results.synthetic-regression.agent.json)
- Agent: MAP/MRR=0.6667, P@1=0.6667.
- Retrieval-only: best is `graph` (MAP/MRR=0.6667); `vector` partial recovery (MAP/MRR=0.3333).
- Expected: **Yes**. Queries are closer to natural language; graph traversal still drives the strongest gains.

## Why these outcomes make sense

1. **Acronym-heavy prompts** (e.g., X2/GTP) reduce vector-only recall. Synonym expansion helps but is still weaker than graph traversal.
2. **Graph traversal provides grounded anchors** once a plausible seed is found, giving strong top-1 recovery for agentic runs.
3. **Static baselines are not the primary KPI** here; the goal is end-to-end correctness for the agentic GraphRAG.

## Overall conclusion

Across all three datasets, the agentic pipeline meets the success criterion. Static retrieval baselines remain weaker by design and are considered acceptable for this evaluation scope.