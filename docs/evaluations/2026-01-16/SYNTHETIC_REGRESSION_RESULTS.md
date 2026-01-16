# Synthetic Regression Evaluation Results (2026-01-16)

This note documents the evaluation artifacts for the synthetic dataset run.

## Artifacts

- [evaluation_results.synthetic-regression.retrieval.json](evaluation_results.synthetic-regression.retrieval.json)
- [evaluation_results.synthetic-regression.agent.json](evaluation_results.synthetic-regression.agent.json)

## Summary

### Agent run (expected to perform well)

- Observed: MAP=0.6667, MRR=0.6667, P@1=0.6667, success rate 100%.
- Tool usage: `hybrid_search` + `graph_traverse` on all queries, plus occasional `vector_search`.
- Status: **Expected and acceptable**; the agent is strongest when it can pivot to graph traversal.

### Retrieval-only baselines

- Best baseline: `graph` (MAP=0.6667, MRR=0.6667).
- `vector` shows partial recovery (MAP/MRR=0.3333), but still trails `graph`.
- Status: **Acceptable**. Static baselines are not the primary KPI for this dataset.

## Notes / Caveats

- This dataset has 3 queries, so small shifts in ranking can move aggregate metrics.

## Conclusion

Agentic GraphRAG performs as expected, with the graph traversal step driving most of the gains.