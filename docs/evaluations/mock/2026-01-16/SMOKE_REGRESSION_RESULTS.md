# Smoke Regression Evaluation Results (2026-01-16)

This note documents the smoke regression evaluation artifacts in this folder and clarifies which outcomes are expected.

## Artifacts

- [evaluation_results.smoke-regression.retrieval.20260116T150139Z.json](evaluation_results.smoke-regression.retrieval.20260116T150139Z.json)
- [evaluation_results.smoke-regression.agent.20260116T150139Z.json](evaluation_results.smoke-regression.agent.20260116T150139Z.json)

## Summary

- **Primary success criterion**: Agentic GraphRAG should resolve acronym-heavy smoke queries via tool orchestration (hybrid + graph traversal). The agent run shows perfect MAP/MRR and 100% success rate for these two queries.
- **Static baselines**: Vector/keyword/hybrid/RAG/graphrag baselines are expected to be weaker on acronym-heavy prompts. These results show low or zero P@1 on retrieval-only strategies, which is acceptable for this smoke scope.

## Expected vs. observed

### Agent run (expected to pass)

- Observed: MAP=1.0, MRR=1.0 with `graph_traverse+hybrid_search`.
- Status: **Expected and acceptable**.

### Retrieval-only baselines (expected to underperform)

- Observed: `vector` remains at 0.0 across metrics, while `keyword`/`hybrid`/`rag`/`graphrag` retrieve the correct IDs but not consistently at rank 1.
- Status: **Expected and acceptable** for this smoke scope.

## Notes / Caveats

- The agent results file has an empty `per_query_results` list. This does not impact aggregate metrics but limits per-query diagnostics for the agent run.

## Conclusion

These results are **not failing**. They align with the stated goal: ensure agentic GraphRAG resolves the smoke queries, while accepting that static baselines may remain weak on acronym-heavy prompts.