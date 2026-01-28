# Test Regression Evaluation Results (2026-01-16)

This note documents the evaluation artifacts for the test dataset run.

## Artifacts

- [evaluation_results.test-regression.retrieval.json](evaluation_results.test-regression.retrieval.json)
- [evaluation_results.test-regression.agent.json](evaluation_results.test-regression.agent.json)

## Summary

### Agent run (expected to pass)

- Observed: MAP=1.0, MRR=1.0, P@1=1.0, success rate 100%.
- Tool usage: `hybrid_search` + `graph_traverse` on both queries.
- Status: **Expected and acceptable**. The agent’s tool orchestration reliably recovers the target test cases.

### Retrieval-only baselines (expected to underperform on acronym-heavy prompts)

- Best baseline: `graph` (MAP=0.50, MRR=0.50).
- `vector` remains at 0.0 across metrics for this dataset.
- Status: **Acceptable** for this scope; static baselines are not the success criterion.

## Notes / Caveats

- This dataset is small (2 queries), so aggregate metrics are sensitive to single-query rank changes.

## Conclusion

The agentic pipeline meets the success criteria. Static baselines are weaker but within expectations.