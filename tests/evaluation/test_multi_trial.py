import types

from agrag.evaluation.agentic_evaluator import AgenticEvaluator, AgentEvaluationResult


class _DummyGraph:
    def invoke(self, *args, **kwargs):
        return {"messages": []}


def test_multiple_trials_per_query(monkeypatch):
    evaluator = AgenticEvaluator(graph=_DummyGraph(), num_trials=5)

    def _fake_evaluate_query(
        self,
        query_id,
        query,
        relevant_ids,
        query_type="unknown",
        difficulty="unknown",
        trial_number=1,
        ground_truth_answer=None,
        initial_state=None,
    ):
        return AgentEvaluationResult(
            query_id=query_id,
            query=query,
            query_type=query_type,
            difficulty=difficulty,
            relevant_ids=relevant_ids,
            trial_number=trial_number,
        )

    monkeypatch.setattr(
        evaluator,
        "evaluate_query",
        types.MethodType(_fake_evaluate_query, evaluator),
    )

    results = evaluator.evaluate_query_with_trials(
        query_id="Q_001",
        query="What tests cover handover?",
        relevant_ids={"TC_001", "TC_002"},
    )

    assert len(results) == 5
    assert [result.trial_number for result in results] == [1, 2, 3, 4, 5]


def test_trial_statistics_aggregation():
    evaluator = AgenticEvaluator(graph=_DummyGraph(), num_trials=3)

    results = [
        AgentEvaluationResult(
            query_id="Q_001",
            query="test",
            query_type="unknown",
            difficulty="simple",
            relevant_ids={"TC_001"},
            metrics={"precision@5": 1.0, "recall@5": 0.5},
            trial_number=1,
        ),
        AgentEvaluationResult(
            query_id="Q_001",
            query="test",
            query_type="unknown",
            difficulty="simple",
            relevant_ids={"TC_001"},
            metrics={"precision@5": 0.5, "recall@5": 0.25},
            trial_number=2,
        ),
        AgentEvaluationResult(
            query_id="Q_001",
            query="test",
            query_type="unknown",
            difficulty="simple",
            relevant_ids={"TC_001"},
            metrics={"precision@5": 0.0, "recall@5": 0.0},
            trial_number=3,
            error="timeout",
        ),
    ]

    stats = evaluator.aggregate_trial_statistics(results)

    assert stats["num_trials"] == 3
    assert stats["success_rate"] == 0.6667
    assert stats["pass_at_k"] == 1.0
    assert stats["pass_pow_k"] == 0.0
    assert "mean_metrics" in stats
    assert "std_metrics" in stats
    assert 0.0 <= stats["stability_score"] <= 1.0