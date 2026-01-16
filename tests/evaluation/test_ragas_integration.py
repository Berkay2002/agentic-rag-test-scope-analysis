import pytest

from agrag.evaluation.ragas_metrics import RagasEvaluator


class _DummyEmbeddingService:
    embeddings = object()


@pytest.fixture
def ragas_evaluator(monkeypatch):
    monkeypatch.setattr("agrag.evaluation.ragas_metrics.get_llm", lambda **_: object())
    monkeypatch.setattr(
        "agrag.evaluation.ragas_metrics.get_embedding_service",
        lambda: _DummyEmbeddingService(),
    )
    monkeypatch.setattr(
        "agrag.evaluation.ragas_metrics.Dataset.from_dict",
        lambda data: data,
    )
    return RagasEvaluator(model_name="dummy", max_retries=3, api_key="test")


@pytest.mark.asyncio
async def test_ragas_with_contexts(ragas_evaluator, monkeypatch):
    query = "What tests cover handover requirements?"
    answer = "TC_HANDOVER_001 and TC_HANDOVER_003 cover handover scenarios."
    contexts = [
        "TestCase TC_HANDOVER_001 verifies basic handover functionality...",
        "TestCase TC_HANDOVER_003 tests edge case scenarios for handover...",
    ]
    ground_truth = "TC_HANDOVER_001 and TC_HANDOVER_003 test handover."

    async def _mock_call_ragas_api(dataset, metrics):
        return {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
            "answer_correctness": 0.85,
        }

    monkeypatch.setattr(ragas_evaluator, "_call_ragas_api", _mock_call_ragas_api)

    result = await ragas_evaluator.evaluate_with_ragas(
        query=query,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
    )

    assert "faithfulness" in result
    assert "answer_relevancy" in result
    assert "context_recall" in result
    assert "context_precision" in result
    assert "answer_correctness" in result
    assert 0.0 <= result["faithfulness"] <= 1.0


@pytest.mark.asyncio
async def test_ragas_without_ground_truth(ragas_evaluator, monkeypatch):
    async def _mock_call_ragas_api(dataset, metrics):
        return {
            "faithfulness": 0.7,
            "answer_relevancy": 0.6,
            "context_precision": 0.5,
            "context_recall": 0.4,
            "answer_correctness": 0.3,
        }

    monkeypatch.setattr(ragas_evaluator, "_call_ragas_api", _mock_call_ragas_api)

    result = await ragas_evaluator.evaluate_with_ragas(
        query="test query",
        answer="test answer",
        contexts=["context 1"],
        ground_truth=None,
    )

    assert "answer_correctness" not in result
    assert "faithfulness" in result


def test_context_deduplication(ragas_evaluator):
    contexts = [
        {"chunk_text": "Same content", "source": "TC_001"},
        {"chunk_text": "Same content", "source": "TC_002"},
        {"chunk_text": "Different content", "source": "TC_003"},
    ]

    formatted = ragas_evaluator.format_contexts_for_ragas(contexts)

    assert len(formatted) == 2
    assert "Same content" in formatted
    assert "Different content" in formatted


@pytest.mark.asyncio
async def test_retry_on_api_failure(ragas_evaluator, monkeypatch):
    call_count = {"count": 0}

    async def _mock_call_ragas_api(dataset, metrics):
        call_count["count"] += 1
        if call_count["count"] < 3:
            raise RuntimeError("Transient API error")
        return {"faithfulness": 0.9, "answer_relevancy": 0.7, "context_precision": 0.6}

    monkeypatch.setattr(ragas_evaluator, "_call_ragas_api", _mock_call_ragas_api)

    result = await ragas_evaluator.evaluate_with_ragas(
        query="test",
        answer="answer",
        contexts=["context"],
    )

    assert call_count["count"] == 3
    assert result["faithfulness"] == 0.9