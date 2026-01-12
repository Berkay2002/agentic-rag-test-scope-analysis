import random

from agrag.data.generators import synthetic as synthetic_module
from agrag.data.generators.synthetic import TelecomDataGenerator


class _StubEmbeddingService:
    def embed_query(self, text: str):
        return [0.0] * 768


def test_evaluation_queries_include_v1_workloads(monkeypatch) -> None:
    monkeypatch.setattr(
        synthetic_module,
        "get_embedding_service",
        lambda: _StubEmbeddingService(),
    )
    random.seed(0)

    gen = TelecomDataGenerator()
    dataset = gen.generate_full_dataset(requirement_count=5, testcase_count=10)

    eval_data = gen.generate_evaluation_dataset(
        test_cases=[e for e in dataset["entities"] if e["id"].startswith("TC_")],
        requirements=[e for e in dataset["entities"] if e["id"].startswith("REQ_")],
        functions=[e for e in dataset["entities"] if e["id"].startswith("FUNC_")],
        relationships=dataset["relationships"],
    )

    query_types = {q["query_type"] for q in eval_data["queries"]}
    assert "change_request_tests" in query_types
    assert "impact_analysis" in query_types
    assert "coverage_by_component" in query_types
    assert "failure_triage" in query_types
