from uuid import uuid4

import pytest

from agrag.config import settings
from agrag.evaluation.langsmith_evaluator import LangSmithEvaluator


@pytest.mark.skipif(
    not settings.langchain_api_key,
    reason="LangSmith API key not configured",
)
class TestLangSmithIntegration:
    @pytest.fixture
    def evaluator(self):
        return LangSmithEvaluator(
            project_name="agrag-test",
            use_ragas=False,
            num_trials=3,
        )

    def test_dataset_upload(self, evaluator):
        queries = [
            {
                "query": "Test query 1",
                "relevant_ids": ["TC_001"],
                "reference_answer": "Answer 1",
            },
            {
                "query": "Test query 2",
                "relevant_ids": ["TC_002", "TC_003"],
                "reference_answer": "Answer 2",
            },
        ]

        dataset_name = "test-dataset"
        version = f"test-{uuid4().hex[:6]}"

        dataset_name_final = evaluator.upload_eval_dataset(
            dataset_name=dataset_name,
            queries=queries,
            version=version,
        )

        assert dataset_name_final == f"{dataset_name}-{version}"

        dataset = evaluator.client.read_dataset(dataset_name=dataset_name_final)
        assert dataset is not None