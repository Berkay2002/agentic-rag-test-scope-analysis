"""LangSmith evaluation utilities for dataset uploads and experiments."""

from __future__ import annotations

import json
import logging
import asyncio
from typing import Any, Callable, Dict, List, Optional

from langsmith import Client

from agrag.config import settings
from agrag.evaluation.utils.entity_extractor import extract_entity_ids
from agrag.evaluation.metrics.metrics import evaluate_retrieval
from agrag.evaluation.evaluators.ragas_metrics import RagasEvaluator
from agrag.evaluation.utils.tool_tracker import ToolTracker

logger = logging.getLogger(__name__)


class LangSmithEvaluator:
    """
    Evaluator that uploads datasets and runs experiments in LangSmith.
    """

    def __init__(
        self,
        project_name: str = "agrag-test-scope-analysis",
        use_ragas: bool = False,
        num_trials: int = 1,
    ):
        """Initialize LangSmith client and configuration."""
        self.client = Client(
            api_key=settings.langchain_api_key,
            api_url=settings.langchain_endpoint,
        )
        self.project_name = project_name
        self.use_ragas = use_ragas
        self.num_trials = max(1, num_trials)
        self.ragas_evaluator = RagasEvaluator(
            model_name=settings.ragas_model,
            max_retries=settings.ragas_max_retries,
        )

    def _dataset_exists(self, dataset_name: str) -> bool:
        if hasattr(self.client, "has_dataset"):
            try:
                return bool(self.client.has_dataset(dataset_name=dataset_name))
            except Exception:
                return False

        try:
            datasets = list(self.client.list_datasets(dataset_name=dataset_name))
        except Exception:
            try:
                datasets = list(self.client.list_datasets())
            except Exception:
                return False

        return any(getattr(ds, "name", None) == dataset_name for ds in datasets)

    def _build_experiment_url(self, experiment_name: str) -> Optional[str]:
        if not experiment_name:
            return None
        endpoint = settings.langchain_endpoint or "https://api.smith.langchain.com"
        ui_base = endpoint.replace("api.smith.langchain.com", "smith.langchain.com").rstrip("/")
        return f"{ui_base}/projects/{experiment_name}"

    def _run_async(self, coroutine: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        if loop.is_running():
            if asyncio.current_task(loop=loop) is not None:
                raise RuntimeError(
                    "Cannot run async evaluation inside a running event loop. "
                    "Use LangSmith async evaluation instead."
                )
            return asyncio.run_coroutine_threadsafe(coroutine, loop).result()
        return loop.run_until_complete(coroutine)

    def upload_eval_dataset(
        self,
        dataset_name: str,
        queries: List[Dict[str, Any]],
        description: Optional[str] = None,
        version: Optional[str] = None,
    ) -> str:
        """
        Upload evaluation queries to LangSmith dataset.

        Returns:
            Dataset name (versioned if specified)
        """
        dataset_name_final = f"{dataset_name}-{version}" if version else dataset_name

        if self._dataset_exists(dataset_name_final):
            return dataset_name_final

        dataset = self.client.create_dataset(
            dataset_name=dataset_name_final,
            description=description,
        )

        examples = []
        for query in queries:
            inputs = {"query": query.get("query", "")}
            outputs = {
                "expected_ids": query.get("relevant_ids", []),
                "reference_answer": query.get("reference_answer"),
            }
            metadata = {
                key: value
                for key, value in {
                    "query_type": query.get("query_type"),
                    "difficulty": query.get("difficulty"),
                }.items()
                if value is not None
            }

            examples.append(
                {
                    "inputs": inputs,
                    "outputs": outputs,
                    "metadata": metadata,
                }
            )

        if examples:
            self.client.create_examples(dataset_id=dataset.id, examples=examples)

        return dataset_name_final

    def run_experiment(
        self,
        dataset_name: str,
        agent_function: Callable,
        experiment_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_concurrency: int = 10,
    ) -> Dict[str, Any]:
        """
        Run LangSmith experiment with multi-trial support.

        Returns:
            {
                "experiment_url": str,
                "summary": {...},
                "results": [...]
            }
        """
        evaluators = self.create_evaluators()
        experiment_prefix = experiment_name or self.project_name
        description = json.dumps(metadata) if metadata else None
        evaluate_kwargs: Dict[str, Any] = {
            "data": dataset_name,
            "evaluators": evaluators,
            "experiment_prefix": experiment_prefix,
            "description": description,
            "max_concurrency": max_concurrency,
        }
        if self.num_trials > 1:
            evaluate_kwargs["num_repetitions"] = self.num_trials

        evaluation = self.client.evaluate(agent_function, **evaluate_kwargs)

        experiment_name_value = getattr(evaluation, "experiment_name", None) or getattr(
            evaluation, "experimentName", None
        )
        experiment_name_value = experiment_name_value or experiment_prefix

        stats = None
        if experiment_name_value:
            try:
                stats = self.client.read_project(
                    project_name=experiment_name_value,
                    include_stats=True,
                )
            except Exception as exc:
                logger.warning("Failed to fetch experiment stats: %s", exc)

        experiment_url = self._build_experiment_url(experiment_name_value)

        return {
            "experiment_url": experiment_url,
            "summary": {
                "num_trials": self.num_trials,
                "experiment_names": [experiment_name_value],
            },
            "results": [
                {
                    "experiment_name": experiment_name_value,
                    "stats": getattr(stats, "dict", lambda: stats)() if stats is not None else None,
                }
            ],
        }

    def create_evaluators(self) -> List[Callable]:
        """
        Create LangSmith-compatible evaluators.

        Returns list of evaluator functions for:
        1. Entity matching (P@k, MAP, MRR)
        2. Ragas metrics (if enabled)
        3. Tool usage validation
        4. Error detection
        """

        def entity_matching_evaluator(run: Any, example: Any) -> Dict[str, Any]:
            retrieved_ids = extract_entity_ids(
                (run.outputs or {}).get("final_answer", ""),
                prioritize_test_cases=True,
            )
            expected_ids = set((example.outputs or {}).get("expected_ids", []) or [])

            metrics_result = evaluate_retrieval(
                retrieved=retrieved_ids,
                relevant=expected_ids,
                k_values=[5],
            )

            precision = metrics_result.get("precision@5", 0.0)
            recall = metrics_result.get("recall@5", 0.0)
            f1 = metrics_result.get("f1@5", 0.0)
            map_score = metrics_result.get("average_precision", 0.0)
            mrr = metrics_result.get("reciprocal_rank", 0.0)

            return {
                "key": "entity_matching",
                "score": f1,
                "comment": (
                    f"P@5={precision:.3f}, R@5={recall:.3f}, "
                    f"F1@5={f1:.3f}, MAP={map_score:.3f}, MRR={mrr:.3f}"
                ),
            }

        def tool_usage_evaluator(run: Any, example: Any) -> Dict[str, Any]:
            tools_used = _extract_tools_from_messages((run.outputs or {}).get("messages"))
            query_type = (example.metadata or {}).get("query_type", "unknown")

            expected_tools = {
                "test_coverage": ["vector_search", "graph_traverse"],
                "impact_analysis": ["graph_traverse"],
                "semantic_search": ["vector_search", "hybrid_search"],
                "exact_match": ["keyword_search"],
            }

            appropriate = any(
                tool in tools_used for tool in expected_tools.get(query_type, [])
            )

            return {
                "key": "tool_appropriateness",
                "score": 1.0 if appropriate else 0.0,
                "comment": f"Used: {', '.join(tools_used) if tools_used else 'none'}",
            }

        def error_detection_evaluator(run: Any, example: Any) -> Dict[str, Any]:
            error = getattr(run, "error", None) or (run.outputs or {}).get("error")
            return {
                "key": "run_error",
                "score": 0.0 if error else 1.0,
                "comment": str(error) if error else "ok",
            }

        evaluators: List[Callable] = [entity_matching_evaluator, tool_usage_evaluator, error_detection_evaluator]

        if self.use_ragas:
            evaluators.append(self._ragas_bridge_evaluator)

        return evaluators

    def _ragas_bridge_evaluator(self, run: Any, example: Any) -> Optional[Dict[str, Any]]:
        if not self.use_ragas:
            return None

        contexts = _extract_contexts_from_run(run)
        formatted_contexts = self.ragas_evaluator.format_contexts_for_ragas(contexts)
        if not formatted_contexts:
            return None

        metrics = self._run_async(
            self.ragas_evaluator.evaluate_with_ragas(
                query=(example.inputs or {}).get("query", ""),
                answer=(run.outputs or {}).get("final_answer", ""),
                contexts=formatted_contexts,
                ground_truth=(example.outputs or {}).get("reference_answer"),
            )
        )

        if not metrics:
            return None

        composite = sum(metrics.values()) / max(1, len(metrics))
        faithfulness = metrics.get("faithfulness")
        relevancy = metrics.get("answer_relevancy")

        return {
            "key": "ragas_composite",
            "score": composite,
            "comment": (
                "Faithfulness={:.3f}, Relevancy={:.3f}".format(
                    faithfulness if faithfulness is not None else 0.0,
                    relevancy if relevancy is not None else 0.0,
                )
            ),
        }


def _extract_contexts_from_run(run: Any) -> List[Dict[str, Any]]:
    outputs = getattr(run, "outputs", None) or {}
    if isinstance(outputs, dict):
        contexts = outputs.get("retrieved_contexts") or outputs.get("contexts") or []
        if isinstance(contexts, list):
            return contexts
    return []


def _extract_tools_from_messages(messages: Any) -> List[str]:
    if not isinstance(messages, list):
        return []

    tracker = ToolTracker()
    stats = tracker.start_query(query_id="langsmith", query="")
    tracker.extract_tool_calls_from_messages(messages, stats)
    return stats.tools_used
