"""Ragas metrics integration for RAG evaluation."""

from __future__ import annotations

import asyncio
import inspect
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Callable, Awaitable, TypeVar

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)

from agrag.config import settings
from agrag.models.embeddings import get_embedding_service
from agrag.models.llm import get_llm

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for exponential backoff on transient API failures."""

    def decorator(func: Callable[..., Awaitable[T]] | Callable[..., T]):
        if inspect.iscoroutinefunction(func):

            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                delay = base_delay
                for attempt in range(1, max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as exc:  # pragma: no cover - defensive logging
                        if attempt >= max_retries:
                            raise
                        logger.warning(
                            "Ragas call failed (attempt %s/%s): %s. Retrying in %.1fs",
                            attempt,
                            max_retries,
                            exc,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        delay *= 2

            return async_wrapper

        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            delay = base_delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # pragma: no cover - defensive logging
                    if attempt >= max_retries:
                        raise
                    logger.warning(
                        "Ragas call failed (attempt %s/%s): %s. Retrying in %.1fs",
                        attempt,
                        max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2

            raise RuntimeError("Ragas call failed after retries")

        return sync_wrapper

    return decorator


class RagasEvaluator:
    """Evaluator for RAG-specific metrics using Ragas with Gemini."""

    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        max_retries: int = 3,
        api_key: Optional[str] = None,
    ):
        """Initialize with Gemini model matching agent configuration."""
        self.model_name = model_name or settings.google_model
        self.max_retries = max_retries
        self.llm = get_llm(model=self.model_name, temperature=0.0, api_key=api_key)
        self.embeddings = get_embedding_service().embeddings

    def _build_metrics(self, include_ground_truth: bool) -> List[Any]:
        metrics = [faithfulness, answer_relevancy, context_precision]
        if include_ground_truth:
            metrics.extend([context_recall, answer_correctness])
        return metrics

    def _configure_metrics(self, metrics: List[Any]) -> None:
        for metric in metrics:
            if hasattr(metric, "llm"):
                metric.llm = self.llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = self.embeddings

    async def _call_ragas_api(self, dataset: Dataset, metrics: List[Any]):
        return await asyncio.to_thread(
            evaluate,
            dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.embeddings,
        )

    def _extract_scores(self, result: Any, metrics: List[Any]) -> Dict[str, float]:
        metric_names = [getattr(metric, "name", str(metric)) for metric in metrics]
        scores: Dict[str, float] = {}

        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            if not df.empty:
                row = df.iloc[0].to_dict()
                for name in metric_names:
                    if name in row and row[name] is not None:
                        scores[name] = float(row[name])
            return scores

        if isinstance(result, dict):
            for name in metric_names:
                value = result.get(name)
                if value is not None:
                    scores[name] = float(value)
            return scores

        raw_result = getattr(result, "result", None)
        if raw_result is not None:
            try:
                row = raw_result.to_pandas().iloc[0].to_dict()
                for name in metric_names:
                    if name in row and row[name] is not None:
                        scores[name] = float(row[name])
            except Exception:  # pragma: no cover - defensive fallback
                pass

        return scores

    async def evaluate_with_ragas(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate using Ragas metrics.

        Returns:
            {
                "faithfulness": float,
                "answer_relevancy": float,
                "context_recall": float,
                "context_precision": float,
                "answer_correctness": float  # if ground_truth provided
            }
        """
        if not contexts:
            return {}

        normalized_contexts = [ctx.strip() for ctx in contexts if isinstance(ctx, str) and ctx.strip()]
        if not normalized_contexts:
            return {}

        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [normalized_contexts],
        }
        include_ground_truth = bool(ground_truth)
        if include_ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)
        metrics = self._build_metrics(include_ground_truth)
        self._configure_metrics(metrics)

        call_with_retry = retry_with_backoff(self.max_retries)(self._call_ragas_api)
        result = await call_with_retry(dataset=dataset, metrics=metrics)
        return self._extract_scores(result, metrics)

    def format_contexts_for_ragas(self, retrieved_contexts: List[Dict[str, Any]]) -> List[str]:
        """
        Convert state contexts to Ragas format (list of strings).

        Deduplicates by content hash to reduce noise.
        Handles missing fields gracefully.
        """
        if not retrieved_contexts:
            return []

        seen_hashes = set()
        formatted: List[str] = []

        for context in retrieved_contexts:
            text: Optional[str] = None
            if isinstance(context, str):
                text = context
            elif isinstance(context, dict):
                text = (
                    context.get("chunk_text")
                    or context.get("content")
                    or context.get("text")
                    or context.get("chunk")
                )

            if not text:
                continue

            normalized = text.strip()
            if not normalized:
                continue

            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                continue

            seen_hashes.add(digest)
            formatted.append(normalized)

        return formatted
