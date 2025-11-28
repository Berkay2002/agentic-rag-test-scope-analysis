"""Utilities for initializing LangGraph checkpointers with graceful fallbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver

try:  # Optional dependency for persistent checkpointing
    from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    PostgresSaver = None  # type: ignore

try:
    import psycopg  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    psycopg = None  # type: ignore

from agrag.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CheckpointerInitResult:
    """Container describing checkpointer initialization outcome."""

    checkpointer: Optional[Any]
    backend: str
    persistent: bool
    error: Optional[Exception] = None


def initialize_checkpointer(
    enable_hitl: bool,
    fallback_to_memory: bool = True,
) -> CheckpointerInitResult:
    """Attempt to initialize a PostgresSaver checkpointer with optional fallback.

    Args:
        enable_hitl: Whether checkpointing is required (HITL/session persistence).
        fallback_to_memory: Whether to fall back to MemorySaver when Postgres is unavailable.

    Returns:
        CheckpointerInitResult describing the initialized saver (if any).
    """
    if not enable_hitl:
        return CheckpointerInitResult(
            checkpointer=None,
            backend="disabled",
            persistent=False,
        )

    last_error: Optional[Exception] = None

    if PostgresSaver and psycopg:
        try:
            conn = psycopg.connect(settings.postgres_connection_string, autocommit=True)
            saver = PostgresSaver(conn)
            saver.setup()
            return CheckpointerInitResult(
                checkpointer=saver,
                backend="postgres",
                persistent=True,
            )
        except Exception as exc:  # pragma: no cover - network configuration varies
            last_error = exc
            logger.warning(
                "Postgres checkpointer unavailable, will use fallback if allowed: %s",
                exc,
            )

    else:  # Missing dependencies, treat as unavailable
        if not PostgresSaver:
            logger.debug("langgraph.checkpoint.postgres.PostgresSaver import failed.")
        if not psycopg:
            logger.debug("psycopg module not available for Postgres checkpointer.")

    if fallback_to_memory:
        logger.info("Falling back to in-memory checkpointer (not persisted).")
        return CheckpointerInitResult(
            checkpointer=MemorySaver(),
            backend="memory",
            persistent=False,
            error=last_error,
        )

    return CheckpointerInitResult(
        checkpointer=None,
        backend="disabled",
        persistent=False,
        error=last_error,
    )


def summarize_error(exc: Exception) -> str:
    """Return a single-line summary for an exception."""
    text = str(exc).strip()
    if not text:
        return exc.__class__.__name__
    first_line = text.splitlines()[0]
    return first_line if first_line else exc.__class__.__name__
