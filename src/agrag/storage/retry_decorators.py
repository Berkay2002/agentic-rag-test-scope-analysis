"""Retry decorators for storage operations with exponential backoff."""

from functools import wraps
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)
from neo4j.exceptions import ServiceUnavailable, TransientError
from psycopg import OperationalError, DatabaseError
from typing import Callable, Any
from agrag.config import settings


def resilient_db_operation(func: Callable) -> Callable:
    """Decorator for database operations with automatic retry."""

    @wraps(func)
    @retry(
        stop=stop_after_attempt(settings.retry_max_attempts),
        wait=wait_exponential(
            min=settings.retry_base_delay,
            max=60,  # Max 60 seconds between retries
            multiplier=2
        ),
        retry=retry_if_exception_type((
            ServiceUnavailable,
            TransientError,
            OperationalError,
            DatabaseError,
            ConnectionError,
            TimeoutError
        )),
        before_sleep=before_sleep_log(logging, logging.WARNING),
        after=after_log(logging, logging.INFO),
        reraise=True
    )
    def wrapper(*args, **kwargs) -> Any:
        return func(*args, **kwargs)

    return wrapper


def with_fallback(fallback_func: Callable):
    """Decorator that provides fallback strategy on persistent failure."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.warning(f"Primary function {func.__name__} failed: {e}")
                if fallback_func:
                    logging.info(f"Falling back to {fallback_func.__name__}")
                    return fallback_func(*args, **kwargs)
                raise
        return wrapper
    return decorator
