"""LLM wrapper for Google Generative AI."""

import logging
from functools import lru_cache

from google.ai.generativelanguage_v1beta.types import GenerationConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

from agrag.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _thinking_level_supported() -> bool:
    """Check whether installed SDK supports the thinking_level parameter."""
    try:
        GenerationConfig().thinking_config.thinking_level = "low"  # type: ignore[attr-defined]
        return True
    except (AttributeError, ValueError):
        return False


def get_llm(
    model: str = None,
    temperature: float = None,
    api_key: str = None,
) -> BaseChatModel:
    """
    Get configured LLM instance.

    Args:
        model: Model name (defaults to settings)
        temperature: Sampling temperature (defaults to settings)
        api_key: Google API key (defaults to settings)

    Returns:
        Configured chat model
    """
    model_name = model or settings.google_model
    temp = temperature if temperature is not None else settings.agent_temperature
    key = api_key or settings.google_api_key
    thinking_level = settings.google_thinking_level
    thinking_budget = settings.google_thinking_budget

    if not key:
        raise ValueError("Google API key must be provided")

    llm_kwargs = {
        "model": model_name,
        "temperature": temp,
        "google_api_key": key,
    }

    if thinking_level:
        if not _thinking_level_supported():
            logger.warning(
                "GOOGLE_THINKING_LEVEL=%s requested but SDK does not support thinking_level; "
                "falling back to thinking budget/default.",
                thinking_level,
            )
            thinking_level = None
        elif thinking_budget is not None:
            logger.info(
                "Both GOOGLE_THINKING_LEVEL and GOOGLE_THINKING_BUDGET are set; "
                "thinking level takes precedence."
            )
            thinking_budget = None

    if thinking_level:
        llm_kwargs["thinking_level"] = thinking_level
    elif thinking_budget is not None:
        llm_kwargs["thinking_budget"] = thinking_budget

    llm = ChatGoogleGenerativeAI(**llm_kwargs)

    log_suffix = f" (temperature={temp}"
    if thinking_level:
        log_suffix += f", thinking_level={thinking_level}"
    elif thinking_budget is not None:
        log_suffix += f", thinking_budget={thinking_budget}"
    log_suffix += ")"
    logger.info(f"LLM initialized: {model_name}{log_suffix}")

    return llm
