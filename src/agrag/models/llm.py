"""LLM wrapper for supported chat providers."""

import logging
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from agrag.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _thinking_level_supported() -> bool:
    """Check whether installed LangChain integration supports thinking_level."""
    try:
        return "thinking_level" in ChatGoogleGenerativeAI.model_fields
    except Exception:
        return False


def get_llm(
    model: str = None,
    temperature: float = None,
    api_key: str = None,
    base_url: str = None,
    organization: str = None,
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
    provider = (settings.llm_provider or "").lower()
    temp = temperature if temperature is not None else settings.agent_temperature

    if provider == "google":
        model_name = model or settings.google_model
        key = api_key or settings.google_api_key
        thinking_level = settings.google_thinking_level
        thinking_budget = settings.google_thinking_budget

        if not key:
            raise ValueError("GOOGLE_API_KEY must be provided")

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
        logger.info(f"LLM initialized: google:{model_name}{log_suffix}")
        return llm

    if provider == "openai":
        model_name = model or settings.openai_model
        key = api_key or settings.openai_api_key
        target_base_url = base_url or settings.openai_base_url
        org = organization or settings.openai_organization

        if not key:
            raise ValueError("OPENAI_API_KEY must be provided")

        llm_kwargs = {
            "model": model_name,
            "temperature": temp,
            "api_key": key,
        }
        if target_base_url:
            llm_kwargs["base_url"] = target_base_url
        if org:
            llm_kwargs["organization"] = org

        llm = ChatOpenAI(**llm_kwargs)
        log_suffix = f" (temperature={temp}"
        if target_base_url:
            log_suffix += f", base_url={target_base_url}"
        log_suffix += ")"
        logger.info(f"LLM initialized: openai:{model_name}{log_suffix}")
        return llm

    raise ValueError(f"Unsupported LLM provider: {provider}")
