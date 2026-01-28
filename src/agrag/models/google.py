"""Google provider integrations for chat and embeddings."""

import logging
from functools import lru_cache
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

from agrag.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _thinking_level_supported() -> bool:
    """Check whether installed LangChain integration supports thinking_level."""
    try:
        return "thinking_level" in ChatGoogleGenerativeAI.model_fields
    except Exception:
        return False


def create_google_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
) -> BaseChatModel:
    """Create a Google chat model instance."""
    model_name = model or settings.google_model
    temp = temperature if temperature is not None else settings.agent_temperature
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


def create_google_embeddings(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Embeddings:
    """Create a Google embeddings instance."""
    model_name = model or settings.google_embedding_model
    key = api_key or settings.google_api_key

    if not key:
        raise ValueError("GOOGLE_API_KEY must be provided for embeddings")

    api_key_value = SecretStr(key) if key else None
    embeddings = GoogleGenerativeAIEmbeddings(
        model=model_name,
        api_key=api_key_value,
        output_dimensionality=settings.embedding_dimensions,
    )
    logger.info("Embeddings initialized: google:%s", model_name)
    return embeddings
