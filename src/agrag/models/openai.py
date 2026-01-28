"""OpenAI-compatible provider integrations for chat and embeddings."""

import logging
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from agrag.config import settings

logger = logging.getLogger(__name__)


def create_openai_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
) -> BaseChatModel:
    """Create an OpenAI-compatible chat model instance."""
    model_name = model or settings.openai_model
    temp = temperature if temperature is not None else settings.agent_temperature
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


def create_openai_embeddings(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
) -> Embeddings:
    """Create an OpenAI-compatible embeddings instance."""
    model_name = model or settings.openai_embedding_model
    key = api_key or settings.openai_embedding_api_key or settings.openai_api_key
    target_base_url = base_url or settings.openai_embedding_base_url or settings.openai_base_url
    org = organization or settings.openai_embedding_organization or settings.openai_organization

    if not key:
        raise ValueError("OPENAI_API_KEY must be provided for embeddings")

    api_key_value = SecretStr(key) if key else None
    embeddings = OpenAIEmbeddings(
        model=model_name,
        api_key=api_key_value,
        base_url=target_base_url,
        organization=org,
    )
    logger.info("Embeddings initialized: openai:%s", model_name)
    return embeddings
