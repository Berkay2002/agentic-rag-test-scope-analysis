"""Factory helpers for provider-backed chat models."""

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from agrag.config import settings
from agrag.models.providers.google import create_google_llm
from agrag.models.providers.openai import create_openai_llm


def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
) -> BaseChatModel:
    """
    Get configured LLM instance.

    Args:
        model: Model name (defaults to settings)
        temperature: Sampling temperature (defaults to settings)
        api_key: Provider API key (defaults to settings)
        base_url: OpenAI-compatible base URL override
        organization: OpenAI organization ID override

    Returns:
        Configured chat model
    """
    provider = (settings.llm_provider or "").lower()

    if provider == "google":
        return create_google_llm(model=model, temperature=temperature, api_key=api_key)

    if provider == "openai":
        return create_openai_llm(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
