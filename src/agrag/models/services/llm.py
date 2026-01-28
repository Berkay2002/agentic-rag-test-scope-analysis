"""LLM wrapper for supported chat providers (backwards-compatible import)."""

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from agrag.models.core.factory import get_llm as _get_llm


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
    return _get_llm(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        organization=organization,
    )
