"""LLM wrapper for Google Generative AI."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
import logging

from agrag.config import settings

logger = logging.getLogger(__name__)


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

    if not key:
        raise ValueError("Google API key must be provided")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temp,
        google_api_key=key,
    )

    logger.info(f"LLM initialized: {model_name} (temperature={temp})")

    return llm
