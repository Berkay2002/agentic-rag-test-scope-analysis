"""Embedding model wrapper for supported providers."""

from typing import List, Optional
import hashlib
import logging
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from agrag.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using configured providers."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize embedding service.

        Args:
            model: Embedding model name (defaults to settings)
            api_key: Google API key (defaults to settings)
        """
        self.model_name = model or settings.google_embedding_model
        self.api_key = api_key or settings.google_api_key

        self.use_mock = _should_use_mock_embeddings()
        self.embeddings: Optional[Embeddings] = None
        mode = "mock" if self.use_mock else "unconfigured"

        if not self.use_mock:
            provider = (settings.embeddings_provider or "").lower()
            if provider == "google":
                if not self.api_key:
                    raise ValueError("GOOGLE_API_KEY must be provided for embeddings")

                api_key_value = SecretStr(self.api_key) if self.api_key else None
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model=self.model_name,
                    api_key=api_key_value,
                    output_dimensionality=settings.embedding_dimensions,
                )
                mode = "google"
            elif provider == "openai":
                model_name = model or settings.openai_embedding_model
                key = api_key or settings.openai_embedding_api_key or settings.openai_api_key
                base_url = settings.openai_embedding_base_url or settings.openai_base_url
                org = settings.openai_embedding_organization or settings.openai_organization

                if not key:
                    raise ValueError("OPENAI_API_KEY must be provided for embeddings")

                api_key_value = SecretStr(key) if key else None
                self.embeddings = OpenAIEmbeddings(
                    model=model_name,
                    api_key=api_key_value,
                    base_url=base_url,
                    organization=org,
                )
                self.model_name = model_name
                mode = "openai"
            else:
                raise ValueError(f"Unsupported embeddings provider: {provider}")

        logger.info(f"Embedding service initialized with model: {self.model_name} (mode={mode})")

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.

        Args:
            text: Query text

        Returns:
            Embedding vector (768-dim)
        """
        try:
            if self.use_mock:
                embedding = _mock_embedding(text, settings.embedding_dimensions)
            else:
                if not self.embeddings:
                    raise RuntimeError("Embedding provider is not initialized")
                embedding = self.embeddings.embed_query(text)
            return _resize_embedding(embedding, settings.embedding_dimensions)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            texts: List of document texts

        Returns:
            List of embedding vectors
        """
        try:
            if self.use_mock:
                embeddings = [_mock_embedding(text, settings.embedding_dimensions) for text in texts]
            else:
                if not self.embeddings:
                    raise RuntimeError("Embedding provider is not initialized")
                embeddings = self.embeddings.embed_documents(texts)
            return [_resize_embedding(vec, settings.embedding_dimensions) for vec in embeddings]
        except Exception as e:
            logger.error(f"Failed to generate document embeddings: {e}")
            raise

    def embed_documents_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
    ) -> List[List[float]]:
        """
        Generate embeddings for documents in batches.

        Args:
            texts: List of document texts
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.info(f"Processing embedding batch {i // batch_size + 1} ({len(batch)} texts)")

            try:
                batch_embeddings = self.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Failed to process batch {i // batch_size + 1}: {e}")
                raise

        return all_embeddings


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    Get or create global embedding service instance.

    Returns:
        EmbeddingService instance
    """
    global _embedding_service

    if _embedding_service is None:
        _embedding_service = EmbeddingService()

    return _embedding_service


def _should_use_mock_embeddings() -> bool:
    """Determine whether to use mock embeddings (avoids external API calls)."""
    mode = os.getenv("AGRAG_EMBEDDINGS_MODE", "").lower()
    if mode in {"mock", "offline", "test"}:
        return True

    if os.getenv("PYTEST_CURRENT_TEST") and os.getenv("AGRAG_ALLOW_EXTERNAL_EMBEDDINGS", "").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return True

    return False


def _resize_embedding(embedding: List[float], target_dim: int) -> List[float]:
    """Resize embeddings to match configured dimensions.

    Truncates if larger, pads with zeros if smaller.
    """
    current_dim = len(embedding)
    if current_dim == target_dim:
        return embedding
    if current_dim > target_dim:
        logger.warning(
            "Embedding dimension mismatch: got %s, expected %s. Truncating to match configuration.",
            current_dim,
            target_dim,
        )
        return embedding[:target_dim]
    logger.warning(
        "Embedding dimension mismatch: got %s, expected %s. Padding to match configuration.",
        current_dim,
        target_dim,
    )
    return embedding + [0.0] * (target_dim - current_dim)


def _mock_embedding(text: str, dimensions: int) -> List[float]:
    """Generate a deterministic mock embedding without external API calls."""
    seed = hashlib.blake2b(text.encode("utf-8"), digest_size=64).digest()
    values: List[float] = []
    counter = 0
    while len(values) < dimensions:
        digest = hashlib.blake2b(seed + counter.to_bytes(4, "big"), digest_size=64).digest()
        for idx in range(0, len(digest), 4):
            if len(values) >= dimensions:
                break
            chunk = digest[idx : idx + 4]
            int_val = int.from_bytes(chunk, "big", signed=False)
            values.append((int_val / 2**32) * 2 - 1)
        counter += 1
    return values
