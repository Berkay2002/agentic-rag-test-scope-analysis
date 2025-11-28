"""Embedding model wrapper for Google Generative AI."""

from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import logging

from agrag.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using Google Generative AI."""

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
    ):
        """
        Initialize embedding service.

        Args:
            model: Embedding model name (defaults to settings)
            api_key: Google API key (defaults to settings)
        """
        self.model_name = model or settings.google_embedding_model
        self.api_key = api_key or settings.google_api_key

        if not self.api_key:
            raise ValueError("Google API key must be provided")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=self.api_key,
        )

        logger.info(f"Embedding service initialized with model: {self.model_name}")

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.

        Args:
            text: Query text

        Returns:
            Embedding vector (768-dim)
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
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
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
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
_embedding_service: EmbeddingService = None


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
