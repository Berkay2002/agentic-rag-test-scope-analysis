"""Semantic text splitter using embeddings-based chunking."""

from typing import List, Dict, Any, Optional
import logging
import numpy as np
from scipy.spatial.distance import cosine

from agrag.data.loaders.base import BaseTextSplitter, Document

logger = logging.getLogger(__name__)


class SemanticSplitter(BaseTextSplitter):
    """
    Semantic text splitter using embeddings to detect topic boundaries.

    Analyzes cosine similarity between adjacent text segments and splits
    where similarity drops below a threshold, indicating a topic shift.
    """

    def __init__(
        self,
        embedding_service=None,
        similarity_threshold: float = 0.75,
        chunk_size: int = 512,
        min_chunk_size: int = 100,
        sentence_split: bool = True,
        **kwargs,
    ):
        """
        Initialize the semantic splitter.

        Args:
            embedding_service: Service for generating embeddings
            similarity_threshold: Minimum similarity to keep segments together
            chunk_size: Maximum chunk size (hard limit)
            min_chunk_size: Minimum chunk size to avoid too small chunks
            sentence_split: Split on sentence boundaries first
            **kwargs: Additional configuration
        """
        super().__init__(chunk_size=chunk_size, **kwargs)

        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.sentence_split = sentence_split

        # Lazy load embedding service to avoid circular imports
        self._embedding_service = embedding_service

    @property
    def embedding_service(self):
        """Lazy load embedding service."""
        if self._embedding_service is None:
            from agrag.models.embeddings import get_embedding_service

            self._embedding_service = get_embedding_service()
        return self._embedding_service

    def split_text(self, text: str) -> List[str]:
        """
        Split text based on semantic boundaries.

        Args:
            text: Input text to split

        Returns:
            List of text chunks
        """
        chunks = self.split_semantic(text)
        return [chunk["content"] for chunk in chunks]

    def split_semantic(
        self,
        text: str,
        file_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Split text into semantic chunks with metadata.

        Args:
            text: Input text
            file_path: Optional file path for metadata

        Returns:
            List of chunk dictionaries
        """
        # First, split into sentences or paragraphs
        segments = self._split_into_segments(text)

        if len(segments) <= 1:
            return [
                {
                    "content": text,
                    "metadata": {
                        "type": "text",
                        "file_path": file_path,
                        "semantic_split": False,
                    },
                }
            ]

        # Generate embeddings for each segment
        try:
            embeddings = self.embedding_service.embed_batch(segments)
        except Exception as e:
            logger.warning(f"Failed to generate embeddings: {e}. Falling back to simple split.")
            return self._fallback_split(text, file_path)

        # Calculate similarity between adjacent segments
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = 1 - cosine(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Find split points where similarity drops below threshold
        split_indices = [0]
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                split_indices.append(i + 1)
        split_indices.append(len(segments))

        # Create chunks from split points
        chunks = []
        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i + 1]

            chunk_segments = segments[start_idx:end_idx]
            chunk_content = " ".join(chunk_segments).strip()

            # Enforce minimum chunk size
            if len(chunk_content) < self.min_chunk_size and chunks:
                # Merge with previous chunk
                chunks[-1]["content"] += " " + chunk_content
                chunks[-1]["metadata"]["end_segment"] = end_idx
            else:
                # Calculate average similarity for this chunk
                chunk_sims = (
                    similarities[start_idx : end_idx - 1] if start_idx < end_idx - 1 else [1.0]
                )
                avg_similarity = np.mean(chunk_sims) if chunk_sims else 1.0

                chunks.append(
                    {
                        "content": chunk_content,
                        "metadata": {
                            "type": "text",
                            "file_path": file_path,
                            "semantic_split": True,
                            "start_segment": start_idx,
                            "end_segment": end_idx,
                            "avg_similarity": float(avg_similarity),
                        },
                    }
                )

        # Enforce maximum chunk size by further splitting if needed
        final_chunks = []
        for chunk in chunks:
            if len(chunk["content"]) > self.chunk_size:
                # Split into smaller chunks
                sub_chunks = self._split_by_size(chunk["content"], file_path)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _split_into_segments(self, text: str) -> List[str]:
        """
        Split text into segments (sentences or paragraphs).

        Args:
            text: Input text

        Returns:
            List of text segments
        """
        if self.sentence_split:
            # Simple sentence splitting (can be improved with nltk)
            import re

            # Split on periods, exclamation marks, question marks
            sentences = re.split(r"(?<=[.!?])\s+", text)
            return [s.strip() for s in sentences if s.strip()]
        else:
            # Split on double newlines (paragraphs)
            paragraphs = text.split("\n\n")
            return [p.strip() for p in paragraphs if p.strip()]

    def _split_by_size(self, text: str, file_path: Optional[str]) -> List[Dict[str, Any]]:
        """
        Split text by size when it exceeds chunk_size.

        Args:
            text: Text to split
            file_path: File path for metadata

        Returns:
            List of chunk dictionaries
        """
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0

        for word in words:
            word_size = len(word) + 1  # +1 for space
            if current_size + word_size > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    {
                        "content": chunk_text,
                        "metadata": {
                            "type": "text",
                            "file_path": file_path,
                            "semantic_split": False,
                            "size_split": True,
                        },
                    }
                )
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        "type": "text",
                        "file_path": file_path,
                        "semantic_split": False,
                        "size_split": True,
                    },
                }
            )

        return chunks

    def _fallback_split(self, text: str, file_path: Optional[str]) -> List[Dict[str, Any]]:
        """
        Fallback to simple size-based splitting if embeddings fail.

        Args:
            text: Text to split
            file_path: File path for metadata

        Returns:
            List of chunk dictionaries
        """
        return self._split_by_size(text, file_path)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents using semantic chunking.

        Args:
            documents: List of Document objects

        Returns:
            List of split Document objects with semantic metadata
        """
        split_docs = []

        for doc in documents:
            file_path = doc.metadata.get("file_path") or doc.metadata.get("source")

            # Split the text
            chunks = self.split_semantic(doc.page_content, file_path)

            # Convert to Document objects
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata.update(chunk["metadata"])
                metadata["chunk_index"] = i
                metadata["total_chunks"] = len(chunks)
                split_docs.append(Document(page_content=chunk["content"], metadata=metadata))

        return split_docs
