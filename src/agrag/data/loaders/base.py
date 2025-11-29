"""Abstract base loader for document ingestion."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Document:
    """Represents a loaded document chunk with metadata."""

    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a Document.

        Args:
            page_content: The text content of the document chunk
            metadata: Additional metadata about the chunk
        """
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    def __init__(self, **kwargs):
        """Initialize the loader with configuration parameters."""
        self.config = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def load(self) -> List[Document]:
        """
        Load documents from the source.

        Returns:
            List of Document objects with content and metadata
        """
        pass

    def validate_path(self, path: Path) -> bool:
        """
        Validate that a path exists and is accessible.

        Args:
            path: Path to validate

        Returns:
            True if path is valid and accessible
        """
        if not path.exists():
            self.logger.error(f"Path does not exist: {path}")
            return False
        return True

    def _create_document(
        self,
        content: str,
        chunk_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Create a Document object with standardized metadata.

        Args:
            content: The text content
            chunk_id: Unique identifier for the chunk
            metadata: Additional metadata

        Returns:
            Document object
        """
        meta = metadata or {}
        meta["chunk_id"] = chunk_id
        meta["loader"] = self.__class__.__name__

        return Document(page_content=content, metadata=meta)


class BaseTextSplitter(ABC):
    """Abstract base class for text splitters."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **kwargs,
    ):
        """
        Initialize the text splitter.

        Args:
            chunk_size: Maximum size of each chunk (in tokens or characters)
            chunk_overlap: Number of overlapping tokens/characters between chunks
            **kwargs: Additional configuration parameters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.config = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Input text to split

        Returns:
            List of text chunks
        """
        pass

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into smaller chunks.

        Args:
            documents: List of documents to split

        Returns:
            List of split documents
        """
        split_docs = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata["chunk_index"] = i
                metadata["total_chunks"] = len(chunks)
                split_docs.append(Document(page_content=chunk, metadata=metadata))

        return split_docs
