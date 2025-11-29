"""BM25 retriever implementation using rank_bm25 library."""

from typing import List, Dict, Any, Optional, Callable
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class BM25RetrieverManager:
    """Manager for BM25 retriever with persistence and document management."""

    def __init__(self, k: int = 10, preprocess_func: Optional[Callable] = None):
        """
        Initialize BM25 retriever manager.

        Args:
            k: Number of results to return (default: 10)
            preprocess_func: Optional preprocessing function for tokenization
        """
        self.k = k
        self.preprocess_func = preprocess_func
        self.retriever: Optional[BM25Retriever] = None
        self.documents: List[Document] = []
        logger.info("BM25RetrieverManager initialized")

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the BM25 index.

        Args:
            documents: List of LangChain Document objects
        """
        self.documents.extend(documents)
        self._rebuild_index()
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add texts to the BM25 index.

        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dictionaries
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        self.add_documents(documents)

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index from current documents."""
        if not self.documents:
            logger.warning("No documents to build BM25 index")
            return

        logger.info(f"Building BM25 index with {len(self.documents)} documents...")

        if self.preprocess_func:
            self.retriever = BM25Retriever.from_documents(
                self.documents,
                k=self.k,
                preprocess_func=self.preprocess_func,
            )
        else:
            self.retriever = BM25Retriever.from_documents(
                self.documents,
                k=self.k,
            )

        logger.info("BM25 index built successfully")

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents using BM25 algorithm.

        Args:
            query: Search query string
            k: Number of results to return (overrides default k)
            metadata_filter: Optional metadata filter

        Returns:
            List of matching documents with metadata
        """
        if not self.retriever:
            logger.warning("BM25 index not built. Building now...")
            self._rebuild_index()

        if not self.retriever:
            return []

        # Update k if provided
        original_k = self.retriever.k
        if k is not None:
            self.retriever.k = k

        # Perform search
        results = self.retriever.invoke(query)

        # Restore original k
        self.retriever.k = original_k

        # Apply metadata filter if provided
        if metadata_filter:
            filtered_results = []
            for doc in results:
                if all(doc.metadata.get(key) == value for key, value in metadata_filter.items()):
                    filtered_results.append(doc)
            results = filtered_results

        # Convert to dict format with BM25 scores
        formatted_results = []
        for rank, doc in enumerate(results, start=1):
            formatted_results.append(
                {
                    "chunk_id": doc.metadata.get("chunk_id", f"doc_{rank}"),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "rank": rank,
                    # BM25 score is implicit in rank ordering
                    "score": 1.0 / rank,  # Simple rank-based score
                }
            )

        return formatted_results

    def clear(self) -> None:
        """Clear all documents and reset the index."""
        self.documents = []
        self.retriever = None
        logger.info("BM25 index cleared")

    def save(self, filepath: str) -> None:
        """
        Save the BM25 index to disk.

        Args:
            filepath: Path to save the index
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "documents": self.documents,
            "k": self.k,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"BM25 index saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load the BM25 index from disk.

        Args:
            filepath: Path to load the index from
        """
        if not Path(filepath).exists():
            logger.warning(f"Index file not found: {filepath}")
            return

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.documents = data["documents"]
        self.k = data["k"]
        self._rebuild_index()

        logger.info(f"BM25 index loaded from {filepath} ({len(self.documents)} documents)")

    def get_document_count(self) -> int:
        """
        Get the total number of documents in the index.

        Returns:
            Number of documents
        """
        return len(self.documents)
