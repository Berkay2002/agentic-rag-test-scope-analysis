"""Storage writers for PostgreSQL, Neo4j, and BM25 indexing."""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import json
import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from agrag.storage.neo4j_client import Neo4jClient
from agrag.storage.postgres_client import PostgresClient
from agrag.storage.bm25_retriever import BM25RetrieverManager
from agrag.config.paths import BM25_INDEX_PATH

logger = logging.getLogger(__name__)


class BaseWriter(ABC):
    """Abstract base class for storage writers with common functionality."""

    def __init__(self):
        """Initialize base writer with stats tracking."""
        self.stats = self._init_stats()

    def _init_stats(self) -> Dict[str, int]:
        """Initialize statistics dictionary.

        Returns:
            Dict with default stat counters
        """
        return {"writes": 0, "failures": 0}

    @abstractmethod
    def _write_entity_impl(self, entity: Dict[str, Any], entity_type: str) -> bool:
        """Implementation-specific entity write logic.

        Args:
            entity: Entity data to write
            entity_type: Type of the entity

        Returns:
            True if write successful, False otherwise
        """
        pass

    def write_entity(self, entity: Dict[str, Any], entity_type: str) -> bool:
        """Write a single entity with error handling.

        Args:
            entity: Entity data to write
            entity_type: Type of the entity

        Returns:
            True if write successful, False otherwise
        """
        try:
            return self._write_entity_impl(entity, entity_type)
        except Exception as e:
            logger.error(
                "%s write failed for %s: %s",
                self.__class__.__name__,
                entity.get("id"),
                e,
            )
            self.stats["failures"] += 1
            return False

    def write_entities_batch(
        self,
        entities: List[Dict[str, Any]],
        entity_type: str,
        batch_size: int = 100,
    ) -> int:
        """Write multiple entities in batch with progress logging.

        Args:
            entities: List of entity data to write
            entity_type: Type of the entities
            batch_size: Batch size for progress logging

        Returns:
            Number of successful writes
        """
        total = len(entities)
        successes = 0

        logger.info(
            "Writing %d %s entities to %s...",
            total,
            entity_type,
            self.__class__.__name__,
        )

        for i, entity in enumerate(entities, 1):
            if self.write_entity(entity, entity_type):
                successes += 1

            if i % batch_size == 0:
                logger.info("Progress: %d/%d entities written", i, total)

        logger.info("%s batch write complete: %d successes", self.__class__.__name__, successes)
        return successes


class GraphWriter(BaseWriter):
    """Write entities to Neo4j when graph materialization is intended."""

    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        """Initialize GraphWriter.

        Args:
            neo4j_client: Neo4j client instance (creates new if not provided)
        """
        super().__init__()
        self.neo4j_client = neo4j_client or Neo4jClient()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _write_neo4j(self, entity: Dict[str, Any], entity_type: str) -> bool:
        """Write entity to Neo4j with retry logic.

        Args:
            entity: Entity data to write
            entity_type: Type of the entity

        Returns:
            True if write successful
        """
        entity_copy = entity.copy()
        if "embedding" in entity_copy:
            del entity_copy["embedding"]
        if isinstance(entity_copy.get("metadata"), dict):
            entity_copy["metadata"] = json.dumps(entity_copy["metadata"])

        query = f"""
        MERGE (n:{entity_type} {{id: $id}})
        SET n += $properties
        RETURN n.id AS id
        """
        result = self.neo4j_client.execute_cypher(
            query, {"id": entity["id"], "properties": entity_copy}
        )
        self.stats["writes"] += 1
        return bool(result)

    def _write_entity_impl(self, entity: Dict[str, Any], entity_type: str) -> bool:
        """Implementation of entity write for Neo4j.

        Args:
            entity: Entity data to write
            entity_type: Type of the entity

        Returns:
            True if write successful
        """
        return self._write_neo4j(entity, entity_type)


class PostgresWriter(BaseWriter):
    """Write entities with embeddings to PostgreSQL for retrieval."""

    def __init__(self, postgres_client: Optional[PostgresClient] = None):
        """Initialize PostgresWriter.

        Args:
            postgres_client: PostgreSQL client instance (creates new if not provided)
        """
        super().__init__()
        self.postgres_client = postgres_client or PostgresClient()
        self.stats["skipped"] = 0  # Add skipped stat for PostgresWriter

    def _build_content(self, entity: Dict[str, Any]) -> str:
        """Build searchable content from entity fields.

        Args:
            entity: Entity data

        Returns:
            Concatenated content string
        """
        content_parts = [
            entity.get("id", ""),
            entity.get("title", ""),
            entity.get("name", ""),
            entity.get("description", ""),
            entity.get("docstring", ""),
            entity.get("signature", ""),
            entity.get("path", ""),
            entity.get("file_path", ""),
        ]
        return " ".join(str(p) for p in content_parts if p)

    def _build_metadata(self, entity: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
        """Build metadata from entity fields.

        Args:
            entity: Entity data
            entity_type: Type of the entity

        Returns:
            Metadata dictionary
        """
        metadata = {"entity_type": entity_type, "entity_id": entity.get("id")}
        for key in [
            "file_path",
            "path",
            "component_id",
            "status",
            "line_start",
            "line_end",
            "category",
            "test_type",
            "priority",
            "result",
            "feature_area",
            "sub_feature",
            "test_suite",
        ]:
            if key in entity and entity[key] is not None:
                metadata[key] = str(entity[key])
        for key in ["source_system", "schema_version", "raw_type"]:
            if key in entity and entity[key] is not None:
                metadata[key] = str(entity[key])
        return metadata

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _write_postgres(self, entity: Dict[str, Any], entity_type: str) -> bool:
        """Write entity to PostgreSQL with retry logic.

        Args:
            entity: Entity data to write
            entity_type: Type of the entity

        Returns:
            True if write successful, False if skipped
        """
        if "embedding" not in entity or entity["embedding"] is None:
            logger.warning("No embedding for %s, skipping PostgreSQL", entity.get("id"))
            self.stats["skipped"] += 1
            return False

        content = self._build_content(entity)
        metadata = self._build_metadata(entity, entity_type)
        chunk_id = f"{entity_type}_{entity['id']}"

        self.postgres_client.insert_document_chunk(
            chunk_id=chunk_id,
            content=content,
            embedding=entity["embedding"],
            metadata=metadata,
        )
        self.stats["writes"] += 1
        return True

    def _write_entity_impl(self, entity: Dict[str, Any], entity_type: str) -> bool:
        """Implementation of entity write for PostgreSQL.

        Args:
            entity: Entity data to write
            entity_type: Type of the entity

        Returns:
            True if write successful
        """
        return self._write_postgres(entity, entity_type)


class BM25Writer(BaseWriter):
    """Write entities to BM25 for keyword search."""

    def __init__(
        self,
        bm25_manager: Optional[BM25RetrieverManager] = None,
        index_path: str = str(BM25_INDEX_PATH),
    ):
        """Initialize BM25Writer.

        Args:
            bm25_manager: BM25 retriever manager instance (creates new if not provided)
            index_path: Path to save the BM25 index
        """
        super().__init__()
        self.bm25_manager = bm25_manager or BM25RetrieverManager()
        self.index_path = index_path

    def _build_content(self, entity: Dict[str, Any]) -> str:
        """Build searchable content from entity fields.

        Args:
            entity: Entity data

        Returns:
            Concatenated content string
        """
        content_parts = [
            entity.get("id", ""),
            entity.get("title", ""),
            entity.get("name", ""),
            entity.get("description", ""),
            entity.get("docstring", ""),
            entity.get("signature", ""),
            entity.get("path", ""),
            entity.get("file_path", ""),
        ]
        tags = entity.get("tags", [])
        if tags:
            content_parts.extend(tags)
        return " ".join(str(p) for p in content_parts if p)

    def _build_metadata(self, entity: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
        """Build metadata from entity fields.

        Args:
            entity: Entity data
            entity_type: Type of the entity

        Returns:
            Metadata dictionary
        """
        metadata = {
            "entity_type": entity_type,
            "entity_id": entity.get("id"),
            "source": entity.get("file_path") or entity.get("path") or "unknown",
        }
        for key in ["source_system", "schema_version", "raw_type"]:
            if key in entity and entity[key] is not None:
                metadata[key] = str(entity[key])
        return metadata

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _write_bm25(self, entity: Dict[str, Any], entity_type: str) -> bool:
        """Write entity to BM25 with retry logic.

        Args:
            entity: Entity data to write
            entity_type: Type of the entity

        Returns:
            True if write successful, False if content is empty
        """
        content = self._build_content(entity)
        if not content.strip():
            return False

        metadata = self._build_metadata(entity, entity_type)
        self.bm25_manager.add_texts(texts=[content], metadatas=[metadata])
        self.stats["writes"] += 1
        return True

    def _write_entity_impl(self, entity: Dict[str, Any], entity_type: str) -> bool:
        """Implementation of entity write for BM25.

        Args:
            entity: Entity data to write
            entity_type: Type of the entity

        Returns:
            True if write successful
        """
        return self._write_bm25(entity, entity_type)

    def persist_index(self, file_path: Optional[str] = None) -> None:
        """Persist the BM25 index to disk.

        Args:
            file_path: Path to save the index (uses default if not provided)
        """
        path = file_path or self.index_path
        try:
            self.bm25_manager.save(path)
            logger.info("BM25 index saved to %s", path)
        except Exception as e:
            logger.error("Failed to save BM25 index: %s", e)
