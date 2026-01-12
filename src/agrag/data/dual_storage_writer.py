"""Storage writers for PostgreSQL, Neo4j, and BM25 indexing."""

from typing import Dict, Any, List, Optional
import json
import logging
import warnings

from tenacity import retry, stop_after_attempt, wait_exponential

from agrag.storage.neo4j_client import Neo4jClient
from agrag.storage.postgres_client import PostgresClient
from agrag.storage.bm25_retriever import BM25RetrieverManager

logger = logging.getLogger(__name__)


class GraphWriter:
    """Write entities to Neo4j when graph materialization is intended."""

    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.stats = {"writes": 0, "failures": 0}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _write_neo4j(self, entity: Dict[str, Any], entity_type: str) -> bool:
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

    def write_entity(self, entity: Dict[str, Any], entity_type: str) -> bool:
        try:
            return self._write_neo4j(entity, entity_type)
        except Exception as e:
            logger.error("Neo4j write failed for %s: %s", entity.get("id"), e)
            self.stats["failures"] += 1
            return False

    def write_entities_batch(
        self,
        entities: List[Dict[str, Any]],
        entity_type: str,
        batch_size: int = 100,
    ) -> int:
        total = len(entities)
        successes = 0

        logger.info("Writing %d %s entities to Neo4j...", total, entity_type)

        for i, entity in enumerate(entities, 1):
            if self.write_entity(entity, entity_type):
                successes += 1

            if i % batch_size == 0:
                logger.info("Progress: %d/%d entities written", i, total)

        logger.info("Neo4j batch write complete: %d successes", successes)
        return successes


class PostgresWriter:
    """Write entities with embeddings to PostgreSQL for retrieval."""

    def __init__(self, postgres_client: Optional[PostgresClient] = None):
        self.postgres_client = postgres_client or PostgresClient()
        self.stats = {"writes": 0, "skipped": 0, "failures": 0}

    def _build_content(self, entity: Dict[str, Any]) -> str:
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
        return metadata

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _write_postgres(self, entity: Dict[str, Any], entity_type: str) -> bool:
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

    def write_entity(self, entity: Dict[str, Any], entity_type: str) -> bool:
        try:
            return self._write_postgres(entity, entity_type)
        except Exception as e:
            logger.error("PostgreSQL write failed for %s: %s", entity.get("id"), e)
            self.stats["failures"] += 1
            return False

    def write_entities_batch(
        self,
        entities: List[Dict[str, Any]],
        entity_type: str,
        batch_size: int = 100,
    ) -> int:
        total = len(entities)
        successes = 0

        logger.info("Writing %d %s entities to PostgreSQL...", total, entity_type)

        for i, entity in enumerate(entities, 1):
            if self.write_entity(entity, entity_type):
                successes += 1

            if i % batch_size == 0:
                logger.info("Progress: %d/%d entities written", i, total)

        logger.info("PostgreSQL batch write complete: %d successes", successes)
        return successes


class BM25Writer:
    """Write entities to BM25 for keyword search."""

    def __init__(
        self,
        bm25_manager: Optional[BM25RetrieverManager] = None,
        index_path: str = "data/bm25_index.pkl",
    ):
        self.bm25_manager = bm25_manager or BM25RetrieverManager()
        self.index_path = index_path
        self.stats = {"writes": 0, "failures": 0}

    def _build_content(self, entity: Dict[str, Any]) -> str:
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
        return {
            "entity_type": entity_type,
            "entity_id": entity.get("id"),
            "source": entity.get("file_path") or entity.get("path") or "unknown",
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _write_bm25(self, entity: Dict[str, Any], entity_type: str) -> bool:
        content = self._build_content(entity)
        if not content.strip():
            return False

        metadata = self._build_metadata(entity, entity_type)
        self.bm25_manager.add_texts(texts=[content], metadatas=[metadata])
        self.stats["writes"] += 1
        return True

    def write_entity(self, entity: Dict[str, Any], entity_type: str) -> bool:
        try:
            return self._write_bm25(entity, entity_type)
        except Exception as e:
            logger.error("BM25 write failed for %s: %s", entity.get("id"), e)
            self.stats["failures"] += 1
            return False

    def write_entities_batch(
        self,
        entities: List[Dict[str, Any]],
        entity_type: str,
        batch_size: int = 100,
    ) -> int:
        total = len(entities)
        successes = 0

        logger.info("Writing %d %s entities to BM25...", total, entity_type)

        for i, entity in enumerate(entities, 1):
            if self.write_entity(entity, entity_type):
                successes += 1

            if i % batch_size == 0:
                logger.info("Progress: %d/%d entities written", i, total)

        logger.info("BM25 batch write complete: %d successes", successes)
        return successes

    def persist_index(self, file_path: Optional[str] = None) -> None:
        path = file_path or self.index_path
        try:
            self.bm25_manager.save(path)
            logger.info("BM25 index saved to %s", path)
        except Exception as e:
            logger.error("Failed to save BM25 index: %s", e)


class DualStorageWriter:
    """Deprecated: use PostgresWriter/BM25Writer/GraphWriter explicitly."""

    def __init__(self, enable_graph: bool = True, bm25_index_path: str = "data/bm25_index.pkl"):
        warnings.warn(
            "DualStorageWriter is deprecated; use PostgresWriter, GraphWriter, and BM25Writer",
            DeprecationWarning,
            stacklevel=2,
        )
        self.postgres_writer = PostgresWriter()
        self.graph_writer = GraphWriter() if enable_graph else None
        self.bm25_writer = BM25Writer(index_path=bm25_index_path)

        self.stats = {
            "neo4j_writes": 0,
            "postgres_writes": 0,
            "bm25_writes": 0,
            "failures": 0,
        }

    def write_entity(
        self,
        entity: Dict[str, Any],
        entity_type: str,
        skip_on_partial_failure: bool = False,
    ) -> Dict[str, bool]:
        results = {"neo4j": False, "postgres": False, "bm25": False}

        try:
            if self.graph_writer:
                results["neo4j"] = self.graph_writer.write_entity(entity, entity_type)
                if results["neo4j"]:
                    self.stats["neo4j_writes"] += 1
            results["postgres"] = self.postgres_writer.write_entity(entity, entity_type)
            if results["postgres"]:
                self.stats["postgres_writes"] += 1
            results["bm25"] = self.bm25_writer.write_entity(entity, entity_type)
            if results["bm25"]:
                self.stats["bm25_writes"] += 1

            if not all(results.values()) and not skip_on_partial_failure:
                self.stats["failures"] += 1
                raise RuntimeError(f"Partial write failure: {results}")

            return results
        except Exception as e:
            logger.error("Write failed for %s: %s", entity.get("id"), e)
            self.stats["failures"] += 1
            if not skip_on_partial_failure:
                raise
            return results

    def write_entities_batch(
        self,
        entities: List[Dict[str, Any]],
        entity_type: str,
        batch_size: int = 100,
    ) -> Dict[str, int]:
        total = len(entities)
        success_count = {"neo4j": 0, "postgres": 0, "bm25": 0}

        logger.info("Writing %d %s entities to dual storage...", total, entity_type)

        for i, entity in enumerate(entities, 1):
            results = self.write_entity(entity, entity_type, skip_on_partial_failure=True)
            for store, succeeded in results.items():
                if succeeded:
                    success_count[store] += 1

            if i % batch_size == 0:
                logger.info("Progress: %d/%d entities written", i, total)

        logger.info("Batch write complete: %s", success_count)
        return success_count

    def persist_bm25_index(self, file_path: str = "data/bm25_index.pkl") -> None:
        self.bm25_writer.persist_index(file_path)

    def get_statistics(self) -> Dict[str, int]:
        return self.stats.copy()
