"""Coordinated dual storage writer with retry logic."""

from typing import Dict, Any, List
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from agrag.storage.neo4j_client import Neo4jClient
from agrag.storage.postgres_client import PostgresClient
from agrag.storage.bm25_retriever import BM25RetrieverManager
from agrag.models.embeddings import get_embedding_service

logger = logging.getLogger(__name__)


class DualStorageWriter:
    """Orchestrates writes to Neo4j, PostgreSQL, and BM25 retriever."""

    def __init__(self):
        """Initialize the dual storage writer."""
        self.neo4j_client = Neo4jClient()
        self.postgres_client = PostgresClient()
        self.bm25_manager = BM25RetrieverManager()
        self.embedding_service = get_embedding_service()

        self.stats = {
            "neo4j_writes": 0,
            "postgres_writes": 0,
            "bm25_writes": 0,
            "failures": 0,
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _write_neo4j(self, entity: Dict[str, Any], entity_type: str) -> bool:
        """Write entity to Neo4j with retry logic."""
        try:
            # Remove embedding before Neo4j write
            entity_copy = entity.copy()
            if "embedding" in entity_copy:
                del entity_copy["embedding"]

            # Upsert (MERGE) for idempotency
            query = f"""
            MERGE (n:{entity_type} {{id: $id}})
            SET n += $properties
            RETURN n.id AS id
            """
            result = self.neo4j_client.execute_cypher(
                query, {"id": entity["id"], "properties": entity_copy}
            )

            self.stats["neo4j_writes"] += 1
            return bool(result)

        except Exception as e:
            logger.error(f"Neo4j write failed for {entity['id']}: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _write_postgres(self, entity: Dict[str, Any], entity_type: str) -> bool:
        """Write entity with embedding to PostgreSQL."""
        try:
            if "embedding" not in entity or entity["embedding"] is None:
                logger.warning(f"No embedding for {entity['id']}, skipping PostgreSQL")
                return False

            # Build content for full-text search
            content_parts = [
                entity.get("id", ""),
                entity.get("name", ""),
                entity.get("description", ""),
                entity.get("docstring", ""),
            ]
            content = " ".join(str(p) for p in content_parts if p)

            # Metadata
            metadata = {
                "entity_type": entity_type,
                "entity_id": entity["id"],
                "file_path": entity.get("file_path"),
                "line_start": entity.get("line_start"),
                "line_end": entity.get("line_end"),
            }

            # Upsert into PostgreSQL
            chunk_id = f"{entity_type}_{entity['id']}"
            self.postgres_client.insert_document_chunk(
                chunk_id=chunk_id,
                content=content,
                embedding=entity["embedding"],
                metadata=metadata,
            )

            self.stats["postgres_writes"] += 1
            return True

        except Exception as e:
            logger.error(f"PostgreSQL write failed for {entity['id']}: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _write_bm25(self, entity: Dict[str, Any], entity_type: str) -> bool:
        """Add entity to BM25 in-memory retriever."""
        try:
            # Build text content for BM25
            content_parts = [
                entity.get("id", ""),
                entity.get("name", ""),
                entity.get("description", ""),
                entity.get("docstring", ""),
                entity.get("signature", ""),
            ]
            content = " ".join(str(p) for p in content_parts if p)

            # Metadata for filtering
            metadata = {
                "entity_type": entity_type,
                "entity_id": entity["id"],
                "source": entity.get("file_path", "unknown"),
            }

            # Add to BM25 retriever
            self.bm25_manager.add_texts(
                texts=[content],
                metadatas=[metadata],
            )

            self.stats["bm25_writes"] += 1
            return True

        except Exception as e:
            logger.error(f"BM25 write failed for {entity['id']}: {e}")
            raise

    def write_entity(
        self,
        entity: Dict[str, Any],
        entity_type: str,
        skip_on_partial_failure: bool = False,
    ) -> Dict[str, bool]:
        """
        Write entity to all storage systems.

        Args:
            entity: Entity dictionary with all fields
            entity_type: Type label (Requirement, TestCase, Function, etc.)
            skip_on_partial_failure: If False, raises on any failure; if True, logs and continues

        Returns:
            Dictionary tracking which writes succeeded
        """
        results = {
            "neo4j": False,
            "postgres": False,
            "bm25": False,
        }

        try:
            # Write to Neo4j (graph structure)
            results["neo4j"] = self._write_neo4j(entity, entity_type)

            # Write to PostgreSQL (vector embeddings)
            results["postgres"] = self._write_postgres(entity, entity_type)

            # Write to BM25 (keyword search)
            results["bm25"] = self._write_bm25(entity, entity_type)

            # Check if all succeeded
            all_succeeded = all(results.values())
            if not all_succeeded:
                logger.warning(f"Partial write for {entity['id']}: {results}")
                if not skip_on_partial_failure:
                    self.stats["failures"] += 1
                    raise RuntimeError(f"Partial write failure: {results}")

            return results

        except Exception as e:
            logger.error(f"Write failed for {entity['id']}: {e}")
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
        """
        Write multiple entities in batches.

        Args:
            entities: List of entity dictionaries
            entity_type: Type of entity
            batch_size: Logging interval

        Returns:
            Statistics: successful writes per storage system
        """
        total = len(entities)
        success_count = {"neo4j": 0, "postgres": 0, "bm25": 0}

        logger.info(f"Writing {total} {entity_type} entities to dual storage...")

        for i, entity in enumerate(entities, 1):
            try:
                results = self.write_entity(entity, entity_type, skip_on_partial_failure=True)

                # Track successes
                for store, succeeded in results.items():
                    if succeeded:
                        success_count[store] += 1

                if i % batch_size == 0:
                    logger.info(f"Progress: {i}/{total} entities written")

            except Exception as e:
                logger.error(f"Failed to write entity {entity.get('id')}: {e}")

        logger.info(f"Batch write complete: {success_count}")
        return success_count

    def persist_bm25_index(self, file_path: str = "data/bm25_index.pkl") -> None:
        """Save BM25 index to disk for persistence."""
        try:
            self.bm25_manager.save(file_path)
            logger.info(f"BM25 index saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")

    def get_statistics(self) -> Dict[str, int]:
        """Get write statistics."""
        return self.stats.copy()
