"""Data ingestion pipeline for loading data into Neo4j and PostgreSQL."""

from typing import List, Dict, Any
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import logging

from agrag.storage.neo4j_client import Neo4jClient
from agrag.storage.postgres_client import PostgresClient
from agrag.data.dual_storage_writer import DualStorageWriter
from agrag.models.embeddings import get_embedding_service

logger = logging.getLogger(__name__)


class DataIngestion:
    """Load data into Neo4j and PostgreSQL."""

    def __init__(self):
        """Initialize ingestion pipeline with database clients."""
        self.neo4j_client = Neo4jClient()
        self.postgres_client = PostgresClient()
        logger.info("Data ingestion pipeline initialized")

    def ingest_entities_neo4j(
        self, entities: List[Dict[str, Any]], entity_type: str, batch_size: int = 100
    ) -> int:
        """
        Batch insert entities into Neo4j.

        Args:
            entities: List of entity dictionaries
            entity_type: Type of entity (Requirement, TestCase, etc.)
            batch_size: Number of entities per batch

        Returns:
            Number of entities inserted
        """
        if not entities:
            return 0

        logger.info(f"Ingesting {len(entities)} {entity_type} entities into Neo4j...")

        # Remove embeddings for Neo4j (stored in PostgreSQL)
        neo4j_entities = []
        for entity in entities:
            entity_copy = entity.copy()
            if "embedding" in entity_copy:
                del entity_copy["embedding"]
            # Convert metadata dict to JSON string for Neo4j
            if "metadata" in entity_copy and isinstance(entity_copy["metadata"], dict):
                import json

                entity_copy["metadata"] = json.dumps(entity_copy["metadata"])
            neo4j_entities.append(entity_copy)

        total_inserted = 0

        # Process in batches
        for i in range(0, len(neo4j_entities), batch_size):
            batch = neo4j_entities[i : i + batch_size]

            # Create Cypher query for batch insert
            query = f"""
            UNWIND $batch AS entity
            MERGE (n:{entity_type} {{id: entity.id}})
            SET n += entity
            RETURN count(n) as count
            """

            result = self.neo4j_client.execute_cypher(query, {"batch": batch})
            count = result[0]["count"] if result else 0
            total_inserted += count

        logger.info(f"Inserted {total_inserted} {entity_type} entities into Neo4j")
        return total_inserted

    def ingest_entities_postgres(self, entities: List[Dict[str, Any]], entity_type: str) -> int:
        """
        Insert entities with embeddings into PostgreSQL.

        Args:
            entities: List of entity dictionaries
            entity_type: Type of entity

        Returns:
            Number of entities inserted
        """
        if not entities:
            return 0

        logger.info(f"Ingesting {len(entities)} {entity_type} entities into PostgreSQL...")

        total_inserted = 0

        for entity in entities:
            # Skip entities without embeddings
            if "embedding" not in entity or entity["embedding"] is None:
                continue

            # Create content for full-text search
            content_parts = [
                entity.get("id", ""),
                entity.get("name", ""),
                entity.get("description", ""),
                entity.get("docstring", ""),
                entity.get("signature", ""),
            ]
            content = " ".join(str(p) for p in content_parts if p)

            # Create metadata
            metadata = {
                "entity_type": entity_type,
                "entity_id": entity["id"],
            }

            # Add additional metadata fields
            for key in ["category", "test_type", "priority", "status", "file_path"]:
                if key in entity:
                    metadata[key] = str(entity[key])

            # Insert into PostgreSQL
            chunk_id = f"{entity_type}_{entity['id']}"
            self.postgres_client.insert_document_chunk(
                chunk_id=chunk_id, content=content, embedding=entity["embedding"], metadata=metadata
            )
            total_inserted += 1

        logger.info(f"Inserted {total_inserted} {entity_type} entities into PostgreSQL")
        return total_inserted

    def ingest_relationships_neo4j(
        self, relationships: List[Dict[str, Any]], batch_size: int = 100
    ) -> int:
        """
        Batch insert relationships into Neo4j.

        Args:
            relationships: List of relationship dictionaries
            batch_size: Number of relationships per batch

        Returns:
            Number of relationships inserted
        """
        if not relationships:
            return 0

        logger.info(f"Ingesting {len(relationships)} relationships into Neo4j...")

        total_inserted = 0

        # Group relationships by type for efficient batching
        from collections import defaultdict

        rels_by_type = defaultdict(list)
        for rel in relationships:
            rels_by_type[rel["relationship_type"]].append(rel)

        # Process each relationship type in batches
        for rel_type, type_rels in rels_by_type.items():
            for i in range(0, len(type_rels), batch_size):
                batch = type_rels[i : i + batch_size]

                # Create Cypher query for batch insert of single relationship type
                query = f"""
                UNWIND $batch AS rel
                MATCH (source {{id: rel.source_id}})
                MATCH (target {{id: rel.target_id}})
                MERGE (source)-[r:{rel_type}]->(target)
                SET r += rel.properties
                RETURN count(r) as count
                """

                try:
                    result = self.neo4j_client.execute_cypher(query, {"batch": batch})
                    count = result[0]["count"] if result else 0
                    total_inserted += count
                except Exception as e:
                    logger.error(f"Error inserting {rel_type} relationships batch: {e}")
                    # Fallback: insert one by one
                    for rel in batch:
                        try:
                            single_query = f"""
                            MATCH (source {{id: $source_id}})
                            MATCH (target {{id: $target_id}})
                            MERGE (source)-[r:{rel_type}]->(target)
                            SET r += $properties
                            RETURN r
                            """
                            self.neo4j_client.execute_cypher(
                                single_query,
                                {
                                    "source_id": rel["source_id"],
                                    "target_id": rel["target_id"],
                                    "properties": rel.get("properties", {}),
                                },
                            )
                            total_inserted += 1
                        except Exception as inner_e:
                            logger.warning(
                                f"Failed to insert relationship {rel['source_id']}-[{rel_type}]->{rel['target_id']}: {inner_e}"
                            )

        logger.info(f"Inserted {total_inserted} relationships into Neo4j")
        return total_inserted

    def ingest_full_dataset(self, dataset: Dict[str, Any]) -> Dict[str, int]:
        """
        Ingest complete dataset into both databases.

        Args:
            dataset: Complete dataset dictionary with entities and relationships

        Returns:
            Dictionary with ingestion counts
        """
        logger.info("Starting full dataset ingestion...")

        entities = dataset.get("entities", [])
        relationships = dataset.get("relationships", [])

        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            # Determine entity type from ID prefix or other fields
            if entity["id"].startswith("REQ_"):
                entity_type = "Requirement"
            elif entity["id"].startswith("TC_"):
                entity_type = "TestCase"
            elif entity["id"].startswith("FUNC_"):
                entity_type = "Function"
            elif entity["id"].startswith("CLASS_"):
                entity_type = "Class"
            elif entity["id"].startswith("MOD_"):
                entity_type = "Module"
            else:
                logger.warning(f"Unknown entity type for ID: {entity['id']}")
                continue

            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)

        # Use rich progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:

            # Ingest entities
            neo4j_counts = {}
            postgres_counts = {}

            total_entities = len(entities)
            entity_task = progress.add_task("[cyan]Ingesting entities...", total=total_entities)

            for entity_type, type_entities in entities_by_type.items():
                # Neo4j
                neo4j_count = self.ingest_entities_neo4j(type_entities, entity_type)
                neo4j_counts[entity_type] = neo4j_count

                # PostgreSQL
                postgres_count = self.ingest_entities_postgres(type_entities, entity_type)
                postgres_counts[entity_type] = postgres_count

                progress.update(entity_task, advance=len(type_entities))

            # Ingest relationships
            rel_task = progress.add_task(
                "[cyan]Ingesting relationships...", total=len(relationships)
            )
            rel_count = self.ingest_relationships_neo4j(relationships)
            progress.update(rel_task, advance=len(relationships))

        results = {
            "neo4j_entities": sum(neo4j_counts.values()),
            "postgres_entities": sum(postgres_counts.values()),
            "relationships": rel_count,
            "details": {
                "neo4j": neo4j_counts,
                "postgres": postgres_counts,
            },
        }

        logger.info("Dataset ingestion complete!")
        logger.info(
            f"Neo4j: {results['neo4j_entities']} entities, {results['relationships']} relationships"
        )
        logger.info(f"PostgreSQL: {results['postgres_entities']} entities with embeddings")

        return results

    def ingest_from_code_repository(self, repo_path: str) -> Dict[str, int]:
        """
        Load and ingest code from a Git repository.

        Args:
            repo_path: Path to the code repository

        Returns:
            Dictionary with ingestion counts
        """
        from agrag.data.loaders import CodeLoader

        logger.info(f"Loading code from repository: {repo_path}")

        # Load code files
        loader = CodeLoader(repo_path, languages=["python"])
        documents = loader.load()

        logger.info(f"Loaded {len(documents)} code chunks from repository")

        # Convert documents to entities
        entities = self._documents_to_entities(documents)

        # Use dual storage writer
        writer = DualStorageWriter()

        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            entity_type = entity.get("entity_type", "Function")
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)

        # Ingest each type
        total_counts = {"neo4j": 0, "postgres": 0, "bm25": 0}
        for entity_type, type_entities in entities_by_type.items():
            counts = writer.write_entities_batch(type_entities, entity_type)
            for key in total_counts:
                total_counts[key] += counts[key]

        # Persist BM25 index
        writer.persist_bm25_index()

        logger.info(f"Repository ingestion complete: {total_counts}")
        return total_counts

    def ingest_from_documents(
        self,
        directory: str,
        use_chunker: bool = True,
        export_format: str = "markdown",
        table_mode: str = "accurate",
        max_num_pages: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Load and ingest requirements/documentation using Docling.

        Args:
            directory: Path to the directory containing documents
            use_chunker: Use Docling HybridChunker for semantic chunking
            export_format: Export format (markdown, text, json, html, doctags)
            table_mode: TableFormer mode (accurate or fast)
            max_num_pages: Maximum pages to process per document

        Returns:
            Dictionary with ingestion counts
        """
        from agrag.data.loaders import MultiDocumentLoader

        logger.info(f"Loading documents from: {directory} (Docling-powered)")

        # Load documents with Docling
        loader = MultiDocumentLoader(
            directory,
            use_chunker=use_chunker,
            export_format=export_format,
            table_mode=table_mode,
            max_num_pages=max_num_pages,
        )
        documents = loader.load()

        logger.info(f"Loaded {len(documents)} document chunks")

        # Convert documents to requirement entities
        entities = self._documents_to_requirements(documents)

        # Use dual storage writer
        writer = DualStorageWriter()
        counts = writer.write_entities_batch(entities, "Requirement")

        # Persist BM25 index
        writer.persist_bm25_index()

        logger.info(f"Document ingestion complete: {counts}")
        return counts

    def _documents_to_entities(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert Document objects to entity dictionaries.

        Args:
            documents: List of Document objects from loaders

        Returns:
            List of entity dictionaries
        """
        entities = []
        embedding_service = get_embedding_service()

        for doc in documents:
            metadata = doc.metadata
            entity_type = metadata.get("type", "function")

            # Generate embedding if not present
            embedding = metadata.get("embedding")
            if not embedding:
                embedding = embedding_service.embed_query(doc.page_content)

            # Build entity based on type
            if entity_type == "function" or entity_type == "method":
                entity = {
                    "id": f"FUNC_{metadata.get('name', 'unknown')}",
                    "name": metadata.get("name", "unknown"),
                    "signature": metadata.get("signature", ""),
                    "code_snippet": doc.page_content[:500],  # Truncate for storage
                    "file_path": metadata.get("file_path", ""),
                    "line_number": metadata.get("line_start", 0),
                    "docstring": metadata.get("docstring", ""),
                    "complexity": metadata.get("complexity", 0),
                    "embedding": embedding,
                    "metadata": {
                        "parent_class": metadata.get("parent_class"),
                        "decorators": metadata.get("decorators", []),
                        "language": metadata.get("language", "python"),
                    },
                    "entity_type": "Function",
                }
            elif entity_type == "class":
                entity = {
                    "id": f"CLASS_{metadata.get('name', 'unknown')}",
                    "name": metadata.get("name", "unknown"),
                    "file_path": metadata.get("file_path", ""),
                    "line_number": metadata.get("line_start", 0),
                    "methods": metadata.get("methods", []),
                    "base_classes": metadata.get("base_classes", []),
                    "docstring": metadata.get("docstring", ""),
                    "embedding": embedding,
                    "metadata": {
                        "language": metadata.get("language", "python"),
                    },
                    "entity_type": "Class",
                }
            else:
                # Default entity
                entity = {
                    "id": f"{entity_type.upper()}_{hash(doc.page_content) % 10000:04d}",
                    "name": metadata.get("name", ""),
                    "description": doc.page_content[:200],
                    "file_path": metadata.get("file_path", ""),
                    "embedding": embedding,
                    "metadata": metadata,
                    "entity_type": entity_type.title(),
                }

            entities.append(entity)

        return entities

    def _documents_to_requirements(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert Document objects to Requirement entities.

        Args:
            documents: List of Document objects

        Returns:
            List of requirement entity dictionaries
        """
        requirements = []
        embedding_service = get_embedding_service()

        for doc in documents:
            metadata = doc.metadata

            # Generate embedding
            embedding = embedding_service.embed_query(doc.page_content)

            # Extract requirement ID from section or generate one
            req_id = metadata.get("section", f"REQ_{hash(doc.page_content) % 10000:04d}")
            if not req_id.startswith("REQ_"):
                req_id = f"REQ_{req_id}"

            requirement = {
                "id": req_id,
                "description": doc.page_content,
                "priority": "MEDIUM",  # Default priority
                "status": "APPROVED",
                "category": (
                    metadata.get("parent_sections", ["general"])[0]
                    if metadata.get("parent_sections")
                    else "general"
                ),
                "embedding": embedding,
                "metadata": {
                    "title": metadata.get("title", ""),
                    "section": metadata.get("section", ""),
                    "parent_sections": metadata.get("parent_sections", []),
                    "file_path": metadata.get("file_path", ""),
                    "line_start": metadata.get("line_start", 0),
                    "line_end": metadata.get("line_end", 0),
                },
            }

            requirements.append(requirement)

        return requirements
