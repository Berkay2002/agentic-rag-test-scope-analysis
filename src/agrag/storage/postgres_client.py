"""PostgreSQL client with pgvector support for vector and keyword search."""

from typing import List, Dict, Any, Optional
import psycopg
from psycopg import Connection
from psycopg.rows import dict_row, DictRow
from psycopg.types.json import Json
from pgvector.psycopg import register_vector
import logging

from agrag.config import settings
from agrag.kg.ontology import POSTGRESQL_SCHEMA

logger = logging.getLogger(__name__)


class PostgresClient:
    """Client for PostgreSQL database with pgvector extension."""

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize PostgreSQL client.

        Args:
            connection_string: PostgreSQL connection string (defaults to settings)
        """
        self.connection_string = connection_string or settings.postgres_connection_string

        if not self.connection_string:
            raise ValueError("PostgreSQL connection string must be provided")

        self.conn: Optional[Connection[DictRow]] = None
        logger.info("PostgreSQL client initialized")

    def connect(self, register_types: bool = True) -> None:
        """Establish connection to PostgreSQL database.

        Args:
            register_types: Whether to register pgvector types (set False if extension not yet created)
        """
        if self.conn is None or self.conn.closed:
            # Note: psycopg.connect with row_factory requires cast due to type stub limitations
            self.conn = psycopg.connect(
                self.connection_string,
                row_factory=dict_row,  # type: ignore[arg-type]
            )
            # Register pgvector types only if extension exists
            if register_types:
                try:
                    register_vector(self.conn)
                except Exception as e:
                    logger.warning(f"Failed to register pgvector types: {e}")
            logger.info("Connected to PostgreSQL database")

    def close(self) -> None:
        """Close the PostgreSQL connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("PostgreSQL connection closed")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def verify_connectivity(self) -> bool:
        """
        Verify connection to PostgreSQL database.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.connect()
            if self.conn is None:
                return False
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1 AS num")
                result = cur.fetchone()
                return result is not None and result["num"] == 1
        except Exception as e:
            logger.error(f"PostgreSQL connectivity check failed: {e}")
            return False

    def setup_schema(self) -> None:
        """
        Set up PostgreSQL schema (create tables, indexes, extensions).

        This should be run once during initial database setup.
        """
        logger.info("Setting up PostgreSQL schema...")

        # Connect without registering vector types first
        self.connect(register_types=False)

        if self.conn is None:
            raise RuntimeError("Failed to establish database connection")

        with self.conn.cursor() as cur:
            # Execute schema SQL (this will create the vector extension)
            cur.execute(POSTGRESQL_SCHEMA)
            self.conn.commit()

        # Now register vector types
        try:
            register_vector(self.conn)
            logger.info("Registered pgvector types")
        except Exception as e:
            logger.warning(f"Failed to register pgvector types: {e}")

        logger.info("PostgreSQL schema setup complete")

    def insert_document_chunk(
        self,
        chunk_id: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Insert a document chunk with embedding.

        Args:
            chunk_id: Unique chunk identifier
            content: Text content
            embedding: Vector embedding (768-dim)
            metadata: Optional metadata (stored as JSONB)

        Returns:
            Inserted chunk_id
        """
        self.connect()

        if self.conn is None:
            raise RuntimeError("Failed to establish database connection")

        query = """
        INSERT INTO document_chunks (chunk_id, content, embedding, metadata)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (chunk_id) DO UPDATE
        SET content = EXCLUDED.content,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
        RETURNING chunk_id
        """

        with self.conn.cursor() as cur:
            cur.execute(
                query,
                (chunk_id, content, embedding, Json(metadata or {})),
            )
            self.conn.commit()
            result = cur.fetchone()
            if result is None:
                raise RuntimeError("Insert failed: no result returned")
            return result["chunk_id"]

    def vector_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using pgvector.

        Args:
            query_embedding: Query embedding vector (768-dim)
            k: Number of results to return
            metadata_filter: Optional metadata filter (JSONB)

        Returns:
            List of similar documents with scores
        """
        self.connect()

        if self.conn is None:
            raise RuntimeError("Failed to establish database connection")

        # Base query - cast to vector type explicitly
        query = """
        SELECT
            chunk_id,
            content,
            metadata,
            embedding <=> %s::vector AS distance,
            1 - (embedding <=> %s::vector) AS similarity
        FROM document_chunks
        """

        params: List[Any] = [query_embedding, query_embedding]

        # Add metadata filter if provided
        if metadata_filter:
            conditions = []
            for key, value in metadata_filter.items():
                conditions.append(f"metadata->>'{key}' = %s")
                params.append(str(value))

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY distance LIMIT %s"
        params.append(k)

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                results = cur.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            # Rollback transaction on error
            if self.conn and not self.conn.closed:
                self.conn.rollback()
            raise e

    def keyword_search(
        self,
        query: str,
        k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search using pg_search extension.

        Uses ParadeDB's pg_search with true BM25 ranking algorithm,
        providing better relevance scoring than traditional TF-IDF.

        Args:
            query: Search query string
            k: Number of results to return
            metadata_filter: Optional metadata filter

        Returns:
            List of matching documents with BM25 scores
        """
        self.connect()

        if self.conn is None:
            raise RuntimeError("Failed to establish database connection")

        # Use pg_search BM25 index with @@@ operator and paradedb.score()
        # The @@@ operator performs BM25 full-text search
        query_sql = """
        SELECT
            chunk_id,
            content,
            metadata,
            paradedb.score(id) AS rank
        FROM document_chunks
        WHERE content @@@ %s
        """

        params: List[Any] = [query]

        # Add metadata filter if provided
        if metadata_filter:
            conditions = []
            for key, value in metadata_filter.items():
                conditions.append(f"metadata->>'{key}' = %s")
                params.append(str(value))

            if conditions:
                query_sql += " AND " + " AND ".join(conditions)

        query_sql += " ORDER BY paradedb.score(id) DESC LIMIT %s"
        params.append(k)

        try:
            with self.conn.cursor() as cur:
                cur.execute(query_sql, params)
                results = cur.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            # Rollback transaction on error
            if self.conn and not self.conn.closed:
                self.conn.rollback()
            raise e

    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        k: int = 10,
        rrf_k: int = 60,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining pgvector and BM25 with RRF fusion.

        Uses Reciprocal Rank Fusion (RRF) to merge:
        - Vector similarity results (pgvector HNSW index)
        - BM25 keyword results (pg_search BM25 index)

        Args:
            query: Search query string
            query_embedding: Query embedding vector
            k: Number of results to return
            rrf_k: RRF constant (default 60)
            metadata_filter: Optional metadata filter

        Returns:
            List of documents ranked by RRF score
        """
        try:
            # Get vector search results
            vector_results = self.vector_search(query_embedding, k * 2, metadata_filter)

            # Get keyword search results
            keyword_results = self.keyword_search(query, k * 2, metadata_filter)
        except Exception as e:
            # Rollback transaction on error
            if self.conn and not self.conn.closed:
                self.conn.rollback()
            raise e

        # Compute RRF scores
        rrf_scores: Dict[str, float] = {}
        rrf_docs: Dict[str, Dict[str, Any]] = {}

        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] = 1 / (rrf_k + rank)
            rrf_docs[chunk_id] = result

        # Process keyword results
        for rank, result in enumerate(keyword_results, start=1):
            chunk_id = result["chunk_id"]
            if chunk_id in rrf_scores:
                rrf_scores[chunk_id] += 1 / (rrf_k + rank)
            else:
                rrf_scores[chunk_id] = 1 / (rrf_k + rank)
                rrf_docs[chunk_id] = result

        # Sort by RRF score and take top k
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        # Build final results
        final_results = []
        for chunk_id, score in sorted_results:
            doc = rrf_docs[chunk_id].copy()
            doc["rrf_score"] = score
            final_results.append(doc)

        return final_results

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document chunk by its ID.

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk data or None if not found
        """
        self.connect()

        if self.conn is None:
            raise RuntimeError("Failed to establish database connection")

        query = "SELECT * FROM document_chunks WHERE chunk_id = %s"

        with self.conn.cursor() as cur:
            cur.execute(query, (chunk_id,))
            result = cur.fetchone()

        return dict(result) if result else None

    def delete_all_chunks(self) -> int:
        """
        Delete all document chunks (use with caution!).

        Returns:
            Number of chunks deleted
        """
        self.connect()

        if self.conn is None:
            raise RuntimeError("Failed to establish database connection")

        query = "DELETE FROM document_chunks RETURNING chunk_id"

        with self.conn.cursor() as cur:
            cur.execute(query)
            deleted = cur.fetchall()
            self.conn.commit()
            count = len(deleted)

        logger.warning(f"Deleted all {count} chunks from PostgreSQL database")
        return count

    def get_chunk_count(self) -> int:
        """
        Get total number of document chunks.

        Returns:
            Total chunk count
        """
        self.connect()

        if self.conn is None:
            raise RuntimeError("Failed to establish database connection")

        query = "SELECT COUNT(*) AS count FROM document_chunks"

        with self.conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchone()

        return int(result["count"]) if result else 0
