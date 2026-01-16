"""Neo4j client for graph operations and vector search."""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver
import logging

from agrag.config import settings
from agrag.storage.retry_decorators import resilient_db_operation
from agrag.kg.registry import get_registry

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Client for Neo4j graph database operations."""

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """
        Initialize Neo4j client.

        Args:
            uri: Neo4j connection URI (defaults to settings)
            username: Neo4j username (defaults to settings)
            password: Neo4j password (defaults to settings)
            database: Neo4j database name (defaults to settings)
        """
        self.uri = uri or settings.neo4j_uri
        self.username = username or settings.neo4j_username
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database

        if not self.uri or not self.password:
            raise ValueError("Neo4j URI and password must be provided")

        self.registry = get_registry()
        self.driver: Driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        logger.info(f"Neo4j client initialized for {self.uri}")

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j client closed")

    def is_healthy(self) -> bool:
        """Check if Neo4j connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            return self.verify_connectivity()
        except Exception:
            return False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def verify_connectivity(self) -> bool:
        """
        Verify connection to Neo4j database.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS num")
                record = result.single()
                return record["num"] == 1
        except Exception as e:
            logger.error(f"Neo4j connectivity check failed: {e}")
            return False

    def setup_schema(self) -> None:
        """
        Set up Neo4j schema (constraints and vector indexes).

        This should be run once during initial database setup.
        """
        logger.info("Setting up Neo4j schema...")

        if not self.verify_connectivity():
            raise ConnectionError(f"Neo4j connectivity check failed for {self.uri}")

        with self.driver.session(database=self.database) as session:
            # Create constraints
            for constraint_query in self.registry.neo4j_constraints():
                try:
                    session.run(constraint_query)
                    logger.info(f"Created constraint: {constraint_query[:50]}...")
                except Exception as e:
                    logger.warning(f"Constraint creation failed (may already exist): {e}")

            # Create vector indexes
            for index_query in self.registry.neo4j_vector_indexes():
                try:
                    session.run(index_query)
                    logger.info("Created vector index")
                except Exception as e:
                    logger.warning(f"Vector index creation failed (may already exist): {e}")

        logger.info("Neo4j schema setup complete")

    @resilient_db_operation
    def create_node(
        self,
        label: str,
        properties: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create a node in Neo4j.

        Args:
            label: Node label
            properties: Node properties

        Returns:
            Created node properties
        """
        label_value = self._normalize_label(label)
        query = f"""
        CREATE (n:{label_value})
        SET n = $properties
        RETURN n
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, properties=properties)
            record = result.single()
            return dict(record["n"]) if record else {}

    @resilient_db_operation
    def create_relationship(
        self,
        source_id: str,
        source_label: str,
        target_id: str,
        target_label: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a relationship between two nodes.

        Args:
            source_id: Source node ID
            source_label: Source node label
            target_id: Target node ID
            target_label: Target node label
            relationship_type: Relationship type
            properties: Optional relationship properties

        Returns:
            Relationship information
        """
        props = properties or {}
        source_label_value = self._normalize_label(source_label)
        target_label_value = self._normalize_label(target_label)
        rel_value = self._normalize_relationship(relationship_type)
        query = f"""
        MATCH (source:{source_label_value} {{id: $source_id}})
        MATCH (target:{target_label_value} {{id: $target_id}})
        CREATE (source)-[r:{rel_value}]->(target)
        SET r = $properties
        RETURN r, source.id AS source_id, target.id AS target_id
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                source_id=source_id,
                target_id=target_id,
                properties=props,
            )
            record = result.single()
            if record:
                return {
                    "source_id": record["source_id"],
                    "target_id": record["target_id"],
                    "type": rel_value,
                    "properties": dict(record["r"]),
                }
            return {}

    @resilient_db_operation
    def vector_search(
        self,
        query_embedding: List[float],
        node_label: str,
        k: int = 10,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using Neo4j vector index.

        Args:
            query_embedding: Query embedding vector (768-dim)
            node_label: Node label to search
            k: Number of results to return
            similarity_threshold: Optional minimum similarity threshold

        Returns:
            List of similar nodes with scores
        """
        label_value = self._normalize_label(node_label)
        index_name = self.registry.neo4j_vector_index_name(label_value)

        query = """
        CALL db.index.vector.queryNodes($index_name, $k, $query_embedding)
        YIELD node, score
        WHERE score >= $threshold
        RETURN node, score
        ORDER BY score DESC
        """

        threshold = similarity_threshold or settings.vector_search_similarity_threshold

        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                index_name=index_name,
                k=k,
                query_embedding=query_embedding,
                threshold=threshold,
            )

            results = []
            for record in result:
                node_data = dict(record["node"])
                results.append(
                    {
                        "node": node_data,
                        "score": record["score"],
                        "id": node_data.get("id"),
                        "label": label_value,
                    }
                )

            return results

    @resilient_db_operation
    def graph_traverse(
        self,
        start_node_id: str,
        start_node_label: str,
        relationship_types: Optional[List[str]] = None,
        depth: int = 2,
        direction: str = "outgoing",  # "outgoing", "incoming", "both"
    ) -> List[Dict[str, Any]]:
        """
        Traverse the graph from a starting node.

        Args:
            start_node_id: Starting node ID
            start_node_label: Starting node label
            relationship_types: Optional list of relationship types to follow
            depth: Maximum traversal depth (1-5)
            direction: Traversal direction

        Returns:
            List of paths with nodes and relationships
        """
        start_label_value = self._normalize_label(start_node_label)
        # Build relationship pattern with quantifier
        if relationship_types:
            rel_types = "|".join([self._normalize_relationship(rt) for rt in relationship_types])
            rel_pattern = f":{rel_types}"
        else:
            rel_pattern = ""

        # Limit depth for safety
        depth = min(max(1, depth), settings.graph_traversal_max_depth)

        # Build direction pattern with quantifier inside brackets
        if direction == "outgoing":
            pattern = f"-[{rel_pattern}*1..{depth}]->"
        elif direction == "incoming":
            pattern = f"<-[{rel_pattern}*1..{depth}]-"
        else:  # both
            pattern = f"-[{rel_pattern}*1..{depth}]-"

        query = f"""
        MATCH path = (start:{start_label_value} {{id: $start_id}})
                     {pattern}
                     (end)
        RETURN path,
               start.id AS start_id,
               end.id AS end_id,
               labels(end) AS end_labels,
               length(path) AS depth
        ORDER BY depth
        LIMIT 100
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, start_id=start_node_id)

            paths = []
            for record in result:
                paths.append(
                    {
                        "start_id": record["start_id"],
                        "end_id": record["end_id"],
                        "end_labels": record["end_labels"],
                        "depth": record["depth"],
                        "path": record["path"],
                    }
                )

            return paths

    @resilient_db_operation
    def execute_cypher(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            Query results as list of dictionaries
        """
        params = parameters or {}

        with self.driver.session(database=self.database) as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    @resilient_db_operation
    def get_node_by_id(
        self,
        node_id: str,
        label: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a node by its ID.

        Args:
            node_id: Node ID
            label: Optional node label for more efficient lookup

        Returns:
            Node properties or None if not found
        """
        if label:
            label_value = self._normalize_label(label)
            query = f"MATCH (n:{label_value} {{id: $node_id}}) RETURN n"
        else:
            query = "MATCH (n {id: $node_id}) RETURN n, labels(n) AS labels"

        with self.driver.session(database=self.database) as session:
            result = session.run(query, node_id=node_id)
            record = result.single()

            if record:
                node_data = dict(record["n"])
                if "labels" in record:
                    node_data["labels"] = record["labels"]
                return node_data

            return None

    def _normalize_label(self, label: Any) -> str:
        if hasattr(label, "value"):
            label = label.value
        normalized = self.registry.normalize_label(str(label))
        if not normalized:
            raise ValueError(f"Unknown node label: {label}")
        return normalized

    def _normalize_relationship(self, relationship: Any) -> str:
        if hasattr(relationship, "value"):
            relationship = relationship.value
        normalized = self.registry.normalize_relationship(str(relationship))
        if not normalized:
            raise ValueError(f"Unknown relationship type: {relationship}")
        return normalized

    def delete_all(self) -> int:
        """
        Delete all nodes and relationships (use with caution!).

        Returns:
            Number of nodes deleted
        """
        query = """
        MATCH (n)
        DETACH DELETE n
        RETURN count(n) AS deleted_count
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            record = result.single()
            count = record["deleted_count"] if record else 0
            logger.warning(f"Deleted all {count} nodes from Neo4j database")
            return count
