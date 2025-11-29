"""Neo4j client for graph operations and vector search."""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver, Session
import logging

from agrag.config import settings
from agrag.kg.ontology import (
    NEO4J_CONSTRAINTS,
    NEO4J_VECTOR_INDEXES,
    NodeLabel,
    RelationshipType,
)

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

        self.driver: Driver = GraphDatabase.driver(
            self.uri, auth=(self.username, self.password)
        )
        logger.info(f"Neo4j client initialized for {self.uri}")

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j client closed")

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

        with self.driver.session(database=self.database) as session:
            # Create constraints
            for constraint_query in NEO4J_CONSTRAINTS:
                try:
                    session.run(constraint_query)
                    logger.info(f"Created constraint: {constraint_query[:50]}...")
                except Exception as e:
                    logger.warning(f"Constraint creation failed (may already exist): {e}")

            # Create vector indexes
            for index_query in NEO4J_VECTOR_INDEXES:
                try:
                    session.run(index_query)
                    logger.info("Created vector index")
                except Exception as e:
                    logger.warning(f"Vector index creation failed (may already exist): {e}")

        logger.info("Neo4j schema setup complete")

    def create_node(
        self,
        label: NodeLabel,
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
        query = f"""
        CREATE (n:{label.value})
        SET n = $properties
        RETURN n
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, properties=properties)
            record = result.single()
            return dict(record["n"]) if record else {}

    def create_relationship(
        self,
        source_id: str,
        source_label: NodeLabel,
        target_id: str,
        target_label: NodeLabel,
        relationship_type: RelationshipType,
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
        query = f"""
        MATCH (source:{source_label.value} {{id: $source_id}})
        MATCH (target:{target_label.value} {{id: $target_id}})
        CREATE (source)-[r:{relationship_type.value}]->(target)
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
                    "type": relationship_type.value,
                    "properties": dict(record["r"]),
                }
            return {}

    def vector_search(
        self,
        query_embedding: List[float],
        node_label: NodeLabel,
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
        index_name = f"{node_label.value.lower()}_embeddings"

        query = f"""
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
                results.append({
                    "node": node_data,
                    "score": record["score"],
                    "id": node_data.get("id"),
                    "label": node_label.value,
                })

            return results

    def graph_traverse(
        self,
        start_node_id: str,
        start_node_label: NodeLabel,
        relationship_types: Optional[List[RelationshipType]] = None,
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
        # Build relationship pattern with quantifier
        if relationship_types:
            rel_types = "|".join([rt.value for rt in relationship_types])
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
        MATCH path = (start:{start_node_label.value} {{id: $start_id}})
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
                paths.append({
                    "start_id": record["start_id"],
                    "end_id": record["end_id"],
                    "end_labels": record["end_labels"],
                    "depth": record["depth"],
                    "path": record["path"],
                })

            return paths

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

    def get_node_by_id(
        self,
        node_id: str,
        label: Optional[NodeLabel] = None,
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
            query = f"MATCH (n:{label.value} {{id: $node_id}}) RETURN n"
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
