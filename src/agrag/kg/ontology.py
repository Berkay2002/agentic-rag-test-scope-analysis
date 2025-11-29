"""Knowledge Graph ontology for test scope analysis.

Defines entity types, relationships, and data models for the software engineering
domain focused on telecommunications test scenarios.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


# Enums for controlled vocabularies


class Priority(str, Enum):
    """Requirement priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RequirementStatus(str, Enum):
    """Requirement lifecycle status."""

    DRAFT = "draft"
    APPROVED = "approved"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"
    DEPRECATED = "deprecated"


class TestType(str, Enum):
    """Types of test cases."""

    UNIT = "unit"
    INTEGRATION = "integration"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    PROTOCOL = "protocol"
    REGRESSION = "regression"


class NodeLabel(str, Enum):
    """Neo4j node labels."""

    REQUIREMENT = "Requirement"
    TEST_CASE = "TestCase"
    FUNCTION = "Function"
    CLASS = "Class"
    MODULE = "Module"


class RelationshipType(str, Enum):
    """Neo4j relationship types."""

    VERIFIES = "VERIFIES"  # TestCase -> Requirement
    COVERS = "COVERS"  # TestCase -> Function
    CALLS = "CALLS"  # Function -> Function
    DEFINED_IN = "DEFINED_IN"  # Function -> Class
    INHERITS_FROM = "INHERITS_FROM"  # Class -> Class
    BELONGS_TO = "BELONGS_TO"  # Class/Function -> Module
    DEPENDS_ON = "DEPENDS_ON"  # Module -> Module
    TESTS = "TESTS"  # TestCase -> TestCase (test dependencies)


# Entity Models


class Requirement(BaseModel):
    """Software requirement entity."""

    id: str = Field(..., description="Unique requirement identifier (e.g., REQ_001)")
    description: str = Field(..., description="Requirement text")
    priority: Priority = Field(default=Priority.MEDIUM, description="Priority level")
    status: RequirementStatus = Field(
        default=RequirementStatus.DRAFT, description="Lifecycle status"
    )
    category: Optional[str] = Field(
        None, description="Requirement category (e.g., 'handover', 'signaling')"
    )
    embedding: Optional[List[float]] = Field(None, description="Vector embedding (768-dim)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "REQ_HANDOVER_001",
                "description": "The system SHALL support LTE handover between adjacent cells with latency < 50ms",
                "priority": "high",
                "status": "approved",
                "category": "handover",
            }
        }


class TestCase(BaseModel):
    """Test case entity."""

    id: str = Field(..., description="Unique test case identifier (e.g., TC_001)")
    name: str = Field(..., description="Test case name")
    description: str = Field(..., description="Test case description")
    test_type: TestType = Field(..., description="Type of test")
    file_path: Optional[str] = Field(None, description="Test file path")
    expected_outcome: Optional[str] = Field(None, description="Expected test outcome")
    preconditions: Optional[str] = Field(None, description="Test preconditions")
    steps: Optional[List[str]] = Field(None, description="Test execution steps")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "TC_HANDOVER_001",
                "name": "Test X2 Handover Success",
                "description": "Verify successful X2 handover between eNodeB cells",
                "test_type": "protocol",
                "file_path": "tests/protocol/test_handover.py",
                "expected_outcome": "Handover completes within 50ms",
            }
        }


class Function(BaseModel):
    """Function/method entity."""

    id: str = Field(..., description="Unique function identifier")
    name: str = Field(..., description="Function name")
    signature: str = Field(..., description="Full function signature")
    code_snippet: Optional[str] = Field(None, description="Function implementation code")
    file_path: str = Field(..., description="Source file path")
    line_number: Optional[int] = Field(None, description="Starting line number")
    docstring: Optional[str] = Field(None, description="Function documentation")
    complexity: Optional[int] = Field(None, description="Cyclomatic complexity")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "FUNC_initiate_handover",
                "name": "initiate_handover",
                "signature": "def initiate_handover(source_cell_id: str, target_cell_id: str) -> bool",
                "file_path": "src/network/handover.py",
                "line_number": 45,
            }
        }


class Class(BaseModel):
    """Class entity."""

    id: str = Field(..., description="Unique class identifier")
    name: str = Field(..., description="Class name")
    file_path: str = Field(..., description="Source file path")
    line_number: Optional[int] = Field(None, description="Starting line number")
    methods: List[str] = Field(default_factory=list, description="List of method names")
    base_classes: List[str] = Field(default_factory=list, description="Parent classes")
    docstring: Optional[str] = Field(None, description="Class documentation")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "CLASS_HandoverManager",
                "name": "HandoverManager",
                "file_path": "src/network/handover.py",
                "methods": ["initiate_handover", "verify_handover", "rollback_handover"],
            }
        }


class Module(BaseModel):
    """Module/package entity."""

    id: str = Field(..., description="Unique module identifier")
    name: str = Field(..., description="Module name")
    file_path: str = Field(..., description="Module file path")
    architectural_component: Optional[str] = Field(None, description="Architecture layer/component")
    description: Optional[str] = Field(None, description="Module description")
    imports: List[str] = Field(default_factory=list, description="Imported modules")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "MOD_network_handover",
                "name": "network.handover",
                "file_path": "src/network/handover.py",
                "architectural_component": "network_layer",
            }
        }


# Relationship Models


class Relationship(BaseModel):
    """Base relationship model."""

    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    relationship_type: RelationshipType = Field(..., description="Relationship type")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship properties")


class VerifiesRelationship(Relationship):
    """TestCase verifies Requirement."""

    relationship_type: Literal[RelationshipType.VERIFIES] = RelationshipType.VERIFIES  # type: ignore[assignment]
    coverage_percentage: Optional[float] = Field(None, description="Coverage percentage (0-100)")


class CoversRelationship(Relationship):
    """TestCase covers Function."""

    relationship_type: Literal[RelationshipType.COVERS] = RelationshipType.COVERS  # type: ignore[assignment]
    direct_coverage: bool = Field(True, description="Direct vs transitive coverage")


class CallsRelationship(Relationship):
    """Function calls Function."""

    relationship_type: Literal[RelationshipType.CALLS] = RelationshipType.CALLS  # type: ignore[assignment]
    call_count: Optional[int] = Field(None, description="Number of calls (if known)")


class DefinedInRelationship(Relationship):
    """Function defined in Class."""

    relationship_type: Literal[RelationshipType.DEFINED_IN] = RelationshipType.DEFINED_IN  # type: ignore[assignment]


class InheritsFromRelationship(Relationship):
    """Class inherits from Class."""

    relationship_type: Literal[RelationshipType.INHERITS_FROM] = RelationshipType.INHERITS_FROM  # type: ignore[assignment]


# Graph Schema Constants

NEO4J_CONSTRAINTS = [
    "CREATE CONSTRAINT requirement_id IF NOT EXISTS FOR (r:Requirement) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT testcase_id IF NOT EXISTS FOR (t:TestCase) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT function_id IF NOT EXISTS FOR (f:Function) REQUIRE f.id IS UNIQUE",
    "CREATE CONSTRAINT class_id IF NOT EXISTS FOR (c:Class) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT module_id IF NOT EXISTS FOR (m:Module) REQUIRE m.id IS UNIQUE",
]

NEO4J_VECTOR_INDEXES = [
    """
    CREATE VECTOR INDEX requirement_embeddings IF NOT EXISTS
    FOR (r:Requirement) ON (r.embedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 768,
        `vector.similarity_function`: 'cosine'
    }}
    """,
    """
    CREATE VECTOR INDEX testcase_embeddings IF NOT EXISTS
    FOR (t:TestCase) ON (t.embedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 768,
        `vector.similarity_function`: 'cosine'
    }}
    """,
    """
    CREATE VECTOR INDEX function_embeddings IF NOT EXISTS
    FOR (f:Function) ON (f.embedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 768,
        `vector.similarity_function`: 'cosine'
    }}
    """,
]

POSTGRESQL_SCHEMA = """
-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pg_search extension for true BM25 full-text search
CREATE EXTENSION IF NOT EXISTS pg_search;

-- Document chunks table for vector and keyword search
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Vector similarity index (HNSW via pgvector)
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- BM25 full-text search index (via pg_search/ParadeDB)
-- This provides true BM25 ranking, not TF-IDF like ts_rank_cd
CREATE INDEX IF NOT EXISTS document_chunks_bm25_idx
ON document_chunks USING bm25 (id, chunk_id, content, metadata)
WITH (key_field = 'id');

-- Metadata GIN index for filtering
CREATE INDEX IF NOT EXISTS document_chunks_metadata_idx
ON document_chunks USING gin(metadata);

-- Entity type index for filtering
CREATE INDEX IF NOT EXISTS document_chunks_entity_type_idx
ON document_chunks ((metadata->>'entity_type'));
"""
