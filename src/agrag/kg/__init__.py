"""Knowledge Graph ontology package."""

from .ontology import (
    # Enums
    Priority,
    RequirementStatus,
    TestType,
    NodeLabel,
    RelationshipType,
    # Entity Models
    ChangeRequest,
    File,
    Component,
    Requirement,
    TestCase,
    Function,
    Class,
    Module,
    # Relationship Models
    Relationship,
    VerifiesRelationship,
    CoversRelationship,
    CallsRelationship,
    DefinedInRelationship,
    InheritsFromRelationship,
    # Schema Constants
    NEO4J_CONSTRAINTS,
    NEO4J_VECTOR_INDEXES,
    POSTGRESQL_SCHEMA,
)
from .registry import OntologyRegistry, get_registry, reload_registry
from .adapter import SourceAdapter

__all__ = [
    "Priority",
    "RequirementStatus",
    "TestType",
    "NodeLabel",
    "RelationshipType",
    "ChangeRequest",
    "File",
    "Component",
    "Requirement",
    "TestCase",
    "Function",
    "Class",
    "Module",
    "Relationship",
    "VerifiesRelationship",
    "CoversRelationship",
    "CallsRelationship",
    "DefinedInRelationship",
    "InheritsFromRelationship",
    "NEO4J_CONSTRAINTS",
    "NEO4J_VECTOR_INDEXES",
    "POSTGRESQL_SCHEMA",
    "OntologyRegistry",
    "get_registry",
    "reload_registry",
    "SourceAdapter",
]
