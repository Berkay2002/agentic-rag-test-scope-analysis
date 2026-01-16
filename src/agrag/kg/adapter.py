"""Source adapters for mapping external data into the canonical ontology."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from agrag.kg.registry import OntologyRegistry, get_registry


class SourceAdapter:
    """Default adapter that normalizes external data to the active ontology."""

    def __init__(
        self,
        registry: Optional[OntologyRegistry] = None,
        source_system: str = "synthetic",
        schema_version: Optional[str] = None,
    ):
        self.registry = registry or get_registry()
        self.source_system = source_system
        self.schema_version = schema_version or self.registry.schema_version

    def normalize_entity(self, entity: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Normalize a raw entity dict to canonical label + provenance."""
        raw_type = entity.get("entity_type") or entity.get("type") or entity.get("label")
        entity_type = None
        if raw_type:
            entity_type = self.registry.normalize_label(str(raw_type))
        if not entity_type:
            entity_type = self.registry.infer_entity_type(entity.get("id"))
        if not entity_type:
            raise ValueError(f"Unable to infer entity type for id={entity.get('id')}")

        normalized = dict(entity)
        normalized["entity_type"] = entity_type
        normalized["source_system"] = self.source_system
        normalized["schema_version"] = self.schema_version
        normalized["raw_type"] = raw_type or entity_type
        return normalized, entity_type

    def normalize_relationship(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a raw relationship dict to canonical type + provenance."""
        raw_type = relationship.get("relationship_type") or relationship.get("type")
        normalized_type = None
        if raw_type:
            normalized_type = self.registry.normalize_relationship(str(raw_type))
        if not normalized_type:
            raise ValueError(
                f"Unable to infer relationship type for relationship={relationship}"
            )

        normalized = dict(relationship)
        normalized["relationship_type"] = normalized_type
        normalized["source_system"] = self.source_system
        normalized["schema_version"] = self.schema_version
        normalized["raw_type"] = raw_type or normalized_type

        properties = normalized.get("properties")
        if not isinstance(properties, dict):
            properties = {}
        properties.setdefault("source_system", self.source_system)
        properties.setdefault("schema_version", self.schema_version)
        properties.setdefault("raw_type", raw_type or normalized_type)
        normalized["properties"] = properties

        return normalized
