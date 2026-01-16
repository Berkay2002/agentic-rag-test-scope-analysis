"""Ontology registry for loading and validating the active KG schema."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agrag.config.settings import settings

_REGISTRY: Optional["OntologyRegistry"] = None


def _default_spec_path() -> Path:
    return Path(__file__).resolve().parent / "specs" / "default.json"


def _to_snake_case(value: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return value.lower()


class OntologyRegistry:
    """Load and validate ontology definitions from a spec file."""

    def __init__(self, spec_path: Optional[str] = None):
        self._spec_path = Path(spec_path) if spec_path else _default_spec_path()
        self._spec = self._load_spec(self._spec_path)

        self.name = self._spec.get("name", "unknown")
        self.schema_version = self._spec.get("schema_version", "unknown")

        self._labels = list(self._spec.get("labels", []))
        self._relationships = list(self._spec.get("relationship_types", []))

        self._label_aliases = dict(self._spec.get("label_aliases", {}))
        self._relationship_aliases = dict(self._spec.get("relationship_aliases", {}))

        self._label_lookup = {label.lower(): label for label in self._labels}
        self._relationship_lookup = {rel.lower(): rel for rel in self._relationships}
        self._label_alias_lookup = {
            alias.lower(): target for alias, target in self._label_aliases.items()
        }
        self._relationship_alias_lookup = {
            alias.lower(): target for alias, target in self._relationship_aliases.items()
        }

        self._id_patterns = dict(self._spec.get("id_patterns", {}))
        self._compiled_patterns = {
            label: re.compile(pattern) for label, pattern in self._id_patterns.items()
        }

        neo4j_spec = self._spec.get("neo4j", {})
        self._neo4j_constraints = list(neo4j_spec.get("constraints", []))
        self._neo4j_vector_indexes = list(neo4j_spec.get("vector_indexes", []))
        self._neo4j_vector_index_names = dict(neo4j_spec.get("vector_index_names", {}))

        postgres_spec = self._spec.get("postgres", {})
        self._postgres_schema = postgres_spec.get("schema_sql", "")

    @staticmethod
    def _load_spec(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Ontology spec not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def spec_path(self) -> Path:
        return self._spec_path

    def labels(self) -> List[str]:
        return list(self._labels)

    def relationship_types(self) -> List[str]:
        return list(self._relationships)

    def id_patterns(self) -> Dict[str, str]:
        return dict(self._id_patterns)

    def normalize_label(self, label: Optional[str]) -> Optional[str]:
        if not label:
            return None
        if label in self._labels:
            return label
        if label in self._label_aliases:
            return self._label_aliases[label]
        label_lower = label.lower()
        if label_lower in self._label_lookup:
            return self._label_lookup[label_lower]
        if label_lower in self._label_alias_lookup:
            return self._label_alias_lookup[label_lower]
        return None

    def normalize_relationship(self, relationship: Optional[str]) -> Optional[str]:
        if not relationship:
            return None
        if relationship in self._relationships:
            return relationship
        if relationship in self._relationship_aliases:
            return self._relationship_aliases[relationship]
        rel_lower = relationship.lower()
        if rel_lower in self._relationship_lookup:
            return self._relationship_lookup[rel_lower]
        if rel_lower in self._relationship_alias_lookup:
            return self._relationship_alias_lookup[rel_lower]
        return None

    def validate_label(self, label: Optional[str]) -> bool:
        return self.normalize_label(label) is not None

    def validate_relationship(self, relationship: Optional[str]) -> bool:
        return self.normalize_relationship(relationship) is not None

    def infer_entity_type(self, entity_id: Optional[str]) -> Optional[str]:
        if not entity_id:
            return None
        for label, pattern in self._compiled_patterns.items():
            if pattern.match(entity_id):
                return label
        return None

    def neo4j_constraints(self) -> List[str]:
        return list(self._neo4j_constraints)

    def neo4j_vector_indexes(self) -> List[str]:
        return list(self._neo4j_vector_indexes)

    def neo4j_vector_index_name(self, label: str) -> str:
        normalized = self.normalize_label(label)
        if not normalized:
            raise ValueError(f"Unknown node label: {label}")
        if normalized in self._neo4j_vector_index_names:
            return self._neo4j_vector_index_names[normalized]
        return f"{_to_snake_case(normalized)}_embeddings"

    def postgres_schema(self) -> str:
        return str(self._postgres_schema)


def get_registry() -> OntologyRegistry:
    """Return a cached ontology registry instance."""
    global _REGISTRY
    if _REGISTRY is None:
        spec_path = settings.ontology_spec_path
        _REGISTRY = OntologyRegistry(spec_path=spec_path)
    return _REGISTRY


def reload_registry(spec_path: Optional[str] = None) -> OntologyRegistry:
    """Reload the registry with a new spec path (useful for tests)."""
    global _REGISTRY
    _REGISTRY = OntologyRegistry(spec_path=spec_path or settings.ontology_spec_path)
    return _REGISTRY
