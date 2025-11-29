"""TGF (Test Governance Framework) CSV loader for Ericsson test scope data.

Loads and parses test execution results from Ericsson's internal testing systems
into the AgRAG knowledge graph format.
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator

from agrag.data.loaders.base import BaseLoader, Document
from agrag.kg.ontology import TestType, NodeLabel, RelationshipType

logger = logging.getLogger(__name__)


class TGFTestRecord(BaseModel):
    """Represents a single test execution record from TGF CSV."""

    test_id: str = Field(..., description="Unique test case identifier")
    test_suite: str = Field(..., description="Test suite/group name")
    test_name: str = Field(..., description="Descriptive test name")
    test_type: str = Field(..., description="Test category (unit/integration/system)")
    feature_area: str = Field(..., description="Feature or component being tested")
    sub_feature: Optional[str] = Field(None, description="Sub-feature or module")
    requirement_ids: List[str] = Field(default_factory=list, description="Related requirement IDs")
    function_names: List[str] = Field(default_factory=list, description="Functions under test")
    result: str = Field(..., description="Test result (PASS/FAIL/SKIP/ERROR)")
    execution_time_ms: Optional[int] = Field(None, description="Execution time in milliseconds")
    timestamp: Optional[str] = Field(None, description="Execution timestamp (ISO 8601)")
    failure_reason: Optional[str] = Field(None, description="Failure description if result != PASS")
    test_file_path: Optional[str] = Field(None, description="Path to test file")
    code_coverage_pct: Optional[float] = Field(None, description="Code coverage percentage")
    priority: str = Field(default="medium", description="Test priority (critical/high/medium/low)")
    tags: List[str] = Field(default_factory=list, description="Additional test tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @validator("requirement_ids", "function_names", "tags", pre=True)
    def parse_semicolon_list(cls, v):
        """Parse semicolon-separated strings into lists."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(";") if item.strip()]
        return v or []

    @validator("test_type")
    def normalize_test_type(cls, v):
        """Normalize test type to standard values."""
        type_mapping = {
            "unit": TestType.UNIT.value,
            "integration": TestType.INTEGRATION.value,
            "system": TestType.SYSTEM.value,
            "performance": TestType.PERFORMANCE.value,
            "protocol": TestType.PROTOCOL.value,
            "regression": TestType.REGRESSION.value,
            "functional": TestType.INTEGRATION.value,
            "acceptance": TestType.SYSTEM.value,
        }
        return type_mapping.get(v.lower(), v.lower())

    @validator("result")
    def normalize_result(cls, v):
        """Normalize test result values."""
        result_mapping = {
            "passed": "PASS",
            "failed": "FAIL",
            "skipped": "SKIP",
            "error": "ERROR",
            "blocked": "SKIP",
            "not_run": "SKIP",
        }
        return result_mapping.get(v.lower(), v.upper())

    def to_test_case_entity(self) -> Dict[str, Any]:
        """Convert TGF record to TestCase entity format."""
        return {
            "id": self.test_id,
            "name": self.test_name,
            "description": f"{self.test_suite}: {self.test_name}",
            "test_type": self.test_type,
            "file_path": self.test_file_path,
            "expected_outcome": "PASS" if self.result == "PASS" else self.failure_reason,
            "metadata": {
                "test_suite": self.test_suite,
                "feature_area": self.feature_area,
                "sub_feature": self.sub_feature,
                "result": self.result,
                "execution_time_ms": self.execution_time_ms,
                "timestamp": self.timestamp,
                "code_coverage_pct": self.code_coverage_pct,
                "priority": self.priority,
                "tags": self.tags,
                "failure_reason": self.failure_reason,
                **self.metadata,
            },
        }


class TGFCSVLoader(BaseLoader):
    """Loads test execution data from Ericsson TGF CSV exports."""

    def __init__(
        self,
        file_path: str,
        encoding: str = "utf-8",
        skip_header: bool = True,
        filter_results: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize TGF CSV loader.

        Args:
            file_path: Path to TGF CSV file
            encoding: File encoding (default: utf-8)
            skip_header: Whether CSV has header row (default: True)
            filter_results: Filter by test results (e.g., ["FAIL", "ERROR"])
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.skip_header = skip_header
        self.filter_results = filter_results
        self.records: List[TGFTestRecord] = []

    def load(self) -> List[Document]:
        """
        Load TGF CSV and convert to Document objects.

        Returns:
            List of Document objects with test case data and relationships
        """
        if not self.validate_path(self.file_path):
            raise FileNotFoundError(f"TGF CSV file not found: {self.file_path}")

        self.logger.info(f"Loading TGF data from {self.file_path}")

        try:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                reader = csv.DictReader(f) if self.skip_header else csv.reader(f)

                for row_idx, row in enumerate(reader, start=1):
                    try:
                        record = self._parse_row(row)

                        # Apply result filter if specified
                        if self.filter_results and record.result not in self.filter_results:
                            continue

                        self.records.append(record)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse row {row_idx}: {e}. Skipping.")
                        continue

            self.logger.info(f"Loaded {len(self.records)} test records from TGF CSV")

            # Convert records to Documents
            documents = self._create_documents()
            self.logger.info(f"Created {len(documents)} documents from TGF data")

            return documents

        except Exception as e:
            self.logger.error(f"Error loading TGF CSV: {e}")
            raise

    def _parse_row(self, row: Dict[str, str]) -> TGFTestRecord:
        """Parse a CSV row into a TGFTestRecord."""
        # Handle optional fields with defaults
        return TGFTestRecord(
            test_id=row["test_id"],
            test_suite=row["test_suite"],
            test_name=row["test_name"],
            test_type=row["test_type"],
            feature_area=row["feature_area"],
            sub_feature=row.get("sub_feature"),
            requirement_ids=row.get("requirement_ids", ""),
            function_names=row.get("function_names", ""),
            result=row["result"],
            execution_time_ms=(
                int(row["execution_time_ms"]) if row.get("execution_time_ms") else None
            ),
            timestamp=row.get("timestamp"),
            failure_reason=row.get("failure_reason"),
            test_file_path=row.get("test_file_path"),
            code_coverage_pct=(
                float(row["code_coverage_pct"]) if row.get("code_coverage_pct") else None
            ),
            priority=row.get("priority", "medium"),
            tags=row.get("tags", ""),
            metadata={k: v for k, v in row.items() if k not in TGFTestRecord.__fields__ and v},
        )

    def _create_documents(self) -> List[Document]:
        """
        Convert TGF records to Document objects with structured metadata.

        Creates documents representing test cases and their relationships
        to requirements and functions.
        """
        documents = []

        for record in self.records:
            # Create main test case document
            test_entity = record.to_test_case_entity()

            # Create content for embedding
            content_parts = [
                f"Test Case: {record.test_name}",
                f"Suite: {record.test_suite}",
                f"Type: {record.test_type}",
                f"Feature: {record.feature_area}",
            ]

            if record.sub_feature:
                content_parts.append(f"Sub-feature: {record.sub_feature}")

            if record.requirement_ids:
                content_parts.append(f"Requirements: {', '.join(record.requirement_ids)}")

            if record.function_names:
                content_parts.append(f"Functions: {', '.join(record.function_names)}")

            if record.failure_reason:
                content_parts.append(f"Failure: {record.failure_reason}")

            content = "\n".join(content_parts)

            # Document metadata includes entity data + relationships
            metadata = {
                "chunk_id": f"test_{record.test_id}",
                "entity_type": NodeLabel.TEST_CASE.value,
                "entity": test_entity,
                "relationships": self._extract_relationships(record),
            }

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        return documents

    def _extract_relationships(self, record: TGFTestRecord) -> List[Dict[str, Any]]:
        """
        Extract relationships from test record.

        Returns:
            List of relationship dicts with type, target node, and properties
        """
        relationships = []

        # TestCase -[:VERIFIES]-> Requirement
        for req_id in record.requirement_ids:
            relationships.append(
                {
                    "type": RelationshipType.VERIFIES.value,
                    "target_label": NodeLabel.REQUIREMENT.value,
                    "target_id": req_id,
                    "properties": {
                        "verified_at": record.timestamp,
                        "result": record.result,
                    },
                }
            )

        # TestCase -[:COVERS]-> Function
        for func_name in record.function_names:
            func_id = f"FUNC_{func_name}"
            relationships.append(
                {
                    "type": RelationshipType.COVERS.value,
                    "target_label": NodeLabel.FUNCTION.value,
                    "target_id": func_id,
                    "properties": {
                        "coverage_pct": record.code_coverage_pct,
                        "execution_time_ms": record.execution_time_ms,
                    },
                }
            )

        return relationships

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded TGF data."""
        if not self.records:
            return {}

        result_counts = {}
        for record in self.records:
            result_counts[record.result] = result_counts.get(record.result, 0) + 1

        feature_areas = set(r.feature_area for r in self.records)
        test_types = set(r.test_type for r in self.records)

        total_requirements = sum(len(r.requirement_ids) for r in self.records)
        total_functions = sum(len(r.function_names) for r in self.records)

        return {
            "total_tests": len(self.records),
            "result_distribution": result_counts,
            "unique_feature_areas": len(feature_areas),
            "unique_test_types": len(test_types),
            "total_requirement_links": total_requirements,
            "total_function_links": total_functions,
            "avg_requirements_per_test": total_requirements / len(self.records),
            "avg_functions_per_test": total_functions / len(self.records),
            "feature_areas": sorted(list(feature_areas)),
            "test_types": sorted(list(test_types)),
        }
