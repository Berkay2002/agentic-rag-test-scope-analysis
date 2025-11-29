"""Synthetic data generator for telecommunications test scope analysis."""

import random
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional
from agrag.kg.ontology import (
    Priority,
    RequirementStatus,
    TestType,
    Requirement,
    TestCase,
    Function,
    Class,
    Module,
)
from agrag.models.embeddings import get_embedding_service
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# TGF-Compatible Constants
# ============================================================================

SUITE_MAPPING = {
    "handover": "Handover Tests",
    "authentication": "Authentication Tests",
    "signaling": "RRC Signaling Tests",
    "protocol": "Protocol Conformance Tests",
    "integration": "Integration Tests",
    "performance": "Performance Tests",
    "security": "Security Tests",
    "mobility": "Mobility Management Tests",
    "data": "Data Session Tests",
    "regression": "Regression Suite",
}

FEATURE_HIERARCHY = {
    "handover": {
        "area": "Handover",
        "sub_features": ["X2", "S1", "Inter-RAT", "Intra-Freq", "Inter-Freq"],
    },
    "authentication": {
        "area": "Authentication",
        "sub_features": ["UserAuth", "TokenRefresh", "SSO", "MFA", "CertValidation"],
    },
    "signaling": {
        "area": "RRC",
        "sub_features": ["Connection", "Release", "Reconfiguration", "Measurement"],
    },
    "protocol": {
        "area": "Protocol",
        "sub_features": ["Conformance", "Interoperability", "Validation", "Compliance"],
    },
    "integration": {
        "area": "Integration",
        "sub_features": ["E2E", "API", "Component", "System"],
    },
    "performance": {
        "area": "Performance",
        "sub_features": ["Throughput", "Latency", "Capacity", "Stress"],
    },
    "security": {
        "area": "Security",
        "sub_features": ["InputValidation", "OutputEncoding", "AccessControl", "Encryption"],
    },
    "mobility": {
        "area": "Mobility",
        "sub_features": ["CellReselection", "Paging", "TAU", "Idle"],
    },
    "data": {
        "area": "DataSession",
        "sub_features": ["PDN", "Bearer", "QoS", "APN"],
    },
}

FAILURE_TEMPLATES = {
    "handover": [
        "Handover timeout after {time}ms",
        "Target cell not found: {cell_id}",
        "X2 connection refused by target eNodeB",
        "UE context transfer failed: {error_code}",
        "Handover preparation timeout",
        "Source cell release failure",
    ],
    "authentication": [
        "Token validation failed: expired",
        "Invalid credentials for user {user_id}",
        "MFA challenge timeout",
        "Certificate chain validation error",
        "Session token revoked",
        "Authentication vector mismatch",
    ],
    "signaling": [
        "RRC message parsing error at offset {offset}",
        "Invalid state transition: {from_state} -> {to_state}",
        "Measurement report timeout",
        "Signaling connection lost",
        "Message sequence number mismatch",
    ],
    "protocol": [
        "Protocol version mismatch: expected {expected}, got {actual}",
        "Invalid message format",
        "Mandatory IE missing: {ie_name}",
        "Sequence violation detected",
    ],
    "integration": [
        "Component integration timeout",
        "API response validation failed",
        "End-to-end path verification failed",
        "Service dependency unavailable: {service}",
    ],
    "performance": [
        "Throughput below threshold: {actual}Mbps < {expected}Mbps",
        "Latency exceeded: {actual}ms > {threshold}ms",
        "Connection dropped under load",
        "Memory threshold exceeded: {usage}%",
        "CPU utilization spike: {cpu}%",
    ],
    "security": [
        "SQL injection vulnerability detected",
        "XSS payload not sanitized: {payload}",
        "Unauthorized access to {resource}",
        "Encryption key validation failed",
        "Access control bypass detected",
    ],
    "mobility": [
        "Cell reselection timeout",
        "Paging response failure",
        "TAU rejection: cause {cause}",
        "Idle mode measurement gap",
    ],
    "data": [
        "PDN connection establishment failed",
        "Bearer modification rejected",
        "QoS negotiation failed: {qos_class}",
        "APN resolution error",
    ],
}

# Execution time ranges per test type (in milliseconds)
EXECUTION_TIME_RANGES = {
    TestType.UNIT: (10, 100),
    TestType.INTEGRATION: (100, 500),
    TestType.SYSTEM: (500, 5000),
    TestType.PERFORMANCE: (5000, 30000),
    TestType.REGRESSION: (50, 200),
    TestType.PROTOCOL: (200, 1000),
}

# Result distribution: 60% PASS, 25% FAIL, 10% ERROR, 5% SKIP
RESULT_DISTRIBUTION = [
    ("PASS", 0.60),
    ("FAIL", 0.25),
    ("ERROR", 0.10),
    ("SKIP", 0.05),
]

# Priority distribution: 10% critical, 30% high, 40% medium, 20% low
PRIORITY_DISTRIBUTION = [
    ("critical", 0.10),
    ("high", 0.30),
    ("medium", 0.40),
    ("low", 0.20),
]


class QueryDifficulty(str, Enum):
    """Difficulty levels for evaluation queries."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    NEGATIVE = "negative"


QUERY_PARAPHRASES = {
    "find_tests_for_requirement": [
        "What tests verify {req_id}?",
        "Find test cases that cover requirement {req_id}",
        "Which tests are linked to {req_id}?",
        "Show tests verifying {req_id}",
        "Tests for requirement {req_id}",
    ],
    "find_failed_tests": [
        "What tests failed?",
        "Show me failing test cases",
        "Which tests have failures?",
        "List tests with FAIL result",
        "Find failed tests",
    ],
    "find_tests_for_feature": [
        "Tests for {feature} feature",
        "What tests cover {feature}?",
        "Find test cases related to {feature}",
        "Show {feature} tests",
        "{feature} test coverage",
    ],
    "find_tests_by_result": [
        "Tests with {result} result",
        "Show {result} tests",
        "List test cases that {result_past}",
        "Find tests that {result_past}",
    ],
    "find_tests_by_priority": [
        "Find {priority} priority tests",
        "Show {priority} tests",
        "List test cases with {priority} priority",
        "{priority} priority test cases",
    ],
    "find_tests_covering_function": [
        "What tests cover function {func_name}?",
        "Tests covering {func_name}",
        "Find tests that exercise {func_name}",
        "Which test cases invoke {func_name}?",
        "Show tests for function {func_name}",
    ],
    "find_functions_covered_by_test": [
        "What functions does {test_id} cover?",
        "Functions covered by test {test_id}",
        "Which functions are exercised by {test_id}?",
        "Show code coverage for {test_id}",
    ],
    "find_tests_in_suite": [
        "Tests in {suite} suite",
        "Show tests from {suite}",
        "List test cases in {suite}",
        "What tests belong to {suite}?",
    ],
    "find_requirements_for_test": [
        "What requirements does {test_id} verify?",
        "Requirements covered by {test_id}",
        "Which requirements are tested by {test_id}?",
    ],
    "find_tests_by_type": [
        "Find {test_type} tests",
        "Show {test_type} test cases",
        "List all {test_type} tests",
        "{test_type} tests in the system",
    ],
}

NEGATIVE_QUERIES = [
    {
        "query": "Tests for quantum encryption feature",
        "relevant_ids": [],
        "difficulty": "negative",
        "query_type": "nonexistent_feature",
        "rationale": "Quantum encryption doesn't exist in telecom domain",
    },
    {
        "query": "Find test case TC_BLOCKCHAIN_001",
        "relevant_ids": [],
        "difficulty": "negative",
        "query_type": "nonexistent_entity",
        "rationale": "No blockchain-related tests in dataset",
    },
    {
        "query": "Tests verifying REQ_AI_ML_001",
        "relevant_ids": [],
        "difficulty": "negative",
        "query_type": "nonexistent_requirement",
        "rationale": "No AI/ML requirements in telecom ontology",
    },
    {
        "query": "Functions covered by GUI tests",
        "relevant_ids": [],
        "difficulty": "negative",
        "query_type": "nonexistent_test_type",
        "rationale": "No GUI tests in telecom system tests",
    },
    {
        "query": "Tests for IPv7 protocol support",
        "relevant_ids": [],
        "difficulty": "negative",
        "query_type": "fictional_protocol",
        "rationale": "IPv7 doesn't exist",
    },
    {
        "query": "Critical tests that all passed with 100% coverage",
        "relevant_ids": [],
        "difficulty": "negative",
        "query_type": "impossible_condition",
        "rationale": "Synthetic data doesn't generate 100% coverage",
    },
    {
        "query": "Tests for 6G network slicing",
        "relevant_ids": [],
        "difficulty": "negative",
        "query_type": "future_technology",
        "rationale": "6G is not yet standardized, not in dataset",
    },
    {
        "query": "Find tests in the 'Deprecated Tests' suite",
        "relevant_ids": [],
        "difficulty": "negative",
        "query_type": "nonexistent_suite",
        "rationale": "No deprecated test suite exists",
    },
]


class TelecomDataGenerator:
    """Generate synthetic telecommunications test data."""

    def __init__(self):
        """Initialize the generator."""
        self.embedding_service = get_embedding_service()

        # Telecommunications domain concepts
        self.protocols = ["LTE", "5G NR", "X2", "S1", "NGAP", "GTP"]
        self.network_elements = ["eNodeB", "gNodeB", "MME", "SGW", "PGW", "AMF", "SMF"]
        self.functions_domain = [
            "handover",
            "authentication",
            "signaling",
            "session",
            "bearer",
            "measurement",
            "configuration",
            "monitoring",
            "alarm",
            "logging",
        ]
        self.test_scenarios = [
            "success",
            "timeout",
            "failure",
            "retry",
            "rollback",
            "boundary",
            "stress",
            "latency",
            "throughput",
        ]

        # Extended categories for TGF compatibility
        self.categories = list(FEATURE_HIERARCHY.keys())

    def _generate_requirement_id(self, category: str, index: int) -> str:
        """Generate a requirement ID."""
        return f"REQ_{category.upper()}_{index:03d}"

    def _generate_test_id(self, category: str, index: int) -> str:
        """Generate a test case ID."""
        return f"TC_{category.upper()}_{index:03d}"

    def _generate_function_id(self, name: str) -> str:
        """Generate a function ID."""
        return f"FUNC_{name}"

    def _generate_class_id(self, name: str) -> str:
        """Generate a class ID."""
        return f"CLASS_{name}"

    def _generate_module_id(self, name: str) -> str:
        """Generate a module ID."""
        return f"MOD_{name}"

    def _generate_timestamp(self, days_back: int = 30) -> str:
        """
        Generate a random ISO 8601 timestamp within the specified number of days.

        Args:
            days_back: Maximum number of days in the past

        Returns:
            ISO 8601 formatted timestamp string
        """
        now = datetime.now()
        random_days = random.uniform(0, days_back)
        random_time = now - timedelta(days=random_days)
        return random_time.isoformat()

    def _generate_result(self) -> str:
        """
        Generate a test result based on weighted distribution.

        Distribution: 60% PASS, 25% FAIL, 10% ERROR, 5% SKIP

        Returns:
            Result string (PASS, FAIL, ERROR, or SKIP)
        """
        roll = random.random()
        cumulative = 0.0
        for result, probability in RESULT_DISTRIBUTION:
            cumulative += probability
            if roll < cumulative:
                return result
        return "PASS"  # Fallback

    def _generate_priority_weighted(self) -> str:
        """
        Generate a priority based on weighted distribution.

        Distribution: 10% critical, 30% high, 40% medium, 20% low

        Returns:
            Priority string
        """
        roll = random.random()
        cumulative = 0.0
        for priority, probability in PRIORITY_DISTRIBUTION:
            cumulative += probability
            if roll < cumulative:
                return priority
        return "medium"  # Fallback

    def _generate_execution_time(self, test_type: TestType) -> int:
        """
        Generate execution time based on test type.

        Args:
            test_type: The type of test (unit, integration, etc.)

        Returns:
            Execution time in milliseconds
        """
        min_time, max_time = EXECUTION_TIME_RANGES.get(test_type, (50, 500))
        return random.randint(min_time, max_time)

    def _generate_failure_reason(self, category: str, result: str) -> Optional[str]:
        """
        Generate a failure reason for FAIL/ERROR results.

        Args:
            category: Test category
            result: Test result (PASS, FAIL, ERROR, SKIP)

        Returns:
            Failure reason string or None for PASS/SKIP
        """
        if result not in ("FAIL", "ERROR"):
            return None

        templates = FAILURE_TEMPLATES.get(category, FAILURE_TEMPLATES.get("integration", []))
        if not templates:
            return "Unknown error occurred"

        template = random.choice(templates)

        # Fill in template placeholders with random values
        replacements = {
            "{time}": str(random.randint(1000, 30000)),
            "{cell_id}": f"CELL_{random.randint(100, 999)}",
            "{error_code}": f"ERR_{random.randint(1000, 9999)}",
            "{user_id}": f"USER_{random.randint(1000, 9999)}",
            "{offset}": str(random.randint(0, 255)),
            "{from_state}": random.choice(["IDLE", "CONNECTED", "INACTIVE"]),
            "{to_state}": random.choice(["ACTIVE", "RELEASED", "SUSPENDED"]),
            "{expected}": str(random.randint(100, 1000)),
            "{actual}": str(random.randint(10, 99)),
            "{threshold}": str(random.randint(50, 200)),
            "{usage}": str(random.randint(85, 99)),
            "{cpu}": str(random.randint(90, 100)),
            "{payload}": "<script>alert('xss')</script>",
            "{resource}": random.choice(["/admin", "/config", "/logs"]),
            "{service}": random.choice(["auth-service", "db-service", "cache-service"]),
            "{ie_name}": random.choice(["UE-Identity", "Cause", "Target-Cell-ID"]),
            "{cause}": str(random.randint(1, 15)),
            "{qos_class}": random.choice(["QCI-1", "QCI-5", "QCI-9"]),
        }

        reason = template
        for key, value in replacements.items():
            reason = reason.replace(key, value)

        return reason

    def _generate_code_coverage(self, result: str) -> float:
        """
        Generate code coverage percentage based on result.

        Args:
            result: Test result

        Returns:
            Code coverage percentage (0.0 to 100.0)
        """
        if result == "SKIP":
            return 0.0
        # 40-98% for PASS/FAIL/ERROR
        return round(random.uniform(40.0, 98.0), 1)

    def _generate_tags(
        self, category: str, scenario: str, result: str, test_type: TestType
    ) -> List[str]:
        """
        Generate tags for a test case.

        Args:
            category: Test category
            scenario: Test scenario
            result: Test result
            test_type: Type of test

        Returns:
            List of tag strings
        """
        tags = [category, scenario, test_type.value.lower()]

        # Add result-based tags
        if result == "FAIL":
            tags.append("needs_fix")
        elif result == "ERROR":
            tags.append("infrastructure_issue")
        elif result == "SKIP":
            tags.append("blocked")

        # Add random feature-based tags
        if random.random() < 0.3:
            tags.append("regression")
        if random.random() < 0.2:
            tags.append("critical_path")
        if random.random() < 0.15:
            tags.append("flaky")

        return list(set(tags))  # Remove duplicates

    def generate_requirements(self, count: int = 50) -> List[Dict[str, Any]]:
        """
        Generate requirement entities.

        Args:
            count: Number of requirements to generate

        Returns:
            List of requirement dictionaries
        """
        logger.info(f"Generating {count} requirements...")
        requirements = []

        categories = ["handover", "authentication", "signaling", "bearer", "measurement"]

        for i in range(count):
            category = random.choice(categories)
            protocol = random.choice(self.protocols)
            element = random.choice(self.network_elements)

            req_id = self._generate_requirement_id(category, i + 1)

            # Generate realistic telecom requirement descriptions
            descriptions = {
                "handover": f"The system SHALL support {protocol} handover between adjacent {element} cells with latency < 50ms",
                "authentication": f"The {element} SHALL authenticate UE using {protocol} authentication procedures",
                "signaling": f"The system SHALL process {protocol} signaling messages with error rate < 0.01%",
                "bearer": f"The {element} SHALL establish {protocol} bearer connections within 100ms",
                "measurement": f"The system SHALL collect {protocol} measurement reports every 200ms",
            }

            description = descriptions[category]

            # Generate embedding
            embedding = self.embedding_service.embed_query(description)

            req = Requirement(
                id=req_id,
                description=description,
                priority=random.choice(list(Priority)),
                status=random.choice(list(RequirementStatus)),
                category=category,
                embedding=embedding,
                metadata={
                    "protocol": protocol,
                    "network_element": element,
                    "generated": True,
                },
            )

            requirements.append(req.model_dump())

        logger.info(f"Generated {len(requirements)} requirements")
        return requirements

    def generate_test_cases(
        self, count: int = 200, requirements: List[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate test case entities with TGF-compatible fields.

        Args:
            count: Number of test cases to generate
            requirements: List of requirements to link to

        Returns:
            List of test case dictionaries with TGF metadata
        """
        logger.info(f"Generating {count} test cases...")
        test_cases = []

        # Use extended categories for better coverage
        categories = list(FEATURE_HIERARCHY.keys())

        for i in range(count):
            category = random.choice(categories)
            scenario = random.choice(self.test_scenarios)
            protocol = random.choice(self.protocols)
            test_type = random.choice(list(TestType))

            test_id = self._generate_test_id(category, i + 1)
            name = f"Test {category.title()} {scenario.title()}"

            # Generate realistic test descriptions
            description = (
                f"Verify {category} behavior during {scenario} scenario using {protocol} protocol"
            )

            # Generate embedding
            embedding = self.embedding_service.embed_query(f"{name} {description}")

            # Generate TGF-compatible fields
            result = self._generate_result()
            priority = self._generate_priority_weighted()
            execution_time_ms = self._generate_execution_time(test_type)
            failure_reason = self._generate_failure_reason(category, result)
            code_coverage_pct = self._generate_code_coverage(result)
            tags = self._generate_tags(category, scenario, result, test_type)
            timestamp = self._generate_timestamp()

            # Get feature area info
            feature_info = FEATURE_HIERARCHY.get(
                category, {"area": category.title(), "sub_features": ["Default"]}
            )
            feature_area = feature_info["area"]
            sub_feature = random.choice(feature_info["sub_features"])
            test_suite = SUITE_MAPPING.get(category, "General Tests")

            test = TestCase(
                id=test_id,
                name=name,
                description=description,
                test_type=test_type,
                file_path=f"tests/{category}/test_{category}_{scenario}.py",
                expected_outcome=f"{scenario.title()} case handled correctly",
                preconditions=f"System initialized with {protocol} configuration",
                steps=[
                    f"Initialize {protocol} connection",
                    f"Trigger {category} {scenario}",
                    "Verify expected outcome",
                    "Cleanup resources",
                ],
                embedding=embedding,
                metadata={
                    # Original fields
                    "protocol": protocol,
                    "scenario": scenario,
                    "generated": True,
                    # TGF-compatible fields
                    "test_suite": test_suite,
                    "feature_area": feature_area,
                    "sub_feature": sub_feature,
                    "result": result,
                    "execution_time_ms": execution_time_ms,
                    "timestamp": timestamp,
                    "failure_reason": failure_reason,
                    "code_coverage_pct": code_coverage_pct,
                    "priority": priority,
                    "tags": tags,
                    "requirement_ids": [],  # Will be populated in generate_relationships
                    "function_names": [],  # Will be populated in generate_relationships
                    "category": category,  # Store for relationship matching
                },
            )

            test_cases.append(test.model_dump())

        logger.info(f"Generated {len(test_cases)} test cases")
        return test_cases

    def generate_code_entities(self) -> Dict[str, List[Dict]]:
        """
        Generate functions, classes, and modules.

        Returns:
            Dictionary with 'functions', 'classes', 'modules' lists
        """
        logger.info("Generating code entities...")

        # Generate modules
        modules = []
        module_names = [
            "network.handover",
            "network.authentication",
            "network.signaling",
            "core.bearer",
            "core.session",
            "utils.measurement",
            "utils.logging",
        ]

        for mod_name in module_names:
            component = mod_name.split(".")[0]
            mod_id = self._generate_module_id(mod_name.replace(".", "_"))

            description = f"Module for {mod_name.split('.')[-1]} functionality"
            embedding = self.embedding_service.embed_query(description)

            module = Module(
                id=mod_id,
                name=mod_name,
                file_path=f"src/{mod_name.replace('.', '/')}.py",
                architectural_component=component,
                description=description,
                imports=[],
                embedding=embedding,
                metadata={"generated": True},
            )
            modules.append(module.model_dump())

        # Generate classes
        classes = []
        class_templates = [
            (
                "HandoverManager",
                "network.handover",
                ["initiate_handover", "verify_handover", "rollback_handover"],
            ),
            (
                "AuthenticationHandler",
                "network.authentication",
                ["authenticate_ue", "verify_credentials", "refresh_token"],
            ),
            (
                "SignalingProcessor",
                "network.signaling",
                ["process_message", "validate_message", "route_message"],
            ),
            (
                "BearerManager",
                "core.bearer",
                ["establish_bearer", "modify_bearer", "release_bearer"],
            ),
            (
                "SessionController",
                "core.session",
                ["create_session", "update_session", "terminate_session"],
            ),
            (
                "MeasurementCollector",
                "utils.measurement",
                ["collect_measurements", "aggregate_data", "export_metrics"],
            ),
            ("Logger", "utils.logging", ["log_event", "log_error", "flush_logs"]),
        ]

        for class_name, module_name, methods in class_templates:
            class_id = self._generate_class_id(class_name)
            file_path = f"src/{module_name.replace('.', '/')}.py"

            docstring = f"{class_name} class for managing {module_name.split('.')[-1]} operations"
            embedding = self.embedding_service.embed_query(docstring)

            cls = Class(
                id=class_id,
                name=class_name,
                file_path=file_path,
                line_number=random.randint(10, 100),
                methods=methods,
                base_classes=[],
                docstring=docstring,
                embedding=embedding,
                metadata={"module": module_name, "generated": True},
            )
            classes.append(cls.model_dump())

        # Generate functions
        functions = []
        function_templates = []

        # Add methods from classes as functions
        for cls in classes:
            for method in cls["methods"]:
                function_templates.append(
                    (
                        method,
                        cls["file_path"],
                        f"def {method}(self, *args, **kwargs) -> bool",
                        cls["name"],
                    )
                )

        # Add standalone utility functions
        utility_functions = [
            (
                "calculate_latency",
                "utils.measurement",
                "def calculate_latency(start_time: float, end_time: float) -> float",
            ),
            (
                "format_message",
                "network.signaling",
                "def format_message(msg_type: str, payload: dict) -> bytes",
            ),
            ("validate_cell_id", "network.handover", "def validate_cell_id(cell_id: str) -> bool"),
        ]

        for func_name, module_name, signature in utility_functions:
            function_templates.append(
                (func_name, f"src/{module_name.replace('.', '/')}.py", signature, None)
            )

        for func_name, file_path, signature, class_name in function_templates:
            func_id = self._generate_function_id(func_name)

            docstring = f"{func_name.replace('_', ' ').title()} function"
            embedding = self.embedding_service.embed_query(f"{func_name} {docstring}")

            func = Function(
                id=func_id,
                name=func_name,
                signature=signature,
                code_snippet=None,
                file_path=file_path,
                line_number=random.randint(50, 300),
                docstring=docstring,
                complexity=random.randint(1, 10),
                embedding=embedding,
                metadata={"class": class_name, "generated": True},
            )
            functions.append(func.model_dump())

        logger.info(
            f"Generated {len(modules)} modules, {len(classes)} classes, {len(functions)} functions"
        )

        return {"modules": modules, "classes": classes, "functions": functions}

    def generate_relationships(
        self, requirements: List[Dict], test_cases: List[Dict], code_entities: Dict[str, List[Dict]]
    ) -> List[Dict[str, Any]]:
        """
        Generate relationships between entities.

        Also backfills requirement_ids and function_names on test case metadata.

        Args:
            requirements: List of requirements
            test_cases: List of test cases
            code_entities: Dictionary with functions, classes, modules

        Returns:
            List of relationship dictionaries
        """
        logger.info("Generating relationships...")
        relationships = []

        functions = code_entities["functions"]
        classes = code_entities["classes"]
        modules = code_entities["modules"]

        # Build lookup maps for backfilling
        test_to_requirements: Dict[str, List[str]] = {t["id"]: [] for t in test_cases}
        test_to_functions: Dict[str, List[str]] = {t["id"]: [] for t in test_cases}

        # VERIFIES: TestCase -> Requirement (80% coverage)
        for test in test_cases:
            if random.random() < 0.8:
                # Link to requirement with same category
                test_category = test["metadata"].get("category", "").lower()
                matching_reqs = [
                    r for r in requirements if r.get("category", "").lower() == test_category
                ]
                if matching_reqs:
                    # Pick 1-2 requirements
                    num_reqs = random.randint(1, min(2, len(matching_reqs)))
                    selected_reqs = random.sample(matching_reqs, num_reqs)
                    for req in selected_reqs:
                        relationships.append(
                            {
                                "source_id": test["id"],
                                "target_id": req["id"],
                                "relationship_type": "VERIFIES",
                                "properties": {"coverage_percentage": random.uniform(70, 100)},
                            }
                        )
                        test_to_requirements[test["id"]].append(req["id"])

        # COVERS: TestCase -> Function (each test covers 1-3 functions)
        for test in test_cases:
            num_functions = random.randint(1, 3)
            covered_funcs = random.sample(functions, min(num_functions, len(functions)))
            for func in covered_funcs:
                relationships.append(
                    {
                        "source_id": test["id"],
                        "target_id": func["id"],
                        "relationship_type": "COVERS",
                        "properties": {"direct_coverage": True},
                    }
                )
                test_to_functions[test["id"]].append(func["name"])

        # CALLS: Function -> Function (function call graph)
        for func in functions:
            if random.random() < 0.3:  # 30% of functions call other functions
                target_func = random.choice([f for f in functions if f["id"] != func["id"]])
                relationships.append(
                    {
                        "source_id": func["id"],
                        "target_id": target_func["id"],
                        "relationship_type": "CALLS",
                        "properties": {"call_count": random.randint(1, 10)},
                    }
                )

        # DEFINED_IN: Function -> Class
        for func in functions:
            class_name = func["metadata"].get("class")
            if class_name:
                class_obj = next((c for c in classes if c["name"] == class_name), None)
                if class_obj:
                    relationships.append(
                        {
                            "source_id": func["id"],
                            "target_id": class_obj["id"],
                            "relationship_type": "DEFINED_IN",
                            "properties": {},
                        }
                    )

        # BELONGS_TO: Class -> Module
        for cls in classes:
            module_name = cls["metadata"].get("module", "").replace(".", "_")
            mod_id = self._generate_module_id(module_name)
            if any(m["id"] == mod_id for m in modules):
                relationships.append(
                    {
                        "source_id": cls["id"],
                        "target_id": mod_id,
                        "relationship_type": "BELONGS_TO",
                        "properties": {},
                    }
                )

        # DEPENDS_ON: Module -> Module
        module_deps = [
            ("network.handover", "core.bearer"),
            ("network.authentication", "core.session"),
            ("network.signaling", "utils.logging"),
        ]

        for src_name, tgt_name in module_deps:
            src_id = self._generate_module_id(src_name.replace(".", "_"))
            tgt_id = self._generate_module_id(tgt_name.replace(".", "_"))
            relationships.append(
                {
                    "source_id": src_id,
                    "target_id": tgt_id,
                    "relationship_type": "DEPENDS_ON",
                    "properties": {},
                }
            )

        # Backfill requirement_ids and function_names on test cases
        for test in test_cases:
            test["metadata"]["requirement_ids"] = test_to_requirements.get(test["id"], [])
            test["metadata"]["function_names"] = test_to_functions.get(test["id"], [])

        logger.info(f"Generated {len(relationships)} relationships")
        return relationships

    def generate_full_dataset(
        self, requirement_count: int = 50, testcase_count: int = 200
    ) -> Dict[str, Any]:
        """
        Generate complete synthetic dataset.

        Args:
            requirement_count: Number of requirements to generate
            testcase_count: Number of test cases to generate

        Returns:
            Complete dataset dictionary
        """
        logger.info("Starting full dataset generation...")

        # Generate entities
        requirements = self.generate_requirements(requirement_count)
        test_cases = self.generate_test_cases(testcase_count, requirements)
        code_entities = self.generate_code_entities()

        # Combine all entities
        all_entities = (
            requirements
            + test_cases
            + code_entities["functions"]
            + code_entities["classes"]
            + code_entities["modules"]
        )

        # Generate relationships
        relationships = self.generate_relationships(requirements, test_cases, code_entities)

        dataset = {
            "entities": all_entities,
            "relationships": relationships,
            "metadata": {
                "requirement_count": len(requirements),
                "testcase_count": len(test_cases),
                "function_count": len(code_entities["functions"]),
                "class_count": len(code_entities["classes"]),
                "module_count": len(code_entities["modules"]),
                "relationship_count": len(relationships),
            },
        }

        logger.info("Dataset generation complete!")
        logger.info(f"Total entities: {len(all_entities)}")
        logger.info(f"Total relationships: {len(relationships)}")

        return dataset

    def generate_evaluation_dataset(
        self,
        test_cases: List[Dict],
        requirements: List[Dict],
        functions: List[Dict],
        relationships: List[Dict],
        num_simple: int = 40,
        num_moderate: int = 35,
        num_complex: int = 25,
        num_negative: int = 8,
        use_paraphrases: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate ground truth evaluation dataset with stratified difficulties.

        Args:
            test_cases: List of test case dictionaries
            requirements: List of requirement dictionaries
            functions: List of function dictionaries
            relationships: List of relationship dictionaries
            num_simple: Number of simple queries to generate
            num_moderate: Number of moderate queries to generate
            num_complex: Number of complex queries to generate
            num_negative: Number of negative/out-of-scope queries
            use_paraphrases: Whether to use query paraphrases

        Returns:
            Evaluation dataset with queries and metadata
        """
        logger.info("Generating evaluation dataset...")
        queries = []
        query_id = 1

        # Build lookup structures
        req_to_tests = self._build_requirement_to_tests_map(relationships)
        func_to_tests = self._build_function_to_tests_map(relationships)
        test_lookup = {t["id"]: t for t in test_cases}
        func_lookup = {f["id"]: f for f in functions}

        # Group tests by various attributes
        tests_by_result = self._group_tests_by_attribute(test_cases, "result")
        tests_by_feature = self._group_tests_by_attribute(test_cases, "feature_area")
        tests_by_suite = self._group_tests_by_attribute(test_cases, "test_suite")
        tests_by_priority = self._group_tests_by_attribute(test_cases, "priority")
        tests_by_type = {t["test_type"]: [] for t in test_cases}
        for t in test_cases:
            tests_by_type[t["test_type"]].append(t["id"])

        # =========================================================================
        # SIMPLE QUERIES (40%)
        # =========================================================================
        simple_queries = []

        # 1. Entity lookup by ID
        sample_tests = random.sample(test_cases, min(5, len(test_cases)))
        for test in sample_tests:
            q = self._create_query(
                query_id,
                f"Find test case {test['id']}",
                [test["id"]],
                QueryDifficulty.SIMPLE,
                "entity_lookup",
            )
            simple_queries.append(q)
            query_id += 1

        # 2. Feature area filter
        for feature, test_ids in list(tests_by_feature.items())[:4]:
            paraphrases = QUERY_PARAPHRASES["find_tests_for_feature"]
            query_text = (
                random.choice(paraphrases).format(feature=feature)
                if use_paraphrases
                else f"Find tests for {feature} feature"
            )
            q = self._create_query(
                query_id, query_text, test_ids, QueryDifficulty.SIMPLE, "feature_filter"
            )
            simple_queries.append(q)
            query_id += 1

        # 3. Result filter
        for result in ["PASS", "FAIL"]:
            if result in tests_by_result:
                paraphrases = (
                    QUERY_PARAPHRASES["find_failed_tests"]
                    if result == "FAIL"
                    else QUERY_PARAPHRASES["find_tests_by_result"]
                )
                if result == "FAIL":
                    query_text = (
                        random.choice(paraphrases) if use_paraphrases else "Find failed tests"
                    )
                else:
                    query_text = f"Find tests with {result} result"
                q = self._create_query(
                    query_id,
                    query_text,
                    tests_by_result[result],
                    QueryDifficulty.SIMPLE,
                    "result_filter",
                )
                simple_queries.append(q)
                query_id += 1

        # 4. Priority filter
        for priority in ["critical", "high"]:
            if priority in tests_by_priority:
                paraphrases = QUERY_PARAPHRASES["find_tests_by_priority"]
                query_text = (
                    random.choice(paraphrases).format(priority=priority)
                    if use_paraphrases
                    else f"Find {priority} priority tests"
                )
                q = self._create_query(
                    query_id,
                    query_text,
                    tests_by_priority[priority],
                    QueryDifficulty.SIMPLE,
                    "priority_filter",
                )
                simple_queries.append(q)
                query_id += 1

        # 5. Test type filter
        for test_type, test_ids in list(tests_by_type.items())[:4]:
            if test_ids:
                paraphrases = QUERY_PARAPHRASES["find_tests_by_type"]
                query_text = (
                    random.choice(paraphrases).format(test_type=test_type)
                    if use_paraphrases
                    else f"Find {test_type} tests"
                )
                q = self._create_query(
                    query_id, query_text, test_ids, QueryDifficulty.SIMPLE, "type_filter"
                )
                simple_queries.append(q)
                query_id += 1

        # Trim to target count
        queries.extend(simple_queries[:num_simple])

        # =========================================================================
        # MODERATE QUERIES (35%)
        # =========================================================================
        moderate_queries = []

        # 1. Requirement coverage - "What tests verify REQ_XXX?"
        for req_id, test_ids in list(req_to_tests.items())[:8]:
            if test_ids:
                paraphrases = QUERY_PARAPHRASES["find_tests_for_requirement"]
                query_text = (
                    random.choice(paraphrases).format(req_id=req_id)
                    if use_paraphrases
                    else f"What tests verify {req_id}?"
                )
                q = self._create_query(
                    query_id,
                    query_text,
                    test_ids,
                    QueryDifficulty.MODERATE,
                    "requirement_coverage",
                    expected_relationship="VERIFIES",
                )
                moderate_queries.append(q)
                query_id += 1

        # 2. Function coverage - "What tests cover function X?"
        for func_id, test_ids in list(func_to_tests.items())[:6]:
            if test_ids:
                func_name = func_lookup.get(func_id, {}).get("name", func_id)
                paraphrases = QUERY_PARAPHRASES["find_tests_covering_function"]
                query_text = (
                    random.choice(paraphrases).format(func_name=func_name)
                    if use_paraphrases
                    else f"What tests cover function {func_name}?"
                )
                q = self._create_query(
                    query_id,
                    query_text,
                    test_ids,
                    QueryDifficulty.MODERATE,
                    "function_coverage",
                    expected_relationship="COVERS",
                )
                moderate_queries.append(q)
                query_id += 1

        # 3. Suite membership
        for suite, test_ids in list(tests_by_suite.items())[:4]:
            paraphrases = QUERY_PARAPHRASES["find_tests_in_suite"]
            query_text = (
                random.choice(paraphrases).format(suite=suite)
                if use_paraphrases
                else f"Tests in {suite} suite"
            )
            q = self._create_query(
                query_id, query_text, test_ids, QueryDifficulty.MODERATE, "suite_membership"
            )
            moderate_queries.append(q)
            query_id += 1

        # 4. Combined filters - "Failed tests in Authentication feature"
        for feature in ["Handover", "Authentication", "Security"]:
            if feature in tests_by_feature:
                failed_in_feature = [
                    tid
                    for tid in tests_by_feature[feature]
                    if test_lookup.get(tid, {}).get("metadata", {}).get("result") == "FAIL"
                ]
                if failed_in_feature:
                    q = self._create_query(
                        query_id,
                        f"Failed tests in {feature} feature",
                        failed_in_feature,
                        QueryDifficulty.MODERATE,
                        "combined_filter",
                    )
                    moderate_queries.append(q)
                    query_id += 1

        # 5. Tag-based queries
        tagged_tests = self._find_tests_with_tag(test_cases, "regression")
        if tagged_tests:
            q = self._create_query(
                query_id,
                "Tests tagged with regression",
                tagged_tests,
                QueryDifficulty.MODERATE,
                "tag_filter",
            )
            moderate_queries.append(q)
            query_id += 1

        queries.extend(moderate_queries[:num_moderate])

        # =========================================================================
        # COMPLEX QUERIES (25%)
        # =========================================================================
        complex_queries = []

        # 1. Multi-hop traversal - "Tests for functions in class X"
        # Find classes and their functions, then tests covering those functions
        class_to_funcs = self._build_class_to_functions_map(relationships)
        for class_id, func_ids in list(class_to_funcs.items())[:3]:
            tests_for_class = set()
            for fid in func_ids:
                tests_for_class.update(func_to_tests.get(fid, []))
            if tests_for_class:
                class_name = class_id.replace("CLASS_", "")
                q = self._create_query(
                    query_id,
                    f"Tests for functions in {class_name} class",
                    list(tests_for_class),
                    QueryDifficulty.COMPLEX,
                    "multi_hop_traversal",
                    expected_relationship="COVERS -> DEFINED_IN",
                )
                complex_queries.append(q)
                query_id += 1

        # 2. Aggregation-style queries (feature areas with failing tests)
        features_with_failures = []
        for feature, test_ids in tests_by_feature.items():
            failed_count = sum(
                1
                for tid in test_ids
                if test_lookup.get(tid, {}).get("metadata", {}).get("result") == "FAIL"
            )
            if failed_count > 0:
                features_with_failures.append((feature, failed_count, test_ids))

        if features_with_failures:
            # "Feature area with the most failing tests"
            features_with_failures.sort(key=lambda x: x[1], reverse=True)
            top_feature = features_with_failures[0][0]
            relevant_tests = [
                tid
                for tid in tests_by_feature[top_feature]
                if test_lookup.get(tid, {}).get("metadata", {}).get("result") == "FAIL"
            ]
            q = self._create_query(
                query_id,
                "Which feature area has the most failing tests?",
                relevant_tests,
                QueryDifficulty.COMPLEX,
                "aggregation",
                notes=f"Expected: {top_feature} with {features_with_failures[0][1]} failures",
            )
            complex_queries.append(q)
            query_id += 1

        # 3. Coverage gaps - "Requirements with no test coverage"
        uncovered_reqs = [
            r["id"]
            for r in requirements
            if r["id"] not in req_to_tests or not req_to_tests[r["id"]]
        ]
        if uncovered_reqs:
            q = self._create_query(
                query_id,
                "Requirements with no test coverage",
                uncovered_reqs,
                QueryDifficulty.COMPLEX,
                "coverage_gap",
                notes="Requirements without VERIFIES relationships",
            )
            complex_queries.append(q)
            query_id += 1

        # 4. Cross-entity queries - "Functions covered by failed authentication tests"
        if "Authentication" in tests_by_feature:
            failed_auth_tests = [
                tid
                for tid in tests_by_feature["Authentication"]
                if test_lookup.get(tid, {}).get("metadata", {}).get("result") == "FAIL"
            ]
            if failed_auth_tests:
                covered_funcs = set()
                for rel in relationships:
                    if (
                        rel["relationship_type"] == "COVERS"
                        and rel["source_id"] in failed_auth_tests
                    ):
                        covered_funcs.add(rel["target_id"])
                if covered_funcs:
                    q = self._create_query(
                        query_id,
                        "Functions covered by failed authentication tests",
                        list(covered_funcs),
                        QueryDifficulty.COMPLEX,
                        "cross_entity",
                        expected_relationship="COVERS",
                    )
                    complex_queries.append(q)
                    query_id += 1

        # 5. Critical failed tests
        critical_failed = [
            t["id"]
            for t in test_cases
            if t.get("metadata", {}).get("priority") == "critical"
            and t.get("metadata", {}).get("result") == "FAIL"
        ]
        if critical_failed:
            q = self._create_query(
                query_id,
                "Critical priority tests that failed",
                critical_failed,
                QueryDifficulty.COMPLEX,
                "compound_filter",
            )
            complex_queries.append(q)
            query_id += 1

        # 6. High coverage tests with errors
        error_tests_high_coverage = [
            t["id"]
            for t in test_cases
            if t.get("metadata", {}).get("result") == "ERROR"
            and t.get("metadata", {}).get("code_coverage_pct", 0) > 80
        ]
        if error_tests_high_coverage:
            q = self._create_query(
                query_id,
                "Tests with ERROR result and high code coverage",
                error_tests_high_coverage,
                QueryDifficulty.COMPLEX,
                "compound_filter",
            )
            complex_queries.append(q)
            query_id += 1

        queries.extend(complex_queries[:num_complex])

        # =========================================================================
        # NEGATIVE QUERIES
        # =========================================================================
        for neg in NEGATIVE_QUERIES[:num_negative]:
            q = {
                "id": f"Q_{query_id:03d}",
                "query": neg["query"],
                "relevant_ids": neg["relevant_ids"],
                "difficulty": neg["difficulty"],
                "query_type": neg["query_type"],
                "notes": neg.get("rationale", ""),
            }
            queries.append(q)
            query_id += 1

        # Build final dataset
        dataset = {
            "queries": queries,
            "metadata": {
                "total_queries": len(queries),
                "difficulty_distribution": {
                    "simple": len([q for q in queries if q["difficulty"] == "simple"]),
                    "moderate": len([q for q in queries if q["difficulty"] == "moderate"]),
                    "complex": len([q for q in queries if q["difficulty"] == "complex"]),
                    "negative": len([q for q in queries if q["difficulty"] == "negative"]),
                },
                "generated_at": datetime.now().isoformat(),
                "source_test_count": len(test_cases),
                "source_requirement_count": len(requirements),
                "source_function_count": len(functions),
            },
        }

        logger.info(f"Generated {len(queries)} evaluation queries")
        logger.info(f"Distribution: {dataset['metadata']['difficulty_distribution']}")

        return dataset

    def _create_query(
        self,
        query_id: int,
        query_text: str,
        relevant_ids: List[str],
        difficulty: QueryDifficulty,
        query_type: str,
        expected_relationship: str = None,
        notes: str = None,
    ) -> Dict[str, Any]:
        """Create a standardized query dictionary."""
        q = {
            "id": f"Q_{query_id:03d}",
            "query": query_text,
            "relevant_ids": relevant_ids,
            "difficulty": difficulty.value,
            "query_type": query_type,
        }
        if expected_relationship:
            q["expected_relationship"] = expected_relationship
        if notes:
            q["notes"] = notes
        return q

    def _build_requirement_to_tests_map(self, relationships: List[Dict]) -> Dict[str, List[str]]:
        """Build mapping from requirement ID to list of test IDs that verify it."""
        mapping = {}
        for rel in relationships:
            if rel["relationship_type"] == "VERIFIES":
                req_id = rel["target_id"]
                test_id = rel["source_id"]
                if req_id not in mapping:
                    mapping[req_id] = []
                mapping[req_id].append(test_id)
        return mapping

    def _build_function_to_tests_map(self, relationships: List[Dict]) -> Dict[str, List[str]]:
        """Build mapping from function ID to list of test IDs that cover it."""
        mapping = {}
        for rel in relationships:
            if rel["relationship_type"] == "COVERS":
                func_id = rel["target_id"]
                test_id = rel["source_id"]
                if func_id not in mapping:
                    mapping[func_id] = []
                mapping[func_id].append(test_id)
        return mapping

    def _build_class_to_functions_map(self, relationships: List[Dict]) -> Dict[str, List[str]]:
        """Build mapping from class ID to list of function IDs defined in it."""
        mapping = {}
        for rel in relationships:
            if rel["relationship_type"] == "DEFINED_IN":
                func_id = rel["source_id"]
                class_id = rel["target_id"]
                if class_id not in mapping:
                    mapping[class_id] = []
                mapping[class_id].append(func_id)
        return mapping

    def _group_tests_by_attribute(self, test_cases: List[Dict], attr: str) -> Dict[str, List[str]]:
        """Group test case IDs by a metadata attribute."""
        grouping = {}
        for t in test_cases:
            value = t.get("metadata", {}).get(attr)
            if value:
                if value not in grouping:
                    grouping[value] = []
                grouping[value].append(t["id"])
        return grouping

    def _find_tests_with_tag(self, test_cases: List[Dict], tag: str) -> List[str]:
        """Find test case IDs that have a specific tag."""
        return [t["id"] for t in test_cases if tag in t.get("metadata", {}).get("tags", [])]
