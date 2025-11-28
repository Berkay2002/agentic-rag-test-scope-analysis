"""Synthetic data generator for telecommunications test scope analysis."""

import random
import json
from typing import List, Dict, Any
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


class TelecomDataGenerator:
    """Generate synthetic telecommunications test data."""

    def __init__(self):
        """Initialize the generator."""
        self.embedding_service = get_embedding_service()
        
        # Telecommunications domain concepts
        self.protocols = ["LTE", "5G NR", "X2", "S1", "NGAP", "GTP"]
        self.network_elements = ["eNodeB", "gNodeB", "MME", "SGW", "PGW", "AMF", "SMF"]
        self.functions_domain = [
            "handover", "authentication", "signaling", "session", "bearer",
            "measurement", "configuration", "monitoring", "alarm", "logging"
        ]
        self.test_scenarios = [
            "success", "timeout", "failure", "retry", "rollback",
            "boundary", "stress", "latency", "throughput"
        ]

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
                }
            )
            
            requirements.append(req.model_dump())
            
        logger.info(f"Generated {len(requirements)} requirements")
        return requirements

    def generate_test_cases(self, count: int = 200, requirements: List[Dict] = None) -> List[Dict[str, Any]]:
        """
        Generate test case entities.
        
        Args:
            count: Number of test cases to generate
            requirements: List of requirements to link to
            
        Returns:
            List of test case dictionaries
        """
        logger.info(f"Generating {count} test cases...")
        test_cases = []
        
        categories = ["handover", "authentication", "signaling", "protocol", "integration"]
        
        for i in range(count):
            category = random.choice(categories)
            scenario = random.choice(self.test_scenarios)
            protocol = random.choice(self.protocols)
            
            test_id = self._generate_test_id(category, i + 1)
            name = f"Test {category.title()} {scenario.title()}"
            
            # Generate realistic test descriptions
            description = f"Verify {category} behavior during {scenario} scenario using {protocol} protocol"
            
            # Generate embedding
            embedding = self.embedding_service.embed_query(f"{name} {description}")
            
            test = TestCase(
                id=test_id,
                name=name,
                description=description,
                test_type=random.choice(list(TestType)),
                file_path=f"tests/{category}/test_{category}_{scenario}.py",
                expected_outcome=f"{scenario.title()} case handled correctly",
                preconditions=f"System initialized with {protocol} configuration",
                steps=[
                    f"Initialize {protocol} connection",
                    f"Trigger {category} {scenario}",
                    "Verify expected outcome",
                    "Cleanup resources"
                ],
                embedding=embedding,
                metadata={
                    "protocol": protocol,
                    "scenario": scenario,
                    "generated": True,
                }
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
            "network.handover", "network.authentication", "network.signaling",
            "core.bearer", "core.session", "utils.measurement", "utils.logging"
        ]
        
        for mod_name in module_names:
            component = mod_name.split('.')[0]
            mod_id = self._generate_module_id(mod_name.replace('.', '_'))
            
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
                metadata={"generated": True}
            )
            modules.append(module.model_dump())
        
        # Generate classes
        classes = []
        class_templates = [
            ("HandoverManager", "network.handover", ["initiate_handover", "verify_handover", "rollback_handover"]),
            ("AuthenticationHandler", "network.authentication", ["authenticate_ue", "verify_credentials", "refresh_token"]),
            ("SignalingProcessor", "network.signaling", ["process_message", "validate_message", "route_message"]),
            ("BearerManager", "core.bearer", ["establish_bearer", "modify_bearer", "release_bearer"]),
            ("SessionController", "core.session", ["create_session", "update_session", "terminate_session"]),
            ("MeasurementCollector", "utils.measurement", ["collect_measurements", "aggregate_data", "export_metrics"]),
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
                metadata={"module": module_name, "generated": True}
            )
            classes.append(cls.model_dump())
        
        # Generate functions
        functions = []
        function_templates = []
        
        # Add methods from classes as functions
        for cls in classes:
            for method in cls["methods"]:
                function_templates.append((
                    method,
                    cls["file_path"],
                    f"def {method}(self, *args, **kwargs) -> bool",
                    cls["name"]
                ))
        
        # Add standalone utility functions
        utility_functions = [
            ("calculate_latency", "utils.measurement", "def calculate_latency(start_time: float, end_time: float) -> float"),
            ("format_message", "network.signaling", "def format_message(msg_type: str, payload: dict) -> bytes"),
            ("validate_cell_id", "network.handover", "def validate_cell_id(cell_id: str) -> bool"),
        ]
        
        for func_name, module_name, signature in utility_functions:
            function_templates.append((
                func_name,
                f"src/{module_name.replace('.', '/')}.py",
                signature,
                None
            ))
        
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
                metadata={
                    "class": class_name,
                    "generated": True
                }
            )
            functions.append(func.model_dump())
        
        logger.info(f"Generated {len(modules)} modules, {len(classes)} classes, {len(functions)} functions")
        
        return {
            "modules": modules,
            "classes": classes,
            "functions": functions
        }

    def generate_relationships(
        self,
        requirements: List[Dict],
        test_cases: List[Dict],
        code_entities: Dict[str, List[Dict]]
    ) -> List[Dict[str, Any]]:
        """
        Generate relationships between entities.
        
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
        
        # VERIFIES: TestCase -> Requirement (80% coverage)
        for test in test_cases:
            if random.random() < 0.8:
                # Link to requirement with same category
                matching_reqs = [r for r in requirements if r.get("category") == test["metadata"].get("protocol", "").lower()]
                if matching_reqs:
                    req = random.choice(matching_reqs)
                    relationships.append({
                        "source_id": test["id"],
                        "target_id": req["id"],
                        "relationship_type": "VERIFIES",
                        "properties": {
                            "coverage_percentage": random.uniform(70, 100)
                        }
                    })
        
        # COVERS: TestCase -> Function (each test covers 1-3 functions)
        for test in test_cases:
            num_functions = random.randint(1, 3)
            covered_funcs = random.sample(functions, min(num_functions, len(functions)))
            for func in covered_funcs:
                relationships.append({
                    "source_id": test["id"],
                    "target_id": func["id"],
                    "relationship_type": "COVERS",
                    "properties": {
                        "direct_coverage": True
                    }
                })
        
        # CALLS: Function -> Function (function call graph)
        for func in functions:
            if random.random() < 0.3:  # 30% of functions call other functions
                target_func = random.choice([f for f in functions if f["id"] != func["id"]])
                relationships.append({
                    "source_id": func["id"],
                    "target_id": target_func["id"],
                    "relationship_type": "CALLS",
                    "properties": {
                        "call_count": random.randint(1, 10)
                    }
                })
        
        # DEFINED_IN: Function -> Class
        for func in functions:
            class_name = func["metadata"].get("class")
            if class_name:
                class_obj = next((c for c in classes if c["name"] == class_name), None)
                if class_obj:
                    relationships.append({
                        "source_id": func["id"],
                        "target_id": class_obj["id"],
                        "relationship_type": "DEFINED_IN",
                        "properties": {}
                    })
        
        # BELONGS_TO: Class -> Module
        for cls in classes:
            module_name = cls["metadata"].get("module", "").replace('.', '_')
            mod_id = self._generate_module_id(module_name)
            if any(m["id"] == mod_id for m in modules):
                relationships.append({
                    "source_id": cls["id"],
                    "target_id": mod_id,
                    "relationship_type": "BELONGS_TO",
                    "properties": {}
                })
        
        # DEPENDS_ON: Module -> Module
        module_deps = [
            ("network.handover", "core.bearer"),
            ("network.authentication", "core.session"),
            ("network.signaling", "utils.logging"),
        ]
        
        for src_name, tgt_name in module_deps:
            src_id = self._generate_module_id(src_name.replace('.', '_'))
            tgt_id = self._generate_module_id(tgt_name.replace('.', '_'))
            relationships.append({
                "source_id": src_id,
                "target_id": tgt_id,
                "relationship_type": "DEPENDS_ON",
                "properties": {}
            })
        
        logger.info(f"Generated {len(relationships)} relationships")
        return relationships

    def generate_full_dataset(
        self,
        requirement_count: int = 50,
        testcase_count: int = 200
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
            requirements +
            test_cases +
            code_entities["functions"] +
            code_entities["classes"] +
            code_entities["modules"]
        )
        
        # Generate relationships
        relationships = self.generate_relationships(
            requirements,
            test_cases,
            code_entities
        )
        
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
            }
        }
        
        logger.info("Dataset generation complete!")
        logger.info(f"Total entities: {len(all_entities)}")
        logger.info(f"Total relationships: {len(relationships)}")
        
        return dataset
