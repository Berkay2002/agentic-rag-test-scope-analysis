# V1 GraphRAG Alignment Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align ontology, synthetic data, ingestion, and evaluation with the v1 thesis plan (ChangeRequest, File, Component; v1 relationships; fixed RAG vs GraphRAG baselines; explainable outputs).

**Architecture:** Extend the ontology with v1 node and edge types, generate deterministic synthetic ground truth for those entities, ingest entity-level documents into Postgres with rich metadata, and add fixed-baseline evaluation paths (retrieval-only and retrieval plus graph). Update agent prompt and CLI parsing to reflect the new entities and graph paths.

**Tech Stack:** Python 3.11, Pydantic, LangGraph, LangChain, Neo4j, Postgres (pgvector, pg_search), Pytest.

---

### Task 1: Expand v1 ontology (entities, relationships, schema)

**Files:**
- Modify: `src/agrag/kg/ontology.py`
- Modify: `src/agrag/kg/__init__.py`
- Test: `tests/unit/test_ontology_v1.py`

**Step 1: Write the failing test**

```python
from agrag.kg.ontology import NodeLabel, RelationshipType


def test_v1_labels_and_relationships() -> None:
    assert NodeLabel.CHANGE_REQUEST.value == "ChangeRequest"
    assert NodeLabel.FILE.value == "File"
    assert NodeLabel.COMPONENT.value == "Component"
    assert RelationshipType.TOUCHES.value == "TOUCHES"
    assert RelationshipType.PART_OF.value == "PART_OF"
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/test_ontology_v1.py -q`
Expected: FAIL with AttributeError for missing enum members.

**Step 3: Write minimal implementation**

```python
class NodeLabel(str, Enum):
    CHANGE_REQUEST = "ChangeRequest"
    FILE = "File"
    COMPONENT = "Component"
    REQUIREMENT = "Requirement"
    TEST_CASE = "TestCase"
    FUNCTION = "Function"
    CLASS = "Class"  # legacy optional
    MODULE = "Module"  # legacy optional


class RelationshipType(str, Enum):
    TOUCHES = "TOUCHES"  # ChangeRequest -> File
    PART_OF = "PART_OF"  # File -> Component
    VERIFIES = "VERIFIES"  # TestCase -> Requirement
    COVERS = "COVERS"  # TestCase -> Function
    DEFINED_IN = "DEFINED_IN"  # Function -> File
    CALLS = "CALLS"  # legacy optional
    INHERITS_FROM = "INHERITS_FROM"  # legacy optional
    BELONGS_TO = "BELONGS_TO"  # legacy optional
    DEPENDS_ON = "DEPENDS_ON"  # legacy optional
    TESTS = "TESTS"  # legacy optional


class ChangeRequest(BaseModel):
    id: str = Field(..., description="Change request ID (e.g., CR_HANDOVER_001)")
    title: str = Field(..., description="Short change request title")
    description: str = Field(..., description="Change request description")
    status: Optional[str] = Field(None, description="Status (open, in_progress, closed)")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class File(BaseModel):
    id: str = Field(..., description="File ID (e.g., FILE_src_network_handover_py)")
    path: str = Field(..., description="File path")
    language: Optional[str] = Field(None, description="Language")
    component_id: Optional[str] = Field(None, description="Owning component ID")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Component(BaseModel):
    id: str = Field(..., description="Component ID (e.g., COMP_NETWORK)")
    name: str = Field(..., description="Component name")
    description: Optional[str] = Field(None, description="Component description")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


NEO4J_CONSTRAINTS = [
    "CREATE CONSTRAINT changerequest_id IF NOT EXISTS FOR (c:ChangeRequest) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT file_id IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE",
    "CREATE CONSTRAINT component_id IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE",
    # existing constraints...
]

NEO4J_VECTOR_INDEXES = [
    """
    CREATE VECTOR INDEX change_request_embeddings IF NOT EXISTS
    FOR (c:ChangeRequest) ON (c.embedding)
    OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
    """,
    """
    CREATE VECTOR INDEX file_embeddings IF NOT EXISTS
    FOR (f:File) ON (f.embedding)
    OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
    """,
    """
    CREATE VECTOR INDEX component_embeddings IF NOT EXISTS
    FOR (c:Component) ON (c.embedding)
    OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
    """,
    # existing indexes...
]
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/test_ontology_v1.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agrag/kg/ontology.py src/agrag/kg/__init__.py tests/unit/test_ontology_v1.py
git commit -m "feat: add v1 ontology entities and relationships"
```

### Task 2: Add v1 entity ID extraction patterns

**Files:**
- Modify: `src/agrag/evaluation/entity_extractor.py`
- Test: `tests/unit/test_entity_extractor.py`

**Step 1: Write the failing test**

```python
from agrag.evaluation.entity_extractor import extract_entity_ids, extract_entity_ids_detailed


def test_extracts_v1_entity_ids() -> None:
    text = "CR_HANDOVER_001 touches FILE_src_network_handover_py in COMP_NETWORK"
    ids = extract_entity_ids(text, prioritize_test_cases=False)
    assert "CR_HANDOVER_001" in ids
    assert "FILE_src_network_handover_py" in ids
    assert "COMP_NETWORK" in ids

    detailed = extract_entity_ids_detailed(text)
    assert detailed.by_type["ChangeRequest"] == ["CR_HANDOVER_001"]
    assert detailed.by_type["File"] == ["FILE_src_network_handover_py"]
    assert detailed.by_type["Component"] == ["COMP_NETWORK"]
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/test_entity_extractor.py -q`
Expected: FAIL with missing patterns.

**Step 3: Write minimal implementation**

```python
ENTITY_PATTERNS = {
    "TestCase": r"TC_[A-Z]+_\d+",
    "Requirement": r"REQ_[A-Z]+_\d+",
    "Function": r"FUNC_[A-Za-z_]+(?:_\d+)?",
    "Class": r"CLASS_[A-Za-z_]+(?:_\d+)?",
    "Module": r"MOD_[A-Za-z_.]+(?:_\d+)?",
    "ChangeRequest": r"CR_[A-Z]+_\d+",
    "File": r"FILE_[A-Za-z0-9_]+",
    "Component": r"COMP_[A-Za-z0-9_]+",
}
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/test_entity_extractor.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agrag/evaluation/entity_extractor.py tests/unit/test_entity_extractor.py
git commit -m "feat: add v1 entity id extraction"
```

### Task 3: Generate v1 entities and relationships in synthetic data

**Files:**
- Modify: `src/agrag/data/generators/synthetic.py`
- Test: `tests/unit/test_synthetic_generator_v1.py`

**Step 1: Write the failing test**

```python
import random
from agrag.data.generators import synthetic as synthetic_module
from agrag.data.generators.synthetic import TelecomDataGenerator


class _StubEmbeddingService:
    def embed_query(self, text: str):
        return [0.0] * 768


def test_generator_emits_v1_entities_and_edges(monkeypatch) -> None:
    monkeypatch.setattr(
        synthetic_module,
        "get_embedding_service",
        lambda: _StubEmbeddingService(),
    )
    random.seed(0)

    gen = TelecomDataGenerator()
    dataset = gen.generate_full_dataset(requirement_count=3, testcase_count=5)

    ids = [entity["id"] for entity in dataset["entities"]]
    assert any(i.startswith("CR_") for i in ids)
    assert any(i.startswith("FILE_") for i in ids)
    assert any(i.startswith("COMP_") for i in ids)

    rel_types = {rel["relationship_type"] for rel in dataset["relationships"]}
    assert "TOUCHES" in rel_types
    assert "PART_OF" in rel_types
    assert "DEFINED_IN" in rel_types
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/test_synthetic_generator_v1.py -q`
Expected: FAIL with missing entities or relationships.

**Step 3: Write minimal implementation**

```python
from agrag.kg.ontology import ChangeRequest, File, Component


def _generate_component_id(self, name: str) -> str:
    return f"COMP_{name.upper()}"


def _generate_file_id(self, path: str) -> str:
    slug = path.replace("/", "_").replace(".", "_")
    return f"FILE_{slug}"


def _generate_change_request_id(self, category: str, index: int) -> str:
    return f"CR_{category.upper()}_{index:03d}"


def generate_components(self) -> List[Dict[str, Any]]:
    components = []
    for name in ["network", "core", "utils"]:
        comp = Component(
            id=self._generate_component_id(name),
            name=name,
            description=f"{name} subsystem",
            embedding=self.embedding_service.embed_query(f"{name} component"),
            metadata={"generated": True},
        )
        components.append(comp.model_dump())
    return components


def generate_files(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    files = []
    seen_paths = set()
    for func in functions:
        path = func["file_path"]
        if path in seen_paths:
            continue
        seen_paths.add(path)
        comp_name = path.split("/")[1] if path.startswith("src/") else "core"
        file_entity = File(
            id=self._generate_file_id(path),
            path=path,
            language="python",
            component_id=self._generate_component_id(comp_name),
            embedding=self.embedding_service.embed_query(path),
            metadata={"generated": True},
        )
        files.append(file_entity.model_dump())
    return files


def generate_change_requests(self, files: List[Dict[str, Any]], count: int = 20):
    change_requests = []
    categories = ["handover", "authentication", "signaling"]
    for i in range(count):
        category = random.choice(categories)
        cr_id = self._generate_change_request_id(category, i + 1)
        title = f"{category.title()} change request"
        description = f"Update {category} handling and validate timeout scenarios"
        cr = ChangeRequest(
            id=cr_id,
            title=title,
            description=description,
            status=random.choice(["open", "in_progress", "closed"]),
            embedding=self.embedding_service.embed_query(f"{title} {description}"),
            metadata={"category": category, "generated": True},
        )
        change_requests.append(cr.model_dump())
    return change_requests
```

Add v1 relationships in `generate_relationships`:

```python
# TOUCHES: ChangeRequest -> File
for cr in change_requests:
    touched_files = random.sample(files, k=min(2, len(files)))
    for file in touched_files:
        relationships.append(
            {
                "source_id": cr["id"],
                "target_id": file["id"],
                "relationship_type": "TOUCHES",
                "properties": {},
            }
        )

# DEFINED_IN: Function -> File
for func in functions:
    file_id = self._generate_file_id(func["file_path"])
    relationships.append(
        {
            "source_id": func["id"],
            "target_id": file_id,
            "relationship_type": "DEFINED_IN",
            "properties": {},
        }
    )

# PART_OF: File -> Component
for file in files:
    comp_id = file.get("component_id")
    if comp_id:
        relationships.append(
            {
                "source_id": file["id"],
                "target_id": comp_id,
                "relationship_type": "PART_OF",
                "properties": {},
            }
        )
```

Update `generate_full_dataset` to include components, files, change_requests in `entities`, and add counts in metadata.

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/test_synthetic_generator_v1.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agrag/data/generators/synthetic.py tests/unit/test_synthetic_generator_v1.py
git commit -m "feat: generate v1 entities and relationships"
```

### Task 4: Add v1 evaluation query workloads

**Files:**
- Modify: `src/agrag/data/generators/synthetic.py`
- Test: `tests/unit/test_evaluation_queries_v1.py`

**Step 1: Write the failing test**

```python
import random
from agrag.data.generators import synthetic as synthetic_module
from agrag.data.generators.synthetic import TelecomDataGenerator


class _StubEmbeddingService:
    def embed_query(self, text: str):
        return [0.0] * 768


def test_evaluation_queries_include_v1_workloads(monkeypatch) -> None:
    monkeypatch.setattr(
        synthetic_module,
        "get_embedding_service",
        lambda: _StubEmbeddingService(),
    )
    random.seed(0)

    gen = TelecomDataGenerator()
    dataset = gen.generate_full_dataset(requirement_count=5, testcase_count=10)

    eval_data = gen.generate_evaluation_dataset(
        test_cases=[e for e in dataset["entities"] if e["id"].startswith("TC_")],
        requirements=[e for e in dataset["entities"] if e["id"].startswith("REQ_")],
        functions=[e for e in dataset["entities"] if e["id"].startswith("FUNC_")],
        relationships=dataset["relationships"],
    )

    query_types = {q["query_type"] for q in eval_data["queries"]}
    assert "change_request_tests" in query_types
    assert "impact_analysis" in query_types
    assert "coverage_by_component" in query_types
    assert "failure_triage" in query_types
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/test_evaluation_queries_v1.py -q`
Expected: FAIL with missing query types.

**Step 3: Write minimal implementation**

Add query templates and builders in `generate_evaluation_dataset`:

```python
QUERY_PARAPHRASES["change_request_tests"] = [
    "Tests impacted by change request {cr_id}",
    "What tests are related to {cr_id}?",
]
QUERY_PARAPHRASES["coverage_by_component"] = [
    "Coverage for {req_id} by component",
    "Which components cover requirement {req_id}?",
]
QUERY_PARAPHRASES["impact_analysis"] = [
    "Impact analysis for {file_id}",
    "Tests impacted by changes in {file_id}",
]
QUERY_PARAPHRASES["failure_triage"] = [
    "Tests related to error {error_code}",
    "Which tests cover failures with {error_code}?",
]
```

Generate query blocks that use v1 relationships:

```python
# change_request_tests: CR -> File -> Function -> TestCase
cr_to_files = self._build_cr_to_files_map(relationships)
file_to_funcs = self._build_file_to_functions_map(relationships)
for cr_id, file_ids in list(cr_to_files.items())[:5]:
    test_ids = set()
    for fid in file_ids:
        for func_id in file_to_funcs.get(fid, []):
            test_ids.update(func_to_tests.get(func_id, []))
    if test_ids:
        q = self._create_query(
            query_id,
            random.choice(QUERY_PARAPHRASES["change_request_tests"]).format(cr_id=cr_id),
            list(test_ids),
            QueryDifficulty.MODERATE,
            "change_request_tests",
        )
        queries.append(q)
        query_id += 1

# coverage_by_component: Requirement -> TestCase -> Function -> File -> Component
file_to_component = self._build_file_to_component_map(relationships)
for req_id, test_ids in list(req_to_tests.items())[:5]:
    components = set()
    for rel in relationships:
        if rel["relationship_type"] == "COVERS" and rel["source_id"] in test_ids:
            func_id = rel["target_id"]
            file_ids = file_to_funcs.get(func_id, [])
            for file_id in file_ids:
                comp_id = file_to_component.get(file_id)
                if comp_id:
                    components.add(comp_id)
    if components:
        q = self._create_query(
            query_id,
            random.choice(QUERY_PARAPHRASES["coverage_by_component"]).format(req_id=req_id),
            list(components),
            QueryDifficulty.COMPLEX,
            "coverage_by_component",
        )
        queries.append(q)
        query_id += 1

# impact_analysis: File -> Function -> TestCase
for file_id, func_ids in list(file_to_funcs.items())[:5]:
    impacted = set()
    for func_id in func_ids:
        impacted.update(func_to_tests.get(func_id, []))
    if impacted:
        q = self._create_query(
            query_id,
            random.choice(QUERY_PARAPHRASES["impact_analysis"]).format(file_id=file_id),
            list(impacted),
            QueryDifficulty.MODERATE,
            "impact_analysis",
        )
        queries.append(q)
        query_id += 1

# failure_triage: pick error codes from failure_reason metadata
error_tests = [
    t for t in test_cases if t.get("metadata", {}).get("failure_reason")
]
for test in error_tests[:5]:
    reason = test.get("metadata", {}).get("failure_reason", "")
    error_code = "ERR_" + reason.split("ERR_")[-1].split()[0] if "ERR_" in reason else "ERR_1000"
    q = self._create_query(
        query_id,
        random.choice(QUERY_PARAPHRASES["failure_triage"]).format(error_code=error_code),
        [test["id"]],
        QueryDifficulty.MODERATE,
        "failure_triage",
    )
    queries.append(q)
    query_id += 1
```

Add helper maps like `_build_cr_to_files_map`, `_build_file_to_functions_map`, `_build_file_to_component_map`.

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/test_evaluation_queries_v1.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agrag/data/generators/synthetic.py tests/unit/test_evaluation_queries_v1.py
git commit -m "feat: add v1 evaluation query workloads"
```

### Task 5: Update ingestion mapping and metadata for v1 entities

**Files:**
- Modify: `src/agrag/data/ingestion.py`
- Modify: `src/agrag/data/dual_storage_writer.py`
- Test: `tests/unit/test_ingestion_mapping.py`

**Step 1: Write the failing test**

```python
from agrag.data.ingestion import DataIngestion


def test_infer_entity_type_v1_prefixes() -> None:
    assert DataIngestion._infer_entity_type("CR_HANDOVER_001") == "ChangeRequest"
    assert DataIngestion._infer_entity_type("FILE_src_network_handover_py") == "File"
    assert DataIngestion._infer_entity_type("COMP_NETWORK") == "Component"
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/test_ingestion_mapping.py -q`
Expected: FAIL with AttributeError.

**Step 3: Write minimal implementation**

```python
class DataIngestion:
    @staticmethod
    def _infer_entity_type(entity_id: str) -> Optional[str]:
        if entity_id.startswith("CR_"):
            return "ChangeRequest"
        if entity_id.startswith("FILE_"):
            return "File"
        if entity_id.startswith("COMP_"):
            return "Component"
        if entity_id.startswith("REQ_"):
            return "Requirement"
        if entity_id.startswith("TC_"):
            return "TestCase"
        if entity_id.startswith("FUNC_"):
            return "Function"
        if entity_id.startswith("CLASS_"):
            return "Class"
        if entity_id.startswith("MOD_"):
            return "Module"
        return None
```

Use `_infer_entity_type` in `ingest_full_dataset` and add metadata fields in `ingest_entities_postgres` and `_entities_to_bm25_documents` for `component_id`, `path`, and `status` if present.

Update `DualStorageWriter` metadata to include `component_id` and `path` when present.

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/test_ingestion_mapping.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agrag/data/ingestion.py src/agrag/data/dual_storage_writer.py tests/unit/test_ingestion_mapping.py
git commit -m "feat: ingest v1 entity types and metadata"
```

### Task 6: Update agent prompt and CLI parsing for v1 entities

**Files:**
- Modify: `src/agrag/core/graph.py`
- Modify: `src/agrag/cli/main.py`
- Modify: `src/agrag/tools/schemas.py`
- Test: `tests/unit/test_cli_parsing_v1.py`

**Step 1: Write the failing test**

```python
from agrag.cli import main as cli_main


def test_parse_result_ids_includes_v1_entities() -> None:
    sample = "1. ID: CR_HANDOVER_001 (Score: 0.9)\n2. ID: FILE_src_network_handover_py (Score: 0.8)\n3. ID: COMP_NETWORK (Score: 0.7)"
    ids = cli_main._parse_result_ids(sample)
    assert "CR_HANDOVER_001" in ids
    assert "FILE_src_network_handover_py" in ids
    assert "COMP_NETWORK" in ids
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/test_cli_parsing_v1.py -q`
Expected: FAIL with missing IDs.

**Step 3: Write minimal implementation**

Update `SYSTEM_PROMPT` in `src/agrag/core/graph.py` to mention ChangeRequest, File, Component and require answers with ranked tests, evidence snippets, graph paths, and uncertainty labels for inferred edges.

Update CLI ID patterns and graph traversal heuristics in `src/agrag/cli/main.py`:

```python
id_patterns = [
    (r"CR_[A-Z]+_\d+", NodeLabel.CHANGE_REQUEST),
    (r"FILE_[A-Za-z0-9_]+", NodeLabel.FILE),
    (r"COMP_[A-Za-z0-9_]+", NodeLabel.COMPONENT),
    (r"REQ_[A-Z]+_\d+", NodeLabel.REQUIREMENT),
    (r"FUNC_[A-Za-z_]+", NodeLabel.FUNCTION),
    (r"TC_[A-Z]+_\d+", NodeLabel.TEST_CASE),
]

if start_node_label in [NodeLabel.CHANGE_REQUEST, NodeLabel.FILE, NodeLabel.COMPONENT]:
    relationship_types = [
        RelationshipType.TOUCHES,
        RelationshipType.DEFINED_IN,
        RelationshipType.COVERS,
        RelationshipType.VERIFIES,
        RelationshipType.PART_OF,
    ]
    direction = "both"
```

Update `_parse_result_ids` and `_parse_graph_result_ids` to include CR_, FILE_, COMP_ patterns, and update tool schema examples in `src/agrag/tools/schemas.py`.

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/test_cli_parsing_v1.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agrag/core/graph.py src/agrag/cli/main.py src/agrag/tools/schemas.py tests/unit/test_cli_parsing_v1.py
git commit -m "feat: align agent prompt and cli parsing with v1 entities"
```

### Task 7: Implement fixed RAG and fixed GraphRAG baselines

**Files:**
- Create: `src/agrag/evaluation/fixed_baselines.py`
- Modify: `src/agrag/cli/main.py`
- Test: `tests/unit/test_fixed_baselines.py`

**Step 1: Write the failing test**

```python
from agrag.evaluation.fixed_baselines import run_fixed_graphrag


class _StubHybridTool:
    def _run(self, query: str, k: int = 10):
        return "1. ID: FILE_src_network_handover_py (Score: 0.8)"


class _StubGraphTool:
    def _run(self, start_node_id, start_node_label, relationship_types, depth, direction):
        return "1. Path (depth 2): FILE_src_network_handover_py -> TC_HANDOVER_001\n   Sequence: File:FILE_src_network_handover_py -> Function:FUNC_initiate_handover -> TestCase:TC_HANDOVER_001"


def test_fixed_graphrag_combines_retrieval_and_graph() -> None:
    ids = run_fixed_graphrag(
        query="handover changes",
        hybrid_tool=_StubHybridTool(),
        graph_tool=_StubGraphTool(),
        k=5,
    )
    assert "FILE_src_network_handover_py" in ids
    assert "TC_HANDOVER_001" in ids
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/test_fixed_baselines.py -q`
Expected: FAIL with ImportError.

**Step 3: Write minimal implementation**

```python
from typing import List
from agrag.kg.ontology import NodeLabel, RelationshipType
from agrag.cli.main import _parse_result_ids, _parse_graph_result_ids


def _infer_label_from_id(entity_id: str) -> NodeLabel:
    if entity_id.startswith("CR_"):
        return NodeLabel.CHANGE_REQUEST
    if entity_id.startswith("FILE_"):
        return NodeLabel.FILE
    if entity_id.startswith("COMP_"):
        return NodeLabel.COMPONENT
    if entity_id.startswith("REQ_"):
        return NodeLabel.REQUIREMENT
    if entity_id.startswith("TC_"):
        return NodeLabel.TEST_CASE
    return NodeLabel.FUNCTION


def run_fixed_rag(query: str, hybrid_tool, k: int = 10) -> List[str]:
    result_str = hybrid_tool._run(query=query, k=k)
    return _parse_result_ids(result_str)


def run_fixed_graphrag(query: str, hybrid_tool, graph_tool, k: int = 10) -> List[str]:
    seed_ids = run_fixed_rag(query=query, hybrid_tool=hybrid_tool, k=k)
    graph_ids: List[str] = []
    for entity_id in seed_ids[:3]:
        label = _infer_label_from_id(entity_id)
        graph_result = graph_tool._run(
            start_node_id=entity_id,
            start_node_label=label,
            relationship_types=[
                RelationshipType.TOUCHES,
                RelationshipType.DEFINED_IN,
                RelationshipType.COVERS,
                RelationshipType.VERIFIES,
                RelationshipType.PART_OF,
            ],
            depth=3,
            direction="both",
        )
        graph_ids.extend(_parse_graph_result_ids(graph_result))
    # de-dupe while preserving order
    seen = set()
    ordered = []
    for entity_id in seed_ids + graph_ids:
        if entity_id not in seen:
            seen.add(entity_id)
            ordered.append(entity_id)
    return ordered
```

Update CLI `evaluate` strategy choices to include `rag` and `graphrag`, and route to `run_fixed_rag`/`run_fixed_graphrag` when selected.

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/test_fixed_baselines.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agrag/evaluation/fixed_baselines.py src/agrag/cli/main.py tests/unit/test_fixed_baselines.py
git commit -m "feat: add fixed rag and graphrag baselines"
```

---

## Progress

- Completed Task 1 (v1 ontology entities/relationships + constraints/indexes). Commit: e2b6be9
- Completed Task 2 (v1 entity ID extraction patterns). Commit: 1e6c78c
- Completed Task 3 (v1 entities + relationships in synthetic generator). Commit: 9ca8b07
- Completed Task 4 (v1 evaluation query workloads). Commit: aa35643
- Completed Task 5 (v1 ingestion mapping + metadata). Commit: 9f42b01
- Completed Task 6 (agent prompt + CLI parsing updates). Commit: aa3afa7
- Completed Task 7 (fixed RAG + GraphRAG baselines). Commit: 7be436f
- Validation: `poetry run pytest tests/unit/test_ontology_v1.py tests/unit/test_entity_extractor.py tests/unit/test_synthetic_generator_v1.py tests/unit/test_evaluation_queries_v1.py tests/unit/test_ingestion_mapping.py tests/unit/test_cli_parsing_v1.py tests/unit/test_fixed_baselines.py -q`, `poetry run ruff check src/ tests/`
- Added E2E coverage for fixed baselines. Commit: 4431f71
- Documented fixed RAG/GraphRAG evaluation commands. Commit: f98d71f
- Fixed baseline tool invocation for StructuredTool wrappers. Commit: 3e5dd79
- Validation: `poetry run ruff check src/agrag/evaluation/fixed_baselines.py tests/integration`, `poetry run pytest tests/integration/test_fixed_baselines_e2e.py`
