# Retrieval and Ontology

## Ontology
Defined in `src/agrag/kg/ontology.py`.

**Entities**
- ChangeRequest, File, Component
- Requirement (priority, status)
- TestCase (test_type, file_path)
- Function (signature, file_path, line_number)
- Class, Module (legacy optional)

**Relationships**
- TOUCHES: ChangeRequest → File
- DEFINED_IN: Function → File
- PART_OF: File → Component
- COVERS: TestCase → Function
- VERIFIES: TestCase → Requirement
- CALLS: Function → Function
- INHERITS_FROM: Class → Class
- BELONGS_TO: Class/Function → Module
- DEPENDS_ON: Module → Module

## Retrieval Tools
- Vector Search: semantic similarity (pgvector HNSW)
- Keyword Search: lexical/BM25 (pg_search)
- Graph Traversal: Cypher path queries (Neo4j)
- Hybrid Search: RRF fusion of vector + keyword

## Tool Defaults (quick)
- Vector/keyword/hybrid: `k` defaults to 10
- Graph traversal: default `depth=2`, direction = outgoing
- Embedding dimensions: 768

## Adding Entities
Update enums → Neo4j constraints → indexes → PostgreSQL schema → generators → tools.
