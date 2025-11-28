# Next Steps for Agentic GraphRAG System

## Current Status

✅ **Phase 0-4 Complete**: All core infrastructure and retrieval tools implemented
🔧 **Environment**: Fully configured with Neo4j Aura, Neon PostgreSQL, Google AI, and LangSmith
📦 **Dependencies**: Installed via Poetry (162 packages)
⚠️ **Issue Fixed**: Pydantic v2 compatibility (replaced `const=True` with `Literal` types)

---

## Immediate Next Steps (Testing & Validation)

### 1. Initialize Database Schemas

**Command:**
```bash
poetry run agrag init
```

**What it does:**
- Creates Neo4j constraints (unique IDs for all entity types)
- Creates Neo4j vector indexes (768-dim HNSW with cosine similarity)
- Creates PostgreSQL tables with pgvector extension
- Sets up full-text search indexes

**Expected output:**
- Neo4j constraints created successfully
- Neo4j vector indexes created successfully
- PostgreSQL schema initialized
- Confirmation messages for each step

**If it fails:**
- Check `.env` credentials (Neo4j URI, PostgreSQL connection string)
- Verify Neo4j Aura instance is running
- Verify Neon PostgreSQL database is accessible
- Check logs for specific errors

---

### 2. Verify System Configuration

**Command:**
```bash
poetry run agrag info
```

**What it does:**
- Displays current configuration
- Shows database connection status
- Lists available tools
- Confirms model settings

**Expected output:**
```
Configuration:
- LLM Model: gemini-2.5-flash-latest
- Embedding Model: models/embedding-001
- Neo4j URI: neo4j+s://YinG5Yt6lBkqg9nt394pbULfIaj2sCnq.databases.neo4j.io
- PostgreSQL: Connected to Neon
- LangSmith: Enabled (project: agrag-test-scope-analysis)

Available Tools:
1. vector_search - Semantic similarity search
2. keyword_search - Lexical BM25-style search
3. graph_traverse - Multi-hop graph traversal
4. hybrid_search - RRF fusion of vector + keyword
```

---

### 3. Test Basic Functionality

Since we don't have data yet, we need to **load sample data first**. See Phase 5 below.

---

## Phase 5: Data Ingestion & Synthetic Data Generation

This is the **critical next phase** to make the system functional.

### 5.1 Create Synthetic Data Generator

**File to create:** `src/agrag/data/generators.py`

**Purpose:** Generate synthetic telecommunications test data

**Implementation tasks:**
- [ ] Define telecommunications domain entities (Requirements, TestCases, Functions, Classes, Modules)
- [ ] Generate realistic requirement IDs (e.g., `REQ_AUTH_005`, `REQ_HANDOVER_012`)
- [ ] Generate test case data with types (unit, integration, protocol, performance)
- [ ] Generate code entities (functions, classes, modules) with realistic names
- [ ] Create relationships (VERIFIES, COVERS, CALLS, DEFINED_IN, INHERITS_FROM, DEPENDS_ON)
- [ ] Generate embeddings for all entities using Google embedding model
- [ ] Export to JSON format for ingestion

**Estimated complexity:** Medium (2-3 hours)

**Example structure:**
```python
# src/agrag/data/generators.py
from typing import List, Dict
from agrag.models import get_embedding_service
from agrag.kg.ontology import *

class TelecomDataGenerator:
    """Generate synthetic telecommunications test data."""

    def generate_requirements(self, count: int = 50) -> List[Dict]:
        """Generate requirement entities."""
        pass

    def generate_test_cases(self, count: int = 200) -> List[Dict]:
        """Generate test case entities."""
        pass

    def generate_code_entities(self) -> Dict[str, List]:
        """Generate functions, classes, modules."""
        pass

    def generate_relationships(self) -> List[Dict]:
        """Generate all relationship types."""
        pass

    def generate_full_dataset(self) -> Dict:
        """Generate complete dataset."""
        pass
```

---

### 5.2 Create Data Ingestion Pipeline

**File to create:** `src/agrag/data/ingestion.py`

**Purpose:** Load generated data into Neo4j and PostgreSQL

**Implementation tasks:**
- [ ] Read JSON dataset
- [ ] Batch insert entities into Neo4j (use `UNWIND` for performance)
- [ ] Batch insert relationships into Neo4j
- [ ] Batch insert embeddings into PostgreSQL
- [ ] Add error handling and retry logic
- [ ] Add progress tracking (use `rich` progress bar)
- [ ] Add validation checks

**Estimated complexity:** Medium (2-3 hours)

**Example structure:**
```python
# src/agrag/data/ingestion.py
from agrag.storage import Neo4jClient, PostgresClient
from rich.progress import Progress

class DataIngestion:
    """Load data into Neo4j and PostgreSQL."""

    def __init__(self):
        self.neo4j_client = Neo4jClient()
        self.postgres_client = PostgresClient()

    def ingest_entities(self, entities: List[Dict], entity_type: str):
        """Batch insert entities into both databases."""
        pass

    def ingest_relationships(self, relationships: List[Dict]):
        """Batch insert relationships into Neo4j."""
        pass

    def ingest_full_dataset(self, dataset: Dict):
        """Ingest complete dataset."""
        pass
```

---

### 5.3 Add CLI Command for Data Generation

**File to modify:** `src/agrag/cli/main.py`

**Add new command:**
```python
@cli.command()
@click.option("--requirements", default=50, help="Number of requirements to generate")
@click.option("--testcases", default=200, help="Number of test cases to generate")
@click.option("--output", default="data/synthetic_dataset.json", help="Output file path")
def generate(requirements: int, testcases: int, output: str):
    """Generate synthetic telecommunications dataset."""
    from agrag.data.generators import TelecomDataGenerator

    logger.info(f"Generating synthetic dataset...")
    generator = TelecomDataGenerator()
    dataset = generator.generate_full_dataset(
        requirement_count=requirements,
        testcase_count=testcases
    )

    # Save to JSON
    with open(output, "w") as f:
        json.dump(dataset, f, indent=2)

    logger.info(f"Dataset saved to {output}")
    click.echo(f"✓ Generated {len(dataset['entities'])} entities and {len(dataset['relationships'])} relationships")

@cli.command()
@click.argument("dataset_path")
def ingest(dataset_path: str):
    """Ingest dataset into Neo4j and PostgreSQL."""
    from agrag.data.ingestion import DataIngestion

    logger.info(f"Loading dataset from {dataset_path}...")
    with open(dataset_path) as f:
        dataset = json.load(f)

    ingestion = DataIngestion()
    ingestion.ingest_full_dataset(dataset)

    click.echo(f"✓ Ingested {len(dataset['entities'])} entities and {len(dataset['relationships'])} relationships")
```

---

### 5.4 Workflow After Phase 5 Implementation

Once Phase 5 is complete:

```bash
# 1. Generate synthetic data
poetry run agrag generate --requirements 50 --testcases 200 --output data/synthetic_dataset.json

# 2. Ingest data into databases
poetry run agrag ingest data/synthetic_dataset.json

# 3. Test queries
poetry run agrag query "What tests cover requirement REQ_AUTH_005?"
poetry run agrag query "Find all handover-related test cases" --stream
poetry run agrag query "Show dependencies for TestLoginTimeout" --checkpoint --thread-id session-1
```

---

## Phase 6: Testing & Quality Assurance

### 6.1 Unit Tests

**Directory:** `tests/`

**Files to create:**
- `tests/test_tools.py` - Test all 4 retrieval tools
- `tests/test_storage.py` - Test Neo4j and PostgreSQL clients
- `tests/test_models.py` - Test LLM and embedding services
- `tests/test_evaluation.py` - Test metric calculations
- `tests/test_agent.py` - Test StateGraph logic

**Run tests:**
```bash
poetry run pytest -v
poetry run pytest --cov=src/agrag --cov-report=html
```

---

### 6.2 Integration Tests

**File to create:** `tests/integration/test_end_to_end.py`

**Test scenarios:**
1. Full query execution from user input to final answer
2. HITL workflow with checkpointing
3. Multi-hop graph traversal with tool chaining
4. Hybrid search with RRF fusion

---

### 6.3 Evaluation Dataset

**File to create:** `data/eval_queries.json`

**Structure:**
```json
[
  {
    "query": "What tests verify REQ_AUTH_005?",
    "relevant_ids": ["TEST_AUTH_001", "TEST_AUTH_002", "TEST_AUTH_003"],
    "expected_tool": "graph_traverse"
  },
  {
    "query": "Find tests related to handover failures",
    "relevant_ids": ["TEST_HANDOVER_001", "TEST_HANDOVER_005"],
    "expected_tool": "hybrid_search"
  }
]
```

**Run evaluation:**
```bash
poetry run agrag evaluate --dataset data/eval_queries.json --output results/eval_results.json
```

---

## Phase 7: Advanced Features

### 7.1 Query Rewriting & Expansion

**File to create:** `src/agrag/tools/query_rewriter.py`

**Features:**
- Expand acronyms (e.g., "LTE" → "Long-Term Evolution")
- Generate synonyms
- Create multi-query variants
- Use LLM for query understanding

---

### 7.2 Caching Layer

**File to create:** `src/agrag/cache/embedding_cache.py`

**Purpose:** Cache embeddings to reduce API costs

**Implementation:**
- Use Redis or local file cache
- Cache query embeddings
- Cache entity embeddings
- TTL-based expiration

---

### 7.3 Batch Processing

**File to create:** `src/agrag/cli/batch.py`

**Purpose:** Process multiple queries in parallel

**Features:**
- Read queries from CSV/JSON
- Parallel execution with ThreadPoolExecutor
- Progress tracking
- Result aggregation

---

## Phase 8: Production Deployment

### 8.1 Docker Compose

**File to create:** `docker-compose.yml`

**Services:**
- Neo4j (local development)
- PostgreSQL with pgvector
- Redis (for caching)
- Application service

---

### 8.2 Kubernetes Deployment

**Directory:** `k8s/`

**Files:**
- `k8s/deployment.yaml` - Application deployment
- `k8s/service.yaml` - Service definition
- `k8s/configmap.yaml` - Configuration
- `k8s/secrets.yaml` - Secrets management

---

### 8.3 CI/CD Pipeline

**File to create:** `.github/workflows/ci.yml`

**Stages:**
1. Lint (black, ruff)
2. Type check (mypy)
3. Unit tests
4. Integration tests
5. Build Docker image
6. Deploy to staging
7. Deploy to production (manual approval)

---

## Phase 9: Monitoring & Observability

### 9.1 Metrics

**File to create:** `src/agrag/monitoring/metrics.py`

**Metrics to track:**
- Query latency (p50, p95, p99)
- Tool usage distribution
- Model call counts
- Database query performance
- Embedding cache hit rate
- Error rates

---

### 9.2 Dashboards

**Tools:**
- LangSmith for trace visualization
- Grafana for system metrics
- Custom dashboard for business metrics

---

## Summary of Immediate Tasks

### Priority 1 (Critical - Do This First)
1. ✅ Fix Pydantic compatibility issues (DONE)
2. 🔄 Initialize databases: `poetry run agrag init`
3. 🔄 Verify configuration: `poetry run agrag info`

### Priority 2 (Essential - Can't Test Without This)
4. 📝 Implement `src/agrag/data/generators.py` - Synthetic data generator
5. 📝 Implement `src/agrag/data/ingestion.py` - Data ingestion pipeline
6. 📝 Add `generate` and `ingest` CLI commands
7. 🏃 Generate and ingest synthetic dataset

### Priority 3 (Important - Validates Implementation)
8. 📝 Write unit tests for all modules
9. 📝 Create evaluation dataset
10. 🏃 Run evaluation suite

### Priority 4 (Nice to Have - Production Ready)
11. 📝 Add query rewriting
12. 📝 Implement caching layer
13. 📝 Create Docker compose setup
14. 📝 Add monitoring and metrics

---

## Quick Start Commands

After Phase 5 is complete, you can use:

```bash
# Initialize everything
poetry run agrag init

# Generate synthetic data
poetry run agrag generate --requirements 50 --testcases 200

# Ingest data
poetry run agrag ingest data/synthetic_dataset.json

# Test queries
poetry run agrag query "What tests cover REQ_AUTH_005?"
poetry run agrag query "Find handover tests" --stream
poetry run agrag query "Dependencies of LoginHandler" --checkpoint

# Run evaluation
poetry run agrag evaluate --dataset data/eval_queries.json

# Check configuration
poetry run agrag info
```

---

## Estimated Timeline

| Phase | Task | Complexity | Estimated Time |
|-------|------|------------|----------------|
| 5.1 | Synthetic data generator | Medium | 2-3 hours |
| 5.2 | Data ingestion pipeline | Medium | 2-3 hours |
| 5.3 | CLI commands | Low | 30 mins |
| 6.1 | Unit tests | Medium | 4-6 hours |
| 6.2 | Integration tests | Medium | 2-3 hours |
| 6.3 | Evaluation dataset | Low | 1-2 hours |
| 7.x | Advanced features | High | 8-12 hours |
| 8.x | Production deployment | Medium | 4-6 hours |
| 9.x | Monitoring | Medium | 3-4 hours |

**Total estimated time:** 25-40 hours for full production-ready system

---

## Questions to Consider

1. **Data Volume**: How much synthetic data should we generate?
   - Recommendation: Start with 50 requirements, 200 test cases, 100 functions

2. **Domain Focus**: Which telecom protocols to focus on?
   - Recommendation: LTE, 5G NR, Handover, Authentication

3. **Evaluation Metrics**: Which metrics are most important?
   - Recommendation: Focus on Precision@10, Recall@10, and MRR

4. **Deployment Target**: Where will this be deployed?
   - Options: Local Docker, Cloud (AWS/GCP), Kubernetes

5. **Cost Optimization**: How to minimize API costs?
   - Recommendation: Implement embedding cache, use batch processing

---

## Resources & Documentation

- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Neo4j Python Driver**: https://neo4j.com/docs/api/python-driver/
- **pgvector**: https://github.com/pgvector/pgvector
- **Google Generative AI**: https://ai.google.dev/docs
- **LangSmith**: https://docs.smith.langchain.com/

---

## Contact & Support

For questions about implementation:
1. Check `README.md` for architecture details
2. Check `IMPLEMENTATION_STATUS.md` for current progress
3. Review code comments in `src/agrag/`

---

**Last Updated**: 2025-11-28
**Current Phase**: Phase 5 (Data Ingestion) - Ready to Begin
**Next Immediate Action**: Run `poetry run agrag init` to initialize databases
