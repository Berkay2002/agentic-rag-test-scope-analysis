# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Agentic GraphRAG system for Test Scope Analysis** - a Master's Thesis research project that combines Knowledge Graphs, Vector Search, and Human-in-the-Loop workflows for analyzing test coverage in telecommunications software systems.

## Key Commands

### Development Setup
```bash
# Install dependencies (requires Poetry 1.8+)
poetry install

# Create environment file
cp .env.example .env
# Edit .env with your API keys and database credentials

# Initialize database schemas
poetry run agrag init
```

### Running Tests
```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/unit/test_vector_search_tool.py

# Run with verbose output
poetry run pytest -v
```

### Code Quality
```bash
# Format code with Black (line length: 100)
poetry run black src/ tests/

# Lint with Ruff
poetry run ruff check src/ tests/

# Fix auto-fixable linting issues
poetry run ruff check --fix src/ tests/
```

### Common Development Tasks
```bash
# Generate synthetic test data
poetry run agrag generate --requirements 50 --testcases 200 --with-eval

# Load documents (AI-powered parsing with Docling)
poetry run agrag load docs /path/to/docs --use-chunker --formats pdf,docx

# Load code repository (AST-based parsing)
poetry run agrag load repo /path/to/repo --languages python,java

# Run evaluation
poetry run agrag evaluate --dataset data/eval_queries.json --strategy all --verbose

# Interactive development/testing
poetry run agrag chat --thread-id dev-session
```

## Architecture Overview

### Core Components

1. **Agent System** (`src/agrag/core/`)
   - Uses LangChain `create_agent` API with modern middleware support
   - Implements ReAct loop with tool execution
   - Supports Human-in-the-Loop via PostgresSaver checkpointing
   - Key files: `graph.py`, `state.py`, `checkpointing.py`

2. **Retrieval Tools** (`src/agrag/tools/`)
   - Four retrieval strategies using `@tool` decorator pattern:
     - Vector Search: PostgreSQL pgvector (semantic similarity)
     - Keyword Search: PostgreSQL pg_search BM25 (lexical matching)
     - Graph Traversal: Neo4j Cypher queries (structural relationships)
     - Hybrid Search: RRF fusion of vector + keyword
   - Tool factory functions for dependency injection

3. **Storage Layer** (`src/agrag/storage/`)
   - Dual architecture: Neo4j (knowledge graph) + PostgreSQL (pgvector + BM25)
   - Clients: `Neo4jClient`, `PostgresClient`, `BM25RetrieverManager`
   - Implements retry logic with exponential backoff

4. **Data Pipeline** (`src/agrag/data/`)
   - Document loading: Docling integration (15+ formats, AI-powered parsing)
   - Code loading: Tree-sitter AST parsing (multi-language support)
   - TGF loading: Ericsson Test Governance Framework CSV parser
   - Splitters: CodeSplitter (AST-based), MarkdownSplitter (header-based)
   - Storage writers: Idempotent upserts with batch processing

5. **Knowledge Graph Ontology** (`src/agrag/kg/`)
   - Entity types: ChangeRequest, File, Component, Requirement, TestCase, Function, Class, Module
   - Relationship types: TOUCHES, PART_OF, VERIFIES, COVERS, CALLS, DEFINED_IN, INHERITS_FROM, BELONGS_TO, DEPENDS_ON
   - Rich metadata support (priorities, test types, signatures)

6. **Evaluation Framework** (`src/agrag/evaluation/`)
   - Metrics: Precision@k, Recall@k, MAP, MRR, F1@k
   - Agentic evaluation with full pipeline testing
   - Entity extraction from natural language responses
   - Tool usage tracking and statistics

### Key Design Patterns

1. **Tool Factory Pattern**: All retrieval tools use factory functions (`create_*_tool`) for dependency injection
2. **Middleware Pattern**: Agent middleware for HITL, PII detection, call limits
3. **Storage Writer Pattern**: Modular writers for different storage backends
4. **Loader Pattern**: Abstract base classes with format-specific implementations

### Important Configuration

- **Python**: 3.11+ required
- **Databases**: Neo4j 5.20+ (with APOC), PostgreSQL 15+ (with pgvector, pg_search)
- **LLM**: Google Gemini (via GOOGLE_API_KEY)
- **Embeddings**: 768-dimensional vectors
- **Code Style**: Black (100 char line length), Ruff linting

### CLI Usage Patterns

1. **Interactive Chat** (recommended for development):
   ```bash
   poetry run agrag chat --thread-id my-session
   # Built-in commands: /help, /stats, /export, /thinking
   ```

2. **Headless Mode** (for scripting):
   ```bash
   poetry run agrag -p "query here" --output-format json
   ```

3. **Safe vs YOLO Mode**:
   - Default: Human approval before each tool execution
   - YOLO: Autonomous execution with `--yolo` flag

### Testing Strategy

- Unit tests in `tests/unit/` for individual components
- Integration tests for database operations
- Evaluation framework for retrieval quality
- Use `--thread-id` for persistent test sessions

### Common Pitfalls to Avoid

1. Always use `poetry run` prefix for commands
2. Initialize databases before first use (`agrag init`)
3. Set up environment variables from `.env.example`
4. Use factory functions when creating tools (not direct instantiation)
5. Batch operations for large datasets to avoid rate limits
6. Use idempotent operations for data ingestion
7. Check middleware configuration when debugging agent behavior