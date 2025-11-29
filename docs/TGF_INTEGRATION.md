# TGF Data Integration Guide

This guide explains how to integrate real test execution data from Ericsson's Test Governance Framework (TGF) into the AgRAG system.

## Overview

The TGF loader (`TGFCSVLoader`) enables you to import test results from Ericsson's internal testing systems into the AgRAG knowledge graph. This creates a comprehensive test scope analysis system combining:

- **Test Cases** from TGF exports
- **Requirements** linked via `requirement_ids`
- **Functions** under test via `function_names`
- **Test Results** (PASS/FAIL/ERROR/SKIP) with execution metrics

## CSV Format

### Required Columns

The TGF CSV export must contain these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `test_id` | String | Unique test identifier | `TC_HANDOVER_001` |
| `test_suite` | String | Test suite/group name | `Handover Tests` |
| `test_name` | String | Descriptive test name | `X2 Handover Success` |
| `test_type` | String | Test category | `protocol`, `integration`, `system`, `performance`, `unit`, `regression` |
| `feature_area` | String | Feature being tested | `Handover`, `Authentication`, `RRC` |
| `result` | String | Test result | `PASS`, `FAIL`, `SKIP`, `ERROR` |

### Optional Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `sub_feature` | String | Sub-feature/module | `X2`, `S1` |
| `requirement_ids` | String | Semicolon-separated requirement IDs | `REQ_HANDOVER_001;REQ_HANDOVER_003` |
| `function_names` | String | Semicolon-separated function names | `initiate_x2_handover;validate_handover_complete` |
| `execution_time_ms` | Integer | Execution time in milliseconds | `245` |
| `timestamp` | String | Execution timestamp (ISO 8601) | `2024-01-15T10:23:45Z` |
| `failure_reason` | String | Failure description if FAIL/ERROR | `Handover timeout after 5000ms` |
| `test_file_path` | String | Path to test file | `tests/protocol/test_handover_x2.py` |
| `code_coverage_pct` | Float | Code coverage percentage | `87.5` |
| `priority` | String | Test priority | `critical`, `high`, `medium`, `low` |
| `tags` | String | Semicolon-separated tags | `x2;handover;protocol` |

### Example CSV

See `data/examples/tgf_sample.csv` for a complete example with 15 test cases covering various scenarios.

## Usage

### 1. Export TGF Data

Export your test results from Ericsson's TGF system to CSV format with the columns above.

**Export options:**
- Full test suite (all results)
- Filtered by date range (last sprint, last release)
- Filtered by feature area (specific component)
- Filtered by result (failed tests only for debugging)

### 2. Load TGF Data into AgRAG

```bash
# Load all test results
poetry run agrag load tgf /path/to/tgf_export.csv

# Load only failed/error tests
poetry run agrag load tgf tests.csv --filter-results FAIL,ERROR

# Load without statistics display
poetry run agrag load tgf tests.csv --no-show-stats
```

### 3. Query Test Scope

Once loaded, you can query the test data:

```bash
# Find failed tests
poetry run agrag query "What tests failed in the handover feature?"

# Analyze test coverage
poetry run agrag query "Which requirements are verified by protocol tests?"

# Find test dependencies
poetry run agrag query "What functions are covered by authentication tests?"

# Debug specific failures
poetry run agrag query "Why did TC_HANDOVER_002 fail?"
```

## Data Model

### Entities Created

The TGF loader creates **TestCase** entities in Neo4j with the following properties:

```python
{
  "id": "TC_HANDOVER_001",
  "name": "X2 Handover Success",
  "description": "Handover Tests: X2 Handover Success",
  "test_type": "protocol",
  "file_path": "tests/protocol/test_handover_x2.py",
  "metadata": {
    "test_suite": "Handover Tests",
    "feature_area": "Handover",
    "sub_feature": "X2",
    "result": "PASS",
    "execution_time_ms": 245,
    "timestamp": "2024-01-15T10:23:45Z",
    "code_coverage_pct": 87.5,
    "priority": "critical",
    "tags": ["x2", "handover", "protocol"]
  }
}
```

### Relationships Created

1. **TestCase -[:VERIFIES]-> Requirement**
   - Created for each requirement_id in the CSV
   - Properties: `verified_at` (timestamp), `result` (PASS/FAIL/etc.)

2. **TestCase -[:COVERS]-> Function**
   - Created for each function_name in the CSV
   - Properties: `coverage_pct`, `execution_time_ms`

### Vector Embeddings

Each test case gets a vector embedding generated from:
- Test name
- Test suite
- Test type
- Feature area
- Sub-feature
- Requirement IDs
- Function names
- Failure reason (if applicable)

This enables semantic search like: "find tests related to handover timeouts"

## Advanced Usage

### Filtering by Result

Load only specific test results for focused analysis:

```bash
# Load only failed tests
poetry run agrag load tgf tests.csv --filter-results FAIL

# Load failures and errors
poetry run agrag load tgf tests.csv --filter-results FAIL,ERROR

# Load skipped tests to analyze coverage gaps
poetry run agrag load tgf tests.csv --filter-results SKIP
```

### Programmatic Usage

```python
from agrag.data.loaders.tgf_loader import TGFCSVLoader
from agrag.data.ingestion import DataIngestion

# Load TGF data
loader = TGFCSVLoader(
    file_path="tgf_export.csv",
    filter_results=["FAIL", "ERROR"]
)
documents = loader.load()

# Get statistics
stats = loader.get_statistics()
print(f"Total tests: {stats['total_tests']}")
print(f"Result distribution: {stats['result_distribution']}")

# Ingest into databases
ingestion = DataIngestion()
# ... (see CLI implementation for full ingestion logic)
```

### Integration with Existing Data

The TGF loader integrates seamlessly with:

1. **Code Repository Data** (`agrag load repo`)
   - Functions from code analysis are linked to test cases
   - Creates bidirectional relationships

2. **Requirements Documents** (`agrag load docs`)
   - Requirements from PDFs/DOCX are linked to test cases
   - Validates requirement coverage

3. **Synthetic Data** (`agrag generate`)
   - Can be used alongside synthetic data for demo/testing
   - Real TGF data overrides synthetic test cases

## Data Validation

The loader performs automatic validation:

1. **Test Type Normalization**
   - Maps various test type names to standard values
   - `functional` → `integration`
   - `acceptance` → `system`

2. **Result Normalization**
   - Standardizes result values
   - `passed` → `PASS`
   - `failed` → `FAIL`
   - `blocked` → `SKIP`

3. **Semicolon List Parsing**
   - Automatically splits `requirement_ids`, `function_names`, `tags`
   - Handles empty values gracefully

4. **Type Coercion**
   - Converts `execution_time_ms` to integer
   - Converts `code_coverage_pct` to float
   - Validates timestamp format

## Statistics and Monitoring

The loader provides comprehensive statistics:

```
=== TGF Data Statistics ===
Total tests: 15

Result Distribution:
  PASS: 11
  FAIL: 2
  ERROR: 1
  SKIP: 1

Feature Areas: 8
Test Types: 5
Avg Requirements/Test: 1.93
Avg Functions/Test: 2.40
```

## Troubleshooting

### Common Issues

**Issue: CSV file not found**
```
✗ CSV file does not exist: /path/to/file.csv
```
**Solution:** Verify the file path is correct and file exists

**Issue: Missing required columns**
```
KeyError: 'test_id'
```
**Solution:** Ensure CSV has all required columns (see CSV Format section)

**Issue: Invalid timestamp format**
```
Warning: Failed to parse row X: Invalid timestamp format
```
**Solution:** Use ISO 8601 format: `YYYY-MM-DDTHH:MM:SSZ`

**Issue: Duplicate test IDs**
```
Neo4j error: Constraint violation on test_id
```
**Solution:** Test IDs must be unique. Clean existing data with `agrag reset`

### Best Practices

1. **Incremental Loading**
   - Reset databases before loading new TGF export
   - Use `agrag reset` to avoid duplicates

2. **Filter by Date**
   - Export TGF data for specific time periods
   - Focus on recent test results for current analysis

3. **Requirement Mapping**
   - Ensure requirement_ids match requirement IDs in your documentation
   - Use consistent naming conventions

4. **Function Mapping**
   - Function names should match actual function names in code
   - Use fully qualified names if needed (e.g., `module.ClassName.method_name`)

5. **Tags for Organization**
   - Use tags to group related tests
   - Examples: `bug`, `regression`, `smoke_test`, `ci`

## Next Steps

After loading TGF data:

1. **Verify Data**
   ```bash
   poetry run agrag load stats
   ```

2. **Test Queries**
   ```bash
   poetry run agrag query "Show test coverage for handover requirements"
   ```

3. **Run Evaluation**
   - Create evaluation dataset with relevant test queries
   - Measure retrieval quality (Precision@k, MAP, MRR)

4. **Continuous Integration**
   - Automate TGF export from your CI/CD pipeline
   - Load fresh test results after each test run
   - Query for regression analysis

## Research Applications

This TGF integration supports research questions:

### RQ1: Knowledge Graph Ontology
- Real-world test data validates ontology design
- Relationship patterns (VERIFIES, COVERS) emerge from actual usage

### RQ2: Retrieval Strategy Comparison
- Benchmark retrieval on real test scope queries
- Compare vector search vs. keyword search on TGF data

### RQ3: Human-in-the-Loop Workflows
- Interactive debugging of test failures
- Agent-assisted test coverage analysis
- HITL verification of requirement links

## Example Queries

```bash
# Find all tests for a feature area
poetry run agrag query "What tests cover the Handover feature?"

# Analyze test failures
poetry run agrag query "Which tests failed and why?"

# Coverage analysis
poetry run agrag query "What requirements have no test coverage?"

# Function coverage
poetry run agrag query "Which functions are tested by security tests?"

# Performance analysis
poetry run agrag query "What is the average execution time for protocol tests?"

# Tag-based search
poetry run agrag query "Find all regression tests marked as critical"
```

## References

- Main documentation: `README.md`
- Developer guide: `AGENTS.md`
- Example data: `data/examples/tgf_sample.csv`
- Loader implementation: `src/agrag/data/loaders/tgf_loader.py`
- CLI command: `src/agrag/cli/main.py` (load_tgf command)
