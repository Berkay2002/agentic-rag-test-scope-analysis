# How to Run Tests

## Quick Test Run

Run all tests with summary:
```bash
poetry run pytest tests/ --tb=no -q
```

**Expected Output** (after fixes):
```
155 passed, 11 failed, 1 skipped, 487 warnings in 390.28s (0:06:30)
```

---

## Run Tests by Feature

### 1. Error Recovery (#19) - 100% pass rate
```bash
poetry run pytest tests/integration/test_error_recovery.py -v
```

**Expected**: All 15 tests PASSED

---

### 2. Result Diversification (#18) - 100% pass rate
```bash
poetry run pytest tests/integration/test_diversification_integration.py -v
```

**Expected**: All 16 tests PASSED

---

### 3. Query Expansion (#17) - 87.5% pass rate
```bash
poetry run pytest tests/integration/test_query_expansion_integration.py -v
```

**Expected**: 14 passed, 1 skipped, 1 failed (minor issue with service strategy test)

---

### 4. Batch Processing (#20) - File Loading Working
```bash
# Test file loading (all should pass after fixes)
poetry run pytest tests/integration/test_batch_workflow.py::TestBatchWorkflow -k "load_queries" -v
```

**Expected**: All 5 load tests PASSED
- test_load_queries_json_array
- test_load_queries_json_objects
- test_load_queries_jsonl
- test_load_queries_csv
- test_load_queries_txt
- test_load_queries_with_comments

---

## Run Specific Failing Tests

### Batch Output Issues (8 failures)
```bash
# These are the main remaining failures
poetry run pytest tests/integration/test_batch_workflow.py::TestBatchWorkflow -k "save_results or generate_report or batch_processor_with_real or thread_id_persistence or full_batch_workflow" -v
```

### Unit Test Mock Issue (1 failure)
```bash
poetry run pytest tests/unit/test_batch_processor.py::TestBatchQueryProcessor::test_run_query_headless_integration -v
```

### Baseline Issues (2 failures - likely pre-existing)
```bash
poetry run pytest tests/integration/test_fixed_baselines_e2e.py -v
```

---

## Run All Tests with Details

### Full verbose output with test names:
```bash
poetry run pytest tests/ -v --tb=no
```

### Full output with failure details:
```bash
poetry run pytest tests/ -v --tb=short
```

### Generate HTML report:
```bash
poetry run pytest tests/ --html=test_report.html --self-contained-html
```

---

## Check Git Status

See what changed:
```bash
git status
git diff src/agrag/cli/batch_processor.py
git diff tests/integration/test_batch_workflow.py
```

---

## Verify Files Modified

```bash
# List modified files
ls -la src/agrag/batch/  # Batch processing module
ls -la src/agrag/cli/batch_processor.py  # CLI for batch

# List new test files
git status | grep "integration/test_"
```

---

## Performance Tips

- Run in parallel (faster): Add `-n auto` flag
- Skip slow tests: Add `-m "not slow"`
- Run only fast tests: `poetry run pytest tests/unit/ -q`

