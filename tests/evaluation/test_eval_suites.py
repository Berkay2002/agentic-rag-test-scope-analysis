import json
from pathlib import Path


def test_eval_suites_registry_is_valid():
    repo_root = Path(__file__).resolve().parents[2]
    registry_path = repo_root / "data" / "eval_suites.json"
    assert registry_path.exists()

    payload = json.loads(registry_path.read_text())
    assert payload.get("version") == 1

    suites = payload.get("suites")
    assert isinstance(suites, list)
    assert suites, "Expected at least one suite"

    for suite in suites:
        assert suite.get("name")
        assert suite.get("type") in {"capability", "regression"}
        assert suite.get("owner")
        datasets = suite.get("datasets", [])
        assert datasets, f"Suite '{suite.get('name')}' has no datasets"

        for dataset in datasets:
            path_value = dataset.get("path")
            assert path_value, f"Suite '{suite.get('name')}' dataset missing path"
            dataset_path = (repo_root / path_value).resolve()
            assert dataset_path.exists(), f"Dataset not found: {dataset_path}"
            assert dataset.get("strategy")
