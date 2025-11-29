"""Standalone test for TGF CSV loader (no dependencies)."""

import csv
from pathlib import Path


def test_tgf_csv_format():
    """Validate TGF sample CSV format and content."""

    print("=" * 60)
    print("TGF CSV Format Validation")
    print("=" * 60)

    csv_path = Path(__file__).parent.parent / "data" / "examples" / "tgf_sample.csv"

    if not csv_path.exists():
        print(f"✗ Sample CSV not found: {csv_path}")
        return False

    print(f"\n✓ Found CSV: {csv_path.name}")

    # Expected columns
    required_cols = ["test_id", "test_suite", "test_name", "test_type", "feature_area", "result"]

    optional_cols = [
        "sub_feature",
        "requirement_ids",
        "function_names",
        "execution_time_ms",
        "timestamp",
        "failure_reason",
        "test_file_path",
        "code_coverage_pct",
        "priority",
        "tags",
    ]

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)

            # Check headers
            headers = reader.fieldnames
            print(f"\n[1] Headers ({len(headers)} columns)")

            missing_required = [col for col in required_cols if col not in headers]
            if missing_required:
                print(f"  ✗ Missing required columns: {missing_required}")
                return False

            print("  ✓ All required columns present")

            extra_cols = [col for col in headers if col not in required_cols + optional_cols]
            if extra_cols:
                print(f"  ⚠ Extra columns: {extra_cols}")

            # Read all rows
            rows = list(reader)
            print(f"\n[2] Data ({len(rows)} test cases)")

            # Analyze data
            results = {}
            feature_areas = set()
            test_types = set()
            req_counts = []
            func_counts = []

            for row in rows:
                # Count results
                result = row["result"]
                results[result] = results.get(result, 0) + 1

                # Track feature areas
                feature_areas.add(row["feature_area"])

                # Track test types
                test_types.add(row["test_type"])

                # Count requirements
                reqs = [r.strip() for r in row["requirement_ids"].split(";") if r.strip()]
                req_counts.append(len(reqs))

                # Count functions
                funcs = [f.strip() for f in row["function_names"].split(";") if f.strip()]
                func_counts.append(len(funcs))

            print("\n[3] Statistics")
            print(f"  Total tests: {len(rows)}")
            print(f"  Feature areas: {len(feature_areas)}")
            print(f"  Test types: {len(test_types)}")

            print("\n  Result distribution:")
            for result, count in sorted(results.items()):
                print(f"    {result}: {count}")

            avg_reqs = sum(req_counts) / len(req_counts) if req_counts else 0
            avg_funcs = sum(func_counts) / len(func_counts) if func_counts else 0
            print(f"\n  Avg requirements/test: {avg_reqs:.2f}")
            print(f"  Avg functions/test: {avg_funcs:.2f}")

            print("\n[4] Sample Records")
            for i, row in enumerate(rows[:3], 1):
                print(f"\n  Test {i}:")
                print(f"    ID: {row['test_id']}")
                print(f"    Name: {row['test_name']}")
                print(f"    Type: {row['test_type']}")
                print(f"    Feature: {row['feature_area']}")
                print(f"    Result: {row['result']}")
                if row["requirement_ids"]:
                    reqs = row["requirement_ids"].split(";")
                    print(f"    Requirements: {', '.join(reqs[:3])}")
                if row["function_names"]:
                    funcs = row["function_names"].split(";")
                    print(f"    Functions: {', '.join(funcs[:2])}")

            print("\n" + "=" * 60)
            print("✓ CSV format validation passed!")
            print("=" * 60)

            return True

    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    success = test_tgf_csv_format()
    sys.exit(0 if success else 1)
