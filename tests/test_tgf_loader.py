"""Test script for TGF CSV loader functionality."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agrag.data.loaders.tgf_loader import TGFCSVLoader


def test_tgf_loader():
    """Test TGF CSV loader with sample data."""

    print("=" * 60)
    print("TGF CSV Loader Test")
    print("=" * 60)

    # Path to sample CSV
    csv_path = Path(__file__).parent.parent / "data" / "examples" / "tgf_sample.csv"

    if not csv_path.exists():
        print(f"✗ Sample CSV not found: {csv_path}")
        return False

    print(f"\n[1] Loading CSV: {csv_path.name}")

    try:
        loader = TGFCSVLoader(file_path=str(csv_path))
        documents = loader.load()

        print(f"✓ Loaded {len(documents)} test cases")

    except Exception as e:
        print(f"✗ Loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Display statistics
    print("\n[2] Statistics")
    stats = loader.get_statistics()

    print(f"  Total tests: {stats['total_tests']}")
    print(f"  Feature areas: {stats['unique_feature_areas']}")
    print(f"  Test types: {stats['unique_test_types']}")
    print("\n  Result distribution:")
    for result, count in stats["result_distribution"].items():
        print(f"    {result}: {count}")

    print(f"\n  Avg requirements/test: {stats['avg_requirements_per_test']:.2f}")
    print(f"  Avg functions/test: {stats['avg_functions_per_test']:.2f}")

    # Show sample document
    print("\n[3] Sample Document")
    if documents:
        doc = documents[0]
        print(f"\n  Chunk ID: {doc.metadata.get('chunk_id')}")
        print(f"  Entity Type: {doc.metadata.get('entity_type')}")
        print("\n  Content Preview:")
        print("  " + "\n  ".join(doc.page_content.split("\n")[:5]))

        entity = doc.metadata.get("entity", {})
        print(f"\n  Entity ID: {entity.get('id')}")
        print(f"  Test Type: {entity.get('test_type')}")
        print(f"  File Path: {entity.get('file_path')}")

        relationships = doc.metadata.get("relationships", [])
        print(f"\n  Relationships: {len(relationships)}")
        for rel in relationships[:3]:
            print(f"    - {rel['type']} -> {rel['target_label']}:{rel['target_id']}")

    # Test filtering
    print("\n[4] Testing Result Filtering")
    loader_filtered = TGFCSVLoader(file_path=str(csv_path), filter_results=["FAIL", "ERROR"])
    docs_filtered = loader_filtered.load()
    print(f"  ✓ Filtered to {len(docs_filtered)} FAIL/ERROR tests")

    # Test record parsing
    print("\n[5] Testing Record Parsing")
    if loader.records:
        record = loader.records[0]
        print(f"  Test ID: {record.test_id}")
        print(f"  Test Name: {record.test_name}")
        print(f"  Test Type: {record.test_type}")
        print(f"  Result: {record.result}")
        print(f"  Requirements: {record.requirement_ids}")
        print(f"  Functions: {record.function_names}")
        print(f"  Tags: {record.tags}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_tgf_loader()
    sys.exit(0 if success else 1)
