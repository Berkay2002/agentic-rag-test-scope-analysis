#!/usr/bin/env python3
"""Demonstration of batch processing with parameters."""

import asyncio
import json
from agrag.cli.batch_processor import BatchProcessor
from agrag.config import settings


async def demo_batch_processing():
    """Demonstrate batch processing with different parameter sets."""
    print("=== Batch Processing Demo ===\n")

    # Store original settings
    original_settings = {
        "max_tool_calls": settings.max_tool_calls,
        "agent_temperature": settings.agent_temperature,
        "enable_pii_detection": settings.enable_pii_detection,
        "default_retrieval_k": settings.default_retrieval_k,
    }

    print("Original settings:")
    for key, value in original_settings.items():
        print(f"  {key}: {value}")

    # Create batch processor with default parameters
    processor = BatchProcessor(
        thread_id="demo-batch-1",
        max_concurrent=2,
        default_params={
            "max_tool_calls": 3,
            "agent_temperature": 0.2,
            "enable_pii_detection": False,
            "debug": False
        }
    )

    # Define queries with different parameter requirements
    queries = [
        {
            "query": "Find tests for REQ_HANDOVER_001",
            "max_tool_calls": 5,  # Override default
            "vector_search_similarity_threshold": 0.8,
        },
        {
            "query": "Search for authentication functions",
            "agent_temperature": 0.8,  # Override default
            "default_retrieval_k": 20,
        },
        "Simple query without parameters",  # Will use defaults
        {
            "query": "Find all bearer establishment tests",
            "max_tool_calls": 10,
            "graph_traversal_max_depth": 5,
            "enable_query_expansion": True,
        }
    ]

    print(f"\nProcessing {len(queries)} queries...")

    # Process queries
    results = await processor.process_queries(queries)

    # Display results
    print(f"\nResults:")
    print(f"  Total queries: {len(results)}")
    print(f"  Successful: {sum(1 for r in results if r.status == 'success')}")
    print(f"  Failed: {sum(1 for r in results if r.status == 'error')}")

    # Show details for each query
    print("\nQuery Details:")
    for i, result in enumerate(results, 1):
        print(f"\nQuery {i}: {result.query[:50]}...")
        print(f"  Status: {result.status}")
        if result.metadata:
            print(f"  Parameters used: {json.dumps(result.metadata, indent=2)}")
        print(f"  Execution time: {result.execution_time_ms:.2f}ms")

    # Generate and display report
    report = processor.generate_report()
    print(f"\nBatch Processing Report:")
    print(f"  Thread ID: {report.thread_id}")
    print(f"  Total queries: {report.total_queries}")
    print(f"  Successful: {report.successful_queries}")
    print(f"  Failed: {report.failed_queries}")
    print(f"  Average execution time: {report.average_execution_time_ms:.2f}ms")
    print(f"  Total time: {report.total_execution_time_ms:.2f}ms")

    # Verify settings were restored
    print(f"\nVerifying settings restoration:")
    all_restored = True
    for key, original_value in original_settings.items():
        current_value = getattr(settings, key)
        print(f"  {key}: {current_value} (original: {original_value})")
        if current_value != original_value:
            print(f"    ❌ NOT RESTORED!")
            all_restored = False

    if all_restored:
        print("\n✅ All settings restored correctly!")
    else:
        print("\n❌ Some settings were not restored!")

    # Save results to file
    output_file = "/tmp/batch_results.json"
    processor.save_results(output_file, format="json")
    print(f"\nResults saved to: {output_file}")

    return all_restored


async def demo_error_handling():
    """Demonstrate error handling with parameter restoration."""
    print("\n=== Error Handling Demo ===\n")

    original_max_tool_calls = settings.max_tool_calls

    print(f"Original max_tool_calls: {original_max_tool_calls}")

    # Create processor that will cause errors
    processor = BatchProcessor(
        thread_id="error-demo",
        default_params={
            "max_tool_calls": 999,
            "debug": True
        }
    )

    # Queries that might cause issues
    queries = [
        {
            "query": "This is a test query",
            "max_tool_calls": 1000,
        }
    ]

    print("Processing queries that might cause errors...")
    results = await processor.process_queries(queries)

    print(f"\nResults after potential errors:")
    for result in results:
        print(f"  Query: {result.query}")
        print(f"  Status: {result.status}")
        if result.error_message:
            print(f"  Error: {result.error_message[:100]}...")

    print(f"\nmax_tool_calls after error handling: {settings.max_tool_calls}")

    if settings.max_tool_calls == original_max_tool_calls:
        print("✅ Settings restored correctly after errors!")
        return True
    else:
        print("❌ Settings not restored after errors!")
        return False


async def main():
    """Run all demos."""
    print("Starting batch processing demonstrations...\n")

    # Run demos
    success1 = await demo_batch_processing()
    success2 = await demo_error_handling()

    # Final summary
    print("\n" + "="*50)
    print("DEMO SUMMARY")
    print("="*50)

    if success1 and success2:
        print("✅ All demonstrations completed successfully!")
        print("\nThe batch processing system is ready for use with:")
        print("  - Parameterized execution per query")
        print("  - Automatic settings restoration")
        print("  - Error handling and recovery")
        print("  - Comprehensive reporting")
    else:
        print("❌ Some demonstrations failed!")

    print("\nTo use in your application:")
    print("  from agrag.cli.batch_processor import BatchProcessor")
    print("  processor = BatchProcessor(thread_id='your-batch')")
    print("  results = await processor.process_queries(queries_with_params)")


if __name__ == "__main__":
    asyncio.run(main())