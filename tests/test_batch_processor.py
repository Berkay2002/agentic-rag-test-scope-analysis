#!/usr/bin/env python3
"""Test script for batch processor with parameters."""

import asyncio
import pytest
from agrag.cli.commands.batch_processor import BatchProcessor
from agrag.config import settings


@pytest.mark.asyncio
async def test_batch_with_params():
    """Test batch processing with parameters."""
    print("Testing batch processor with params...")

    # Store original settings
    original_values = {
        "max_tool_calls": settings.max_tool_calls,
        "agent_temperature": settings.agent_temperature,
        "enable_pii_detection": settings.enable_pii_detection,
    }

    print("Original settings:")
    for key, value in original_values.items():
        print(f"  {key}: {value}")

    # Create batch processor
    processor = BatchProcessor(
        thread_id="test-batch",
        max_concurrent=2,
        default_params={
            "max_tool_calls": 3,
            "agent_temperature": 0.3,
            "enable_pii_detection": False,
            "debug": True
        }
    )

    # Test queries
    queries = [
        "What is the max_tool_calls setting?",
        "Is PII detection enabled?",
        "What is the agent temperature?"
    ]

    # Process queries
    results = await processor.process_queries(queries)

    print(f"\nProcessed {len(results)} queries")
    print(f"Successful: {sum(1 for r in results if r.status == 'success')}")
    print(f"Failed: {sum(1 for r in results if r.status == 'error')}")

    # Verify settings were restored
    print("\nSettings after batch processing:")
    for key, value in original_values.items():
        current_value = getattr(settings, key)
        print(f"  {key}: {current_value}")
        assert current_value == original_values[key], f"{key} not restored!"

    print("\n✓ Settings restored correctly after batch processing!")

    # Show sample results
    if results:
        print("\nSample results:")
        for i, result in enumerate(results[:2]):
            print(f"  Query {i+1}: {result.query}")
            print(f"    Status: {result.status}")
            print(f"    Response: {result.response[:50]}...")
            print(f"    Execution time: {result.execution_time_ms:.2f}ms")


@pytest.mark.asyncio
async def test_batch_with_different_params():
    """Test that different queries can have different parameters."""
    print("\nTesting batch with per-query parameters...")

    # Create batch processor
    processor = BatchProcessor(
        thread_id="test-batch-2",
        max_concurrent=1  # Sequential for easier debugging
    )

    # Queries with different parameters
    queries = [
        {
            "query": "Query with low temperature",
            "max_tool_calls": 5,
            "agent_temperature": 0.1
        },
        {
            "query": "Query with high temperature",
            "max_tool_calls": 10,
            "agent_temperature": 0.9
        }
    ]

    # Process queries
    results = await processor.process_queries(queries)

    print(f"Processed {len(results)} queries with different parameters")
    for i, result in enumerate(results):
        print(f"  Query {i+1}: {result.query}")
        print(f"    Parameters: {result.metadata}")


if __name__ == "__main__":
    print("=== Testing Batch Processor with Parameters ===\n")

    asyncio.run(test_batch_with_params())
    asyncio.run(test_batch_with_different_params())

    print("\n=== Batch processor tests completed! ===")
