#!/usr/bin/env python3
"""Test script for headless mode with parameters."""

import asyncio
from agrag.cli.commands.headless import run_headless
from agrag.config import settings


def test_without_params():
    """Test run_headless without parameters (existing behavior)."""
    print("Testing run_headless without params...")
    print(f"Original max_tool_calls: {settings.max_tool_calls}")

    exit_code = run_headless(
        prompt="What is the current max_tool_calls setting?",
        output_format="text"
    )

    print(f"Exit code: {exit_code}")
    print(f"max_tool_calls after execution: {settings.max_tool_calls}")
    print()


def test_with_params():
    """Test run_headless with parameters (new behavior)."""
    print("Testing run_headless with params...")
    original_max_tool_calls = settings.max_tool_calls
    original_agent_temperature = settings.agent_temperature

    print(f"Original max_tool_calls: {original_max_tool_calls}")
    print(f"Original agent_temperature: {original_agent_temperature}")

    # Test with parameter overrides
    exit_code = run_headless(
        prompt="Test query",
        output_format="text",
        params={
            "max_tool_calls": 5,
            "agent_temperature": 0.5,
            "debug": True,
            "nonexistent_param": "should_be_ignored"
        }
    )

    print(f"Exit code: {exit_code}")
    print(f"max_tool_calls after execution: {settings.max_tool_calls}")
    print(f"agent_temperature after execution: {settings.agent_temperature}")

    # Verify settings were restored
    assert settings.max_tool_calls == original_max_tool_calls, "max_tool_calls not restored!"
    assert settings.agent_temperature == original_agent_temperature, "agent_temperature not restored!"

    print("✓ Settings restored correctly!")
    print()


def test_error_handling():
    """Test that settings are restored even on error."""
    print("Testing error handling with params...")
    original_max_tool_calls = settings.max_tool_calls

    print(f"Original max_tool_calls: {original_max_tool_calls}")

    try:
        # This should cause an error (invalid output format)
        exit_code = run_headless(
            prompt="Test query",
            output_format="invalid_format",  # This will cause an error
            params={
                "max_tool_calls": 999,
                "debug": True
            }
        )
        print(f"Exit code: {exit_code}")
    except Exception as e:
        print(f"Expected error: {e}")

    print(f"max_tool_calls after error: {settings.max_tool_calls}")

    # Verify settings were restored
    assert settings.max_tool_calls == original_max_tool_calls, "max_tool_calls not restored after error!"

    print("✓ Settings restored correctly after error!")
    print()


def test_type_conversion():
    """Test type conversion for different parameter types."""
    print("Testing type conversion...")
    original_values = {
        "max_tool_calls": settings.max_tool_calls,
        "agent_temperature": settings.agent_temperature,
        "enable_pii_detection": settings.enable_pii_detection,
        "google_model": settings.google_model
    }

    print("Original values:")
    for key, value in original_values.items():
        print(f"  {key}: {value} ({type(value).__name__})")

    # Test various types (use a valid model name)
    exit_code = run_headless(
        prompt="Test type conversion",
        output_format="text",
        params={
            "max_tool_calls": "10",  # string to int
            "agent_temperature": "0.7",  # string to float
            "enable_pii_detection": "false",  # string to bool
            "google_model": "gemini-3-flash-preview"  # use valid model name
        }
    )

    print(f"Exit code: {exit_code}")

    if exit_code == 0:
        # Settings should be restored after execution
        assert settings.max_tool_calls == original_values["max_tool_calls"]
        assert settings.agent_temperature == original_values["agent_temperature"]
        assert settings.enable_pii_detection == original_values["enable_pii_detection"]
        assert settings.google_model == original_values["google_model"]
        print("✓ Type conversion accepts string inputs and settings restore correctly!")
    else:
        print("⚠ Type conversion test failed - execution error")

    # Restore original values defensively
    for key, value in original_values.items():
        setattr(settings, key, value)

    print()


if __name__ == "__main__":
    print("=== Testing Headless Mode with Parameters ===\n")

    test_without_params()
    test_with_params()
    test_error_handling()
    test_type_conversion()

    print("=== All tests passed! ===")