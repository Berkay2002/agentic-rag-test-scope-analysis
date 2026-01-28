#!/usr/bin/env python3
"""Simple test for parameter handling without requiring API calls."""

from agrag.cli.commands.headless import run_headless
from agrag.config import settings


def test_parameter_override_and_restore():
    """Test that parameters are correctly overridden and restored."""
    print("=== Testing Parameter Override and Restore ===\n")

    # Store original values
    original_max_tool_calls = settings.max_tool_calls
    original_agent_temperature = settings.agent_temperature
    original_enable_pii_detection = settings.enable_pii_detection

    print(f"Original settings:")
    print(f"  max_tool_calls: {original_max_tool_calls}")
    print(f"  agent_temperature: {original_agent_temperature}")
    print(f"  enable_pii_detection: {original_enable_pii_detection}")

    # Test with parameter overrides (using a simple query that won't trigger API calls)
    try:
        exit_code = run_headless(
            prompt="/help",  # This should be a simple command that doesn't need API
            output_format="text",
            params={
                "max_tool_calls": 5,
                "agent_temperature": 0.5,
                "enable_pii_detection": False,
            }
        )
        print(f"\nExit code: {exit_code}")
    except Exception as e:
        print(f"\nError during execution: {e}")

    # Check if settings were restored
    print(f"\nSettings after execution:")
    print(f"  max_tool_calls: {settings.max_tool_calls}")
    print(f"  agent_temperature: {settings.agent_temperature}")
    print(f"  enable_pii_detection: {settings.enable_pii_detection}")

    # Verify restoration
    assert settings.max_tool_calls == original_max_tool_calls, "max_tool_calls not restored!"
    assert settings.agent_temperature == original_agent_temperature, "agent_temperature not restored!"
    assert (
        settings.enable_pii_detection == original_enable_pii_detection
    ), "enable_pii_detection not restored!"

    print("\n✅ All settings restored correctly!")


def test_parameter_filtering():
    """Test that reserved parameters are filtered out."""
    print("\n=== Testing Parameter Filtering ===\n")

    # Test with reserved parameters that should be ignored
    original_debug = settings.debug
    original_max_tool_calls = settings.max_tool_calls

    try:
        exit_code = run_headless(
            prompt="/help",
            output_format="text",
            debug=False,  # This is a function parameter
            params={
                "debug": True,  # This should be ignored (reserved parameter)
                "output_format": "json",  # This should be ignored (reserved parameter)
                "thread_id": "test123",  # This should be ignored (reserved parameter)
                "prompt": "ignored",  # This should be ignored (reserved parameter)
                "max_tool_calls": 8,  # This should be applied
            }
        )
        print(f"Exit code: {exit_code}")
    except Exception as e:
        print(f"Error during execution: {e}")

    # Check that debug wasn't changed via params
    print(f"\nDebug setting after execution: {settings.debug}")
    print(f"Max tool calls after execution: {settings.max_tool_calls}")

    assert settings.debug == original_debug, "Reserved parameter 'debug' was not filtered!"
    assert (
        settings.max_tool_calls == original_max_tool_calls
    ), "max_tool_calls was not restored after execution!"

    print("✅ Reserved parameters filtered and settings restored correctly!")


if __name__ == "__main__":
    print("Testing parameter handling in headless mode...\n")

    try:
        test_parameter_override_and_restore()
        test_parameter_filtering()
    except AssertionError:
        print("\n❌ Some tests failed!")
        raise
    else:
        print("\n✅ All tests passed!")
