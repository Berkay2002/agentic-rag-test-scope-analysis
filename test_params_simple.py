#!/usr/bin/env python3
"""Simple test for parameter handling without requiring API calls."""

from agrag.cli.headless import run_headless
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
    success = True
    if settings.max_tool_calls != original_max_tool_calls:
        print("❌ max_tool_calls not restored!")
        success = False
    if settings.agent_temperature != original_agent_temperature:
        print("❌ agent_temperature not restored!")
        success = False
    if settings.enable_pii_detection != original_enable_pii_detection:
        print("❌ enable_pii_detection not restored!")
        success = False

    if success:
        print("\n✅ All settings restored correctly!")

    return success


def test_parameter_filtering():
    """Test that reserved parameters are filtered out."""
    print("\n=== Testing Parameter Filtering ===\n")

    # Test with reserved parameters that should be ignored
    original_debug = settings.debug

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

    if settings.debug != original_debug:
        print("❌ Reserved parameter 'debug' was not filtered!")
        return False
    elif settings.max_tool_calls == 8:
        print("✅ Non-reserved parameter was applied correctly!")
        return True
    else:
        print("❌ Non-reserved parameter was not applied!")
        return False


if __name__ == "__main__":
    print("Testing parameter handling in headless mode...\n")

    success1 = test_parameter_override_and_restore()
    success2 = test_parameter_filtering()

    # Restore settings to original values
    settings.max_tool_calls = 35  # Default value
    settings.agent_temperature = 1.0  # Default value
    settings.enable_pii_detection = True  # Default value

    if success1 and success2:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")