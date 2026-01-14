#!/usr/bin/env python3
"""Direct test of parameter handling logic without execution."""

from agrag.config import settings


def test_parameter_logic():
    """Test the parameter handling logic directly."""
    print("=== Testing Parameter Logic Directly ===\n")

    # Store original values
    original_max_tool_calls = settings.max_tool_calls
    original_agent_temperature = settings.agent_temperature
    original_enable_pii_detection = settings.enable_pii_detection

    print(f"Original settings:")
    print(f"  max_tool_calls: {original_max_tool_calls}")
    print(f"  agent_temperature: {original_agent_temperature}")
    print(f"  enable_pii_detection: {original_enable_pii_detection}")

    # Simulate what happens in run_headless
    params = {
        "max_tool_calls": "15",  # string to int
        "agent_temperature": "0.8",  # string to float
        "enable_pii_detection": "false",  # string to bool
        "debug": True,  # reserved parameter - should be ignored
        "output_format": "json",  # reserved parameter - should be ignored
    }

    original_values = {}

    # Apply parameter overrides if provided
    if params:
        for key, value in params.items():
            # Skip reserved parameters that are function arguments
            if key in ("output_format", "thread_id", "debug", "prompt"):
                print(f"  Skipping reserved parameter: {key}")
                continue

            # Check if the setting exists
            if hasattr(settings, key):
                # Store original value
                original_values[key] = getattr(settings, key)

                # Handle type conversion based on the original value type
                original_value = original_values[key]
                if original_value is not None:
                    target_type = type(original_value)
                    try:
                        # Convert value to the correct type
                        if target_type == bool and isinstance(value, str):
                            # Handle boolean conversion from string
                            value = value.lower() in ("true", "1", "yes", "on")
                            print(f"  Converted {key} from '{original_value}' to {value} (bool)")
                        elif target_type in (int, float) and value == "":
                            # Skip empty strings for numeric types
                            print(f"  Skipping empty value for {key}")
                            continue
                        else:
                            old_value = value
                            value = target_type(value)
                            print(f"  Converted {key} from '{old_value}' to {value} ({target_type.__name__})")
                    except (ValueError, TypeError) as e:
                        print(f"  ⚠ Could not convert {key}='{value}' to {target_type.__name__}: {e}")
                        continue

                # Set the new value
                setattr(settings, key, value)
                print(f"  ✓ Set {key} = {value}")
            else:
                print(f"  ⚠ Setting {key} not found in settings")

    print(f"\nSettings after overrides:")
    print(f"  max_tool_calls: {settings.max_tool_calls}")
    print(f"  agent_temperature: {settings.agent_temperature}")
    print(f"  enable_pii_detection: {settings.enable_pii_detection}")

    # Simulate restoration
    print(f"\nRestoring original settings...")
    if original_values:
        for key, value in original_values.items():
            setattr(settings, key, value)
            print(f"  ✓ Restored {key} = {value}")

    print(f"\nSettings after restoration:")
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


if __name__ == "__main__":
    try:
        test_parameter_logic()
    except AssertionError:
        print("\n❌ Parameter logic test failed!")
        raise
    else:
        print("\n✅ Parameter logic test passed!")
