"""Shared utility functions for CLI modules."""

from typing import Any


def extract_message_content(content: Any) -> str:
    """Extract text content from message payloads.

    Handles multiple content formats:
    - Plain strings
    - List of content blocks (Gemini format with dict/text parts)
    - Dict with 'text' key
    - Other types (converted to string)

    Args:
        content: Message content (string, list of blocks, dict, or other).

    Returns:
        Extracted text content as a string.
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle content blocks (e.g., Gemini format)
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)
    elif isinstance(content, dict) and "text" in content:
        return str(content["text"])
    return str(content)
