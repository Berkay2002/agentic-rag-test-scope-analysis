"""Shared utility functions for CLI modules."""

from typing import Any, Dict, List, Tuple


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


def extract_reasoning_and_answer(content: Any) -> Tuple[List[str], str]:
    """Extract reasoning blocks and final answer from message content.
    
    This function separates "thinking" or "reasoning" blocks from the final
    answer text, enabling separate display of agent reasoning process.
    
    Args:
        content: Message content (can be string, list of blocks, or dict).
        
    Returns:
        Tuple of (reasoning_blocks, answer_text):
        - reasoning_blocks: List of reasoning/thinking content strings
        - answer_text: The final answer text
    """
    reasoning_blocks: List[str] = []
    answer_parts: List[str] = []
    
    if isinstance(content, str):
        # Plain string - treat as answer
        return ([], content)
    
    elif isinstance(content, list):
        # Handle content blocks (Gemini format with type metadata)
        for part in content:
            if isinstance(part, dict):
                # Check for reasoning/thinking blocks
                part_type = part.get("type", "").lower()
                text = part.get("text", "")
                
                # Gemini thinking blocks have type "thinking" or contain "thought" metadata
                if part_type in ("thinking", "thought", "reasoning"):
                    if text:
                        reasoning_blocks.append(text)
                elif "text" in part:
                    # Regular text block - part of answer
                    answer_parts.append(text)
            elif isinstance(part, str):
                # Plain string part - treat as answer
                answer_parts.append(part)
    
    elif isinstance(content, dict):
        # Single block - check type
        part_type = content.get("type", "").lower()
        text = content.get("text", str(content))
        
        if part_type in ("thinking", "thought", "reasoning"):
            reasoning_blocks.append(text)
        else:
            answer_parts.append(text)
    
    else:
        # Other type - convert to string and treat as answer
        answer_parts.append(str(content))
    
    answer_text = "\n".join(answer_parts) if answer_parts else ""
    return (reasoning_blocks, answer_text)
