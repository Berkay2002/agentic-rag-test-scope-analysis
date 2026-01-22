"""Unit tests for reasoning extraction from AI messages."""

import pytest
from agrag.cli.utils import extract_reasoning_and_answer, extract_message_content


def test_extract_plain_string():
    """Test extracting content from plain string."""
    content = "This is a plain text answer."
    reasoning, answer = extract_reasoning_and_answer(content)
    
    assert reasoning == []
    assert answer == "This is a plain text answer."


def test_extract_with_reasoning_blocks():
    """Test extracting reasoning blocks from structured content."""
    content = [
        {"type": "thinking", "text": "Let me think about this..."},
        {"type": "thinking", "text": "I need to analyze the requirements."},
        {"type": "text", "text": "Here is my final answer."},
    ]
    
    reasoning, answer = extract_reasoning_and_answer(content)
    
    assert len(reasoning) == 2
    assert "Let me think about this..." in reasoning[0]
    assert "I need to analyze the requirements." in reasoning[1]
    assert answer == "Here is my final answer."


def test_extract_without_reasoning():
    """Test extracting content without reasoning blocks."""
    content = [
        {"type": "text", "text": "This is part one."},
        {"type": "text", "text": "This is part two."},
    ]
    
    reasoning, answer = extract_reasoning_and_answer(content)
    
    assert reasoning == []
    assert "This is part one." in answer
    assert "This is part two." in answer


def test_extract_mixed_content():
    """Test extracting mixed content with reasoning and text."""
    content = [
        {"type": "thinking", "text": "Analyzing query..."},
        {"type": "text", "text": "First part of answer."},
        {"type": "text", "text": "Second part of answer."},
    ]
    
    reasoning, answer = extract_reasoning_and_answer(content)
    
    assert len(reasoning) == 1
    assert "Analyzing query..." in reasoning[0]
    assert "First part of answer." in answer
    assert "Second part of answer." in answer


def test_extract_dict_with_thinking_type():
    """Test extracting single dict with thinking type."""
    content = {"type": "thinking", "text": "Internal deliberation..."}
    
    reasoning, answer = extract_reasoning_and_answer(content)
    
    assert len(reasoning) == 1
    assert "Internal deliberation..." in reasoning[0]
    assert answer == ""


def test_extract_dict_with_text_type():
    """Test extracting single dict with text type."""
    content = {"type": "text", "text": "Final answer text."}
    
    reasoning, answer = extract_reasoning_and_answer(content)
    
    assert reasoning == []
    assert answer == "Final answer text."


def test_backward_compatibility_extract_message_content():
    """Test that extract_message_content still works for backward compatibility."""
    # Plain string
    assert extract_message_content("test") == "test"
    
    # List of blocks
    content = [
        {"type": "thinking", "text": "thinking..."},
        {"type": "text", "text": "answer"},
    ]
    result = extract_message_content(content)
    assert "thinking..." in result
    assert "answer" in result
    
    # Dict
    content = {"text": "content"}
    assert extract_message_content(content) == "content"


def test_extract_with_string_parts():
    """Test extracting content with plain string parts in list."""
    content = [
        "Plain string part",
        {"type": "text", "text": "Dict text part"},
    ]
    
    reasoning, answer = extract_reasoning_and_answer(content)
    
    assert reasoning == []
    assert "Plain string part" in answer
    assert "Dict text part" in answer


def test_extract_empty_content():
    """Test extracting from empty content."""
    reasoning, answer = extract_reasoning_and_answer([])
    assert reasoning == []
    assert answer == ""
    
    reasoning, answer = extract_reasoning_and_answer({})
    assert reasoning == []
    assert answer == ""


def test_extract_with_alternative_reasoning_types():
    """Test extracting with alternative reasoning type names."""
    content = [
        {"type": "thought", "text": "A thought..."},
        {"type": "reasoning", "text": "Some reasoning..."},
        {"type": "text", "text": "Final answer."},
    ]
    
    reasoning, answer = extract_reasoning_and_answer(content)
    
    assert len(reasoning) == 2
    assert "A thought..." in reasoning[0]
    assert "Some reasoning..." in reasoning[1]
    assert answer == "Final answer."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
