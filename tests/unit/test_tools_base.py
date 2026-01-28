"""Unit tests for the tools base module."""

import pytest
from unittest.mock import Mock

from agrag.tools.shared.base import (
    BaseToolWrapper,
    format_search_results_header,
    format_search_result_item,
    format_search_results_footer,
    build_metadata_with_chunk_id,
    extract_score_or_default,
)


class TestBaseToolWrapper:
    """Tests for BaseToolWrapper class."""

    def test_wrapper_delegates_name(self):
        """Test that name property delegates to underlying tool."""
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        wrapper = BaseToolWrapper(mock_tool)
        
        assert wrapper.name == "test_tool"

    def test_wrapper_delegates_description(self):
        """Test that description property delegates to underlying tool."""
        mock_tool = Mock()
        mock_tool.description = "Test tool description"
        wrapper = BaseToolWrapper(mock_tool)
        
        assert wrapper.description == "Test tool description"

    def test_wrapper_delegates_invoke(self):
        """Test that invoke method delegates to underlying tool."""
        mock_tool = Mock()
        mock_tool.invoke.return_value = "result"
        wrapper = BaseToolWrapper(mock_tool)
        
        result = wrapper.invoke("test_input")
        
        assert result == "result"
        mock_tool.invoke.assert_called_once_with("test_input")

    def test_wrapper_delegates_arbitrary_attributes(self):
        """Test that arbitrary attributes delegate to underlying tool."""
        mock_tool = Mock()
        mock_tool.custom_attr = "custom_value"
        wrapper = BaseToolWrapper(mock_tool)
        
        assert wrapper.custom_attr == "custom_value"


class TestFormatSearchResultsHeader:
    """Tests for format_search_results_header function."""

    def test_basic_header_formatting(self):
        """Test basic header formatting without additional info."""
        result = format_search_results_header(
            query="test query",
            total_results=5,
            retrieval_time_ms=123.45,
            search_type="Vector Search",
        )
        
        assert "Vector Search Results - found 5 items in 123.45ms:" in result
        assert "Query: test query" in result

    def test_header_with_additional_info(self):
        """Test header formatting with additional info."""
        result = format_search_results_header(
            query="test query",
            total_results=5,
            retrieval_time_ms=123.45,
            search_type="Hybrid Search",
            additional_info="RRF fusion",
        )
        
        assert "Hybrid Search Results (RRF fusion) - found 5 items in 123.45ms:" in result
        assert "Query: test query" in result


class TestFormatSearchResultItem:
    """Tests for format_search_result_item function."""

    def test_basic_item_formatting(self):
        """Test basic item formatting without entity type."""
        result = format_search_result_item(
            index=1,
            result_id="TC_001",
            score=0.85,
            score_label="Similarity",
            content="Test content here",
        )
        
        assert "1. Entity ID: TC_001 (Similarity: 0.8500)" in result
        assert "Snippet: Test content here..." in result

    def test_item_with_entity_type(self):
        """Test item formatting with entity type."""
        result = format_search_result_item(
            index=2,
            result_id="REQ_001",
            score=0.92,
            score_label="RRF Score",
            content="Requirement content",
            entity_type="Requirement",
        )
        
        assert "2. Entity ID: REQ_001 (RRF Score: 0.9200)" in result
        assert "Entity Type: Requirement" in result

    def test_item_with_custom_preview_length(self):
        """Test item formatting with custom content preview length."""
        long_content = "a" * 500
        result = format_search_result_item(
            index=1,
            result_id="TEST_001",
            score=0.75,
            score_label="Rank",
            content=long_content,
            content_preview_length=100,
        )
        
        assert "a" * 100 + "..." in result
        assert len(result.split("Snippet: ")[1].split("...")[0]) == 100


class TestFormatSearchResultsFooter:
    """Tests for format_search_results_footer function."""

    def test_footer_with_note(self):
        """Test footer formatting with a note."""
        result = format_search_results_footer("Note: This is a test note.")
        
        assert "\nNote: This is a test note." == result

    def test_footer_without_note(self):
        """Test footer formatting without a note."""
        result = format_search_results_footer(None)
        
        assert result == ""


class TestBuildMetadataWithChunkId:
    """Tests for build_metadata_with_chunk_id function."""

    def test_with_both_metadata_and_chunk_id(self):
        """Test building metadata with both existing metadata and chunk_id."""
        metadata = {"entity_type": "TestCase", "entity_id": "TC_001"}
        result = build_metadata_with_chunk_id(metadata, "chunk_123")
        
        assert result["entity_type"] == "TestCase"
        assert result["entity_id"] == "TC_001"
        assert result["chunk_id"] == "chunk_123"

    def test_with_none_metadata(self):
        """Test building metadata with None metadata."""
        result = build_metadata_with_chunk_id(None, "chunk_123")
        
        assert result == {"chunk_id": "chunk_123"}

    def test_with_none_chunk_id(self):
        """Test building metadata with None chunk_id."""
        metadata = {"entity_type": "Requirement"}
        result = build_metadata_with_chunk_id(metadata, None)
        
        assert result == {"entity_type": "Requirement"}

    def test_with_both_none(self):
        """Test building metadata with both None."""
        result = build_metadata_with_chunk_id(None, None)
        
        assert result == {}


class TestExtractScoreOrDefault:
    """Tests for extract_score_or_default function."""

    def test_with_valid_float(self):
        """Test extracting a valid float score."""
        result = extract_score_or_default(0.85)
        
        assert result == 0.85
        assert isinstance(result, float)

    def test_with_integer(self):
        """Test extracting an integer score."""
        result = extract_score_or_default(5)
        
        assert result == 5.0
        assert isinstance(result, float)

    def test_with_none(self):
        """Test extracting None returns default."""
        result = extract_score_or_default(None)
        
        assert result == 0.0

    def test_with_none_and_custom_default(self):
        """Test extracting None with custom default."""
        result = extract_score_or_default(None, default=1.0)
        
        assert result == 1.0

    def test_with_string_number(self):
        """Test extracting a string number."""
        result = extract_score_or_default("0.95")
        
        assert result == 0.95
        assert isinstance(result, float)
