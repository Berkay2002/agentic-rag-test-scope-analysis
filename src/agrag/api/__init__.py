"""
API module for Agentic GraphRAG system.

This module provides programmatic interfaces for batch processing and other API operations.
"""

from .batch import BatchQueryProcessor, load_queries_from_file

__all__ = ["BatchQueryProcessor", "load_queries_from_file"]