"""Middleware configurations for the AgRAG agent.

This module provides pre-configured middleware for common use cases.
Middleware intercepts execution at strategic points in the agent lifecycle.

Currently exposed helpers:
- PII/safety middleware constructors (LangChain middleware, optional dependency)

Usage:
    from agrag.middleware import get_pii_middleware, get_safety_middleware

    # Get pre-configured PII protection (not applied by default)
    pii_middleware = get_pii_middleware()

    # Create agent with middleware
    from agrag.core import create_agent_graph
    agent = create_agent_graph(middleware=pii_middleware)
"""

from .pii import get_pii_middleware, get_safety_middleware

__all__ = [
    "get_pii_middleware",
    "get_safety_middleware",
]
