"""Shared utilities for search tools."""

from typing import List
import re


def extract_signal_tokens(query: str) -> List[str]:
    """Extract high-signal tokens (acronyms, alphanumerics) from the query."""
    tokens = re.findall(r"[A-Za-z0-9\-]+", query)
    signal_tokens = []
    for token in tokens:
        if len(token) < 2:
            continue
        if token.isupper() or any(ch.isdigit() for ch in token):
            signal_tokens.append(token)
    return signal_tokens