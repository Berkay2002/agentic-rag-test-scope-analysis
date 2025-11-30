"""PII and safety middleware configurations.

This module provides pre-configured middleware for detecting and handling
Personally Identifiable Information (PII) and other sensitive data.

NOTE: These middleware are NOT applied by default. They must be explicitly
passed to create_agent_graph() via the middleware parameter.

Example:
    from agrag.middleware import get_pii_middleware
    from agrag.core import create_agent_graph

    # Apply PII protection
    agent = create_agent_graph(middleware=get_pii_middleware())
"""

from typing import List, Optional

from langchain.agents.middleware import PIIMiddleware


def get_pii_middleware(
    redact_emails: bool = True,
    redact_credit_cards: bool = True,
    redact_ips: bool = False,
    apply_to_input: bool = True,
    apply_to_output: bool = True,
    apply_to_tool_results: bool = False,
) -> List[PIIMiddleware]:
    """
    Get pre-configured PII detection middleware.

    This middleware detects and handles common PII types in conversations.
    By default, it redacts emails and credit cards from both inputs and outputs.

    Args:
        redact_emails: Whether to redact email addresses (default: True)
        redact_credit_cards: Whether to redact credit card numbers (default: True)
        redact_ips: Whether to redact IP addresses (default: False)
        apply_to_input: Check user messages before model call (default: True)
        apply_to_output: Check AI messages after model call (default: True)
        apply_to_tool_results: Check tool results (default: False)

    Returns:
        List of configured PIIMiddleware instances

    Example:
        >>> middleware = get_pii_middleware()
        >>> agent = create_agent_graph(middleware=middleware)

        # User input "My email is john@example.com" becomes:
        # "My email is [REDACTED_EMAIL]"
    """
    middleware = []

    if redact_emails:
        middleware.append(
            PIIMiddleware(
                "email",
                strategy="redact",
                apply_to_input=apply_to_input,
                apply_to_output=apply_to_output,
                apply_to_tool_results=apply_to_tool_results,
            )
        )

    if redact_credit_cards:
        middleware.append(
            PIIMiddleware(
                "credit_card",
                strategy="mask",  # Show last 4 digits: ****-****-****-1234
                apply_to_input=apply_to_input,
                apply_to_output=apply_to_output,
                apply_to_tool_results=apply_to_tool_results,
            )
        )

    if redact_ips:
        middleware.append(
            PIIMiddleware(
                "ip",
                strategy="redact",
                apply_to_input=apply_to_input,
                apply_to_output=apply_to_output,
                apply_to_tool_results=apply_to_tool_results,
            )
        )

    return middleware


def get_safety_middleware(
    block_api_keys: bool = True,
    block_passwords: bool = True,
    custom_patterns: Optional[List[dict]] = None,
) -> List[PIIMiddleware]:
    """
    Get safety middleware that blocks sensitive patterns.

    Unlike redaction, blocking raises an exception when sensitive
    data is detected, preventing it from being processed entirely.

    Args:
        block_api_keys: Block API key patterns (default: True)
        block_passwords: Block common password patterns (default: True)
        custom_patterns: List of custom patterns to block, each dict should have:
            - name: Name for the pattern
            - detector: Regex pattern string

    Returns:
        List of configured PIIMiddleware instances

    Example:
        >>> middleware = get_safety_middleware()
        >>> agent = create_agent_graph(middleware=middleware)

        # User input containing "sk-abc123..." will raise an exception
    """
    middleware = []

    if block_api_keys:
        # OpenAI-style API keys
        middleware.append(
            PIIMiddleware(
                "openai_api_key",
                detector=r"sk-[a-zA-Z0-9]{32,}",
                strategy="block",
                apply_to_input=True,
            )
        )
        # Google API keys
        middleware.append(
            PIIMiddleware(
                "google_api_key",
                detector=r"AIza[a-zA-Z0-9_-]{35}",
                strategy="block",
                apply_to_input=True,
            )
        )

    if block_passwords:
        # Common password field patterns (password=, pwd=, etc.)
        middleware.append(
            PIIMiddleware(
                "password_field",
                detector=r"(?:password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]+",
                strategy="block",
                apply_to_input=True,
            )
        )

    # Add custom patterns
    if custom_patterns:
        for pattern in custom_patterns:
            middleware.append(
                PIIMiddleware(
                    pattern["name"],
                    detector=pattern["detector"],
                    strategy="block",
                    apply_to_input=True,
                )
            )

    return middleware


def get_telecom_pii_middleware() -> List[PIIMiddleware]:
    """
    Get PII middleware specifically for telecommunications data.

    This includes patterns commonly found in telecom test data:
    - IMSI (International Mobile Subscriber Identity)
    - IMEI (International Mobile Equipment Identity)
    - Phone numbers
    - IP addresses

    Returns:
        List of configured PIIMiddleware instances for telecom data
    """
    middleware = []

    # IMSI: 15-digit number starting with MCC/MNC
    middleware.append(
        PIIMiddleware(
            "imsi",
            detector=r"\b\d{15}\b",  # Simplified pattern
            strategy="mask",
            apply_to_input=True,
            apply_to_output=True,
        )
    )

    # IMEI: 15-digit number
    middleware.append(
        PIIMiddleware(
            "imei",
            detector=r"\b\d{15}\b",
            strategy="mask",
            apply_to_input=True,
            apply_to_output=True,
        )
    )

    # Phone numbers (international format)
    middleware.append(
        PIIMiddleware(
            "phone_number",
            detector=r"\+?\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{4}",
            strategy="mask",
            apply_to_input=True,
            apply_to_output=True,
        )
    )

    # IP addresses
    middleware.append(
        PIIMiddleware(
            "ip",
            strategy="redact",
            apply_to_input=True,
            apply_to_output=True,
        )
    )

    return middleware
