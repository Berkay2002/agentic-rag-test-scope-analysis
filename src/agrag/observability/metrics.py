"""Metrics collection for error recovery and robustness."""

from dataclasses import dataclass, field
from typing import Dict, List, Any
from datetime import datetime


@dataclass
class ErrorMetrics:
    """Collection of error recovery metrics."""
    total_operations: int = 0
    failed_operations: int = 0
    retried_operations: int = 0
    fallback_activations: int = 0
    retry_attempts: Dict[int, int] = field(default_factory=dict)
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    latency_by_operation: Dict[str, List[float]] = field(default_factory=dict)

    def record_operation(
        self,
        operation: str,
        success: bool,
        retry_count: int = 0,
        fallback_used: bool = False,
        error_type: str = None,
        latency_ms: float = 0
    ):
        """Record metrics for an operation.

        Args:
            operation: Name/type of the operation
            success: Whether the operation succeeded
            retry_count: Number of retry attempts made
            fallback_used: Whether a fallback was used
            error_type: Type of error if failed
            latency_ms: Operation latency in milliseconds
        """
        self.total_operations += 1
        if not success:
            self.failed_operations += 1
        if retry_count > 0:
            self.retried_operations += 1
            self.retry_attempts[retry_count] = self.retry_attempts.get(retry_count, 0) + 1
        if fallback_used:
            self.fallback_activations += 1
        if error_type:
            self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

        if operation not in self.latency_by_operation:
            self.latency_by_operation[operation] = []
        self.latency_by_operation[operation].append(latency_ms)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dictionary containing summary metrics including:
            - total_operations: Total number of operations
            - success_rate: Success rate as a fraction
            - retried_operations: Number of operations that were retried
            - avg_retries_per_failed_op: Average retries per failed operation
            - fallback_activations: Number of times fallback was used
            - errors_by_type: Breakdown of errors by type
        """
        success_rate = (self.total_operations - self.failed_operations) / max(self.total_operations, 1)
        avg_retries = sum(k * v for k, v in self.retry_attempts.items()) / max(self.retried_operations, 1) if self.retried_operations else 0

        return {
            "total_operations": self.total_operations,
            "success_rate": success_rate,
            "retried_operations": self.retried_operations,
            "avg_retries_per_failed_op": avg_retries,
            "fallback_activations": self.fallback_activations,
            "errors_by_type": self.errors_by_type,
        }