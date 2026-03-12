"""Boundary-Aware Self-Distillation (BASD) training package."""

from .types import (
    BoundaryResult,
    DistillBatchOutput,
    Example,
    RolloutRecord,
    StepSpan,
    TokenMetrics,
)

__all__ = [
    "Example",
    "StepSpan",
    "RolloutRecord",
    "TokenMetrics",
    "BoundaryResult",
    "DistillBatchOutput",
]
