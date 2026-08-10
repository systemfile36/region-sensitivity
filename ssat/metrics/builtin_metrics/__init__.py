"""Built-in metric implementations and the default registry wiring."""

from __future__ import annotations

from ssat.metrics.builtin_metrics.flips import (
    DEFAULT_TOPK,
    FlipCorrectToWrong,
    FlipWrongToCorrect,
    PredChanged,
    TopkExit,
)
from ssat.metrics.registry import MetricRegistry

__all__ = [
    "DEFAULT_TOPK",
    "FlipCorrectToWrong",
    "FlipWrongToCorrect",
    "PredChanged",
    "TopkExit",
    "default_metric_registry",
]


def default_metric_registry() -> MetricRegistry:
    """Return a fresh registry containing only v1 first-priority metrics."""

    registry = MetricRegistry()
    registry.register(FlipCorrectToWrong())
    registry.register(FlipWrongToCorrect())
    registry.register(PredChanged())
    registry.register(TopkExit())
    return registry
