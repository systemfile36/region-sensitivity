"""Built-in metric implementations and the default registry wiring."""

from __future__ import annotations

from ssat.metrics.builtin_metrics.continuous import (
    GtLogitDrop,
    GtProbDrop,
    GtRankWorsening,
    LOSS_EPSILON,
    LossIncrease,
    MarginDrop,
)
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
    "GtLogitDrop",
    "GtProbDrop",
    "GtRankWorsening",
    "LOSS_EPSILON",
    "LossIncrease",
    "MarginDrop",
    "PredChanged",
    "TopkExit",
    "default_metric_registry",
]


def default_metric_registry() -> MetricRegistry:
    """Return a fresh registry containing every v1 built-in metric."""

    registry = MetricRegistry()
    registry.register(FlipCorrectToWrong())
    registry.register(FlipWrongToCorrect())
    registry.register(PredChanged())
    registry.register(TopkExit())
    registry.register(GtProbDrop())
    registry.register(GtLogitDrop())
    registry.register(MarginDrop())
    registry.register(LossIncrease())
    registry.register(GtRankWorsening())
    return registry
