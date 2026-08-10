"""Metrics-engine contracts for the SSAT toolkit."""

from ssat.metrics.builtin_metrics import (
    DEFAULT_TOPK,
    FlipCorrectToWrong,
    FlipWrongToCorrect,
    PredChanged,
    TopkExit,
    default_metric_registry,
)
from ssat.metrics.dump_reader import DumpHandle, JoinedFrame
from ssat.metrics.errors import (
    MetricsCorruptionError,
    MetricsError,
    MetricsRegistryError,
    MetricsSchemaError,
)
from ssat.metrics.normalize import NormalizedOutput, normalize_output
from ssat.metrics.registry import Metric, MetricRegistry, MetricResult
from ssat.metrics.types import (
    ClassMetrics,
    ExclusionReason,
    ItemMetrics,
    RegionGeometryRef,
    RegionMetrics,
    SampleMetrics,
    SpatialProfile,
)

__all__ = [
    "ClassMetrics",
    "DEFAULT_TOPK",
    "DumpHandle",
    "ExclusionReason",
    "FlipCorrectToWrong",
    "FlipWrongToCorrect",
    "ItemMetrics",
    "JoinedFrame",
    "Metric",
    "MetricRegistry",
    "MetricResult",
    "MetricsCorruptionError",
    "MetricsError",
    "MetricsRegistryError",
    "MetricsSchemaError",
    "NormalizedOutput",
    "PredChanged",
    "RegionGeometryRef",
    "RegionMetrics",
    "SampleMetrics",
    "SpatialProfile",
    "TopkExit",
    "default_metric_registry",
    "normalize_output",
]
