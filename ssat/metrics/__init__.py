"""Metrics-engine contracts for the SSAT toolkit."""

from ssat.metrics.dump_reader import DumpHandle, JoinedFrame
from ssat.metrics.errors import MetricsCorruptionError, MetricsError, MetricsSchemaError
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
    "DumpHandle",
    "ExclusionReason",
    "ItemMetrics",
    "JoinedFrame",
    "MetricsCorruptionError",
    "MetricsError",
    "MetricsSchemaError",
    "RegionGeometryRef",
    "RegionMetrics",
    "SampleMetrics",
    "SpatialProfile",
]
