"""UI-independent application facade for SSAT."""

from ssat.application.application import AuditApplication, PreparedRun
from ssat.application.types import (
    ApplicationError,
    ApplicationErrorCode,
    ApplicationEvent,
    CancellationToken,
    ComputeMetricsRequest,
    ComputeMetricsResult,
    DumpSectionSummary,
    DumpSummary,
    EstimateRequest,
    EstimateResult,
    EventSink,
    IndexRebuildResult,
    InspectRequest,
    RebuildIndexRequest,
    RunRequest,
    RunResult,
)

__all__ = [
    "ApplicationError",
    "ApplicationErrorCode",
    "ApplicationEvent",
    "AuditApplication",
    "CancellationToken",
    "ComputeMetricsRequest",
    "ComputeMetricsResult",
    "DumpSectionSummary",
    "DumpSummary",
    "EstimateRequest",
    "EstimateResult",
    "EventSink",
    "IndexRebuildResult",
    "InspectRequest",
    "PreparedRun",
    "RebuildIndexRequest",
    "RunRequest",
    "RunResult",
]
