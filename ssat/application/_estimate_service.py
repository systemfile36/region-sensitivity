"""``estimate`` body for ``AuditApplication`` (facade delegation)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ssat.application import _session_service
from ssat.application.types import (
    ApplicationError,
    ApplicationErrorCode,
    EstimateRequest,
    EstimateResult,
    EventSink,
)
from ssat.core.estimate import CostEstimator

if TYPE_CHECKING:
    from ssat.application.application import AuditApplication


def estimate(
    app: AuditApplication,
    request: EstimateRequest,
    *,
    event_sink: EventSink | None = None,
) -> EstimateResult:
    """Return a structured standalone estimate without prompting."""

    if not isinstance(request, EstimateRequest):
        raise TypeError("request must be an EstimateRequest")
    _session_service.emit(app, event_sink, "started", "config")
    context = _session_service.build_context(app, request.config, request.base_dir)
    mode: Literal["create", "resume", "none"] = "none"
    resume = None
    if request.dump is not None:
        dump = request.dump.expanduser().resolve()
        mode = _session_service.output_mode(dump)
        if mode != "resume":
            raise ApplicationError(
                ApplicationErrorCode.OUTPUT,
                "estimate --dump requires an existing valid dump",
            )
        resume = _session_service.resume_for_context(app, dump, context)
    _session_service.emit(app, event_sink, "completed", "config")
    _session_service.emit(app, event_sink, "started", "preflight")
    report = CostEstimator(request.options).estimate(
        context.resolved,
        context.builder,
        context.source,
        context.adapter,
        resume_index=resume,
        region_resolver=context.region_resolver,
    )
    _session_service.emit(app, event_sink, "completed", "preflight")
    return EstimateResult(report, mode)
