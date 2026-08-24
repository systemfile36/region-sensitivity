"""``prepare_run``/``execute_run`` bodies for ``AuditApplication`` (facade delegation)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ssat.application import _session_service
from ssat.application.locking import OutputLockedError, output_lock
from ssat.application.types import (
    ApplicationError,
    ApplicationErrorCode,
    CancellationToken,
    EstimateResult,
    EventSink,
    RunRequest,
    RunResult,
)
from ssat.core.dump import DumpWriter
from ssat.core.estimate import CostEstimator
from ssat.core.resume import ResumeIndex
from ssat.core.runtime import RuntimeCancelledError, run_audit

if TYPE_CHECKING:
    from ssat.application.application import AuditApplication, PreparedRun


def prepare_run(
    app: AuditApplication,
    request: RunRequest,
    *,
    event_sink: EventSink | None = None,
) -> PreparedRun:
    """Resolve and profile an audit without creating a new dump."""

    from ssat.application.application import PreparedRun

    if not isinstance(request, RunRequest):
        raise TypeError("request must be a RunRequest")
    _session_service.emit(app, event_sink, "started", "config")
    context = _session_service.build_context(app, request.config, request.base_dir)
    output = request.output.expanduser().resolve()
    mode = _session_service.output_mode(output)
    resume = (
        _session_service.resume_for_context(app, output, context) if mode == "resume" else None
    )
    _session_service.emit(app, event_sink, "completed", "config")
    _session_service.emit(app, event_sink, "started", "preflight")
    report = CostEstimator(request.estimate_options).estimate(
        context.resolved,
        context.builder,
        context.source,
        context.adapter,
        resume_index=resume,
        region_resolver=context.region_resolver,
    )
    result = EstimateResult(report=report, dump_mode=mode)
    fingerprint = _session_service.preflight_fingerprint(context, output, mode)
    _session_service.emit(
        app,
        event_sink,
        "completed",
        "preflight",
        completed=report.pending_perturbed_items,
        total=report.total_perturbed_items,
    )
    return PreparedRun(app, request, context, result, mode, fingerprint)


def execute_run(
    app: AuditApplication,
    prepared: PreparedRun,
    *,
    confirmation_granted: bool = False,
    event_sink: EventSink | None = None,
    cancellation: CancellationToken | None = None,
) -> RunResult:
    """Execute one prepared audit after confirmation and stale-state checks."""

    from ssat.application.application import PreparedRun

    if not isinstance(prepared, PreparedRun) or prepared._application is not app:
        raise ApplicationError(
            ApplicationErrorCode.EXECUTION,
            "prepared run belongs to a different application",
        )
    if prepared._closed or prepared._consumed:
        raise ApplicationError(
            ApplicationErrorCode.EXECUTION,
            "prepared run is closed or already consumed",
        )
    if prepared.confirmation_required and not confirmation_granted:
        raise ApplicationError(
            ApplicationErrorCode.CONFIRMATION_REQUIRED,
            "audit preflight requires explicit confirmation",
        )
    output = prepared.request.output.expanduser().resolve()
    if cancellation is not None and cancellation.is_cancelled:
        prepared._consumed = True
        _session_service.emit(app, event_sink, "cancelled", "run")
        return RunResult("cancelled", output, prepared.estimate, None)

    try:
        with output_lock(output):
            try:
                current = _session_service.preflight_fingerprint(
                    prepared.context,
                    output,
                    prepared.mode,
                )
            except Exception as error:
                raise ApplicationError(
                    ApplicationErrorCode.STALE_PREFLIGHT,
                    "configuration, source, adapter, or dump is no longer available",
                ) from error
            if current != prepared.fingerprint:
                raise ApplicationError(
                    ApplicationErrorCode.STALE_PREFLIGHT,
                    "configuration, source, or dump changed after preflight",
                )
            _session_service.emit(app, event_sink, "started", "run")
            prepared._consumed = True
            writer = DumpWriter(
                output,
                prepared.context.resolved,
                code_version=app._code_version,
                mode=prepared.mode,
            )
            try:
                resume = ResumeIndex.open(output)
                summary = run_audit(
                    prepared.context.resolved,
                    prepared.context.builder,
                    prepared.context.source,
                    prepared.context.adapter,
                    writer,
                    resume,
                    region_resolver=prepared.context.region_resolver,
                    progress_callback=lambda completed, total: _session_service.emit(
                        app,
                        event_sink,
                        "progress",
                        "run",
                        completed=completed,
                        total=total,
                    ),
                    cancel_requested=(
                        None
                        if cancellation is None
                        else lambda: cancellation.is_cancelled
                    ),
                )
            except RuntimeCancelledError:
                writer.close(success=False)
                _session_service.emit(app, event_sink, "cancelled", "run")
                return RunResult("cancelled", output, prepared.estimate, None)
            except Exception:
                writer.close(success=False)
                raise
            else:
                writer.close(success=True)
    except ApplicationError:
        raise
    except OutputLockedError as error:
        raise ApplicationError(ApplicationErrorCode.LOCKED, str(error)) from error
    except Exception as error:
        raise ApplicationError(
            ApplicationErrorCode.EXECUTION,
            f"audit execution failed: {error}",
        ) from error

    _session_service.emit(
        app,
        event_sink,
        "completed",
        "run",
        completed=summary.records_written,
        total=summary.records_written,
    )
    return RunResult("completed", output, prepared.estimate, summary)
