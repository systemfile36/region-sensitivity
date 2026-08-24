"""``compute_metrics`` body for ``AuditApplication`` (facade delegation)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ssat.application.types import (
    ApplicationError,
    ApplicationErrorCode,
    ComputeMetricsRequest,
    ComputeMetricsResult,
)
from ssat.metrics.aggregate import aggregate_item_metrics
from ssat.metrics.dump_reader import DumpHandle
from ssat.metrics.errors import MetricsRegistryError
from ssat.metrics.store import save_metrics

if TYPE_CHECKING:
    from ssat.application.application import AuditApplication


def compute_metrics(app: AuditApplication, request: ComputeMetricsRequest) -> ComputeMetricsResult:
    """Compute and persist every registered metric for an existing dump.

    This is the Application-layer counterpart of what experiment scripts
    (e.g. experiments/synthetic_shortcut/run_audit.py) and test fixtures
    previously had to hand-roll themselves by opening a DumpHandle
    directly: it registers every metric in this application's metric
    registry — every v1 built-in metric by default, or a caller-supplied
    registry passed as ``metric_registry`` to ``AuditApplication.__init__``
    (v1 scope intentionally has no per-metric selection flag within one
    registry) — and stores the result under metrics_dir (default:
    <dump>/metrics).
    """

    dump = request.dump.expanduser().resolve(strict=True)
    metrics_dir = (request.metrics_dir or dump / "metrics").expanduser().resolve()
    try:
        handle = DumpHandle(dump)
        joined = handle.joined()
        resolved_config = handle.manifest.resolved_config

        registry = app._metric_registry
        if request.primary_metric not in registry.names:
            # Fail before the (potentially expensive, whole-dump)
            # compute_item_metrics call below rather than only inside
            # aggregate_item_metrics/save_metrics, which both re-check
            # this same condition but only after every item has already
            # been scored (ssat/metrics/aggregate.py).
            raise MetricsRegistryError(
                f"primary_metric not registered: {request.primary_metric}"
            )
        item_metrics = registry.compute_item_metrics(
            joined, adapter_spec=resolved_config.adapter_spec
        )
        result = aggregate_item_metrics(
            item_metrics,
            joined,
            registry,
            resolved_config,
            primary_metric=request.primary_metric,
        )
        manifest = save_metrics(
            metrics_dir,
            item_metrics,
            result,
            registry=registry,
            primary_metric=request.primary_metric,
            source_run_manifest_path=handle.manifest_path,
            exclusion_summary=handle.summary(),
        )
    except Exception as error:
        # Catches both ssat.metrics.errors.MetricsError (schema/corruption/
        # registry failures raised by the metrics engine itself, including
        # the MetricsRegistryError raised above) and anything DumpHandle
        # surfaces while reading the raw dump -- both map to the same
        # METRICS error code, so there is no need to distinguish them here.
        raise ApplicationError(
            ApplicationErrorCode.METRICS, f"cannot compute metrics: {error}"
        ) from error

    return ComputeMetricsResult(
        dump=dump,
        metrics_dir=metrics_dir,
        primary_metric=manifest.metric_config.primary_metric,
        metric_names=tuple(metric.name for metric in manifest.registered_metrics),
        n_item_metric_rows=len(item_metrics),
        # isoformat() here, not the raw datetime -- see ComputeMetricsResult's
        # docstring on why to_primitive() cannot carry a datetime through.
        computed_at=manifest.computed_at.isoformat(),
    )
