"""Console and JSON presentation helpers kept outside application services."""

from __future__ import annotations

import json
from typing import Any

from ssat.application import (
    AnalyzeResult,
    ComputeMetricsResult,
    DumpSummary,
    EstimateResult,
    IndexRebuildResult,
    RunResult,
)


def json_text(value: Any) -> str:
    """Serialize one public result deterministically for machine consumers."""

    payload = value.to_dict() if hasattr(value, "to_dict") else value
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def format_estimate(result: EstimateResult) -> str:
    report = result.report
    duration = _duration(report.estimated_remaining_seconds)
    total_dump = _bytes(report.estimated_total_dump_bytes)
    remaining_dump = _bytes(report.estimated_remaining_dump_bytes)
    lines = [
        "SSAT estimate",
        f"  dump mode: {result.dump_mode}",
        f"  clean samples: {report.pending_clean_samples:,} pending / {report.total_clean_samples:,} total",
        f"  perturbed items: {report.pending_perturbed_items:,} pending / {report.total_perturbed_items:,} total",
        f"  chunks: {report.pending_chunks:,} pending / {report.total_chunks:,} total",
        f"  estimated remaining time: {duration}",
        f"  estimated dump: {remaining_dump} remaining / {total_dump} total",
        f"  confirmation required: {'yes' if report.confirmation_required else 'no'}",
    ]
    if report.sanity is not None:
        accuracy = (
            "n/a" if report.sanity.accuracy is None else f"{report.sanity.accuracy:.2%}"
        )
        lines.append(f"  sanity top-1 accuracy: {accuracy}")
    for advisory in report.advisories:
        lines.append(f"  advisory [{advisory.code.value}]: {advisory.message}")
    for recommendation in report.recommendations:
        lines.append(f"  recommendation: {recommendation}")
    return "\n".join(lines)


def format_run(result: RunResult) -> str:
    if result.status == "cancelled":
        return f"SSAT run cancelled; durable partial results: {result.output}"
    summary = result.summary
    if summary is None:  # pragma: no cover - guarded by RunResult status
        return f"SSAT run completed: {result.output}"
    return "\n".join(
        [
            "SSAT run completed",
            f"  output: {result.output}",
            f"  records written: {summary.records_written:,}",
            f"  OOM events: {summary.oom_events:,}",
            f"  final batch size: {summary.final_batch_size:,}",
        ]
    )


def format_dump_summary(summary: DumpSummary) -> str:
    statuses = ", ".join(
        f"{name}={count:,}" for name, count in summary.total_counts_by_status.items()
    ) or "none"
    return "\n".join(
        [
            "SSAT dump",
            f"  path: {summary.dump}",
            f"  schema/code: {summary.schema_version} / {summary.code_version}",
            f"  model: {summary.model_id}",
            f"  clean rows: {summary.clean.rows:,}",
            f"  perturbed rows: {summary.perturbed.rows:,}",
            f"  statuses: {statuses}",
            f"  resumes: {summary.resume_count:,}",
            f"  finished: {summary.finished_at or 'no'}",
            f"  manifest counts match: {'yes' if summary.manifest_counts_match else 'no'}",
        ]
    )


def format_metrics(result: ComputeMetricsResult) -> str:
    return "\n".join(
        [
            "SSAT metrics computed",
            f"  dump: {result.dump}",
            f"  metrics dir: {result.metrics_dir}",
            f"  primary metric: {result.primary_metric}",
            f"  metrics: {', '.join(result.metric_names)}",
            f"  item-metric rows: {result.n_item_metric_rows:,}",
            f"  computed at: {result.computed_at}",
        ]
    )


def format_analysis(result: AnalyzeResult) -> str:
    coverage = result.coverage_report
    grades = ", ".join(f"{name}={count:,}" for name, count in result.grade_distribution.items()) or "none"
    return "\n".join(
        [
            "SSAT control/stability analysis computed",
            f"  dump: {result.dump}",
            f"  metrics dir: {result.metrics_dir}",
            f"  analysis dir: {result.analysis_dir}",
            f"  reliability rows: {result.n_reliability_rows:,}",
            f"  reliability grades: {grades}",
            f"  anchors: {coverage.n_anchors:,}",
            f"  conditions insufficient: {coverage.n_conditions_insufficient:,}",
            f"  controls unmatched: {coverage.n_controls_unmatched:,}",
            f"  area mismatch warnings: {coverage.n_area_mismatch_warnings:,}",
            f"  computed at: {result.computed_at}",
        ]
    )


def format_rebuild(result: IndexRebuildResult) -> str:
    return "\n".join(
        [
            "SSAT index rebuilt",
            f"  dump: {result.dump}",
            f"  indexed items: {result.indexed_items:,}",
        ]
    )


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    resolved = float(value)
    for unit in units:
        if resolved < 1024 or unit == units[-1]:
            return f"{resolved:.1f} {unit}"
        resolved /= 1024
    return f"{value} B"  # pragma: no cover
