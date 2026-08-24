"""``export_labels`` body for ``AuditApplication`` (facade delegation)."""

from __future__ import annotations

from ssat.application.types import (
    ApplicationError,
    ApplicationErrorCode,
    ExportLabelsRequest,
    ExportLabelsResult,
)
from ssat.report import export_risk_labels, load_assembled_report_for_labels


def export_labels(request: ExportLabelsRequest) -> ExportLabelsResult:
    """Export a risk-label file (JSONL + optional CSV) for an already-generated report.

    Unlike ``generate_report`` (R0-R4, opens the dump/MetricsStore/
    AnalysisStore), this reads only ``request.report_dir``'s
    already-written ``data/report_model.json``/``data/sample_rankings
    .csv``/``report_manifest.json`` (``ssat.report.labels.load_
    assembled_report_for_labels``) and never reopens any of those three
    stores -- the deliberate design of exporting what was already
    computed rather than rerunning R0, which is also why this is a
    separate CLI subcommand rather than something ``generate_report``
    runs automatically.

    Args:
        request: ``report_dir`` plus the label-export options.

    Raises:
        ApplicationError: With ``ApplicationErrorCode.EXPORT_LABELS``,
            wrapping ``ReportDataError`` (missing/pre-plan
            ``report_model.json``) or any filesystem failure.
    """

    report_dir = request.report_dir.expanduser().resolve(strict=True)
    output_dir = (request.output_dir or report_dir / "labels").expanduser().resolve()

    try:
        loaded, source_report_manifest_hash = load_assembled_report_for_labels(report_dir)
        result = export_risk_labels(
            loaded,
            output_dir,
            only_clean_correct=request.only_clean_correct,
            write_csv=request.csv,
            source_report_manifest_hash=source_report_manifest_hash,
        )
    except Exception as error:
        # Catches ReportDataError (missing/pre-plan report_model.json,
        # ssat.report.labels module docstring) and any filesystem
        # failure surfaced along the way, mirroring generate_report's
        # identical single-error-code rationale above.
        raise ApplicationError(
            ApplicationErrorCode.EXPORT_LABELS, f"cannot export labels: {error}"
        ) from error

    return ExportLabelsResult(
        labels_path=result.labels_path,
        manifest_path=result.manifest_path,
        csv_path=result.csv_path,
        n_labels=result.manifest.n_labels,
        n_positive=result.manifest.n_positive,
        n_negative_or_none=(
            result.manifest.n_negative if result.manifest.is_binary_primary_metric else None
        ),
        generated_at=result.manifest.generated_at,
    )
