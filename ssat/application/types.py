"""UI-independent requests, results, events, and errors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from threading import Event
from typing import Any, Callable, Literal, Mapping, TypeAlias

from ssat.analysis.indexer import DEFAULT_AREA_MATCH_TOLERANCE
from ssat.analysis.interval import DEFAULT_N_BOOTSTRAP, DEFAULT_RANDOM_SEED
from ssat.analysis.reliability import DEFAULT_SEED_CV_THRESHOLD, DEFAULT_Z_VS_CONTROL_THRESHOLD
from ssat.analysis.types import AvailableAnalyses, CoverageReport
from ssat.core.estimate import EstimateOptions, EstimateReport
from ssat.core.runtime import ExecutionSummary
from ssat.metrics.aggregate import DEFAULT_PRIMARY_METRIC


ConfigValue: TypeAlias = str | Path | Mapping[str, Any]


class ApplicationErrorCode(str, Enum):
    CONFIG = "config_error"
    PROVIDER = "provider_error"
    OUTPUT = "output_error"
    CONFIRMATION_REQUIRED = "confirmation_required"
    STALE_PREFLIGHT = "stale_preflight"
    LOCKED = "output_locked"
    CANCELLED = "cancelled"
    EXECUTION = "execution_error"
    CORRUPTION = "dump_corruption"
    METRICS = "metrics_error"
    ANALYSIS = "analysis_error"
    REPORT = "report_error"
    EXPORT_LABELS = "export_labels_error"


class ApplicationError(RuntimeError):
    """Stable application-boundary failure suitable for any presentation layer."""

    def __init__(self, code: ApplicationErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ApplicationEvent:
    """One non-sensitive lifecycle or progress event."""

    kind: str
    phase: str
    completed: int | None = None
    total: int | None = None
    message: str | None = None


EventSink: TypeAlias = Callable[[ApplicationEvent], None]


class CancellationToken:
    """Thread-safe cooperative cancellation token."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True, slots=True)
class RunRequest:
    config: ConfigValue
    output: Path
    base_dir: Path | None = None
    estimate_options: EstimateOptions = field(default_factory=EstimateOptions)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", Path(self.output))


@dataclass(frozen=True, slots=True)
class EstimateRequest:
    config: ConfigValue
    dump: Path | None = None
    base_dir: Path | None = None
    options: EstimateOptions = field(default_factory=EstimateOptions)

    def __post_init__(self) -> None:
        if self.dump is not None:
            object.__setattr__(self, "dump", Path(self.dump))


@dataclass(frozen=True, slots=True)
class InspectRequest:
    dump: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "dump", Path(self.dump))


@dataclass(frozen=True, slots=True)
class RebuildIndexRequest:
    dump: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "dump", Path(self.dump))


@dataclass(frozen=True, slots=True)
class ComputeMetricsRequest:
    """Ask for every built-in metric to be computed and persisted for one dump.

    ``metrics_dir`` defaults to ``<dump>/metrics`` when omitted -- co-located
    with the dump itself, matching how ``inspect``/``rebuild-index`` only
    need the dump path and nothing else. Overriding it lets a caller
    recompute metrics (e.g. with a different ``primary_metric``) into a
    fresh location without disturbing a previous computation.
    """

    dump: Path
    metrics_dir: Path | None = None
    primary_metric: str = DEFAULT_PRIMARY_METRIC

    def __post_init__(self) -> None:
        object.__setattr__(self, "dump", Path(self.dump))
        if self.metrics_dir is not None:
            object.__setattr__(self, "metrics_dir", Path(self.metrics_dir))


@dataclass(frozen=True, slots=True)
class ComputeMetricsResult:
    """Summarize one persisted metrics-computation run.

    This intentionally does not carry the full ``ssat.metrics.store
    .MetricsManifest`` object -- that dataclass has a ``datetime`` field,
    which ``to_primitive`` below cannot serialize (the same reason
    ``DumpSummary`` stores its timestamps as pre-formatted ISO strings
    rather than passing datetimes through). ``computed_at`` is therefore
    already isoformat()-ted by the caller before this is constructed.
    """

    dump: Path
    metrics_dir: Path
    primary_metric: str
    metric_names: tuple[str, ...]
    # Count of persisted item_metrics.parquet rows -- one row per (perturbed
    # item, applicable metric), i.e. up to len(metric_names) rows per audited
    # item, not a count of distinct items. Named to match what it literally
    # measures rather than implying a 1:1 item count, which would
    # under-describe what registry.compute_item_metrics() actually returns.
    n_item_metric_rows: int
    computed_at: str

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class AnalyzeRequest:
    """Ask for the control/stability analysis (``ssat.analysis`` A0-A6) to be
    computed and persisted for one existing dump + metrics store pair.

    ``metrics_dir``/``analysis_dir`` default to ``<dump>/metrics``/
    ``<dump>/analysis`` -- the same co-located-by-default convention
    ``ComputeMetricsRequest.metrics_dir`` uses. Scoped to one dump+metrics
    pair, matching ``AnalysisReader``/``compute_metrics``'s own scope;
    combining several runs into one analysis (as
    experiments/synthetic_shortcut/analyze_control_stability.py does) stays
    a script-level concern.
    """

    dump: Path
    metrics_dir: Path | None = None
    analysis_dir: Path | None = None
    primary_metric: str = DEFAULT_PRIMARY_METRIC
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    random_seed: int = DEFAULT_RANDOM_SEED
    z_vs_control_threshold: float = DEFAULT_Z_VS_CONTROL_THRESHOLD
    seed_cv_threshold: float = DEFAULT_SEED_CV_THRESHOLD
    area_match_tolerance: float = DEFAULT_AREA_MATCH_TOLERANCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "dump", Path(self.dump))
        if self.metrics_dir is not None:
            object.__setattr__(self, "metrics_dir", Path(self.metrics_dir))
        if self.analysis_dir is not None:
            object.__setattr__(self, "analysis_dir", Path(self.analysis_dir))


@dataclass(frozen=True, slots=True)
class AnalyzeResult:
    """Summarize one persisted control/stability analysis run.

    ``computed_at`` is already isoformat()-ted by the caller before this is
    constructed -- same reason as ``ComputeMetricsResult.computed_at``.
    """

    dump: Path
    metrics_dir: Path
    analysis_dir: Path
    available_analyses: AvailableAnalyses
    coverage_report: CoverageReport
    grade_distribution: dict[str, int]
    n_reliability_rows: int
    computed_at: str

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class ReportRequest:
    """Ask for a human-facing ``report.html`` (+CSV/JSON/SVG assets) for one dump+metrics(+analysis) pair.

    ``metrics_dir``/``analysis_dir``/``report_dir`` default to
    ``<dump>/metrics``/``<dump>/analysis``/``<dump>/report`` -- the same
    co-located-by-default convention ``ComputeMetricsRequest``/
    ``AnalyzeRequest`` already use.

    Unlike ``AnalyzeRequest.analysis_dir`` (whose absence makes ``analyze()``
    fail), an ``analysis_dir`` that does not exist as a directory here is
    not an error: ``AuditApplication.generate_report`` treats it exactly
    like ``analysis_dir=None``, assembling a report with every
    analysis-derived section explicitly marked unavailable rather than
    failing (``ssat.report.assembler`` module docstring).
    """

    dump: Path
    metrics_dir: Path | None = None
    analysis_dir: Path | None = None
    report_dir: Path | None = None
    primary_metric: str = DEFAULT_PRIMARY_METRIC
    top_k: int = 20
    bottom_k: int = 20

    def __post_init__(self) -> None:
        object.__setattr__(self, "dump", Path(self.dump))
        if self.metrics_dir is not None:
            object.__setattr__(self, "metrics_dir", Path(self.metrics_dir))
        if self.analysis_dir is not None:
            object.__setattr__(self, "analysis_dir", Path(self.analysis_dir))
        if self.report_dir is not None:
            object.__setattr__(self, "report_dir", Path(self.report_dir))


@dataclass(frozen=True, slots=True)
class ReportResult:
    """Summarize one generated report.

    ``analysis_dir`` is ``None`` exactly when the source run had no
    ``ssat analyze`` output available for this report (see
    ``ReportRequest.analysis_dir``'s docstring) -- every analysis-derived
    section of the generated report is then explicitly marked unavailable
    rather than silently absent.

    ``secondary_report_html`` is the auxiliary "Question Driven" report
    (``ssat.report.html_renderer.render_secondary_report``) -- always
    written alongside ``report_dir / "report.html"``, not behind a flag.
    """

    dump: Path
    metrics_dir: Path
    analysis_dir: Path | None
    report_dir: Path
    secondary_report_html: Path
    n_samples: int
    n_regions: int
    grade_distribution: dict[str, int]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class ExportLabelsRequest:
    """Ask for a risk-label file (JSONL + optional CSV) for one already-generated report.

    Deliberately takes only ``report_dir``, not ``dump``/``metrics_dir``/
    ``analysis_dir`` like ``ReportRequest`` -- this is ``report_model.json``
    read back off disk, not a fresh assembly (R0 never reruns).
    ``report_dir`` must be a directory a prior
    ``AuditApplication.generate_report()`` call wrote to.
    """

    report_dir: Path
    output_dir: Path | None = None
    only_clean_correct: bool = True
    csv: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "report_dir", Path(self.report_dir))
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))


@dataclass(frozen=True, slots=True)
class ExportLabelsResult:
    """Summarize one persisted risk-label export.

    ``n_negative_or_none`` is ``None`` exactly when the source report's
    primary metric was continuous -- no row carries ``risk_label`` at all,
    so a plain ``0`` here would misrepresent "not computed" as "computed and
    zero" (the same distinction ``ssat.report.labels.LabelsManifest.
    n_negative`` already makes, just renamed at this boundary for clarity
    to a caller who never sees that dataclass).
    """

    labels_path: Path
    manifest_path: Path
    csv_path: Path | None
    n_labels: int
    n_positive: int
    n_negative_or_none: int | None
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class EstimateResult:
    report: EstimateReport
    dump_mode: Literal["create", "resume", "none"]

    def to_dict(self) -> dict[str, Any]:
        return {"dump_mode": self.dump_mode, "report": to_primitive(self.report)}


@dataclass(frozen=True, slots=True)
class RunResult:
    status: Literal["completed", "cancelled"]
    output: Path
    estimate: EstimateResult
    summary: ExecutionSummary | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output": str(self.output),
            "estimate": self.estimate.to_dict(),
            "summary": to_primitive(self.summary),
        }


@dataclass(frozen=True, slots=True)
class DumpSectionSummary:
    rows: int
    counts_by_status: dict[str, int]


@dataclass(frozen=True, slots=True)
class DumpSummary:
    dump: Path
    schema_version: str
    code_version: str
    model_id: str
    started_at: str
    finished_at: str | None
    resume_count: int
    clean: DumpSectionSummary
    perturbed: DumpSectionSummary
    total_counts_by_status: dict[str, int]
    manifest_counts_match: bool

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class IndexRebuildResult:
    dump: Path
    indexed_items: int
    summary: DumpSummary

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def to_primitive(value: Any) -> Any:
    """Recursively convert public dataclasses, enums, and paths to JSON values."""

    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return to_primitive(asdict(value))
    if isinstance(value, Mapping):
        return {str(to_primitive(key)): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_primitive(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"cannot serialize application value {type(value).__name__}")
