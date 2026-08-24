"""Reusable application service orchestrating the SSAT core."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from ssat import __version__
from ssat.application import (
    _analysis_service,
    _dump_service,
    _estimate_service,
    _labels_service,
    _metrics_service,
    _run_service,
)
from ssat.application._session_service import _ExecutionContext
from ssat.application.types import (
    AnalyzeRequest,
    AnalyzeResult,
    ApplicationError,
    ApplicationErrorCode,
    CancellationToken,
    ComputeMetricsRequest,
    ComputeMetricsResult,
    DumpSummary,
    EstimateRequest,
    EstimateResult,
    EventSink,
    ExportLabelsRequest,
    ExportLabelsResult,
    IndexRebuildResult,
    InspectRequest,
    RebuildIndexRequest,
    ReportRequest,
    ReportResult,
    RunRequest,
    RunResult,
)
from ssat.core.adapter import AdapterProviderRegistry, default_adapter_provider_registry
from ssat.core.source import SourceProviderRegistry, default_source_provider_registry
from ssat.metrics.builtin_metrics import default_metric_registry
from ssat.metrics.registry import MetricRegistry
from ssat.report import (
    ClassificationAdapter,
    ReportDataAssembler,
    apply_asset_manifest,
    export as export_report,
    link_assets,
    render_fill_strategy_correlation,
    render_region_bar,
    render_report,
    render_secondary_report,
    render_vulnerability_histogram,
)
from ssat.utils.logger_factory import get_logger


CODE_VERSION = __version__


class PreparedRun:
    """One-shot, process-local preflight session owned by an application."""

    def __init__(
        self,
        application: AuditApplication,
        request: RunRequest,
        context: _ExecutionContext,
        estimate: EstimateResult,
        mode: Literal["create", "resume"],
        fingerprint: str,
    ) -> None:
        self._application = application
        self.request = request
        self._context: _ExecutionContext | None = context
        self.estimate = estimate
        self.mode = mode
        self.fingerprint = fingerprint
        self._closed = False
        self._consumed = False

    @property
    def confirmation_required(self) -> bool:
        return self.estimate.report.confirmation_required

    @property
    def context(self) -> _ExecutionContext:
        if self._context is None:
            raise ApplicationError(
                ApplicationErrorCode.EXECUTION,
                "prepared run resources are closed",
            )
        return self._context

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        context = self._context
        self._context = None
        if context is not None:
            context.adapter.cleanup_after_oom()
        self._closed = True

    def __enter__(self) -> PreparedRun:
        if self._closed:
            raise ApplicationError(
                ApplicationErrorCode.EXECUTION,
                "prepared run is already closed",
            )
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False


class AuditApplication:
    """UI-independent facade for configuration, preflight, execution, and dumps."""

    def __init__(
        self,
        adapter_registry: AdapterProviderRegistry | None = None,
        *,
        source_registry: SourceProviderRegistry | None = None,
        metric_registry: MetricRegistry | None = None,
        code_version: str = CODE_VERSION,
    ) -> None:
        if not code_version:
            raise ValueError("code_version must not be empty")
        self._registry = adapter_registry or default_adapter_provider_registry()
        self._source_registry = source_registry or default_source_provider_registry()
        self._metric_registry = metric_registry or default_metric_registry()
        self._code_version = code_version
        self._logger = get_logger(__name__)

    def prepare_run(
        self,
        request: RunRequest,
        *,
        event_sink: EventSink | None = None,
    ) -> PreparedRun:
        """Resolve and profile an audit without creating a new dump."""

        return _run_service.prepare_run(self, request, event_sink=event_sink)

    def execute_run(
        self,
        prepared: PreparedRun,
        *,
        confirmation_granted: bool = False,
        event_sink: EventSink | None = None,
        cancellation: CancellationToken | None = None,
    ) -> RunResult:
        """Execute one prepared audit after confirmation and stale-state checks."""

        return _run_service.execute_run(
            self,
            prepared,
            confirmation_granted=confirmation_granted,
            event_sink=event_sink,
            cancellation=cancellation,
        )

    def estimate(
        self,
        request: EstimateRequest,
        *,
        event_sink: EventSink | None = None,
    ) -> EstimateResult:
        """Return a structured standalone estimate without prompting."""

        return _estimate_service.estimate(self, request, event_sink=event_sink)

    def inspect(self, request: InspectRequest) -> DumpSummary:
        """Summarize authoritative dump rows and manifest provenance."""

        return _dump_service.inspect(request)

    def compute_metrics(self, request: ComputeMetricsRequest) -> ComputeMetricsResult:
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

        return _metrics_service.compute_metrics(self, request)

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResult:
        """Compute and persist the control/stability analysis for an existing dump+metrics pair.

        This is the Application-layer counterpart of what experiment scripts
        (experiments/synthetic_shortcut/analyze_control_stability.py,
        validate_reliability_thresholds_full.py) previously had to hand-roll:
        opening an AnalysisReader, indexing anchors/controls, and running
        A2-A6 in sequence. Scoped to one dump+metrics pair, matching
        AnalysisReader/compute_metrics's own scope; combining several runs
        into one item_values frame stays a script-level concern.
        """

        return _analysis_service.analyze(request)

    def generate_report(self, request: ReportRequest) -> ReportResult:
        """Assemble ``report.html`` (+CSV/JSON/SVG assets) for an existing dump+metrics(+analysis) pair.

        This is the Application-layer counterpart of what
        ``ssat.report.html_renderer``'s own module docstring reserves for
        "some earlier orchestration step" -- every ``ssat.report`` module
        (R0-R4) is a pure, single-purpose transformer that never opens more
        than one store or renders more than one kind of output; wiring them
        into one end-to-end pipeline is this method's job. The pipeline
        order is R0 (assemble) -> R3 (link_assets/apply_asset_manifest,
        per-sample heatmap/thumbnail PNGs) -> R2 (render the three chart
        SVGs) -> R1 (export the full population as JSON/CSV) -> R4 (render
        report.html + report_manifest.json) -> R4 again for the auxiliary
        ``report_question_driven.html`` (``render_secondary_report``) -- R4
        must run last (twice) because its templates reference every asset
        ref R2/R3 fill in and every file R1 writes, and
        ``render_secondary_report`` specifically must run after
        ``render_report`` because it reuses the ``assets/css``/``assets/js``
        the latter writes rather than duplicating them.

        Chart rendering (R2) is not gated by
        ``TaskPresentationAdapter.applicable_charts()`` here: the
        vulnerability histogram and region bar chart are unconditional
        already-computed-data visualizations (``ssat.report.charts`` itself
        draws an explicit "no data" placeholder rather than omitting them
        when empty, so missing data is never silently omitted), and
        ``render_fill_strategy_correlation`` already decides
        for itself whether a rank-correlation SVG exists at all (``None``
        when ``rank_correlation_rows`` is empty) -- ``applicable_charts()``
        has no additional decision left to make for either.

        ``fill_strategy_stability_available`` is passed to
        ``ClassificationAdapter`` as ``analysis_dir is not None``: the
        actual flag lives in ``AnalysisManifest.available_analyses``, only
        readable *inside* ``ReportDataAssembler.assemble()`` (which itself
        needs an already-built adapter), so there is no cheaper way to know
        it beforehand without a second, redundant analysis-store open. This
        mirrors the same proxy ``tests/integration/test_report_synthetic_
        dump.py``'s own ``_assembler()`` test helper already uses for this
        exact situation; the adapter only consumes this flag inside
        ``applicable_charts()``, which (see above) this method never calls.
        """

        dump = request.dump.expanduser().resolve(strict=True)
        metrics_dir = (request.metrics_dir or dump / "metrics").expanduser().resolve()
        analysis_dir = (request.analysis_dir or dump / "analysis").expanduser().resolve()
        if not analysis_dir.is_dir():
            analysis_dir = None
        report_dir = (request.report_dir or dump / "report").expanduser().resolve()

        try:
            adapter = ClassificationAdapter(
                primary_metric=request.primary_metric,
                fill_strategy_stability_available=analysis_dir is not None,
            )
            assembler = ReportDataAssembler(
                dump,
                metrics_dir,
                analysis_dir,
                adapter=adapter,
                top_k=request.top_k,
                bottom_k=request.bottom_k,
            )
            assembled = assembler.assemble(request.primary_metric)

            asset_manifest = link_assets(
                assembled, dump, metrics_dir, report_dir, primary_metric=request.primary_metric
            )
            linked = apply_asset_manifest(assembled, asset_manifest)

            charts_dir = report_dir / "assets" / "img" / "charts"
            charts_dir.mkdir(parents=True, exist_ok=True)

            (charts_dir / "vulnerability_histogram.svg").write_text(
                render_vulnerability_histogram(linked.full_sample_rankings), encoding="utf-8"
            )
            (charts_dir / "region_bar.svg").write_text(
                render_region_bar(linked.model.region_summary.rows), encoding="utf-8"
            )
            fill_strategy_ref = None
            fill_strategy_svg = render_fill_strategy_correlation(assembled.rank_correlation_rows)
            if fill_strategy_svg is not None:
                (charts_dir / "fill_strategy_correlation.svg").write_text(
                    fill_strategy_svg, encoding="utf-8"
                )
                fill_strategy_ref = "assets/img/charts/fill_strategy_correlation.svg"

            model = replace(
                linked.model,
                vulnerability_distribution=replace(
                    linked.model.vulnerability_distribution,
                    histogram_asset_ref="assets/img/charts/vulnerability_histogram.svg",
                ),
                region_summary=replace(
                    linked.model.region_summary,
                    chart_asset_ref="assets/img/charts/region_bar.svg",
                ),
                fill_strategy_correlation_asset_ref=fill_strategy_ref,
            )
            final = replace(linked, model=model)

            export_report(final, report_dir / "data")
            render_report(model, report_dir, top_k=request.top_k, bottom_k=request.bottom_k)
            # Always generated alongside the main report, not behind a flag:
            # render_secondary_report computes nothing new, it only
            # reorganizes the same `model` R4 just rendered into
            # report.html, so there is no meaningful "opt out" case to
            # support -- and it must run after render_report, which is what
            # writes the assets/css, assets/js this auxiliary page's own
            # <link>/<script> tags reference (render_secondary_report's own
            # docstring).
            secondary_report_html = render_secondary_report(
                model, report_dir, top_k=request.top_k, bottom_k=request.bottom_k
            )
        except Exception as error:
            # Catches everything R0-R4 can raise: ReportDataError (bad
            # primary_metric), AnalysisCorruptionError (a *present but
            # stale* analysis_dir -- see module docstring on why only a
            # *missing* analysis_dir is downgraded to None, not a corrupt
            # one), and any DumpHandle/MetricsStore/filesystem failure
            # surfaced along the way -- all map to the same REPORT error
            # code, mirroring compute_metrics/analyze's identical rationale.
            raise ApplicationError(
                ApplicationErrorCode.REPORT, f"cannot generate report: {error}"
            ) from error

        return ReportResult(
            dump=dump,
            metrics_dir=metrics_dir,
            analysis_dir=analysis_dir,
            report_dir=report_dir,
            secondary_report_html=secondary_report_html,
            n_samples=model.run_summary.n_samples,
            n_regions=len(model.region_summary.rows),
            grade_distribution=dict(model.region_summary.reliability_distribution),
            generated_at=model.meta.generated_at,
        )

    def export_labels(self, request: ExportLabelsRequest) -> ExportLabelsResult:
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

        return _labels_service.export_labels(request)

    def rebuild_index(self, request: RebuildIndexRequest) -> IndexRebuildResult:
        """Rebuild the perturbed index and return the resulting summary."""

        return _dump_service.rebuild_index(self, request)
