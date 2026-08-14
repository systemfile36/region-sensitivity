"""Reusable application service orchestrating the SSAT core."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from ssat import __version__
from ssat.analysis.control import compare_to_controls
from ssat.analysis.indexer import ComparisonIndexer
from ssat.analysis.interval import compute_intervals
from ssat.analysis.reader import AnalysisReader
from ssat.analysis.reliability import compute_reliability
from ssat.analysis.stability import compute_seed_stability, compute_strategy_stability
from ssat.analysis.store import save_analysis
from ssat.analysis.strategy_profile import compute_strategy_profile
from ssat.application.config import (
    ApplicationConfigError,
    LoadedApplicationConfig,
    load_application_config,
)
from ssat.application.locking import OutputLockedError, output_lock
from ssat.application.types import (
    AnalyzeRequest,
    AnalyzeResult,
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
    ReportRequest,
    ReportResult,
    RunRequest,
    RunResult,
)
from ssat.core.adapter import (
    AdapterProviderError,
    AdapterProviderRegistry,
    ModelAdapter,
    default_adapter_provider_registry,
)
from ssat.core.config import ConfigResolver, ResolvedConfig, ResolvedSkeletonSourceConfig
from ssat.core.dump import DumpReader, DumpWriter
from ssat.core.dump.manifest import load_manifest
from ssat.core.estimate import CostEstimator
from ssat.core.plan import PlanBuilder, RegionExpander
from ssat.core.plan.hashing import canonical_json
from ssat.core.region import RegionResolver
from ssat.core.region.skeleton_provider import SkeletonRegionProvider
from ssat.core.region.skeleton_store import (
    SkeletonBBoxStore,
    SkeletonDataError,
    load_skeleton_bbox_store,
)
from ssat.core.resume import ResumeIndex
from ssat.core.runtime import RuntimeCancelledError, run_audit
from ssat.core.source.base import SampleSource
from ssat.core.types import ItemStatus
from ssat.metrics.aggregate import aggregate_item_metrics
from ssat.metrics.builtin_metrics import default_metric_registry
from ssat.metrics.dump_reader import DumpHandle
from ssat.metrics.errors import MetricsRegistryError
from ssat.metrics.store import save_metrics
from ssat.report import (
    ClassificationAdapter,
    ReportDataAssembler,
    apply_asset_manifest,
    export as export_report,
    link_assets,
    render_fill_strategy_correlation,
    render_region_bar,
    render_report,
    render_vulnerability_histogram,
)
from ssat.utils.io import sha256_bytes, sha256_file
from ssat.utils.logger_factory import get_logger


CODE_VERSION = __version__


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    loaded: LoadedApplicationConfig
    adapter: ModelAdapter
    resolved: ResolvedConfig
    source: SampleSource
    builder: PlanBuilder
    region_resolver: RegionResolver


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
        code_version: str = CODE_VERSION,
    ) -> None:
        if not code_version:
            raise ValueError("code_version must not be empty")
        self._registry = adapter_registry or default_adapter_provider_registry()
        self._code_version = code_version
        self._logger = get_logger(__name__)

    def prepare_run(
        self,
        request: RunRequest,
        *,
        event_sink: EventSink | None = None,
    ) -> PreparedRun:
        """Resolve and profile an audit without creating a new dump."""

        if not isinstance(request, RunRequest):
            raise TypeError("request must be a RunRequest")
        self._emit(event_sink, "started", "config")
        context = self._build_context(request.config, request.base_dir)
        output = request.output.expanduser().resolve()
        mode = self._output_mode(output)
        resume = self._resume_for_context(output, context) if mode == "resume" else None
        self._emit(event_sink, "completed", "config")
        self._emit(event_sink, "started", "preflight")
        report = CostEstimator(request.estimate_options).estimate(
            context.resolved,
            context.builder,
            context.source,
            context.adapter,
            resume_index=resume,
            region_resolver=context.region_resolver,
        )
        result = EstimateResult(report=report, dump_mode=mode)
        fingerprint = self._preflight_fingerprint(context, output, mode)
        self._emit(
            event_sink,
            "completed",
            "preflight",
            completed=report.pending_perturbed_items,
            total=report.total_perturbed_items,
        )
        return PreparedRun(self, request, context, result, mode, fingerprint)

    def execute_run(
        self,
        prepared: PreparedRun,
        *,
        confirmation_granted: bool = False,
        event_sink: EventSink | None = None,
        cancellation: CancellationToken | None = None,
    ) -> RunResult:
        """Execute one prepared audit after confirmation and stale-state checks."""

        if not isinstance(prepared, PreparedRun) or prepared._application is not self:
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
            self._emit(event_sink, "cancelled", "run")
            return RunResult("cancelled", output, prepared.estimate, None)

        try:
            with output_lock(output):
                try:
                    current = self._preflight_fingerprint(
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
                self._emit(event_sink, "started", "run")
                prepared._consumed = True
                writer = DumpWriter(
                    output,
                    prepared.context.resolved,
                    code_version=self._code_version,
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
                        progress_callback=lambda completed, total: self._emit(
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
                    self._emit(event_sink, "cancelled", "run")
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

        self._emit(
            event_sink,
            "completed",
            "run",
            completed=summary.records_written,
            total=summary.records_written,
        )
        return RunResult("completed", output, prepared.estimate, summary)

    def estimate(
        self,
        request: EstimateRequest,
        *,
        event_sink: EventSink | None = None,
    ) -> EstimateResult:
        """Return a structured standalone estimate without prompting."""

        if not isinstance(request, EstimateRequest):
            raise TypeError("request must be an EstimateRequest")
        self._emit(event_sink, "started", "config")
        context = self._build_context(request.config, request.base_dir)
        mode: Literal["create", "resume", "none"] = "none"
        resume = None
        if request.dump is not None:
            dump = request.dump.expanduser().resolve()
            mode = self._output_mode(dump)
            if mode != "resume":
                raise ApplicationError(
                    ApplicationErrorCode.OUTPUT,
                    "estimate --dump requires an existing valid dump",
                )
            resume = self._resume_for_context(dump, context)
        self._emit(event_sink, "completed", "config")
        self._emit(event_sink, "started", "preflight")
        report = CostEstimator(request.options).estimate(
            context.resolved,
            context.builder,
            context.source,
            context.adapter,
            resume_index=resume,
            region_resolver=context.region_resolver,
        )
        self._emit(event_sink, "completed", "preflight")
        return EstimateResult(report, mode)

    def inspect(self, request: InspectRequest) -> DumpSummary:
        """Summarize authoritative dump rows and manifest provenance."""

        try:
            dump = request.dump.expanduser().resolve(strict=True)
            reader = DumpReader(dump)
            manifest = reader.read_manifest()
            clean = reader.read_clean()
            perturbed = reader.read_perturbed()
            clean_observed = Counter(row["status"] for row in clean.to_pylist())
            perturbed_observed = Counter(row["status"] for row in perturbed.to_pylist())
            clean_counts = {
                status.value: clean_observed.get(status.value, 0) for status in ItemStatus
            }
            perturbed_counts = {
                status.value: perturbed_observed.get(status.value, 0)
                for status in ItemStatus
            }
            total_counts = {
                status.value: clean_counts[status.value] + perturbed_counts[status.value]
                for status in ItemStatus
            }
            manifest_counts = {
                status.value: count for status, count in manifest.counts_by_status.items()
            }
            return DumpSummary(
                dump=dump,
                schema_version=manifest.schema_version,
                code_version=manifest.code_version,
                model_id=manifest.adapter_spec.model_id,
                started_at=manifest.started_at.isoformat(),
                finished_at=(
                    None if manifest.finished_at is None else manifest.finished_at.isoformat()
                ),
                resume_count=len(manifest.resume_events),
                clean=DumpSectionSummary(clean.num_rows, clean_counts),
                perturbed=DumpSectionSummary(
                    perturbed.num_rows,
                    perturbed_counts,
                ),
                total_counts_by_status=total_counts,
                manifest_counts_match=manifest_counts == total_counts,
            )
        except Exception as error:
            raise ApplicationError(
                ApplicationErrorCode.CORRUPTION,
                f"cannot inspect dump: {error}",
            ) from error

    def compute_metrics(self, request: ComputeMetricsRequest) -> ComputeMetricsResult:
        """Compute and persist every built-in metric for an existing dump.

        This is the Application-layer counterpart of what experiment scripts
        (e.g. experiments/synthetic_shortcut/run_audit.py) and test fixtures
        previously had to hand-roll themselves by opening a DumpHandle
        directly: it always registers every built-in metric via
        default_metric_registry() (v1 scope intentionally has no per-metric
        selection flag -- see docs/IMPLE_PLAN_METRIC_DESIGN_v1.md section 8),
        and stores the result under metrics_dir (default: <dump>/metrics).
        """

        dump = request.dump.expanduser().resolve(strict=True)
        metrics_dir = (request.metrics_dir or dump / "metrics").expanduser().resolve()
        try:
            handle = DumpHandle(dump)
            joined = handle.joined()
            resolved_config = handle.manifest.resolved_config

            registry = default_metric_registry()
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

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResult:
        """Compute and persist the control/stability analysis for an existing dump+metrics pair.

        This is the Application-layer counterpart of what experiment scripts
        (experiments/synthetic_shortcut/analyze_control_stability.py,
        validate_reliability_thresholds_full.py) previously had to hand-roll:
        opening an AnalysisReader, indexing anchors/controls, and running
        A2-A6 in sequence (docs/IMPLE_PLAN_CONTROL_STABILITY_v1.md section
        2.2 deferred this until that repeated-call-pattern was observed --
        it has been). Scoped to one dump+metrics pair, matching
        AnalysisReader/compute_metrics's own scope; combining several runs
        into one item_values frame stays a script-level concern.
        """

        dump = request.dump.expanduser().resolve(strict=True)
        metrics_dir = (request.metrics_dir or dump / "metrics").expanduser().resolve()
        analysis_dir = (request.analysis_dir or dump / "analysis").expanduser().resolve()
        try:
            reader = AnalysisReader(dump, metrics_dir)
            item_values = reader.item_values()
            indexer = ComparisonIndexer(
                reader.item_context(), area_match_tolerance=request.area_match_tolerance
            )

            control_rows = compare_to_controls(
                item_values,
                indexer.control_pairs,
                area_match_tolerance=request.area_match_tolerance,
            )
            seed_rows = compute_seed_stability(item_values)
            strategy_rows, rank_rows = compute_strategy_stability(
                item_values, primary_metric=request.primary_metric
            )
            profile_rows = compute_strategy_profile(
                strategy_rows, rank_rows, primary_metric=request.primary_metric
            )
            interval_rows = compute_intervals(
                item_values, n_bootstrap=request.n_bootstrap, random_seed=request.random_seed
            )
            reliability_rows = compute_reliability(
                control_rows,
                seed_rows,
                strategy_rows,
                interval_rows,
                z_vs_control_threshold=request.z_vs_control_threshold,
                seed_cv_threshold=request.seed_cv_threshold,
            )
            available = reader.available_analyses()

            manifest = save_analysis(
                analysis_dir,
                control_rows=control_rows,
                seed_rows=seed_rows,
                strategy_rows=strategy_rows,
                rank_correlation_rows=rank_rows,
                strategy_profile_rows=profile_rows,
                interval_rows=interval_rows,
                reliability_rows=reliability_rows,
                coverage_report=indexer.coverage_report,
                available_analyses=available,
                thresholds={
                    "z_vs_control_threshold": request.z_vs_control_threshold,
                    "seed_cv_threshold": request.seed_cv_threshold,
                    "area_match_tolerance": request.area_match_tolerance,
                },
                n_bootstrap=request.n_bootstrap,
                random_seed=request.random_seed,
                source_metrics_manifest_path=metrics_dir / "metrics_manifest.json",
            )
        except Exception as error:
            # Catches AnalysisReader's unwrapped MetricsCorruptionError/
            # MetricsSchemaError (module docstring on ssat.analysis.reader),
            # ssat.analysis.errors.* raised by A1-A6/A7 themselves, and
            # anything DumpHandle surfaces -- all map to the same ANALYSIS
            # error code, mirroring compute_metrics's identical rationale.
            raise ApplicationError(
                ApplicationErrorCode.ANALYSIS, f"cannot compute analysis: {error}"
            ) from error

        return AnalyzeResult(
            dump=dump,
            metrics_dir=metrics_dir,
            analysis_dir=analysis_dir,
            available_analyses=available,
            coverage_report=indexer.coverage_report,
            grade_distribution=dict(manifest.grade_distribution),
            n_reliability_rows=len(reliability_rows),
            computed_at=manifest.computed_at.isoformat(),
        )

    def generate_report(self, request: ReportRequest) -> ReportResult:
        """Assemble ``report.html`` (+CSV/JSON/SVG assets) for an existing dump+metrics(+analysis) pair.

        This is the Application-layer counterpart of what
        ``ssat.report.html_renderer``'s own module docstring reserves for
        "some earlier orchestration step" -- every ``ssat.report`` module
        (R0-R4) is a pure, single-purpose transformer that never opens more
        than one store or renders more than one kind of output; wiring them
        into one end-to-end pipeline is this method's job
        (IMPLE_PLAN_REPORTING_v1.md §5 단계7). The pipeline order is R0
        (assemble) -> R3 (link_assets/apply_asset_manifest, per-sample
        heatmap/thumbnail PNGs) -> R2 (render the three chart SVGs) -> R1
        (export the full population as JSON/CSV) -> R4 (render report.html
        + report_manifest.json) -- R4 must run last because its template
        references every asset ref R2/R3 fill in and every file R1 writes.

        Chart rendering (R2) is not gated by
        ``TaskPresentationAdapter.applicable_charts()`` here: the
        vulnerability histogram and region bar chart are unconditional
        already-computed-data visualizations (``ssat.report.charts`` itself
        draws an explicit "no data" placeholder rather than omitting them
        when empty, matching design's "결측이 조용히 생략되지 않음"
        principle), and ``render_fill_strategy_correlation`` already decides
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
            n_samples=model.run_summary.n_samples,
            n_regions=len(model.region_summary.rows),
            grade_distribution=dict(model.region_summary.reliability_distribution),
            generated_at=model.meta.generated_at,
        )

    def rebuild_index(self, request: RebuildIndexRequest) -> IndexRebuildResult:
        """Rebuild the perturbed index and return the resulting summary."""

        dump = request.dump.expanduser().resolve()
        try:
            with output_lock(dump):
                ResumeIndex.rebuild(dump)
            summary = self.inspect(InspectRequest(dump))
            return IndexRebuildResult(dump, summary.perturbed.rows, summary)
        except OutputLockedError as error:
            raise ApplicationError(ApplicationErrorCode.LOCKED, str(error)) from error
        except ApplicationError:
            raise
        except Exception as error:
            raise ApplicationError(
                ApplicationErrorCode.CORRUPTION,
                f"cannot rebuild dump index: {error}",
            ) from error

    def _build_context(
        self,
        config: object,
        base_dir: Path | None,
    ) -> _ExecutionContext:
        try:
            loaded = load_application_config(
                config,  # type: ignore[arg-type]
                self._registry,
                base_dir=base_dir,
            )
            adapter = self._registry.build(
                loaded.adapter_config,
                base_dir=loaded.base_dir,
            )
            resolved = ConfigResolver().resolve(
                loaded.audit,
                adapter,
                loaded.sample_source,
                base_dir=loaded.base_dir,
                config_source=loaded.config_source,
                source_provenance=loaded.source_provenance,
            )
            skeleton_store = self._load_skeleton_store(resolved.skeleton_source)
            region_expander = RegionExpander(
                SkeletonRegionProvider(skeleton_store)
                if skeleton_store is not None
                else None
            )
            return _ExecutionContext(
                loaded,
                adapter,
                resolved,
                loaded.sample_source,
                PlanBuilder(resolved, loaded.sample_source, region_expander=region_expander),
                RegionResolver(skeleton_store=skeleton_store),
            )
        except ApplicationConfigError as error:
            raise ApplicationError(ApplicationErrorCode.CONFIG, str(error)) from error
        except AdapterProviderError as error:
            raise ApplicationError(ApplicationErrorCode.PROVIDER, str(error)) from error
        except Exception as error:
            raise ApplicationError(
                ApplicationErrorCode.CONFIG,
                f"failed to resolve audit configuration: {error}",
            ) from error

    @staticmethod
    def _load_skeleton_store(
        skeleton_source: ResolvedSkeletonSourceConfig | None,
    ) -> SkeletonBBoxStore | None:
        """Load the skeleton bbox store referenced by a resolved config.

        Args:
            skeleton_source: Resolved, hash-verified reference, or ``None``.

        Returns:
            The loaded store, or ``None`` if no ``skeleton_source`` was
            configured.

        Raises:
            ApplicationConfigError: If the referenced file is unreadable or
                fails re-verification against its resolved hash.
        """

        if skeleton_source is None:
            return None
        try:
            return load_skeleton_bbox_store(
                skeleton_source.bbox_data,
                expected_hash=skeleton_source.bbox_data_hash,
            )
        except SkeletonDataError as error:
            raise ApplicationConfigError(
                f"failed to load skeleton_source data: {error}"
            ) from error

    def _resume_for_context(
        self,
        output: Path,
        context: _ExecutionContext,
    ) -> ResumeIndex:
        try:
            manifest = load_manifest(output / "run_manifest.json")
            if manifest.resolved_config != context.resolved:
                raise ApplicationError(
                    ApplicationErrorCode.OUTPUT,
                    "existing dump resolved_config does not match the request",
                )
            if manifest.adapter_spec != context.adapter.describe():
                raise ApplicationError(
                    ApplicationErrorCode.OUTPUT,
                    "existing dump adapter does not match the request",
                )
            if manifest.code_version != self._code_version:
                raise ApplicationError(
                    ApplicationErrorCode.OUTPUT,
                    "existing dump code_version does not match this application",
                )
            return ResumeIndex.open(output)
        except ApplicationError:
            raise
        except Exception as error:
            raise ApplicationError(
                ApplicationErrorCode.OUTPUT,
                f"cannot resume existing dump: {error}",
            ) from error

    @staticmethod
    def _output_mode(output: Path) -> Literal["create", "resume"]:
        if not output.exists():
            return "create"
        if not output.is_dir():
            raise ApplicationError(
                ApplicationErrorCode.OUTPUT,
                f"output is not a directory: {output}",
            )
        if not any(output.iterdir()):
            return "create"
        if (output / "run_manifest.json").is_file():
            try:
                load_manifest(output / "run_manifest.json")
            except Exception as error:
                raise ApplicationError(
                    ApplicationErrorCode.OUTPUT,
                    f"output is not a valid dump: {error}",
                ) from error
            return "resume"
        raise ApplicationError(
            ApplicationErrorCode.OUTPUT,
            f"non-empty output is not an SSAT dump: {output}",
        )

    def _preflight_fingerprint(
        self,
        context: _ExecutionContext,
        output: Path,
        expected_mode: Literal["create", "resume"],
    ) -> str:
        loaded = context.loaded
        if sha256_file(loaded.source_provenance.manifest) != loaded.source_provenance.manifest_hash:
            return "source-manifest-changed"
        if loaded.config_source is not None and loaded.config_hash is not None:
            if sha256_file(loaded.config_source) != loaded.config_hash:
                return "configuration-changed"
        current_mode = self._output_mode(output)
        payload = {
            "expected_mode": expected_mode,
            "current_mode": current_mode,
            "source_hash": loaded.source_provenance.manifest_hash,
            "config_hash": loaded.config_hash,
            "resolved_config": context.resolved,
            "adapter_spec": context.adapter.describe(),
            "output": self._output_snapshot(output, current_mode),
        }
        return sha256_bytes(canonical_json(payload).encode("utf-8"))

    @staticmethod
    def _output_snapshot(
        output: Path,
        mode: Literal["create", "resume"],
    ) -> list[tuple[str, int, int]]:
        if mode == "create":
            return []
        rows = []
        for directory in ("clean", "perturbed", "index"):
            for path in sorted((output / directory).glob("*.parquet")):
                stat = path.stat()
                rows.append((path.relative_to(output).as_posix(), stat.st_size, stat.st_mtime_ns))
        manifest = output / "run_manifest.json"
        stat = manifest.stat()
        rows.append(("run_manifest.json", stat.st_size, stat.st_mtime_ns))
        return rows

    def _emit(
        self,
        sink: EventSink | None,
        kind: str,
        phase: str,
        *,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        if sink is None:
            return
        try:
            sink(ApplicationEvent(kind, phase, completed, total))
        except Exception:
            self._logger.warning("application.event_sink_failed phase=%s", phase)
