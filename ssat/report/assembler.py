"""R0 ReportDataAssembler: join MetricsStore + AnalysisStore + run_manifest into ReportModel.

This is the reporting layer's bottleneck module (IMPLE_PLAN_REPORTING_v1.md
§5 단계 2 "이 계획서의 병목") — every join/aggregation policy §1's six gaps
identified gets fixed here, in code, once:

- Gap#1: ``reliability.parquet`` is treated as the de-facto source of every
  AnchorKey that exists (``AnchorTable`` itself is never persisted).
- Gap#2: ``RegionRow.region_kind``/areas come from MetricsStore's
  ``region_metrics.parquet``, not AnalysisStore.
- Gap#3: When several anchors share a ``region_key``, ``RegionRow.
  reliability_grade`` takes the worst grade across them (``UNRELIABLE >
  LOW > MODERATE > HIGH``), with ``reliability_distribution`` alongside it
  so that worst-case choice cannot be mistaken for "every anchor is this
  bad" (see :func:`_worst_grade`).
- Gap#4: ``SampleCard.top_regions[*].degradation`` is read verbatim from
  ``spatial_profile.parquet`` — the exact same source R3's heatmaps render
  from, so a card's text and its image can never contradict each other.
- Gap#5: :meth:`ReportDataAssembler.assemble` returns an
  :class:`AssembledReport`, not a bare ``ReportModel`` — ``ReportModel``
  only carries the top-K/bottom-K slice, while ``AssembledReport.
  full_sample_rankings`` carries every sample (§R1 "전량은 항상 CSV로 접근
  가능").
- Gap#6: every reduction performed here (means, percentiles, distribution
  counts) is a straightforward, uncontested arithmetic summary of numbers
  the metrics/analysis engines already computed — never a new statistic
  (design §0 "계산하지 않고 조립한다").

``analysis_dir=None`` is a first-class input, not an error path: a run
that never had ``ssat analyze`` executed against it still assembles a
complete ``ReportModel``, with every analysis-derived field explicitly
marked unavailable (``None``/empty, plus a ``note`` on the scorecard's
control-comparison card) rather than silently omitted (design §6.2 C1).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

import ssat
from ssat.analysis.store import (
    AnalysisManifest,
    load_analysis,
    verify_source_metrics,
)
from ssat.analysis.types import (
    ControlComparisonRow,
    FlagValue,
    ReliabilityGrade,
    ReliabilityRow,
)
from ssat.metrics.dump_reader import DumpHandle
from ssat.metrics.store import MetricsManifest, load_metrics
from ssat.metrics.types import RegionMetrics, SampleMetrics, SpatialProfile
from ssat.report.adapters import DetectionAdapter, TaskPresentationAdapter
from ssat.report.errors import ReportDataError
from ssat.report.types import (
    REPORT_SCHEMA_VERSION,
    FlaggedItem,
    MetricCard,
    ProvenanceInfo,
    RegionRow,
    RegionSummary,
    ReliabilitySpotlight,
    ReportGrade,
    ReportMeta,
    ReportModel,
    ReportSchemaVersions,
    RunSummary,
    SampleCard,
    SampleRankings,
    TaskKind,
    TopRegionEntry,
    VulnerabilityDistribution,
    VulnerabilitySummaryStats,
)
from ssat.utils.io import sha256_file

_GradeKeyT = TypeVar("_GradeKeyT")

# Worst-first: the grade a group of anchors is reduced to is whichever of
# these appears first among them (§1 격차#3's worst-case policy).
_GRADE_SEVERITY_ORDER = (
    ReportGrade.UNRELIABLE,
    ReportGrade.LOW,
    ReportGrade.MODERATE,
    ReportGrade.HIGH,
)


@dataclass(frozen=True, slots=True)
class AssembledReport:
    """R0's full output: the top-K/bottom-K ``ReportModel`` plus every sample (§1 격차#5).

    Attributes:
        model: The JSON-serializable report R1 exports and R4 renders.
        full_sample_rankings: Every sample the run produced, sorted by
            ``vulnerability_score`` descending (unscored samples last, see
            :meth:`ReportDataAssembler._build_full_rankings`) — consumed by
            R1's ``sample_rankings.csv`` (full population) and R2's
            histogram (a top-K-only slice cannot represent a distribution).
            Not serialized as part of ``model``.
    """

    model: ReportModel
    full_sample_rankings: tuple[SampleCard, ...]


@dataclass(frozen=True, slots=True)
class _AnalysisContext:
    """The subset of one AnalysisStore load that R0 actually consumes.

    ``seed_rows``/``strategy_rows``/``rank_correlation_rows``/
    ``strategy_profile_rows``/``interval_rows``/``coverage_report`` are
    loaded (``load_analysis`` returns all nine outputs together) but not
    threaded any further — R0's v1 scope only surfaces control-comparison
    and reliability data (design §3's ``ReportModel`` schema has no field
    for the others yet).
    """

    control_rows: tuple[ControlComparisonRow, ...]
    reliability_rows: tuple[ReliabilityRow, ...]
    manifest: AnalysisManifest


class ReportDataAssembler:
    """R0: the only reporting-layer module that opens all three stores at once."""

    def __init__(
        self,
        dump_dir: Path,
        metrics_dir: Path,
        analysis_dir: Path | None = None,
        *,
        adapter: TaskPresentationAdapter,
        top_k: int = 20,
        bottom_k: int = 20,
        region_top_k: int = 5,
    ) -> None:
        """Configure the three source locations and the task/selection policy.

        Args:
            dump_dir: Root of the source dump, opened only through
                ``ssat.metrics.dump_reader.DumpHandle`` (§3.3 "dump 접근은
                DumpHandle을 통해서만").
            metrics_dir: MetricsStore directory for this dump.
            analysis_dir: AnalysisStore directory for this dump+metrics
                pair, or ``None`` when no ``ssat analyze`` run exists —
                every analysis-derived field is then assembled as
                explicitly unavailable rather than raising (design §6.2 C1).
            adapter: Task-specific card/field/chart translator (R5).
            top_k: Number of most-vulnerable samples to render as
                ``SampleCard``s (design §R0 default).
            bottom_k: Number of most-robust samples to render as
                ``SampleCard``s.
            region_top_k: Number of a sample's most-affected regions to
                surface in its ``SampleCard.top_regions`` (undetermined by
                design/plan; confirmed with the user for this stage).

        Raises:
            ValueError: If top_k, bottom_k, or region_top_k is negative.
        """

        for name, value in (
            ("top_k", top_k),
            ("bottom_k", bottom_k),
            ("region_top_k", region_top_k),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        self._dump_dir = Path(dump_dir)
        self._metrics_dir = Path(metrics_dir)
        self._analysis_dir = Path(analysis_dir) if analysis_dir is not None else None
        self._adapter = adapter
        self._top_k = top_k
        self._bottom_k = bottom_k
        self._region_top_k = region_top_k

    def assemble(self, primary_metric: str) -> AssembledReport:
        """Join the three stores into one ``AssembledReport``.

        Args:
            primary_metric: Registered metric name to build every
                metric-scoped section from (scorecard, rankings, region
                summary) — must be one of the source MetricsStore's
                registered metrics.

        Raises:
            ReportDataError: If primary_metric is not registered in the
                source MetricsStore.
            ssat.analysis.errors.AnalysisCorruptionError: If ``analysis_dir``
                is given but its AnalysisStore no longer matches the source
                MetricsStore (propagated from ``verify_source_metrics``
                unwrapped — mapping to a report-layer error is the
                Application layer's job, design §5 단계 7, mirroring how
                ``AuditApplication.analyze`` already handles this boundary).
        """

        if not primary_metric:
            raise ValueError("primary_metric must not be empty")

        run_manifest = DumpHandle(self._dump_dir).manifest
        _item_metrics, aggregation, metrics_manifest = load_metrics(self._metrics_dir)

        registered_names = {metric.name for metric in metrics_manifest.registered_metrics}
        if primary_metric not in registered_names:
            raise ReportDataError(
                f"primary_metric not found in metrics store: {primary_metric}"
            )

        sample_rows = [
            row for row in aggregation.sample_metrics if row.metric_name == primary_metric
        ]
        region_rows = [
            row for row in aggregation.region_metrics if row.metric_name == primary_metric
        ]
        spatial_rows = [
            row for row in aggregation.spatial_profile if row.metric_name == primary_metric
        ]

        analysis = self._load_analysis_context()

        full_rankings = self._build_full_rankings(sample_rows, spatial_rows, analysis, primary_metric)
        model = ReportModel(
            meta=self._build_meta(run_manifest, metrics_manifest, analysis),
            run_summary=self._build_run_summary(run_manifest, metrics_manifest, sample_rows),
            scorecard=tuple(self._build_scorecard(sample_rows, analysis, primary_metric)),
            vulnerability_distribution=self._build_vulnerability_distribution(full_rankings),
            sample_rankings=self._build_sample_rankings(full_rankings),
            region_summary=self._build_region_summary(region_rows, analysis, primary_metric),
            reliability_spotlight=self._build_reliability_spotlight(analysis),
            provenance=self._build_provenance(analysis),
        )
        return AssembledReport(model=model, full_sample_rankings=full_rankings)

    # --- source loading ----------------------------------------------------

    def _load_analysis_context(self) -> _AnalysisContext | None:
        if self._analysis_dir is None:
            return None
        (
            control_rows,
            _seed_rows,
            _strategy_rows,
            _rank_correlation_rows,
            _strategy_profile_rows,
            _interval_rows,
            reliability_rows,
            _coverage_report,
            manifest,
        ) = load_analysis(self._analysis_dir)
        verify_source_metrics(manifest, self._metrics_dir / "metrics_manifest.json")
        return _AnalysisContext(
            control_rows=tuple(control_rows),
            reliability_rows=tuple(reliability_rows),
            manifest=manifest,
        )

    # --- run_summary ---------------------------------------------------------

    def _build_run_summary(
        self,
        run_manifest: Any,
        metrics_manifest: MetricsManifest,
        sample_rows: Sequence[SampleMetrics],
    ) -> RunSummary:
        resolved_config = run_manifest.resolved_config
        duration_seconds = None
        finished_at = run_manifest.finished_at
        if finished_at is not None:
            started_at = run_manifest.started_at
            duration_seconds = (finished_at - started_at).total_seconds()
        return RunSummary(
            dataset_name=_dataset_name(resolved_config),
            n_samples=len(sample_rows),
            n_regions_per_sample=len(resolved_config.regions),
            n_conditions=len(resolved_config.perturbations),
            duration_seconds=duration_seconds,
            failure_rate=_failure_rate(metrics_manifest.exclusion_summary),
            model_id=resolved_config.adapter_spec.model_id,
            preprocessing_desc=resolved_config.adapter_spec.preprocessing_desc,
        )

    # --- scorecard -------------------------------------------------------------

    def _build_scorecard(
        self,
        sample_rows: Sequence[SampleMetrics],
        analysis: _AnalysisContext | None,
        primary_metric: str,
    ) -> list[MetricCard]:
        cards = list(self._adapter.summarize_performance(sample_rows))
        cards.append(self._control_comparison_card(analysis, primary_metric))
        return cards

    def _control_comparison_card(
        self, analysis: _AnalysisContext | None, primary_metric: str
    ) -> MetricCard:
        """Build the one control-comparison card R0 itself owns (design §5 단계 2).

        Unlike the adapter's cards (MetricsStore-derived, task-specific),
        this reads AnalysisStore — control comparison is a task-agnostic
        concept R5 has no reason to know about. Its value is the mean
        ``z_vs_control`` across every matched anchor: the same standardized
        quantity A6 thresholds to derive reliability grades, so the number
        on this card and the grades elsewhere in the report share one
        justification (confirmed with the user for this stage).
        """

        if analysis is None:
            return MetricCard(
                key="control_comparison",
                label="Mean Z vs Control",
                value=None,
                unit="",
                higher_is_better=False,
                note="분석 미실행: 대조군 비교가 실행되지 않았습니다.",
            )
        values = [
            row.z_vs_control
            for row in analysis.control_rows
            if row.metric_name == primary_metric
            and row.control_available is FlagValue.TRUE
            and row.z_vs_control is not None
        ]
        if not values:
            return MetricCard(
                key="control_comparison",
                label="Mean Z vs Control",
                value=None,
                unit="",
                higher_is_better=False,
                note="해당 없음: 매칭된 대조군이 없습니다.",
            )
        return MetricCard(
            key="control_comparison",
            label="Mean Z vs Control",
            value=sum(values) / len(values),
            unit="",
            higher_is_better=False,
        )

    # --- sample rankings ---------------------------------------------------

    def _build_full_rankings(
        self,
        sample_rows: Sequence[SampleMetrics],
        spatial_rows: Sequence[SpatialProfile],
        analysis: _AnalysisContext | None,
        primary_metric: str,
    ) -> tuple[SampleCard, ...]:
        """Sort every sample by vulnerability_score descending (§1 격차#5).

        Samples with no computed ``vulnerability_score`` (zero valid
        primary-metric items) cannot be ranked at all; they are kept —
        dropping them would silently shrink ``n_samples`` — but sorted after
        every scored sample, and excluded from both top-K and bottom-K so
        neither gallery mislabels "unranked" as "most vulnerable" or "most
        robust" (see :meth:`_build_sample_rankings`).
        """

        scored = sorted(
            (row for row in sample_rows if row.vulnerability_score is not None),
            key=lambda row: row.vulnerability_score,  # type: ignore[arg-type,return-value]
            reverse=True,
        )
        unscored = sorted(
            (row for row in sample_rows if row.vulnerability_score is None),
            key=lambda row: row.sample_id,
        )

        highlighted_ids: set[str] = set()
        if self._top_k:
            highlighted_ids.update(row.sample_id for row in scored[: self._top_k])
        if self._bottom_k:
            highlighted_ids.update(row.sample_id for row in scored[-self._bottom_k :])

        grades_by_sample = (
            _group_report_grades(
                analysis.reliability_rows, primary_metric, key=lambda row: row.anchor_key.sample_id
            )
            if analysis is not None
            else {}
        )
        grades_by_sample_region = (
            _group_report_grades(
                analysis.reliability_rows,
                primary_metric,
                key=lambda row: (row.anchor_key.sample_id, row.anchor_key.region_key),
            )
            if analysis is not None
            else {}
        )
        spatial_by_sample: dict[str, list[SpatialProfile]] = defaultdict(list)
        for row in spatial_rows:
            if row.sample_id in highlighted_ids:
                spatial_by_sample[row.sample_id].append(row)

        cards = []
        for row in (*scored, *unscored):
            top_regions: tuple[TopRegionEntry, ...] = ()
            if row.sample_id in highlighted_ids:
                top_regions = self._top_regions_for_sample(
                    spatial_by_sample.get(row.sample_id, ()), grades_by_sample_region
                )
            cards.append(
                SampleCard(
                    sample_id=row.sample_id,
                    gt_label=row.gt_label,
                    clean_correct=row.clean_correct,
                    vulnerability_score=row.vulnerability_score,
                    reliability_grade=(
                        _worst_grade(grades_by_sample.get(row.sample_id, ()))
                        if analysis is not None
                        else None
                    ),
                    heatmap_asset_ref=None,
                    thumbnail_asset_ref=None,
                    top_regions=top_regions,
                    task_extra=self._adapter.sample_extra_fields(row),
                )
            )
        return tuple(cards)

    def _top_regions_for_sample(
        self,
        spatial_rows: Sequence[SpatialProfile],
        grades_by_sample_region: dict[tuple[str, str], list[ReportGrade]],
    ) -> tuple[TopRegionEntry, ...]:
        scored = sorted(
            (row for row in spatial_rows if row.degradation is not None),
            key=lambda row: row.degradation,  # type: ignore[arg-type,return-value]
            reverse=True,
        )
        unscored = [row for row in spatial_rows if row.degradation is None]
        ordered = (*scored, *unscored)
        entries = []
        for row in ordered[: self._region_top_k]:
            grades = grades_by_sample_region.get((row.sample_id, row.region_key), ())
            entries.append(
                TopRegionEntry(
                    region_key=row.region_key,
                    degradation=row.degradation,
                    reliability_grade=_worst_grade(grades) if grades else None,
                )
            )
        return tuple(entries)

    def _build_sample_rankings(self, full_rankings: tuple[SampleCard, ...]) -> SampleRankings:
        scored = [card for card in full_rankings if card.vulnerability_score is not None]
        most_vulnerable = tuple(scored[: self._top_k]) if self._top_k else ()
        most_robust = tuple(reversed(scored[-self._bottom_k :])) if self._bottom_k else ()
        return SampleRankings(most_vulnerable=most_vulnerable, most_robust=most_robust)

    def _build_vulnerability_distribution(
        self, full_rankings: tuple[SampleCard, ...]
    ) -> VulnerabilityDistribution:
        scores = [
            card.vulnerability_score
            for card in full_rankings
            if card.vulnerability_score is not None
        ]
        if not scores:
            stats = VulnerabilitySummaryStats(mean=None, median=None, p90=None, p99=None)
        else:
            array = np.asarray(scores, dtype=float)
            stats = VulnerabilitySummaryStats(
                mean=float(np.mean(array)),
                median=float(np.median(array)),
                p90=float(np.percentile(array, 90)),
                p99=float(np.percentile(array, 99)),
            )
        return VulnerabilityDistribution(histogram_asset_ref=None, summary_stats=stats)

    # --- region summary -----------------------------------------------------

    def _build_region_summary(
        self,
        region_rows: Sequence[RegionMetrics],
        analysis: _AnalysisContext | None,
        primary_metric: str,
    ) -> RegionSummary:
        """Join ``region_metrics.parquet`` (source of which regions exist, §1 격차#2) with grades.

        Iterating ``region_rows`` (already primary-metric-filtered
        ``region_metrics.parquet``) rather than ``reliability.parquet`` is
        what keeps control-only regions — already excluded from N3
        aggregation — out of ``region_summary.rows`` (§1 부수 확인).
        """

        grades_by_region = (
            _group_report_grades(
                analysis.reliability_rows, primary_metric, key=lambda row: row.anchor_key.region_key
            )
            if analysis is not None
            else {}
        )

        rows = []
        for row in sorted(region_rows, key=lambda row: row.region_key):
            grades = grades_by_region.get(row.region_key, ())
            rows.append(
                RegionRow(
                    region_key=row.region_key,
                    region_kind=row.region_kind.value,
                    intended_area_px=row.intended_area_px,
                    effective_area_px=row.effective_area_px,
                    mean_degradation=row.metric_mean,
                    flip_rate=row.flip_rate,
                    n_valid=row.n_valid,
                    reliability_grade=_worst_grade(grades) if analysis is not None else None,
                    reliability_distribution=(
                        dict(Counter(grade.value for grade in grades))
                        if analysis is not None
                        else {}
                    ),
                )
            )
        dataset_distribution = (
            dict(analysis.manifest.grade_distribution) if analysis is not None else {}
        )
        return RegionSummary(rows=tuple(rows), reliability_distribution=dataset_distribution)

    # --- reliability spotlight -----------------------------------------------

    def _build_reliability_spotlight(
        self, analysis: _AnalysisContext | None
    ) -> ReliabilitySpotlight:
        """Surface every UNRELIABLE-grade anchor across every metric (design §5 단계 2 step7).

        Unlike the per-metric-filtered sections above, this is intentionally
        *not* filtered to ``primary_metric`` — the plan's own wording
        ("reliability.parquet에서 reliability_grade == UNRELIABLE인 행 전부")
        surfaces every unreliable finding, regardless of which metric
        flagged it, since this section's purpose is "이 결과는 믿지 말라"
        for the whole run.
        """

        if analysis is None:
            return ReliabilitySpotlight(flagged_examples=())
        flagged = [
            FlaggedItem(
                anchor_key_repr=(
                    f"{row.anchor_key.sample_id}::{row.anchor_key.region_key}::"
                    f"{row.anchor_key.invert_mask}"
                ),
                reason_summary=_reason_summary(row),
                reliability_reasons=tuple(row.reliability_reasons),
            )
            for row in analysis.reliability_rows
            if row.reliability_grade is ReliabilityGrade.UNRELIABLE
        ]
        return ReliabilitySpotlight(flagged_examples=tuple(flagged))

    # --- provenance / meta ---------------------------------------------------

    def _build_provenance(self, analysis: _AnalysisContext | None) -> ProvenanceInfo:
        analysis_dir_str = None
        analysis_manifest_hash = None
        thresholds: dict[str, float] = {}
        if analysis is not None and self._analysis_dir is not None:
            analysis_dir_str = str(self._analysis_dir.resolve())
            analysis_manifest_hash = sha256_file(self._analysis_dir / "analysis_manifest.json")
            thresholds = dict(analysis.manifest.thresholds)
        return ProvenanceInfo(
            dump_path=str(self._dump_dir.resolve()),
            metrics_dir=str(self._metrics_dir.resolve()),
            analysis_dir=analysis_dir_str,
            metrics_manifest_hash=sha256_file(self._metrics_dir / "metrics_manifest.json"),
            analysis_manifest_hash=analysis_manifest_hash,
            thresholds=thresholds,
        )

    def _build_meta(
        self,
        run_manifest: Any,
        metrics_manifest: MetricsManifest,
        analysis: _AnalysisContext | None,
    ) -> ReportMeta:
        schema_versions = ReportSchemaVersions(
            dump=run_manifest.schema_version,
            metrics=metrics_manifest.metrics_schema_version,
            analysis=analysis.manifest.analysis_schema_version if analysis is not None else None,
            report=REPORT_SCHEMA_VERSION,
        )
        task_kind = (
            TaskKind.DETECTION if isinstance(self._adapter, DetectionAdapter) else TaskKind.CLASSIFICATION
        )
        return ReportMeta(
            run_id=self._dump_dir.resolve().name,
            generated_at=datetime.now(timezone.utc).isoformat(),
            tool_version=ssat.__version__,
            schema_versions=schema_versions,
            task_kind=task_kind,
        )


# --- module-level helpers ---------------------------------------------------


def _dataset_name(resolved_config: Any) -> str:
    """Derive a display dataset name absent from every upstream schema (§1 부수 확인).

    Typed as ``Any`` rather than ``ssat.core.config.schema.ResolvedConfig``:
    importing that module would cross the §3.3 dependency rule this file is
    already careful about elsewhere (``ssat.core.dump``/``ssat.analysis.
    reader`` are named explicitly, but the same "only through the metrics
    engine's façade" spirit applies to the config types ``RunManifest``
    carries) — this reads it via plain attribute access instead.
    """

    source_provenance = resolved_config.source_provenance
    if source_provenance is not None:
        return source_provenance.manifest.parent.name
    config_source = resolved_config.config_source
    if config_source is not None:
        return config_source.stem
    return "unknown"


def _failure_rate(exclusion_summary: Mapping[str, object]) -> float | None:
    """Compute the perturbed-item failure rate from a metrics manifest's exclusion summary."""

    total = exclusion_summary.get("total_perturbed_items")
    excluded = exclusion_summary.get("items_excluded_perturbed_failed")
    if not total:
        return None
    return excluded / total  # type: ignore[operator]


def _worst_grade(grades: Sequence[ReportGrade]) -> ReportGrade | None:
    """Reduce a group of anchors' grades to the single worst one (§1 격차#3).

    ``UNRELIABLE > LOW > MODERATE > HIGH`` — the grade design treats as most
    concerning wins, mirroring ``ssat.analysis`` A6's own severity framing.
    """

    if not grades:
        return None
    return min(grades, key=_GRADE_SEVERITY_ORDER.index)


def _group_report_grades(
    rows: Sequence[ReliabilityRow],
    primary_metric: str,
    *,
    key: Callable[[ReliabilityRow], _GradeKeyT],
) -> dict[_GradeKeyT, list[ReportGrade]]:
    """Group reliability rows' grades by an arbitrary key, filtered to one metric.

    Shared by every per-region/per-sample/per-(sample, region) worst-case
    grouping this module performs — the only thing that differs between
    them is ``key``.
    """

    grouped: dict[_GradeKeyT, list[ReportGrade]] = defaultdict(list)
    for row in rows:
        if row.metric_name != primary_metric:
            continue
        grouped[key(row)].append(ReportGrade(row.reliability_grade.value))
    return grouped


def _reason_summary(row: ReliabilityRow) -> str:
    """One-line human-readable summary for a FlaggedItem, from existing reasons only."""

    if row.reliability_reasons:
        return "; ".join(row.reliability_reasons)
    return f"{row.reliability_grade.value} ({row.metric_name})"
