"""R0 ReportDataAssembler: join MetricsStore + AnalysisStore + run_manifest into ReportModel.

This is the reporting layer's bottleneck module — every join/aggregation
policy gap identified below gets fixed here, in code, once:

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
  full_sample_rankings`` carries every sample, so the full population
  remains reachable via CSV.
- Gap#6: every reduction performed here (means, percentiles, distribution
  counts) is a straightforward, uncontested arithmetic summary of numbers
  the metrics/analysis engines already computed — never a new statistic.
  This module assembles; it does not compute.

**Spatial concentration.** The report had no dataset-level answer to "does
this model repeatedly depend on one fixed location, or is sensitivity
spread across many?" — only per-sample/per-region worst-case grades.
``RegionRow.top_region_share``/``high_rate`` and the new ``ReportModel.
spatial_concentration`` section (:func:`_dataset_top_region_by_sample`,
:func:`_build_spatial_concentration`) close that gap the same way Gap#3
closed the worst-case-rollup gap: an argmax-per-sample reduction of
``SpatialProfile.degradation`` (already computed), then a ``Counter``/
normalized-entropy reduction of that histogram — arithmetic summary, not new
model inference, staying inside Gap#6's boundary.

``analysis_dir=None`` is a first-class input, not an error path: a run
that never had ``ssat analyze`` executed against it still assembles a
complete ``ReportModel``, with every analysis-derived field explicitly
marked unavailable (``None``/empty, plus a ``note`` on the scorecard's
control-comparison card) rather than silently omitted.

**Two additions closing gaps in already-assembled data.** Implementing R4
surfaced two fields with no alternative data source, both threaded through
here rather than routed around:

- ``AssembledReport.rank_correlation_rows`` — R2's
  ``render_fill_strategy_correlation`` needs raw ``rank_correlation.parquet``
  rows (``RankCorrelationRow``), but :meth:`ReportDataAssembler.
  _load_analysis_context` was discarding them (bound to ``_rank_correlation_
  rows``) even though ``adapter.applicable_charts()`` already lists
  ``"fill_strategy_correlation_heatmap"`` as a real, renderable chart when
  fill-strategy stability was run. They are now threaded through
  ``AssembledReport`` at the same "extra data alongside ``model``, not
  serialized into it" level as ``full_sample_rankings`` (Gap#5) — R2 needs
  the *unfiltered* raw rows, not a ``ReportModel``-carried summary.
  ``RankCorrelationRow`` has no ``metric_name`` field (it is dataset-wide),
  so unlike every other analysis collection here these rows are never
  filtered by ``primary_metric``.
- ``ProvenanceInfo.run_manifest_hash`` — R4's ``report_manifest.json``
  requires ``source_manifest_hashes: {run, metrics, analysis}``, but
  :meth:`_build_provenance` only ever populated the latter two. Computed
  the same way ``metrics_manifest_hash`` already was, via ``sha256_file``
  on ``DumpHandle.manifest_path``.

**Semantic region profiling.** ``ReportModel.semantic_summary``/
``class_semantic_matrix``/``semantic_concentration`` add a
``semantic_group`` axis alongside ``region_key`` — a user-declared
grouping of concrete region families (``ResolvedRegionConfig.
semantic_group``) into one meaning-bearing unit (e.g.
``"left_arm"``/``"right_arm"`` -> ``"upper_limb"``). Every value here is
again an arithmetic reduction of already-computed numbers (Gap#6):
``SpatialProfile.degradation`` averaged within a sample across a group's
concrete regions, then across samples/semantic_groups exactly the way
:func:`_dataset_top_region_by_sample`/:func:`_build_spatial_concentration`
already reduce the ``region_key`` axis (see :func:`_dataset_top_semantic_
group_by_sample`/:func:`_build_semantic_concentration`).

Two data-availability decisions confirmed with the user, since MetricsStore
(N3, frozen schema) has no counterpart to plug in directly:

- ``ClassSemanticRow.flip_rate`` is always ``None``. A per-``(gt_label,
  semantic_group)`` cell would need a flip signal at the ``(sample,
  region)`` grain, but N3 only carries flip at the whole-sample grain
  (``SampleMetrics.flip_rate``, blind to region) or the whole-region grain
  (``RegionMetrics.flip_rate``, blind to sample/label) — never both axes
  at once. Approximating with either would silently misrepresent a
  different quantity as this cell's flip rate, so it is left unavailable
  rather than guessed (:func:`_build_class_semantic_matrix`).
- ``ProvenanceInfo.class_semantic_excluded_no_gt_label`` is a dedicated
  field, not a ``thresholds`` entry — that mapping is documented as
  "thresholds used to derive grades", and an exclusion count is not a
  threshold.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
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
    RankCorrelationRow,
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
    ClassSemanticRow,
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
    SemanticConcentration,
    SemanticGroupRow,
    SpatialConcentration,
    TaskKind,
    TopRegionEntry,
    VulnerabilityDistribution,
    VulnerabilitySummaryStats,
)
from ssat.utils.io import sha256_file

_GradeKeyT = TypeVar("_GradeKeyT")

# Worst-first: the grade a group of anchors is reduced to is whichever of
# these appears first among them (worst-case policy).
_GRADE_SEVERITY_ORDER = (
    ReportGrade.UNRELIABLE,
    ReportGrade.LOW,
    ReportGrade.MODERATE,
    ReportGrade.HIGH,
)


@dataclass(frozen=True, slots=True)
class AssembledReport:
    """R0's full output: the top-K/bottom-K ``ReportModel`` plus every sample.

    Attributes:
        model: The JSON-serializable report R1 exports and R4 renders.
        full_sample_rankings: Every sample the run produced, sorted by
            ``vulnerability_score`` descending (unscored samples last, see
            :meth:`ReportDataAssembler._build_full_rankings`) — consumed by
            R1's ``sample_rankings.csv`` (full population) and R2's
            histogram (a top-K-only slice cannot represent a distribution).
            Not serialized as part of ``model``.
        rank_correlation_rows: Every ``rank_correlation.parquet`` row from
            the source AnalysisStore, unfiltered; empty when
            ``analysis_dir=None``. For R2's ``render_fill_strategy_
            correlation`` — the same "extra data alongside ``model``"
            level as ``full_sample_rankings``, since these raw op-pair
            rows have no ``ReportModel`` field of their own (only a
            rendered SVG's ref does, ``ReportModel.
            fill_strategy_correlation_asset_ref``).
        sample_semantic_degradation: Every ``(sample_id, semantic_group)``
            pair with a determinable mean degradation, from
            :func:`_sample_semantic_group_degradation` — the same "extra
            data alongside ``model``" level as ``full_sample_rankings``.
            For a future labels module that needs the per-sample values
            ``ReportModel.semantic_summary``/``class_semantic_matrix``
            only ever expose pre-aggregated.
    """

    model: ReportModel
    full_sample_rankings: tuple[SampleCard, ...]
    rank_correlation_rows: tuple[RankCorrelationRow, ...] = ()
    sample_semantic_degradation: Mapping[tuple[str, str], float] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True, slots=True)
class _AnalysisContext:
    """The subset of one AnalysisStore load that R0 actually consumes.

    ``seed_rows``/``strategy_rows``/``strategy_profile_rows``/
    ``interval_rows``/``coverage_report`` are loaded (``load_analysis``
    returns all nine outputs together) but not threaded any further — R0's
    v1 scope only surfaces control-comparison, reliability, and
    rank-correlation data (``ReportModel`` schema has no field for the
    rest yet).
    """

    control_rows: tuple[ControlComparisonRow, ...]
    reliability_rows: tuple[ReliabilityRow, ...]
    rank_correlation_rows: tuple[RankCorrelationRow, ...]
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
                ``ssat.metrics.dump_reader.DumpHandle``.
            metrics_dir: MetricsStore directory for this dump.
            analysis_dir: AnalysisStore directory for this dump+metrics
                pair, or ``None`` when no ``ssat analyze`` run exists —
                every analysis-derived field is then assembled as
                explicitly unavailable rather than raising.
            adapter: Task-specific card/field/chart translator (R5).
            top_k: Number of most-vulnerable samples to render as
                ``SampleCard``s.
            bottom_k: Number of most-robust samples to render as
                ``SampleCard``s.
            region_top_k: Number of a sample's most-affected regions to
                surface in its ``SampleCard.top_regions``.

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
                Application layer's job, mirroring how ``AuditApplication.
                analyze`` already handles this boundary).
        """

        if not primary_metric:
            raise ValueError("primary_metric must not be empty")

        handle = DumpHandle(self._dump_dir)
        run_manifest = handle.manifest
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
        top_region_by_sample = _dataset_top_region_by_sample(spatial_rows)
        region_keys = {row.region_key for row in region_rows}
        scorecard = tuple(self._build_scorecard(sample_rows, analysis, primary_metric))

        # --- semantic_group axis ---
        semantic_group_by_region_id = _semantic_group_by_region_id(run_manifest.resolved_config)
        # Scoped to region_keys (already control-excluded, same population
        # _build_spatial_concentration's entropy normalizer uses) rather than
        # every resolved_config.regions family -- a control-comparison-only
        # family (RANDOM_AREA_MATCH, never in region_metrics.parquet) would
        # otherwise inflate this count and defeat the n_semantic_groups <= 1
        # gate for an otherwise-ungrouped run.
        n_semantic_groups = len(
            {
                semantic_group_by_region_id.get(region_id, region_id)
                for region_id in {_region_id_from_region_key(key) for key in region_keys}
            }
        )
        sample_semantic_degradation = _sample_semantic_group_degradation(
            spatial_rows, semantic_group_by_region_id
        )
        top_semantic_group_by_sample = _dataset_top_semantic_group_by_sample(
            sample_semantic_degradation
        )
        is_binary_primary_metric = _is_binary_primary_metric(scorecard)
        grades_by_semantic_group = (
            _group_report_grades(
                analysis.reliability_rows,
                primary_metric,
                key=lambda row: semantic_group_by_region_id.get(
                    _region_id_from_region_key(row.anchor_key.region_key),
                    _region_id_from_region_key(row.anchor_key.region_key),
                ),
            )
            if analysis is not None
            else {}
        )
        semantic_summary = _build_semantic_summary(
            sample_semantic_degradation,
            semantic_group_by_region_id,
            region_rows,
            grades_by_semantic_group,
            is_binary_primary_metric,
        )
        gt_label_by_sample = {card.sample_id: card.gt_label for card in full_rankings}
        class_semantic_matrix, excluded_no_gt_label = _build_class_semantic_matrix(
            sample_semantic_degradation, gt_label_by_sample
        )

        model = ReportModel(
            meta=self._build_meta(run_manifest, metrics_manifest, analysis),
            run_summary=self._build_run_summary(run_manifest, metrics_manifest, sample_rows),
            scorecard=scorecard,
            vulnerability_distribution=self._build_vulnerability_distribution(full_rankings),
            sample_rankings=self._build_sample_rankings(full_rankings),
            region_summary=self._build_region_summary(
                region_rows, analysis, primary_metric, top_region_by_sample
            ),
            spatial_concentration=_build_spatial_concentration(top_region_by_sample, region_keys),
            semantic_summary=semantic_summary,
            class_semantic_matrix=class_semantic_matrix,
            semantic_concentration=_build_semantic_concentration(
                top_semantic_group_by_sample, n_semantic_groups
            ),
            fill_strategy_correlation_asset_ref=None,
            reliability_spotlight=self._build_reliability_spotlight(analysis),
            provenance=self._build_provenance(analysis, handle, excluded_no_gt_label),
        )
        rank_correlation_rows = analysis.rank_correlation_rows if analysis is not None else ()
        return AssembledReport(
            model=model,
            full_sample_rankings=full_rankings,
            rank_correlation_rows=rank_correlation_rows,
            sample_semantic_degradation=MappingProxyType(dict(sample_semantic_degradation)),
        )

    # --- source loading ----------------------------------------------------

    def _load_analysis_context(self) -> _AnalysisContext | None:
        if self._analysis_dir is None:
            return None
        (
            control_rows,
            _seed_rows,
            _strategy_rows,
            rank_correlation_rows,
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
            rank_correlation_rows=tuple(rank_correlation_rows),
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
        """Build the one control-comparison card R0 itself owns.

        Unlike the adapter's cards (MetricsStore-derived, task-specific),
        this reads AnalysisStore — control comparison is a task-agnostic
        concept R5 has no reason to know about. Its value is the mean
        ``z_vs_control`` across every matched anchor: the same standardized
        quantity A6 thresholds to derive reliability grades, so the number
        on this card and the grades elsewhere in the report share one
        justification.
        """

        if analysis is None:
            return MetricCard(
                key="control_comparison",
                label="Mean Z vs Control",
                value=None,
                unit="",
                higher_is_better=False,
                note="Analysis not run: control comparison was not executed.",
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
                note="N/A: no matched controls.",
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
        """Sort every sample by vulnerability_score descending.

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
        top_region_by_sample: Mapping[str, str],
    ) -> RegionSummary:
        """Join ``region_metrics.parquet`` (source of which regions exist) with grades.

        Iterating ``region_rows`` (already primary-metric-filtered
        ``region_metrics.parquet``) rather than ``reliability.parquet`` is
        what keeps control-only regions — already excluded from N3
        aggregation — out of ``region_summary.rows``.

        ``top_region_by_sample`` (from :func:`_dataset_top_region_by_sample`,
        already computed once in :meth:`assemble` and shared with
        :func:`_build_spatial_concentration`) drives each row's
        ``top_region_share``, which lets a region table sort/color by how
        often a region is *the* answer across the dataset, independent of
        the worst-case ``reliability_grade``.
        """

        grades_by_region = (
            _group_report_grades(
                analysis.reliability_rows, primary_metric, key=lambda row: row.anchor_key.region_key
            )
            if analysis is not None
            else {}
        )
        top_region_counts = Counter(top_region_by_sample.values())
        n_scored_samples = len(top_region_by_sample)

        rows = []
        for row in sorted(region_rows, key=lambda row: row.region_key):
            grades = grades_by_region.get(row.region_key, ())
            distribution = (
                dict(Counter(grade.value for grade in grades)) if analysis is not None else {}
            )
            n_graded = sum(distribution.values())
            rows.append(
                RegionRow(
                    region_key=row.region_key,
                    region_id=_region_id_from_region_key(row.region_key),
                    region_kind=row.region_kind.value,
                    intended_area_px=row.intended_area_px,
                    effective_area_px=row.effective_area_px,
                    mean_degradation=row.metric_mean,
                    flip_rate=row.flip_rate,
                    n_valid=row.n_valid,
                    reliability_grade=_worst_grade(grades) if analysis is not None else None,
                    reliability_distribution=distribution,
                    top_region_share=(
                        top_region_counts.get(row.region_key, 0) / n_scored_samples
                        if n_scored_samples
                        else None
                    ),
                    high_rate=(
                        distribution.get(ReportGrade.HIGH.value, 0) / n_graded
                        if n_graded
                        else None
                    ),
                )
            )
        dataset_distribution = (
            dict(analysis.manifest.grade_distribution) if analysis is not None else {}
        )
        return RegionSummary(
            rows=tuple(rows), reliability_distribution=dataset_distribution, chart_asset_ref=None
        )

    # --- reliability spotlight -----------------------------------------------

    def _build_reliability_spotlight(
        self, analysis: _AnalysisContext | None
    ) -> ReliabilitySpotlight:
        """Surface every UNRELIABLE-grade anchor across every metric.

        Unlike the per-metric-filtered sections above, this is intentionally
        *not* filtered to ``primary_metric`` — it surfaces every unreliable
        finding, regardless of which metric flagged it, since this
        section's purpose is "don't trust this result" for the whole run.
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

    def _build_provenance(
        self,
        analysis: _AnalysisContext | None,
        handle: DumpHandle,
        class_semantic_excluded_no_gt_label: int,
    ) -> ProvenanceInfo:
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
            run_manifest_hash=sha256_file(handle.manifest_path),
            metrics_manifest_hash=sha256_file(self._metrics_dir / "metrics_manifest.json"),
            analysis_manifest_hash=analysis_manifest_hash,
            class_semantic_excluded_no_gt_label=class_semantic_excluded_no_gt_label,
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
    """Derive a display dataset name absent from every upstream schema.

    Typed as ``Any`` rather than ``ssat.core.config.schema.ResolvedConfig``:
    importing that module would cross the dependency rule this file is
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
    """Reduce a group of anchors' grades to the single worst one.

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


def _dataset_top_region_by_sample(spatial_rows: Sequence[SpatialProfile]) -> dict[str, str]:
    """Reduce every sample to its single most-degraded region, dataset-wide.

    Generalizes the same sort-and-take-first idiom
    :meth:`ReportDataAssembler._top_regions_for_sample` already uses for the
    top-K/bottom-K gallery, but over *every* sample in ``spatial_rows``
    (already primary-metric-filtered by :meth:`ReportDataAssembler.assemble`
    before this is called) rather than only ``highlighted_ids`` — the
    dataset-wide population :func:`_build_spatial_concentration` and
    ``RegionRow.top_region_share`` both need. A sample contributes no
    entry when every one of its regions has ``degradation is None`` (no
    valid item), matching the "unavailable, not zero" convention every other
    reduction in this module follows. Ties are broken by ``region_key``
    ascending for a deterministic result.

    Returns:
        A mapping from ``sample_id`` to its top ``region_key``, covering
        only samples with at least one scored region.
    """

    by_sample: dict[str, list[SpatialProfile]] = defaultdict(list)
    for row in spatial_rows:
        by_sample[row.sample_id].append(row)

    top_region: dict[str, str] = {}
    for sample_id, rows in by_sample.items():
        scored = [row for row in rows if row.degradation is not None]
        if not scored:
            continue
        best = min(scored, key=lambda row: (-row.degradation, row.region_key))  # type: ignore[operator]
        top_region[sample_id] = best.region_key
    return top_region


def _build_spatial_concentration(
    top_region_by_sample: Mapping[str, str], region_keys: Sequence[str]
) -> SpatialConcentration:
    """Reduce the per-sample top-region histogram to dominant-share/entropy scalars.

    Both quantities are plain arithmetic over ``top_region_by_sample``
    (itself already a reduction of already-computed ``SpatialProfile.
    degradation`` values, :func:`_dataset_top_region_by_sample`) — no new
    model inference, matching R0's "assembles, does not compute new
    statistics" boundary (module docstring Gap#6). ``region_keys`` is the
    *possible* location count (every region_key this run has a
    ``region_metrics.parquet`` row for), the entropy normalizer: "how spread
    out are top regions, relative to how spread out they could be."

    Args:
        top_region_by_sample: Every sample with a determinable top region,
            from :func:`_dataset_top_region_by_sample`.
        region_keys: Every region_key present in this run's region_summary.

    Returns:
        ``dominant_region_key``/``dominant_region_share``/``spatial_entropy``
        all ``None`` when ``top_region_by_sample`` is empty;
        ``spatial_entropy`` additionally ``None`` when fewer than two
        distinct ``region_keys`` exist (entropy is undefined, not zero, with
        nothing to spread across).
    """

    n_scored_samples = len(top_region_by_sample)
    if n_scored_samples == 0:
        return SpatialConcentration(
            dominant_region_key=None,
            dominant_region_share=None,
            spatial_entropy=None,
            n_scored_samples=0,
        )

    counts = Counter(top_region_by_sample.values())
    dominant_region_key, dominant_count = counts.most_common(1)[0]
    dominant_region_share = dominant_count / n_scored_samples

    spatial_entropy = None
    if len(region_keys) > 1:
        probabilities = [count / n_scored_samples for count in counts.values()]
        raw_entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
        # Clamp away floating-point drift at the extremes (e.g. a perfectly
        # uniform distribution computing to 1.0000000000000002) so this
        # always satisfies SpatialConcentration's [0, 1] validation.
        spatial_entropy = min(1.0, max(0.0, raw_entropy / math.log(len(region_keys))))

    return SpatialConcentration(
        dominant_region_key=dominant_region_key,
        dominant_region_share=dominant_region_share,
        spatial_entropy=spatial_entropy,
        n_scored_samples=n_scored_samples,
    )


# --- semantic_group axis ---------------------------------------------------


def _region_id_from_region_key(region_key: str) -> str:
    """Recover a concrete region's family identity from its ``region_key``.

    Always safe: ``RegionId`` (``ssat.core.config.schema``) forbids ``"::"``,
    so this split never ambiguously cuts through the family name itself
    (verified by code inspection rather than assumed).
    """

    return region_key.split("::", 1)[0]


def _semantic_group_by_region_id(resolved_config: Any) -> dict[str, str]:
    """Build the ``region_id -> semantic_group`` map from the run's resolved config.

    A family with no declared ``semantic_group`` falls back to its own
    ``region_id`` — the common case for a plain grid/explicit run, where
    every family is its own, singleton semantic group.

    Typed ``resolved_config: Any`` for the same reason :func:`_dataset_name`
    is: this reads ``ResolvedConfig.regions[*].region_id``/
    ``semantic_group`` via plain attribute access rather than importing
    ``ssat.core.config.schema``.
    """

    return {
        family.region_id: (family.semantic_group or family.region_id)
        for family in resolved_config.regions
    }


def _is_binary_primary_metric(scorecard: Sequence[MetricCard]) -> bool:
    """Reuse the adapter's existing binary/continuous determination.

    ``ClassificationAdapter.summarize_performance`` always emits a
    ``"flip_rate"`` card, but only gives it a real ``value`` when the
    primary metric is binary (``value=None`` plus a "continuous metric"
    note otherwise, ``ssat.report.adapters._flip_rate_card``) — that
    non-``None`` check is the exact determination this function reuses
    rather than re-deriving "is this metric binary" from scratch.
    """

    return any(card.key == "flip_rate" and card.value is not None for card in scorecard)


def _sample_semantic_group_degradation(
    spatial_rows: Sequence[SpatialProfile], semantic_group_by_region_id: Mapping[str, str]
) -> dict[tuple[str, str], float]:
    """Reduce each sample's concrete-region degradations to one value per semantic_group.

    When a semantic_group folds together several concrete region families
    (e.g. ``"left_arm"``/``"right_arm"`` -> ``"upper_limb"``), a sample
    contributes one degradation value per family; this averages those
    within the sample — a worst-case/max rollup is reserved for grade
    badges only. Regions with ``degradation is None`` are skipped,
    matching every other "unavailable, not zero" reduction here.

    Returns:
        A mapping from ``(sample_id, semantic_group)`` to the sample's mean
        degradation across that group's concrete regions, covering only
        pairs with at least one valid value.
    """

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in spatial_rows:
        if row.degradation is None:
            continue
        region_id = _region_id_from_region_key(row.region_key)
        semantic_group = semantic_group_by_region_id.get(region_id, region_id)
        grouped[(row.sample_id, semantic_group)].append(row.degradation)
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def _dataset_top_semantic_group_by_sample(
    sample_semantic_degradation: Mapping[tuple[str, str], float]
) -> dict[str, str]:
    """Reduce every sample to its single most-degraded semantic_group, dataset-wide.

    The semantic-axis counterpart of :func:`_dataset_top_region_by_sample`,
    over :func:`_sample_semantic_group_degradation`'s per-``(sample,
    semantic_group)`` averages instead of raw per-region values. Ties are
    broken by ``semantic_group`` ascending for a deterministic result.

    Returns:
        A mapping from ``sample_id`` to its top ``semantic_group``, covering
        only samples with at least one scored semantic_group.
    """

    by_sample: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (sample_id, semantic_group), value in sample_semantic_degradation.items():
        by_sample[sample_id].append((semantic_group, value))

    top_semantic_group: dict[str, str] = {}
    for sample_id, entries in by_sample.items():
        best_group, _best_value = min(entries, key=lambda entry: (-entry[1], entry[0]))
        top_semantic_group[sample_id] = best_group
    return top_semantic_group


def _build_semantic_concentration(
    top_semantic_group_by_sample: Mapping[str, str], n_semantic_groups: int
) -> SemanticConcentration:
    """Reduce the per-sample top-semantic_group histogram to dominant-share/entropy scalars.

    The semantic-axis counterpart of :func:`_build_spatial_concentration`.
    Unlike that function, the ``n_semantic_groups <= 1`` gate is checked
    first and unconditionally forces the graceful-degradation marker
    (enforced again at the type level by ``SemanticConcentration.
    __post_init__``) — the common case for a run that never declared
    ``regions[].semantic_group``, where every family collapses to its own
    singleton group.

    Args:
        top_semantic_group_by_sample: Every sample with a determinable top
            semantic_group, from :func:`_dataset_top_semantic_group_by_sample`.
        n_semantic_groups: Distinct semantic_group count among the region
            families this run actually reports on (the gate; control-
            comparison-only families excluded, see :meth:`ReportDataAssembler.
            assemble`), not merely how many appear in ``top_semantic_group_
            by_sample``.

    Returns:
        The graceful-degradation marker when ``n_semantic_groups <= 1`` or
        no sample has a determinable top semantic_group; the real
        dominant-share/normalized-entropy reduction otherwise.
    """

    if n_semantic_groups <= 1:
        return SemanticConcentration(
            dominant_semantic_group=None,
            dominant_semantic_group_share=None,
            semantic_group_entropy=None,
            n_semantic_groups=n_semantic_groups,
            n_scored_samples=0,
        )

    n_scored_samples = len(top_semantic_group_by_sample)
    if n_scored_samples == 0:
        return SemanticConcentration(
            dominant_semantic_group=None,
            dominant_semantic_group_share=None,
            semantic_group_entropy=None,
            n_semantic_groups=n_semantic_groups,
            n_scored_samples=0,
        )

    counts = Counter(top_semantic_group_by_sample.values())
    dominant_semantic_group, dominant_count = counts.most_common(1)[0]
    dominant_semantic_group_share = dominant_count / n_scored_samples

    probabilities = [count / n_scored_samples for count in counts.values()]
    raw_entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    # Clamp away floating-point drift at the extremes, same as
    # _build_spatial_concentration. n_semantic_groups >= 2 here (the <= 1
    # gate above already returned), so this normalizer is always defined.
    semantic_group_entropy = min(1.0, max(0.0, raw_entropy / math.log(n_semantic_groups)))

    return SemanticConcentration(
        dominant_semantic_group=dominant_semantic_group,
        dominant_semantic_group_share=dominant_semantic_group_share,
        semantic_group_entropy=semantic_group_entropy,
        n_semantic_groups=n_semantic_groups,
        n_scored_samples=n_scored_samples,
    )


def _build_semantic_summary(
    sample_semantic_degradation: Mapping[tuple[str, str], float],
    semantic_group_by_region_id: Mapping[str, str],
    region_rows: Sequence[RegionMetrics],
    grades_by_semantic_group: Mapping[str, Sequence[ReportGrade]],
    is_binary_primary_metric: bool,
) -> tuple[SemanticGroupRow, ...]:
    """Build one ``SemanticGroupRow`` per semantic_group with at least one scored sample.

    Mirrors :meth:`ReportDataAssembler._build_region_summary`'s join
    pattern, one axis coarser: ``mean_degradation``/``n_samples`` reduce
    :func:`_sample_semantic_group_degradation`; ``high_rate`` regroups the
    same per-anchor reliability grades ``RegionRow.high_rate`` already
    uses, just keyed by semantic_group instead of region_key (Gap#3's
    worst-case-adjacent, no-new-statistic pattern); ``flip_rate`` averages
    ``RegionMetrics.flip_rate`` across the group's concrete regions,
    populated only when ``is_binary_primary_metric`` — never from
    sample-grain data, since no ``(sample, semantic_group)``-grain flip
    signal exists (module docstring).

    Args:
        sample_semantic_degradation: From :func:`_sample_semantic_group_
            degradation`.
        semantic_group_by_region_id: From :func:`_semantic_group_by_region_id`.
        region_rows: This run's primary-metric-filtered ``region_metrics.
            parquet`` rows, the ``flip_rate`` source.
        grades_by_semantic_group: Reliability grades already regrouped by
            semantic_group (empty when no analysis run exists).
        is_binary_primary_metric: From :func:`_is_binary_primary_metric`.

    Returns:
        Rows sorted by semantic_group for a deterministic result.
    """

    region_ids_by_group: dict[str, list[str]] = defaultdict(list)
    for region_id, semantic_group in semantic_group_by_region_id.items():
        region_ids_by_group[semantic_group].append(region_id)

    degradation_by_group: dict[str, list[float]] = defaultdict(list)
    samples_by_group: dict[str, set[str]] = defaultdict(set)
    for (sample_id, semantic_group), value in sample_semantic_degradation.items():
        degradation_by_group[semantic_group].append(value)
        samples_by_group[semantic_group].add(sample_id)

    flip_values_by_group: dict[str, list[float]] = defaultdict(list)
    if is_binary_primary_metric:
        for row in region_rows:
            if row.flip_rate is None:
                continue
            region_id = _region_id_from_region_key(row.region_key)
            semantic_group = semantic_group_by_region_id.get(region_id, region_id)
            flip_values_by_group[semantic_group].append(row.flip_rate)

    rows = []
    for semantic_group in sorted(degradation_by_group):
        values = degradation_by_group[semantic_group]
        grades = grades_by_semantic_group.get(semantic_group, ())
        n_graded = len(grades)
        high_count = sum(1 for grade in grades if grade is ReportGrade.HIGH)
        flip_values = flip_values_by_group.get(semantic_group, [])
        rows.append(
            SemanticGroupRow(
                semantic_group=semantic_group,
                region_ids=tuple(
                    sorted(region_ids_by_group.get(semantic_group, (semantic_group,)))
                ),
                n_samples=len(samples_by_group[semantic_group]),
                mean_degradation=sum(values) / len(values) if values else None,
                high_rate=(high_count / n_graded) if n_graded else None,
                flip_rate=(sum(flip_values) / len(flip_values)) if flip_values else None,
            )
        )
    return tuple(rows)


def _build_class_semantic_matrix(
    sample_semantic_degradation: Mapping[tuple[str, str], float],
    gt_label_by_sample: Mapping[str, int | None],
) -> tuple[tuple[ClassSemanticRow, ...], int]:
    """Build the ``(gt_label, semantic_group)`` cross-tabulation.

    Joins :func:`_sample_semantic_group_degradation`'s per-sample values
    with each sample's ``gt_label`` (from ``full_sample_rankings``, already
    computed) and regroups by the coarser ``(gt_label, semantic_group)``
    key — a table answering "for this action class, which body part
    matters most?"

    Samples with ``gt_label is None`` are excluded (their body-part
    contribution cannot be attributed to a class) rather than folded into a
    misleading "unknown class" row; ``ClassSemanticRow.flip_rate`` is always
    ``None`` (module docstring — no ``(sample, semantic_group)``-grain flip
    signal exists in N3).

    Returns:
        ``(rows, n_excluded)`` — rows sorted by ``(gt_label, semantic_
        group)``, and the count of *distinct* samples excluded for having no
        ``gt_label`` (not occurrences, since one such sample can contribute
        to several semantic_groups).
    """

    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    samples_by_cell: dict[tuple[int, str], set[str]] = defaultdict(set)
    excluded_sample_ids: set[str] = set()
    for (sample_id, semantic_group), value in sample_semantic_degradation.items():
        gt_label = gt_label_by_sample.get(sample_id)
        if gt_label is None:
            excluded_sample_ids.add(sample_id)
            continue
        grouped[(gt_label, semantic_group)].append(value)
        samples_by_cell[(gt_label, semantic_group)].add(sample_id)

    rows = tuple(
        ClassSemanticRow(
            gt_label=gt_label,
            semantic_group=semantic_group,
            n_samples=len(samples_by_cell[(gt_label, semantic_group)]),
            mean_degradation=sum(values) / len(values) if values else None,
            flip_rate=None,
        )
        for (gt_label, semantic_group), values in sorted(grouped.items())
    )
    return rows, len(excluded_sample_ids)


def _reason_summary(row: ReliabilityRow) -> str:
    """One-line human-readable summary for a FlaggedItem, from existing reasons only."""

    if row.reliability_reasons:
        return "; ".join(row.reliability_reasons)
    return f"{row.reliability_grade.value} ({row.metric_name})"
