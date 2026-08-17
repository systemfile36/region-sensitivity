"""C1 integration tests for R0 ReportDataAssembler (IMPLE_PLAN_REPORTING_v1.md §5 단계2).

Builds one dump+metrics pair through the real pipeline (``synthetic_dump_
builder``, core never executed) and one AnalysisStore assembled by hand from
directly-constructed A2/A3(c)/A5 rows fed through the real A6
``compute_reliability`` — mirroring ``tests/unit/test_analysis_reliability.
py``'s own precedent for getting an exact, deterministic
``ReliabilityGrade`` without depending on the bootstrap/threshold numerics
the full ``ssat analyze`` pipeline would otherwise require engineering
around. This keeps every assertion below tied to hand-computed expected
values instead of "whatever the pipeline happened to produce".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from synthetic_dump_builder import (
    build_resolved_config,
    clean_record,
    compute_and_save_metrics,
    perturbed_record,
    write_dump,
)

from ssat.analysis.reliability import compute_reliability
from ssat.analysis.store import save_analysis
from ssat.analysis.types import (
    AnchorKey,
    AvailableAnalyses,
    ConditionKey,
    ControlComparisonRow,
    CoverageReport,
    FlagValue,
    IntervalRow,
    StrategyStabilityRow,
)
from ssat.core.config.schema import ResolvedRegionConfig
from ssat.core.types import PerturbationOp, RegionKind
from ssat.metrics.builtin_metrics.continuous import GtLogitDrop
from ssat.metrics.registry import MetricRegistry
from ssat.report import ClassificationAdapter, ReportDataAssembler, ReportGrade

_METRIC_NAME = "gt_logit_drop"
_MAIN_REGION_KEY = "grid::grid/r0/c0"
_CONTROL_REGION_KEY = "control::control/0"
_CLEAN_GT_LOGIT = 10.0

# sample_id -> perturbed gt-logit (degradation = _CLEAN_GT_LOGIT - this value).
_PERTURBED_GT_LOGIT_BY_SAMPLE = {
    "s0": 0.0,  # degradation 10.0 -- most vulnerable, HIGH-grade anchor
    "s1": 3.0,  # degradation 7.0  -- UNRELIABLE-grade anchor
    "s2": 6.0,  # degradation 4.0
    "s3": 8.0,  # degradation 2.0
    "s4": 9.5,  # degradation 0.5  -- most robust
}
_EXPECTED_DEGRADATION = {
    sample_id: _CLEAN_GT_LOGIT - perturbed
    for sample_id, perturbed in _PERTURBED_GT_LOGIT_BY_SAMPLE.items()
}


def _registry() -> MetricRegistry:
    registry = MetricRegistry()
    registry.register(GtLogitDrop())
    return registry


def _build_dump_and_metrics(tmp_path: Path) -> tuple[Path, Path]:
    """Build 5 samples, each with one "grid" (target) item and one "control" item.

    Every sample's target degradation is a distinct, hand-picked value
    (module constants above) so top-K/bottom-K selection and every
    downstream numeric assertion can be checked against a value computed by
    hand rather than re-derived from the pipeline.
    """

    config = build_resolved_config(
        tmp_path,
        regions=(
            ResolvedRegionConfig(region_id="grid", kind=RegionKind.GRID, params={"rows": 1, "cols": 1}),
            ResolvedRegionConfig(region_id="control", kind=RegionKind.RANDOM_AREA_MATCH, params={}),
        ),
    )

    clean_records = [
        clean_record(sample_id, logits=np.array([_CLEAN_GT_LOGIT, 0.0]), gt_label=0)
        for sample_id in _PERTURBED_GT_LOGIT_BY_SAMPLE
    ]
    perturbed_records = []
    index = 0
    for sample_id, perturbed_gt_logit in _PERTURBED_GT_LOGIT_BY_SAMPLE.items():
        perturbed_records.append(
            perturbed_record(
                index,
                sample_id=sample_id,
                region_id="grid",
                region_instance_id="grid/r0/c0",
                logits=np.array([perturbed_gt_logit, 0.0]),
                perturb_op=PerturbationOp.CONSTANT_FILL,
            )
        )
        index += 1
        perturbed_records.append(
            perturbed_record(
                index,
                sample_id=sample_id,
                region_id="control",
                region_instance_id="control/0",
                logits=np.array([5.0, 0.0]),
                region_kind=RegionKind.RANDOM_AREA_MATCH,
                is_control=True,
            )
        )
        index += 1

    dump_root = tmp_path / "dump"
    write_dump(
        dump_root,
        config,
        clean_records=tuple(clean_records),
        perturbed_records=tuple(perturbed_records),
    )
    metrics_dir = tmp_path / "metrics"
    compute_and_save_metrics(
        dump_root, config, metrics_dir, registry=_registry(), primary_metric=_METRIC_NAME
    )
    return dump_root, metrics_dir


def _build_analysis(tmp_path: Path, metrics_dir: Path) -> Path:
    """Hand-assemble an AnalysisStore with exactly one HIGH and one UNRELIABLE anchor.

    Both anchors share ``_MAIN_REGION_KEY`` (§1 격차#3's worst-case policy
    test target). ``compute_reliability`` (the real A6 scorer) is fed
    directly-constructed A2/A3(c)/A5 rows rather than the noisy bootstrap
    pipeline, exactly as ``tests/unit/test_analysis_reliability.py`` already
    does for the same reason: a deterministic grade from hand-picked inputs.
    """

    anchor_high = AnchorKey(sample_id="s0", region_key=_MAIN_REGION_KEY, invert_mask=False)
    anchor_unreliable = AnchorKey(sample_id="s1", region_key=_MAIN_REGION_KEY, invert_mask=False)
    condition = ConditionKey(perturb_op="constant_fill", perturb_params_hash="hash")

    strategy_rows = [
        StrategyStabilityRow(
            anchor_key=anchor_high,
            metric_name=_METRIC_NAME,
            strategy_signs={"constant_fill": 1, "mean_fill": 1},
            strategy_values={"constant_fill": 10.0, "mean_fill": 9.0},
            sign_agreement_ratio=1.0,
            n_strategies=2,
        ),
        StrategyStabilityRow(
            anchor_key=anchor_unreliable,
            metric_name=_METRIC_NAME,
            strategy_signs={"constant_fill": 1, "mean_fill": -1},
            strategy_values={"constant_fill": 7.0, "mean_fill": -2.0},
            sign_agreement_ratio=0.5,
            n_strategies=2,
        ),
    ]
    control_rows = [
        ControlComparisonRow(
            target_anchor_key=anchor_high,
            condition_key=condition,
            metric_name=_METRIC_NAME,
            control_available=FlagValue.TRUE,
            area_matched=FlagValue.TRUE,
            control_mean=1.0,
            control_std=0.5,
            n_controls=3,
            excess=9.0,
            ratio=10.0,
            z_vs_control=3.0,  # exceeds DEFAULT_Z_VS_CONTROL_THRESHOLD (2.0)
        ),
        ControlComparisonRow(
            target_anchor_key=anchor_unreliable,
            condition_key=condition,
            metric_name=_METRIC_NAME,
            control_available=FlagValue.TRUE,
            area_matched=FlagValue.TRUE,
            control_mean=1.0,
            control_std=0.5,
            n_controls=3,
            excess=6.0,
            ratio=7.0,
            z_vs_control=5.0,
        ),
        ControlComparisonRow(
            # A different region_key than _MAIN_REGION_KEY on purpose: this
            # row exists only to prove the control-comparison scorecard card
            # excludes unavailable rows from its mean (test below); giving
            # it its own anchor keeps compute_reliability from adding a
            # third (LOW-grade) row to _MAIN_REGION_KEY's distribution,
            # which would confuse the worst-case-policy test above.
            target_anchor_key=AnchorKey(sample_id="s2", region_key="aux::0", invert_mask=False),
            condition_key=condition,
            metric_name=_METRIC_NAME,
            control_available=FlagValue.UNAVAILABLE,
            area_matched=FlagValue.UNAVAILABLE,
            control_mean=None,
            control_std=None,
            n_controls=0,
            excess=None,
            ratio=None,
            z_vs_control=None,
        ),
    ]
    interval_rows = [
        IntervalRow(
            region_key=_MAIN_REGION_KEY,
            metric=_METRIC_NAME,
            point_estimate=5.0,
            ci_low=1.0,
            ci_high=9.0,
            ci_method="percentile",
            n_bootstrap=100,
            excludes_zero=True,
        )
    ]

    reliability_rows = compute_reliability(control_rows, [], strategy_rows, interval_rows)

    analysis_dir = tmp_path / "analysis"
    save_analysis(
        analysis_dir,
        control_rows=control_rows,
        seed_rows=[],
        strategy_rows=strategy_rows,
        rank_correlation_rows=[],
        strategy_profile_rows=[],
        interval_rows=interval_rows,
        reliability_rows=reliability_rows,
        coverage_report=CoverageReport(
            n_anchors=5, n_conditions_insufficient=3, n_controls_unmatched=0, n_area_mismatch_warnings=0
        ),
        available_analyses=AvailableAnalyses(
            control_comparison=True,
            fill_strategy_stability=True,
            seed_stability=False,
            jitter_stability=False,
        ),
        thresholds={"z_vs_control_threshold": 2.0, "seed_cv_threshold": 0.2, "area_match_tolerance": 0.1},
        n_bootstrap=100,
        random_seed=0,
        source_metrics_manifest_path=metrics_dir / "metrics_manifest.json",
    )
    return analysis_dir


def _assembler(dump_root: Path, metrics_dir: Path, analysis_dir: Path | None) -> ReportDataAssembler:
    adapter = ClassificationAdapter(
        primary_metric=_METRIC_NAME, fill_strategy_stability_available=analysis_dir is not None
    )
    return ReportDataAssembler(
        dump_root, metrics_dir, analysis_dir, adapter=adapter, top_k=2, bottom_k=2
    )


# --- sample rankings (§1 격차#5) -------------------------------------------


def test_top_k_bottom_k_selected_by_vulnerability_score_and_full_rankings_not_truncated(
    tmp_path: Path,
) -> None:
    dump_root, metrics_dir = _build_dump_and_metrics(tmp_path)
    analysis_dir = _build_analysis(tmp_path, metrics_dir)

    assembled = _assembler(dump_root, metrics_dir, analysis_dir).assemble(_METRIC_NAME)

    assert len(assembled.full_sample_rankings) == 5
    assert assembled.model.run_summary.n_samples == 5

    rankings = assembled.model.sample_rankings
    assert [card.sample_id for card in rankings.most_vulnerable] == ["s0", "s1"]
    assert [card.vulnerability_score for card in rankings.most_vulnerable] == [
        _EXPECTED_DEGRADATION["s0"],
        _EXPECTED_DEGRADATION["s1"],
    ]
    assert [card.sample_id for card in rankings.most_robust] == ["s4", "s3"]
    assert [card.vulnerability_score for card in rankings.most_robust] == [
        _EXPECTED_DEGRADATION["s4"],
        _EXPECTED_DEGRADATION["s3"],
    ]
    assert len(rankings.most_vulnerable) + len(rankings.most_robust) <= 4


# --- region summary worst-case policy (§1 격차#3) ---------------------------


def test_region_row_worst_case_grade_and_distribution(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_dump_and_metrics(tmp_path)
    analysis_dir = _build_analysis(tmp_path, metrics_dir)

    assembled = _assembler(dump_root, metrics_dir, analysis_dir).assemble(_METRIC_NAME)

    rows = assembled.model.region_summary.rows
    assert [row.region_key for row in rows] == [_MAIN_REGION_KEY]
    row = rows[0]
    assert row.reliability_grade is ReportGrade.UNRELIABLE
    assert row.reliability_distribution == {"high": 1, "unreliable": 1}
    assert row.n_valid == 5
    assert row.mean_degradation == sum(_EXPECTED_DEGRADATION.values()) / 5
    assert row.flip_rate is None  # gt_logit_drop is a continuous metric

    # Dataset-level distribution (AnalysisManifest.grade_distribution, reused
    # verbatim rather than recomputed) covers every reliability row in the
    # store, including the LOW-grade "aux::0" anchor synthesized only to
    # exercise the control-comparison card below -- unlike the row-level
    # distribution above, this one is not filtered to _MAIN_REGION_KEY.
    assert assembled.model.region_summary.reliability_distribution == {
        "high": 1,
        "low": 1,
        "unreliable": 1,
    }


def test_spatial_concentration_and_region_top_region_share(tmp_path: Path) -> None:
    """Every sample's only non-control region is _MAIN_REGION_KEY -- a maximally concentrated fixture.

    With exactly one region_key in the run, spatial_entropy is undefined
    (not zero) -- there is nothing to spread across (report layout redesign,
    docs/report_layout_improve/AGENTS_OPINION_1.md).
    """

    dump_root, metrics_dir = _build_dump_and_metrics(tmp_path)
    analysis_dir = _build_analysis(tmp_path, metrics_dir)

    assembled = _assembler(dump_root, metrics_dir, analysis_dir).assemble(_METRIC_NAME)

    concentration = assembled.model.spatial_concentration
    assert concentration.dominant_region_key == _MAIN_REGION_KEY
    assert concentration.dominant_region_share == 1.0
    assert concentration.n_scored_samples == 5
    assert concentration.spatial_entropy is None

    row = assembled.model.region_summary.rows[0]
    assert row.top_region_share == 1.0
    assert row.high_rate == 0.5  # 1 high / (1 high + 1 unreliable)


def test_control_only_region_excluded_from_region_summary(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_dump_and_metrics(tmp_path)
    analysis_dir = _build_analysis(tmp_path, metrics_dir)

    assembled = _assembler(dump_root, metrics_dir, analysis_dir).assemble(_METRIC_NAME)

    region_keys = {row.region_key for row in assembled.model.region_summary.rows}
    assert _CONTROL_REGION_KEY not in region_keys


# --- sample cards / top_regions (§1 격차#4) ---------------------------------


def test_sample_card_top_regions_matches_spatial_profile_and_grade(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_dump_and_metrics(tmp_path)
    analysis_dir = _build_analysis(tmp_path, metrics_dir)

    assembled = _assembler(dump_root, metrics_dir, analysis_dir).assemble(_METRIC_NAME)

    most_vulnerable_by_id = {card.sample_id: card for card in assembled.model.sample_rankings.most_vulnerable}
    s0_card = most_vulnerable_by_id["s0"]
    assert len(s0_card.top_regions) == 1
    assert s0_card.top_regions[0].region_key == _MAIN_REGION_KEY
    assert s0_card.top_regions[0].degradation == _EXPECTED_DEGRADATION["s0"]
    assert s0_card.top_regions[0].reliability_grade is ReportGrade.HIGH
    assert s0_card.reliability_grade is ReportGrade.HIGH

    s1_card = most_vulnerable_by_id["s1"]
    assert s1_card.top_regions[0].degradation == _EXPECTED_DEGRADATION["s1"]
    assert s1_card.top_regions[0].reliability_grade is ReportGrade.UNRELIABLE
    assert s1_card.reliability_grade is ReportGrade.UNRELIABLE

    # s2 has no reliability row at all -- "not evaluated" must stay None,
    # never a stand-in grade (design §6.2 "unavailable ≠ false").
    most_robust_by_id = {card.sample_id: card for card in assembled.model.sample_rankings.most_robust}
    s3_card = most_robust_by_id["s3"]
    assert s3_card.reliability_grade is None
    assert s3_card.top_regions[0].reliability_grade is None


# --- scorecard control-comparison card --------------------------------------


def test_control_comparison_card_averages_only_available_z_vs_control(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_dump_and_metrics(tmp_path)
    analysis_dir = _build_analysis(tmp_path, metrics_dir)

    assembled = _assembler(dump_root, metrics_dir, analysis_dir).assemble(_METRIC_NAME)

    card = next(card for card in assembled.model.scorecard if card.key == "control_comparison")
    assert card.value == (3.0 + 5.0) / 2
    assert card.note is None


# --- reliability spotlight ----------------------------------------------------


def test_reliability_spotlight_flags_the_unreliable_anchor(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_dump_and_metrics(tmp_path)
    analysis_dir = _build_analysis(tmp_path, metrics_dir)

    assembled = _assembler(dump_root, metrics_dir, analysis_dir).assemble(_METRIC_NAME)

    flagged = assembled.model.reliability_spotlight.flagged_examples
    assert len(flagged) == 1
    assert flagged[0].anchor_key_repr == f"s1::{_MAIN_REGION_KEY}::False"


# --- analysis_dir=None: explicit unavailability, not silent omission -------


def test_analysis_dir_none_marks_every_analysis_derived_field_unavailable(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_dump_and_metrics(tmp_path)

    assembled = _assembler(dump_root, metrics_dir, None).assemble(_METRIC_NAME)
    model = assembled.model

    assert model.reliability_spotlight.flagged_examples == ()

    control_card = next(card for card in model.scorecard if card.key == "control_comparison")
    assert control_card.value is None
    assert control_card.note == "분석 미실행: 대조군 비교가 실행되지 않았습니다."

    assert len(model.region_summary.rows) == 1
    region_row = model.region_summary.rows[0]
    assert region_row.reliability_grade is None
    assert region_row.reliability_distribution == {}
    assert model.region_summary.reliability_distribution == {}

    for card in (*model.sample_rankings.most_vulnerable, *model.sample_rankings.most_robust):
        assert card.reliability_grade is None
        for entry in card.top_regions:
            assert entry.reliability_grade is None

    assert model.provenance.analysis_dir is None
    assert model.provenance.analysis_manifest_hash is None
    assert model.meta.schema_versions.analysis is None

    # spatial_concentration/top_region_share are derived from spatial_profile
    # alone (never AnalysisStore), so they stay populated even with no
    # analysis run -- unlike every reliability-grade-derived field above.
    assert model.spatial_concentration.dominant_region_key == _MAIN_REGION_KEY
    assert model.spatial_concentration.dominant_region_share == 1.0
    assert region_row.top_region_share == 1.0
    assert region_row.high_rate is None  # empty distribution -- no analysis run

    # Not truncated by the missing analysis run -- assembling without an
    # analysis store must not change ranking/count behavior at all.
    assert len(assembled.full_sample_rankings) == 5
