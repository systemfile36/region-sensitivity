"""C1 integration tests for R0 ReportDataAssembler.

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
import pytest
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

    Both anchors share ``_MAIN_REGION_KEY`` (the worst-case policy
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


# --- sample rankings ------------------------------------------------------


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


# --- region summary worst-case policy ---------------------------------------


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
    (not zero) -- there is nothing to spread across.
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


# --- sample cards / top_regions ---------------------------------------------


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
    # never a stand-in grade ("unavailable ≠ false").
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
    assert control_card.note == "Analysis not run: control comparison was not executed."

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


# --- semantic_group axis ------------------------------------------------------

# sample_id -> (gt_label, {region_id: perturbed gt-logit}). gt_label=0
# samples degrade most under lower_limb occlusion (a "foot action" class);
# gt_label=1 samples degrade most under upper_limb occlusion (a "hand
# action" class) -- clean gt-logit is always 10.0, so degradation = 10.0
# minus the perturbed value below.
_LIMB_GT_LOGIT_BY_SAMPLE: dict[str, tuple[int, dict[str, float]]] = {
    "s0": (0, {"left_arm": 9.9, "right_arm": 9.9, "left_leg": 9.15, "right_leg": 9.15}),
    "s1": (0, {"left_arm": 9.9, "right_arm": 9.9, "left_leg": 9.35, "right_leg": 9.35}),
    "s2": (1, {"left_arm": 9.15, "right_arm": 9.15, "left_leg": 9.85, "right_leg": 9.85}),
    "s3": (1, {"left_arm": 9.35, "right_arm": 9.35, "left_leg": 9.95, "right_leg": 9.95}),
}
_LIMB_SEMANTIC_GROUP = {
    "left_arm": "upper_limb",
    "right_arm": "upper_limb",
    "left_leg": "lower_limb",
    "right_leg": "lower_limb",
}


def _limb_logits(gt_label: int, value: float) -> np.ndarray:
    logits = [0.0, 0.0]
    logits[gt_label] = value
    return np.array(logits)


def _build_dump_and_metrics_semantic_groups(tmp_path: Path) -> tuple[Path, Path]:
    """4 region families grouped into upper_limb/lower_limb via ``semantic_group``.

    Reproduces, end-to-end through the real dump/metrics/assemble pipeline,
    the exact "foot action class" scenario already hand-verified at the
    pure-function level in tests/unit/test_report_assembler.py.
    """

    config = build_resolved_config(
        tmp_path,
        regions=tuple(
            ResolvedRegionConfig(
                region_id=region_id,
                kind=RegionKind.GRID,
                params={"rows": 1, "cols": 1},
                semantic_group=semantic_group,
            )
            for region_id, semantic_group in _LIMB_SEMANTIC_GROUP.items()
        ),
    )

    clean_records = [
        clean_record(sample_id, logits=_limb_logits(gt_label, 10.0), gt_label=gt_label)
        for sample_id, (gt_label, _by_region) in _LIMB_GT_LOGIT_BY_SAMPLE.items()
    ]
    perturbed_records = []
    index = 0
    for sample_id, (gt_label, by_region) in _LIMB_GT_LOGIT_BY_SAMPLE.items():
        for region_id, perturbed_value in by_region.items():
            perturbed_records.append(
                perturbed_record(
                    index,
                    sample_id=sample_id,
                    region_id=region_id,
                    region_instance_id=f"{region_id}/r0/c0",
                    logits=_limb_logits(gt_label, perturbed_value),
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


def test_semantic_summary_and_class_semantic_matrix_reproduce_foot_action_class_pattern(
    tmp_path: Path,
) -> None:
    dump_root, metrics_dir = _build_dump_and_metrics_semantic_groups(tmp_path)

    assembled = _assembler(dump_root, metrics_dir, None).assemble(_METRIC_NAME)
    model = assembled.model

    assert model.semantic_concentration.n_semantic_groups == 2
    assert model.semantic_concentration.n_scored_samples == 4

    summary_by_group = {row.semantic_group: row for row in model.semantic_summary}
    assert set(summary_by_group) == {"upper_limb", "lower_limb"}
    assert summary_by_group["upper_limb"].region_ids == ("left_arm", "right_arm")
    assert summary_by_group["lower_limb"].region_ids == ("left_leg", "right_leg")
    assert summary_by_group["upper_limb"].flip_rate is None  # gt_logit_drop is continuous

    matrix_by_cell = {
        (row.gt_label, row.semantic_group): row for row in model.class_semantic_matrix
    }
    # The core finding: for gt_label=0 (foot action), lower_limb dominates;
    # for gt_label=1 (hand action), upper_limb dominates.
    assert (
        matrix_by_cell[(0, "lower_limb")].mean_degradation
        > matrix_by_cell[(0, "upper_limb")].mean_degradation
    )
    assert (
        matrix_by_cell[(1, "upper_limb")].mean_degradation
        > matrix_by_cell[(1, "lower_limb")].mean_degradation
    )
    assert all(row.n_samples == 2 for row in model.class_semantic_matrix)
    assert all(row.flip_rate is None for row in model.class_semantic_matrix)
    assert model.provenance.class_semantic_excluded_no_gt_label == 0

    # AssembledReport.sample_semantic_degradation carries the per-sample values
    # class_semantic_matrix was built from, for the future labels.py export.
    assert assembled.sample_semantic_degradation[("s0", "lower_limb")] == pytest.approx(0.85)
    assert assembled.sample_semantic_degradation[("s2", "upper_limb")] == pytest.approx(0.85)


def test_ungrouped_run_gates_semantic_concentration_but_still_computes_trivial_rows(
    tmp_path: Path,
) -> None:
    """A plain grid+control run (no regions[].semantic_group) -- the common, self-evident case.

    The "control" family (RANDOM_AREA_MATCH, never in region_metrics.parquet)
    must not inflate n_semantic_groups -- it is excluded from the semantic
    axis the same way it is already excluded from region_summary.
    """

    dump_root, metrics_dir = _build_dump_and_metrics(tmp_path)

    assembled = _assembler(dump_root, metrics_dir, None).assemble(_METRIC_NAME)
    model = assembled.model

    concentration = model.semantic_concentration
    assert concentration.n_semantic_groups == 1
    assert concentration.dominant_semantic_group is None
    assert concentration.dominant_semantic_group_share is None
    assert concentration.semantic_group_entropy is None
    assert concentration.n_scored_samples == 0

    # Computed, not display-suppressed -- one self-evident row/cell.
    assert len(model.semantic_summary) == 1
    assert model.semantic_summary[0].semantic_group == "grid"
    assert model.semantic_summary[0].region_ids == ("grid",)
    assert len(model.class_semantic_matrix) == 1


def test_unlabeled_sample_reaches_r0_without_crashing_and_contributes_nothing(
    tmp_path: Path,
) -> None:
    """A ``gt_label=None`` sample must not crash dump/metrics/report assembly.

    Was previously untestable end-to-end: ``MetricRegistry.
    compute_item_metrics`` called ``int(row.gt_label)`` unconditionally,
    crashing the whole run the moment any sample's ``gt_label`` was unknown
    -- a pre-existing N1/N2 limitation orthogonal to this plan, fixed
    alongside it (``ssat.metrics.types.ExclusionReason.GT_LABEL_UNKNOWN``).

    With that fixed, a label-free sample ("s4" here, alongside the labeled
    "foot action class" fixture) now reaches R0 cleanly, but contributes
    *nothing* to the semantic axis: every one of its items is unavailable
    (no currently registered metric is computable without a reference
    class), so its ``SpatialProfile.degradation`` is ``None`` everywhere,
    and it never even reaches ``_sample_semantic_group_degradation``'s
    output -- there is no per-sample value to exclude by
    ``_build_class_semantic_matrix``, so ``class_semantic_excluded_no_gt_
    label`` stays ``0`` here. That provenance counter's "excluded despite
    having data" path is a defensive branch for a case classification
    metrics can never actually produce; it is instead fully covered at the
    pure-function level in tests/unit/test_report_assembler.py::
    test_build_class_semantic_matrix_excludes_samples_with_no_gt_label,
    which constructs that otherwise-unreachable input by hand.
    """

    config = build_resolved_config(
        tmp_path,
        regions=tuple(
            ResolvedRegionConfig(
                region_id=region_id,
                kind=RegionKind.GRID,
                params={"rows": 1, "cols": 1},
                semantic_group=semantic_group,
            )
            for region_id, semantic_group in _LIMB_SEMANTIC_GROUP.items()
        ),
    )

    clean_records = [
        clean_record(sample_id, logits=_limb_logits(gt_label, 10.0), gt_label=gt_label)
        for sample_id, (gt_label, _by_region) in _LIMB_GT_LOGIT_BY_SAMPLE.items()
    ]
    clean_records.append(
        clean_record("s4", logits=_limb_logits(0, 10.0), gt_label=None)
    )
    perturbed_records = []
    index = 0
    for sample_id, (gt_label, by_region) in _LIMB_GT_LOGIT_BY_SAMPLE.items():
        for region_id, perturbed_value in by_region.items():
            perturbed_records.append(
                perturbed_record(
                    index,
                    sample_id=sample_id,
                    region_id=region_id,
                    region_instance_id=f"{region_id}/r0/c0",
                    logits=_limb_logits(gt_label, perturbed_value),
                )
            )
            index += 1
    for region_id in _LIMB_SEMANTIC_GROUP:
        perturbed_records.append(
            perturbed_record(
                index,
                sample_id="s4",
                region_id=region_id,
                region_instance_id=f"{region_id}/r0/c0",
                logits=_limb_logits(0, 5.0),
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

    # The whole point: this must not raise.
    assembled = _assembler(dump_root, metrics_dir, None).assemble(_METRIC_NAME)
    model = assembled.model

    assert "s4" not in {sample_id for sample_id, _group in assembled.sample_semantic_degradation}
    matrix_by_cell = {
        (row.gt_label, row.semantic_group): row for row in model.class_semantic_matrix
    }
    # Identical to the labeled-only fixture (test_semantic_summary_and_
    # class_semantic_matrix_reproduce_foot_action_class_pattern) -- s4
    # changed nothing about the labeled samples' own results.
    assert matrix_by_cell[(0, "lower_limb")].n_samples == 2
    assert matrix_by_cell[(1, "upper_limb")].n_samples == 2
    assert model.provenance.class_semantic_excluded_no_gt_label == 0
