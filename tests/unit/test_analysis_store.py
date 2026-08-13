"""Tests for A7 AnalysisStore (ssat/analysis/store.py)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ssat.analysis.errors import AnalysisCorruptionError, AnalysisSchemaError
from ssat.analysis.store import load_analysis, save_analysis, verify_source_metrics
from ssat.analysis.types import (
    Alignment,
    AnchorKey,
    AvailableAnalyses,
    ConditionKey,
    ControlComparisonRow,
    CoverageReport,
    FlagValue,
    IntervalRow,
    RankCorrelationRow,
    ReliabilityGrade,
    ReliabilityRow,
    SeedStabilityRow,
    StrategyProfileRow,
    StrategyStabilityRow,
)
from ssat.utils.io import sha256_file, write_json_atomic

_METRIC = "margin_drop"


def _anchor(region_key: str = "grid::grid/r0/c0", sample_id: str = "s1") -> AnchorKey:
    return AnchorKey(sample_id=sample_id, region_key=region_key, invert_mask=False)


def _condition(op: str = "constant_fill") -> ConditionKey:
    return ConditionKey(perturb_op=op, perturb_params_hash="hash")


def _control_row(anchor: AnchorKey) -> ControlComparisonRow:
    return ControlComparisonRow(
        target_anchor_key=anchor,
        condition_key=_condition(),
        metric_name=_METRIC,
        control_available=FlagValue.TRUE,
        area_matched=FlagValue.TRUE,
        control_mean=1.0,
        control_std=0.5,
        n_controls=3,
        excess=2.0,
        ratio=3.0,
        z_vs_control=4.0,
    )


def _seed_row(anchor: AnchorKey) -> SeedStabilityRow:
    return SeedStabilityRow(
        anchor_key=anchor,
        condition_key=_condition(),
        metric_name=_METRIC,
        seed_mean=1.0,
        seed_std=0.1,
        seed_cv=0.1,
        n_seeds=3,
    )


def _strategy_row(anchor: AnchorKey) -> StrategyStabilityRow:
    return StrategyStabilityRow(
        anchor_key=anchor,
        metric_name=_METRIC,
        strategy_signs={"constant_fill": 1, "blur": 1},
        strategy_values={"constant_fill": 2.0, "blur": 3.0},
        sign_agreement_ratio=1.0,
        n_strategies=2,
    )


def _rank_row() -> RankCorrelationRow:
    return RankCorrelationRow(
        op_a="blur",
        op_b="constant_fill",
        spearman=0.5,
        n_regions=4,
        spearman_excl_top1=0.3,
        scope="full_dataset",
    )


def _profile_row() -> StrategyProfileRow:
    return StrategyProfileRow(
        perturb_op="blur",
        preserves_statistics=True,
        preserves_local_texture=False,
        is_global_operation=True,
        cluster_id=0,
        mean_corr_within=0.8,
        mean_corr_across=0.1,
        alignment=Alignment.ALIGNED,
        mean_degradation_excl_top=1.5,
        sign_ratio_positive=0.75,
        n_anchors=4,
    )


def _interval_row() -> IntervalRow:
    return IntervalRow(
        region_key="grid::grid/r0/c0",
        metric=_METRIC,
        point_estimate=5.0,
        ci_low=1.0,
        ci_high=9.0,
        ci_method="percentile",
        n_bootstrap=1000,
        excludes_zero=True,
    )


def _reliability_row(anchor: AnchorKey, grade: ReliabilityGrade = ReliabilityGrade.HIGH) -> ReliabilityRow:
    return ReliabilityRow(
        anchor_key=anchor,
        metric_name=_METRIC,
        sign_consistent=FlagValue.TRUE,
        exceeds_control=FlagValue.TRUE,
        seed_stable=FlagValue.TRUE,
        jitter_stable=FlagValue.UNAVAILABLE,
        multi_strategy=FlagValue.TRUE,
        ci_excludes_zero=FlagValue.TRUE,
        area_matched=FlagValue.TRUE,
        reliability_grade=grade,
        reliability_reasons=(
            "sign consistent across 2 operator(s)",
            "exceeds control (z=4.00 > threshold 2)",
            "seed-stable (max cv=0.100 < 0.2)",
            "jitter stability unavailable (core has no jitter support)",
            "reproduced in 2 of 2 operators",
            "bootstrap CI excludes zero ([1.000, 9.000])",
            "control area match within tolerance",
        ),
    )


def _available_analyses() -> AvailableAnalyses:
    return AvailableAnalyses(
        control_comparison=True,
        fill_strategy_stability=True,
        seed_stability=True,
        jitter_stability=False,
    )


def _metrics_manifest_path(tmp_path: Path) -> Path:
    path = tmp_path / "metrics" / "metrics_manifest.json"
    write_json_atomic(path, {"metrics_schema_version": "1.0.0"})
    return path


def _save(tmp_path: Path, *, reliability_rows=None):
    anchor = _anchor()
    metrics_manifest_path = _metrics_manifest_path(tmp_path)
    analysis_dir = tmp_path / "analysis"

    control_rows = [_control_row(anchor)]
    seed_rows = [_seed_row(anchor)]
    strategy_rows = [_strategy_row(anchor)]
    rank_correlation_rows = [_rank_row()]
    strategy_profile_rows = [_profile_row()]
    interval_rows = [_interval_row()]
    if reliability_rows is None:
        reliability_rows = [_reliability_row(anchor)]
    coverage_report = CoverageReport(
        n_anchors=5, n_conditions_insufficient=1, n_controls_unmatched=0, n_area_mismatch_warnings=0
    )

    manifest = save_analysis(
        analysis_dir,
        control_rows=control_rows,
        seed_rows=seed_rows,
        strategy_rows=strategy_rows,
        rank_correlation_rows=rank_correlation_rows,
        strategy_profile_rows=strategy_profile_rows,
        interval_rows=interval_rows,
        reliability_rows=reliability_rows,
        coverage_report=coverage_report,
        available_analyses=_available_analyses(),
        thresholds={"z_vs_control_threshold": 2.0, "seed_cv_threshold": 0.2},
        n_bootstrap=1000,
        random_seed=0,
        source_metrics_manifest_path=metrics_manifest_path,
        computed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    return (
        analysis_dir,
        metrics_manifest_path,
        manifest,
        {
            "control_rows": control_rows,
            "seed_rows": seed_rows,
            "strategy_rows": strategy_rows,
            "rank_correlation_rows": rank_correlation_rows,
            "strategy_profile_rows": strategy_profile_rows,
            "interval_rows": interval_rows,
            "reliability_rows": reliability_rows,
            "coverage_report": coverage_report,
        },
    )


# --- round trip ----------------------------------------------------------


def test_round_trip_preserves_every_row_type(tmp_path: Path) -> None:
    analysis_dir, _, manifest, originals = _save(tmp_path)

    (
        control_rows,
        seed_rows,
        strategy_rows,
        rank_correlation_rows,
        strategy_profile_rows,
        interval_rows,
        reliability_rows,
        coverage_report,
        loaded_manifest,
    ) = load_analysis(analysis_dir)

    assert control_rows == originals["control_rows"]
    assert seed_rows == originals["seed_rows"]
    assert strategy_rows == originals["strategy_rows"]
    assert rank_correlation_rows == originals["rank_correlation_rows"]
    assert strategy_profile_rows == originals["strategy_profile_rows"]
    assert interval_rows == originals["interval_rows"]
    assert reliability_rows == originals["reliability_rows"]
    assert coverage_report == originals["coverage_report"]
    assert loaded_manifest == manifest
    assert loaded_manifest.available_analyses == _available_analyses()


def test_strategy_signs_and_reliability_reasons_round_trip_exactly(tmp_path: Path) -> None:
    analysis_dir, _, _, originals = _save(tmp_path)

    _, _, strategy_rows, _, _, _, reliability_rows, _, _ = load_analysis(analysis_dir)

    assert strategy_rows[0].strategy_signs == originals["strategy_rows"][0].strategy_signs
    assert strategy_rows[0].strategy_values == originals["strategy_rows"][0].strategy_values
    assert reliability_rows[0].reliability_reasons == originals["reliability_rows"][0].reliability_reasons
    assert isinstance(reliability_rows[0].reliability_reasons, tuple)


def test_empty_row_lists_round_trip(tmp_path: Path) -> None:
    metrics_manifest_path = _metrics_manifest_path(tmp_path)
    analysis_dir = tmp_path / "analysis"

    save_analysis(
        analysis_dir,
        control_rows=[],
        seed_rows=[],
        strategy_rows=[],
        rank_correlation_rows=[],
        strategy_profile_rows=[],
        interval_rows=[],
        reliability_rows=[],
        coverage_report=CoverageReport(0, 0, 0, 0),
        available_analyses=AvailableAnalyses(False, False, False, False),
        thresholds={},
        n_bootstrap=1000,
        random_seed=0,
        source_metrics_manifest_path=metrics_manifest_path,
    )

    result = load_analysis(analysis_dir)
    for rows in result[:7]:
        assert rows == []
    assert result[7] == CoverageReport(0, 0, 0, 0)
    assert result[8].grade_distribution == {}


# --- manifest / provenance ------------------------------------------------


def test_grade_distribution_computed_from_reliability_rows(tmp_path: Path) -> None:
    anchor_a = _anchor(sample_id="s1")
    anchor_b = _anchor(sample_id="s2")
    anchor_c = _anchor(sample_id="s3")
    rows = [
        _reliability_row(anchor_a, ReliabilityGrade.HIGH),
        _reliability_row(anchor_b, ReliabilityGrade.HIGH),
        _reliability_row(anchor_c, ReliabilityGrade.UNRELIABLE),
    ]

    _, _, manifest, _ = _save(tmp_path, reliability_rows=rows)

    assert manifest.grade_distribution == {"high": 2, "unreliable": 1}


def test_source_metrics_manifest_hash_matches_actual_file(tmp_path: Path) -> None:
    _, metrics_manifest_path, manifest, _ = _save(tmp_path)

    assert manifest.source_metrics_manifest_hash == sha256_file(metrics_manifest_path)


def test_verify_source_metrics_detects_change(tmp_path: Path) -> None:
    _, metrics_manifest_path, manifest, _ = _save(tmp_path)

    verify_source_metrics(manifest, metrics_manifest_path)  # no raise yet

    write_json_atomic(metrics_manifest_path, {"metrics_schema_version": "2.0.0"})

    with pytest.raises(AnalysisCorruptionError):
        verify_source_metrics(manifest, metrics_manifest_path)


def test_analysis_schema_version_mismatch_rejected(tmp_path: Path) -> None:
    analysis_dir, _, _, _ = _save(tmp_path)

    manifest_path = analysis_dir / "analysis_manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["analysis_schema_version"] = "0.0.1"
    write_json_atomic(manifest_path, payload)

    with pytest.raises(AnalysisSchemaError):
        load_analysis(analysis_dir)
