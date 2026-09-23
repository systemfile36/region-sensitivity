"""Parity and unit tests for the A1 grade engine and feature builder."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synthetic_dump_builder import (
    build_resolved_config,
    clean_record,
    compute_and_save_analysis,
    compute_and_save_metrics,
    perturbed_record,
    write_dump,
)

from experiments.revision_1.a1_threshold.build_features import (
    build_features,
    replicate_intervals,
    target_item_values,
    bootstrap_draw_counts,
)
from experiments.revision_1.common.grade_engine import (
    CI_PERCENTILES,
    FALSE,
    TRUE,
    UNAVAILABLE,
    GradeParams,
    RunScales,
    combine_grade,
    compute_scales,
    flag_labels,
    grade_anchors,
    grade_labels,
)
from experiments.revision_1.common.runs import RunRef
from ssat.analysis.interval import compute_intervals
from ssat.analysis.reader import AnalysisReader
from ssat.analysis.reliability import _grade
from ssat.analysis.types import FlagValue
from ssat.core.config.schema import ResolvedRegionConfig
from ssat.core.types import PerturbationOp, RegionKind
from ssat.metrics.builtin_metrics.continuous import GtLogitDrop, MarginDrop
from ssat.metrics.registry import MetricRegistry

OPS = (PerturbationOp.CONSTANT_FILL, PerturbationOp.MEAN_FILL, PerturbationOp.BLUR)
METRICS = ("margin_drop", "gt_logit_drop")
FLAGS = ("sign_consistent", "exceeds_control", "multi_strategy", "ci_excludes_zero")

# Anchors with a deliberately degenerate control or strategy setup.
ZERO_STD = ("s0", 0, 0)  # identical control values -> z undefined
ONE_CONTROL = ("s1", 0, 1)  # n_controls = 1 -> z undefined
NO_CONTROL = ("s2", 1, 1)  # no control items at all
SINGLE_OP = ("s3", 1, 0)  # one operator only -> multi_strategy FALSE
ZERO_EFFECT = ("s3", 0, 0)  # mean_fill leaves the logits unchanged -> sign 0


def _registry() -> MetricRegistry:
    registry = MetricRegistry()
    registry.register(MarginDrop())
    registry.register(GtLogitDrop())
    return registry


def _cell_params(row: int, col: int) -> dict[str, int]:
    return {"col_index": col, "cols": 2, "row_index": row, "rows": 2}


def _build_run(tmp_path: Path, **analyze_kwargs: object) -> RunRef:
    """Write a 4-sample, 2x2-grid, 3-operator dump with K=3 controls and analyze it."""

    rng = np.random.default_rng(7)
    config = build_resolved_config(
        tmp_path,
        regions=(ResolvedRegionConfig(region_id="grid", kind=RegionKind.GRID, params={"rows": 2, "cols": 2}),),
    )
    clean_logits = {f"s{index}": np.array([3.0, 1.0, 0.5]) + rng.normal(0, 0.2, 3) for index in range(4)}
    records = []
    counter = itertools.count()
    for sample_id, (row, col), op in itertools.product(clean_logits, itertools.product(range(2), range(2)), OPS):
        anchor = (sample_id, row, col)
        if anchor == SINGLE_OP and op is not OPS[0]:
            continue
        instance = f"grid/r{row}/c{col}"
        effect = rng.choice([0.05, 0.4, 1.5])
        for seed in range(2):
            logits = clean_logits[sample_id] - np.array([effect, 0.0, 0.0]) + rng.normal(0, 0.1, 3)
            if anchor == ZERO_EFFECT and op is PerturbationOp.MEAN_FILL:
                logits = clean_logits[sample_id]
            records.append(
                perturbed_record(
                    next(counter),
                    sample_id=sample_id,
                    region_id="grid",
                    region_instance_id=instance,
                    region_params=_cell_params(row, col),
                    logits=logits,
                    perturb_op=op,
                    seed_used=seed + 1,
                )
            )
        n_controls = {ONE_CONTROL: 1, NO_CONTROL: 0}.get(anchor, 3)
        shared = clean_logits[sample_id] - np.array([0.3, 0.0, 0.0])
        for control_index in range(n_controls):
            logits = shared if anchor == ZERO_STD else clean_logits[sample_id] - rng.normal(0.2, 0.3, 3)
            records.append(
                perturbed_record(
                    next(counter),
                    sample_id=sample_id,
                    region_id="control:grid:0",
                    region_instance_id=f"control:{instance}:0:{control_index}",
                    region_kind=RegionKind.RANDOM_AREA_MATCH,
                    region_params={
                        "control_index": control_index,
                        "control_request_index": 0,
                        "target_region": {
                            "kind": "grid",
                            "params": _cell_params(row, col),
                            "region_id": "grid",
                            "region_instance_id": instance,
                        },
                    },
                    logits=logits,
                    perturb_op=op,
                    is_control=True,
                    seed_used=100 + control_index,
                )
            )
    dump = tmp_path / "dump"
    write_dump(
        dump,
        config,
        clean_records=tuple(clean_record(sample_id, logits=logits) for sample_id, logits in clean_logits.items()),
        perturbed_records=tuple(records),
    )
    metrics = tmp_path / "metrics"
    analysis = tmp_path / "analysis"
    compute_and_save_metrics(dump, config, metrics, registry=_registry(), primary_metric="margin_drop")
    compute_and_save_analysis(dump, metrics, analysis, primary_metric="margin_drop", **analyze_kwargs)
    return RunRef(
        name="fixture",
        dump=dump,
        metrics=metrics,
        analysis=analysis,
        dataset="synthetic",
        model="fixture",
        protocol="crop_free",
        n_controls=3,
    )


@pytest.fixture(scope="module")
def default_run(tmp_path_factory: pytest.TempPathFactory) -> RunRef:
    return _build_run(tmp_path_factory.mktemp("a1_default"))


def _assert_parity(features: pd.DataFrame, params: GradeParams) -> pd.DataFrame:
    graded = grade_anchors(features, params, scales=compute_scales(features))
    for flag in FLAGS:
        np.testing.assert_array_equal(flag_labels(graded[flag]), features[f"stored__{flag}"].to_numpy(), err_msg=flag)
    np.testing.assert_array_equal(grade_labels(graded["grade"]), features["stored__reliability_grade"].to_numpy())
    return graded


# --- parity with ssat analyze ------------------------------------------------


@pytest.mark.parametrize("metric", METRICS)
def test_default_grades_match_ssat_analyze(default_run: RunRef, metric: str) -> None:
    features, meta = build_features(default_run, metric)
    graded = _assert_parity(features, GradeParams())

    assert meta["interval_parity_95"]["bounds_bitwise_equal"]
    by_anchor = features.set_index(["sample_id", "region_key"])
    exceeds = pd.Series(graded["exceeds_control"].to_numpy(), index=by_anchor.index)
    for sample_id, row, col in (ZERO_STD, ONE_CONTROL, NO_CONTROL):
        assert exceeds[(sample_id, f"grid::grid/r{row}/c{col}")] == UNAVAILABLE
    multi = pd.Series(graded["multi_strategy"].to_numpy(), index=by_anchor.index)
    assert multi[(SINGLE_OP[0], f"grid::grid/r{SINGLE_OP[1]}/c{SINGLE_OP[2]}")] == FALSE
    assert {TRUE, FALSE} <= set(graded["sign_consistent"])
    assert {TRUE, FALSE} <= set(graded["exceeds_control"])


def test_non_default_z_threshold_matches_ssat_analyze(tmp_path: Path) -> None:
    run = _build_run(tmp_path, z_vs_control_threshold=0.5, seed_cv_threshold=0.05)
    features, _ = build_features(run, "margin_drop")
    _assert_parity(features, GradeParams(z_threshold=0.5))

    seed_cv = features["seed_cv_max"].to_numpy()
    expected = np.where(np.isnan(seed_cv), "unavailable", np.where(seed_cv < 0.05, "true", "false"))
    np.testing.assert_array_equal(expected, features["stored__seed_stable"].to_numpy())


@pytest.mark.parametrize("level", ["90", "99", "999"])
def test_replicated_intervals_match_compute_intervals(default_run: RunRef, level: str) -> None:
    low_pct, high_pct = CI_PERCENTILES[level]
    full = AnalysisReader(default_run.dump, default_run.metrics).item_values()
    expected = {
        row.region_key: row
        for row in compute_intervals(full, ci_low_pct=low_pct, ci_high_pct=high_pct)
        if row.metric == "gt_logit_drop"
    }
    replicated = replicate_intervals(
        target_item_values(default_run, "gt_logit_drop"), bootstrap_draw_counts(default_run.metrics), "gt_logit_drop"
    ).set_index("region_key")
    assert set(replicated.index) == set(expected)
    for region_key, row in expected.items():
        assert replicated.loc[region_key, f"ci{level}__low"] == row.ci_low
        assert replicated.loc[region_key, f"ci{level}__high"] == row.ci_high
        assert replicated.loc[region_key, "point_estimate"] == row.point_estimate


# --- engine rules -------------------------------------------------------------


def _features(op_values: list[list[float]], conditions: list[tuple[float, float, int, bool]] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(op_values, columns=[f"op__{index}" for index in range(len(op_values[0]))])
    for index, (excess, std0, n, available) in enumerate(conditions or []):
        frame[f"cond{index}__excess"] = excess
        frame[f"cond{index}__std0"] = std0
        frame[f"cond{index}__n"] = n
        frame[f"cond{index}__available"] = available
    for level in CI_PERCENTILES:
        frame[f"ci{level}__low"] = 0.1
        frame[f"ci{level}__high"] = 0.2
    return frame


SCALES = RunScales(sign_scale=1.0, std_scale=0.5)


def test_sign_deadband_turns_small_values_into_zero_signs() -> None:
    features = _features([[0.02, 0.5, 0.6], [0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]])
    default = grade_anchors(features, GradeParams(), scales=SCALES)
    assert list(default["sign_consistent"]) == [TRUE, TRUE, UNAVAILABLE]
    assert list(default["multi_strategy"]) == [TRUE, TRUE, UNAVAILABLE]

    banded = grade_anchors(features, GradeParams(sign_alpha=0.05), scales=SCALES)
    assert list(banded["sign_consistent"]) == [FALSE, TRUE, UNAVAILABLE]

    abstain = grade_anchors(features, GradeParams(sign_alpha=0.05, zero_abstain=True), scales=SCALES)
    assert list(abstain["sign_consistent"]) == [TRUE, UNAVAILABLE, UNAVAILABLE]
    assert list(abstain["multi_strategy"]) == [TRUE, FALSE, UNAVAILABLE]


def test_min_multi_strategy_counts_the_dominant_sign() -> None:
    features = _features([[0.5, 0.6, -0.1]])
    assert grade_anchors(features, GradeParams(min_multi_strategy=2), scales=SCALES)["multi_strategy"][0] == TRUE
    assert grade_anchors(features, GradeParams(min_multi_strategy=3), scales=SCALES)["multi_strategy"][0] == FALSE


def test_ddof_and_std_floor_change_z() -> None:
    features = _features(
        [[1.0]] * 4,
        conditions=[(1.0, 0.4, 2, True)],
    )
    features.loc[1, "cond0__std0"] = 0.0
    features.loc[2, "cond0__n"] = 1
    features.loc[3, "cond0__available"] = False

    default = grade_anchors(features, GradeParams(), scales=SCALES)
    assert default["max_z"][0] == pytest.approx(2.5)
    assert list(default["exceeds_control"]) == [TRUE, UNAVAILABLE, UNAVAILABLE, UNAVAILABLE]

    ddof = grade_anchors(features, GradeParams(ddof=1), scales=SCALES)
    assert ddof["max_z"][0] == pytest.approx(1.0 / (0.4 * np.sqrt(2)))
    assert ddof["exceeds_control"][0] == FALSE

    floored = grade_anchors(features, GradeParams(std_floor_beta=0.1), scales=SCALES)
    assert floored["max_z"][1] == pytest.approx(1.0 / 0.05)
    assert list(floored["exceeds_control"]) == [TRUE, TRUE, UNAVAILABLE, UNAVAILABLE]


def test_max_z_takes_the_best_evaluable_condition() -> None:
    features = _features([[1.0]], conditions=[(1.0, 1.0, 3, True), (3.0, 1.0, 3, True), (9.0, 1.0, 3, False)])
    assert grade_anchors(features, GradeParams(), scales=SCALES)["max_z"][0] == 3.0


def test_ci_level_selects_the_matching_bounds() -> None:
    features = _features([[1.0]])
    features["ci99__low"] = -0.1
    features["ci95__low"] = np.nan
    assert grade_anchors(features, GradeParams(ci_level="90"), scales=SCALES)["ci_excludes_zero"][0] == TRUE
    assert grade_anchors(features, GradeParams(ci_level="99"), scales=SCALES)["ci_excludes_zero"][0] == FALSE
    assert grade_anchors(features, GradeParams(ci_level="95"), scales=SCALES)["ci_excludes_zero"][0] == UNAVAILABLE


def test_combine_grade_matches_reliability_grade_rule() -> None:
    codes = {FlagValue.TRUE: TRUE, FlagValue.FALSE: FALSE, FlagValue.UNAVAILABLE: UNAVAILABLE}
    combos = list(itertools.product(FlagValue, repeat=4))
    arrays = [np.array([codes[combo[index]] for combo in combos], dtype=np.int8) for index in range(4)]
    grades = grade_labels(combine_grade(*arrays))
    assert list(grades) == [_grade(*combo).value for combo in combos]


def test_compute_scales_uses_target_values_and_evaluable_stds() -> None:
    features = _features(
        [[1.0, -3.0], [np.nan, 2.0]],
        conditions=[(0.0, 0.2, 3, True)],
    )
    features.loc[1, "cond0__std0"] = 0.6
    features.loc[1, "cond0__available"] = False
    scales = compute_scales(features)
    assert scales.sign_scale == 2.0
    assert scales.std_scale == pytest.approx(0.2)


# --- sweep settings and summaries ----------------------------------------------


def test_settings_follow_the_protocol_and_skip_z_parameters_without_controls() -> None:
    from experiments.revision_1.a1_threshold.run_sweep import load_protocol, settings_for, z_curve_values

    protocol = load_protocol()
    imagenet = {setting_id: params for setting_id, _, _, params in settings_for(protocol, "imagenet")}
    ntu = {setting_id for setting_id, _, _, _ in settings_for(protocol, "ntu60")}
    synthetic = {setting_id for setting_id, _, _, _ in settings_for(protocol, "synthetic")}

    assert imagenet["default"] == GradeParams()
    assert imagenet["oat:ci_level=99"] == GradeParams(ci_level="99")
    assert imagenet["preset:conservative"] == GradeParams(
        z_threshold=3.0, sign_alpha=0.05, ci_level="99", ddof=1, std_floor_beta=0.10
    )
    assert imagenet["alt:zero_abstain:sign_alpha=0.05"] == GradeParams(sign_alpha=0.05, zero_abstain=True)
    assert not any(setting.startswith(("oat:z_threshold", "oat:ddof", "oat:std_floor_beta")) for setting in ntu)
    assert "oat:min_multi_strategy=5" in synthetic and "oat:min_multi_strategy=5" not in imagenet
    curve = z_curve_values(protocol)
    assert curve[0] == 0.0 and curve[-1] == 6.0 and len(curve) == 121


def test_sensitivity_ranking_uses_the_worst_registered_value() -> None:
    from experiments.revision_1.a1_threshold.summarize import sensitivity_ranking

    agreement = pd.DataFrame(
        {
            "run": ["r"] * 4,
            "dataset": ["imagenet"] * 4,
            "setting_id": ["default", "oat:z_threshold=1.0", "oat:z_threshold=3.0", "oat:ddof=1"],
            "param": ["default", "z_threshold", "z_threshold", "ddof"],
            "value": [None, 1.0, 3.0, 1],
            "agreement": [1.0, 0.9, 0.95, 0.99],
            "delta_ge2": [0.0, 0.1, 0.05, 0.01],
            "kappa_linear": [1.0, 0.8, 0.9, 0.98],
        }
    )
    ranking = sensitivity_ranking(agreement).set_index("param")
    assert ranking.loc["z_threshold", "max_disagreement"] == pytest.approx(0.1)
    assert ranking.loc["z_threshold", "at_value"] == 1.0
    assert ranking.loc["z_threshold", "rank"] == 1 and ranking.loc["ddof", "rank"] == 2
