"""Tests for A6 ReliabilityScorer (ssat/analysis/reliability.py)."""

from __future__ import annotations

from collections import Counter

import pytest

from ssat.analysis.reliability import compute_reliability
from ssat.analysis.types import (
    AnchorKey,
    ConditionKey,
    ControlComparisonRow,
    FlagValue,
    IntervalRow,
    ReliabilityGrade,
    SeedStabilityRow,
    StrategyStabilityRow,
)

_METRIC = "margin_drop"


def _anchor(region_key: str = "grid::grid/r0/c0", sample_id: str = "s1") -> AnchorKey:
    return AnchorKey(sample_id=sample_id, region_key=region_key, invert_mask=False)


def _condition(op: str = "constant_fill") -> ConditionKey:
    return ConditionKey(perturb_op=op, perturb_params_hash="hash")


def _strategy_row(
    anchor: AnchorKey, values_by_op: dict[str, float], *, metric: str = _METRIC
) -> StrategyStabilityRow:
    signs = {op: (1 if v > 0 else (-1 if v < 0 else 0)) for op, v in values_by_op.items()}
    n = len(values_by_op)
    _, top_count = Counter(signs.values()).most_common(1)[0]
    return StrategyStabilityRow(
        anchor_key=anchor,
        metric_name=metric,
        strategy_signs=signs,
        strategy_values=dict(values_by_op),
        sign_agreement_ratio=top_count / n,
        n_strategies=n,
    )


def _control_row(
    anchor: AnchorKey,
    *,
    op: str = "constant_fill",
    metric: str = _METRIC,
    area_matched: FlagValue = FlagValue.TRUE,
    z_vs_control: float | None = 0.0,
    control_mean: float = 0.0,
    control_std: float = 1.0,
    n_controls: int = 3,
    excess: float = 0.0,
) -> ControlComparisonRow:
    return ControlComparisonRow(
        target_anchor_key=anchor,
        condition_key=_condition(op),
        metric_name=metric,
        control_available=FlagValue.TRUE,
        area_matched=area_matched,
        control_mean=control_mean,
        control_std=control_std,
        n_controls=n_controls,
        excess=excess,
        ratio=None,
        z_vs_control=z_vs_control,
    )


def _seed_row(
    anchor: AnchorKey,
    *,
    op: str = "constant_fill",
    metric: str = _METRIC,
    seed_mean: float = 1.0,
    seed_std: float | None = 0.1,
    seed_cv: float | None = 0.1,
    n_seeds: int = 3,
) -> SeedStabilityRow:
    return SeedStabilityRow(
        anchor_key=anchor,
        condition_key=_condition(op),
        metric_name=metric,
        seed_mean=seed_mean,
        seed_std=seed_std,
        seed_cv=seed_cv,
        n_seeds=n_seeds,
    )


def _interval_row(
    region_key: str, *, metric: str = _METRIC, ci_low: float = 1.0, ci_high: float = 9.0
) -> IntervalRow:
    return IntervalRow(
        region_key=region_key,
        metric=metric,
        point_estimate=(ci_low + ci_high) / 2,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_method="percentile",
        n_bootstrap=100,
        excludes_zero=ci_low > 0.0 or ci_high < 0.0,
    )


# --- B2 scenarios (design §4.3 / plan §5 단계7) -----------------------------


def test_all_conditions_agree_and_strong_evidence_yields_high_grade() -> None:
    anchor = _anchor()
    strategy_row = _strategy_row(anchor, {"constant_fill": 2.0, "blur": 3.0, "mean_fill": 1.5})
    control_row = _control_row(anchor, z_vs_control=5.0)
    seed_row = _seed_row(anchor, seed_cv=0.05)
    interval_row = _interval_row(anchor.region_key, ci_low=1.0, ci_high=9.0)

    result = compute_reliability([control_row], [seed_row], [strategy_row], [interval_row])

    assert len(result) == 1
    row = result[0]
    assert row.sign_consistent is FlagValue.TRUE
    assert row.exceeds_control is FlagValue.TRUE
    assert row.multi_strategy is FlagValue.TRUE
    assert row.ci_excludes_zero is FlagValue.TRUE
    assert row.seed_stable is FlagValue.TRUE
    assert row.jitter_stable is FlagValue.UNAVAILABLE
    assert row.reliability_grade is ReliabilityGrade.HIGH
    assert len(row.reliability_reasons) == 7


def test_opposite_signs_yields_unreliable_regardless_of_other_flags() -> None:
    anchor = _anchor()
    strategy_row = _strategy_row(anchor, {"constant_fill": 2.0, "blur": -3.0, "mean_fill": 1.5})
    control_row = _control_row(anchor, z_vs_control=5.0)
    interval_row = _interval_row(anchor.region_key, ci_low=1.0, ci_high=9.0)

    result = compute_reliability([control_row], [], [strategy_row], [interval_row])

    row = result[0]
    assert row.sign_consistent is FlagValue.FALSE
    assert "sign differs" in row.reliability_reasons[0]
    assert row.reliability_grade is ReliabilityGrade.UNRELIABLE


def test_target_equals_control_yields_excess_zero_and_exceeds_control_false() -> None:
    anchor = _anchor()
    strategy_row = _strategy_row(anchor, {"constant_fill": 1.0})
    control_row = _control_row(anchor, control_mean=1.0, excess=0.0, z_vs_control=0.0)

    result = compute_reliability([control_row], [], [strategy_row], [])

    row = result[0]
    assert row.exceeds_control is FlagValue.FALSE


def test_no_control_items_yields_unavailable_not_false_and_moderate_grade() -> None:
    anchor = _anchor()
    strategy_row = _strategy_row(anchor, {"constant_fill": 2.0, "blur": 3.0})
    interval_row = _interval_row(anchor.region_key, ci_low=1.0, ci_high=9.0)

    result = compute_reliability([], [], [strategy_row], [interval_row])

    row = result[0]
    assert row.exceeds_control is FlagValue.UNAVAILABLE
    assert row.area_matched is FlagValue.UNAVAILABLE
    assert "unavailable" in row.reliability_reasons[1]
    # unavailable != false: grade must not be dragged down to unreliable/low.
    assert row.reliability_grade is ReliabilityGrade.MODERATE


def test_single_condition_yields_insufficient_seed_stability_and_definite_multi_strategy_false() -> None:
    anchor = _anchor()
    strategy_row = _strategy_row(anchor, {"constant_fill": 2.0})
    seed_row = _seed_row(anchor, seed_cv=None, seed_std=None, n_seeds=1)

    result = compute_reliability([], [seed_row], [strategy_row], [])

    row = result[0]
    assert row.seed_stable is FlagValue.UNAVAILABLE
    assert row.multi_strategy is FlagValue.FALSE
    assert row.sign_consistent is FlagValue.TRUE


def test_area_mismatch_control_yields_area_matched_false() -> None:
    anchor = _anchor()
    strategy_row = _strategy_row(anchor, {"constant_fill": 2.0})
    control_row = _control_row(anchor, area_matched=FlagValue.FALSE)

    result = compute_reliability([control_row], [], [strategy_row], [])

    row = result[0]
    assert row.area_matched is FlagValue.FALSE


# --- grade boundary ----------------------------------------------------


def test_low_grade_when_effect_marginal_and_not_reproduced() -> None:
    anchor = _anchor()
    strategy_row = _strategy_row(anchor, {"constant_fill": 2.0})  # single op
    control_row = _control_row(anchor, z_vs_control=0.5)  # below default threshold 2.0

    result = compute_reliability([control_row], [], [strategy_row], [])

    row = result[0]
    assert row.exceeds_control is FlagValue.FALSE
    assert row.multi_strategy is FlagValue.FALSE
    assert row.reliability_grade is ReliabilityGrade.LOW


def test_z_vs_control_threshold_is_configurable() -> None:
    anchor = _anchor()
    strategy_row = _strategy_row(anchor, {"constant_fill": 2.0, "blur": 3.0})
    control_row = _control_row(anchor, z_vs_control=1.5)

    default_result = compute_reliability([control_row], [], [strategy_row], [])
    lenient_result = compute_reliability(
        [control_row], [], [strategy_row], [], z_vs_control_threshold=1.0
    )

    assert default_result[0].exceeds_control is FlagValue.FALSE
    assert lenient_result[0].exceeds_control is FlagValue.TRUE
