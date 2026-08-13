"""Tests for A2 ControlComparator (ssat/analysis/control.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ssat.analysis.control import compare_to_controls
from ssat.analysis.types import AnchorKey, ConditionKey, ControlPairRow, FlagValue, MatchMethod
from ssat.utils.io import sha256_bytes

_TARGET_ANCHOR = AnchorKey(sample_id="s1", region_key="grid::grid/r0/c0", invert_mask=False)
_CONDITION = ConditionKey(perturb_op="constant_fill", perturb_params_hash=sha256_bytes(b"{}"))


def _item_value_row(
    *,
    sample_id: str,
    region_id: str,
    region_instance_id: str,
    is_control: bool,
    degradation: float | None,
    available: bool = True,
    metric_name: str = "margin_drop",
    perturb_op: str = "constant_fill",
    perturb_params_json: str = "{}",
    invert_mask: bool = False,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "region_id": region_id,
        "region_instance_id": region_instance_id,
        "invert_mask": invert_mask,
        "perturb_op": perturb_op,
        "perturb_params_json": perturb_params_json,
        "is_control": is_control,
        "metric_name": metric_name,
        "degradation": degradation,
        "available": available,
    }


def _item_values(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _target_row(degradation: float, *, available: bool = True) -> dict[str, object]:
    return _item_value_row(
        sample_id="s1",
        region_id="grid",
        region_instance_id="grid/r0/c0",
        is_control=False,
        degradation=degradation,
        available=available,
    )


def _control_anchor(index: int) -> AnchorKey:
    return AnchorKey(
        sample_id="s1",
        region_key=f"control:grid:0::control:grid/r0/c0:0:{index}",
        invert_mask=False,
    )


def _control_row(index: int, degradation: float | None, *, available: bool = True) -> dict[str, object]:
    return _item_value_row(
        sample_id="s1",
        region_id="control:grid:0",
        region_instance_id=f"control:grid/r0/c0:0:{index}",
        is_control=True,
        degradation=degradation,
        available=available,
    )


def _control_pair(
    index: int,
    *,
    match_method: MatchMethod = MatchMethod.EXACT_REFERENCE,
    area_match_ratio: float | None = 1.0,
) -> ControlPairRow:
    return ControlPairRow(
        target_anchor_key=_TARGET_ANCHOR,
        control_anchor_key=_control_anchor(index),
        condition_key=_CONDITION,
        match_method=match_method,
        area_match_ratio=area_match_ratio,
    )


def test_excess_ratio_z_match_hand_computed_values() -> None:
    target_value = 1.0
    control_values = [0.2, 0.3, 0.4]
    rows = [_target_row(target_value)]
    rows.extend(_control_row(i, value) for i, value in enumerate(control_values))
    control_pairs = [_control_pair(i) for i in range(len(control_values))]

    result = compare_to_controls(_item_values(rows), control_pairs)

    assert len(result) == 1
    row = result[0]
    expected_mean = float(np.mean(control_values))
    expected_std = float(np.std(control_values))
    assert row.target_anchor_key == _TARGET_ANCHOR
    assert row.condition_key == _CONDITION
    assert row.control_available is FlagValue.TRUE
    assert row.n_controls == 3
    assert row.control_mean == pytest.approx(expected_mean)
    assert row.control_std == pytest.approx(expected_std)
    assert row.excess == pytest.approx(target_value - expected_mean)
    assert row.ratio == pytest.approx(target_value / expected_mean)
    assert row.z_vs_control == pytest.approx((target_value - expected_mean) / expected_std)
    assert row.area_matched is FlagValue.TRUE


def test_zero_control_std_yields_none_z() -> None:
    rows = [_target_row(1.0), _control_row(0, 0.5), _control_row(1, 0.5), _control_row(2, 0.5)]
    control_pairs = [_control_pair(i) for i in range(3)]

    result = compare_to_controls(_item_values(rows), control_pairs)

    row = result[0]
    assert row.control_std == pytest.approx(0.0)
    assert row.z_vs_control is None
    assert row.control_mean == pytest.approx(0.5)
    assert row.excess == pytest.approx(0.5)
    assert row.ratio == pytest.approx(2.0)


def test_near_zero_control_mean_yields_none_ratio() -> None:
    rows = [_target_row(1.0), _control_row(0, 1e-8), _control_row(1, -1e-8)]
    control_pairs = [_control_pair(0), _control_pair(1)]

    result = compare_to_controls(_item_values(rows), control_pairs)

    row = result[0]
    assert row.ratio is None
    assert row.excess == pytest.approx(1.0)


def test_target_equal_to_control_mean_yields_zero_excess() -> None:
    rows = [_target_row(0.5), _control_row(0, 0.4), _control_row(1, 0.5), _control_row(2, 0.6)]
    control_pairs = [_control_pair(i) for i in range(3)]

    result = compare_to_controls(_item_values(rows), control_pairs)

    row = result[0]
    assert row.excess == pytest.approx(0.0)


def test_no_matched_controls_yields_unavailable() -> None:
    rows = [_target_row(1.0)]

    result = compare_to_controls(_item_values(rows), control_pairs=[])

    assert len(result) == 1
    row = result[0]
    assert row.control_available is FlagValue.UNAVAILABLE
    assert row.area_matched is FlagValue.UNAVAILABLE
    assert row.control_mean is None
    assert row.control_std is None
    assert row.n_controls == 0
    assert row.excess is None
    assert row.ratio is None
    assert row.z_vs_control is None


def test_matched_control_without_metric_value_is_unavailable() -> None:
    # A1 matched this control structurally, but its item never produced a
    # usable value for this metric (perturbation failure) -- availability
    # tracks values, not the structural match.
    rows = [_target_row(1.0), _control_row(0, None, available=False)]
    control_pairs = [_control_pair(0)]

    result = compare_to_controls(_item_values(rows), control_pairs)

    row = result[0]
    assert row.control_available is FlagValue.UNAVAILABLE
    assert row.area_matched is FlagValue.UNAVAILABLE
    assert row.n_controls == 0


def test_area_mismatch_flags_area_matched_false() -> None:
    rows = [_target_row(1.0), _control_row(0, 0.5)]
    control_pairs = [_control_pair(0, area_match_ratio=2.0)]

    result = compare_to_controls(_item_values(rows), control_pairs)

    row = result[0]
    assert row.control_available is FlagValue.TRUE
    assert row.area_matched is FlagValue.FALSE
    assert row.control_mean == pytest.approx(0.5)


def test_single_control_below_n_controls_threshold_yields_none_z() -> None:
    rows = [_target_row(1.0), _control_row(0, 0.5)]
    control_pairs = [_control_pair(0)]

    result = compare_to_controls(_item_values(rows), control_pairs)

    row = result[0]
    assert row.n_controls == 1
    assert row.z_vs_control is None
