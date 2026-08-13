"""Tests for A5 IntervalEstimator (ssat/analysis/interval.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ssat.analysis.interval import CI_METHOD, compute_intervals


def _item_value_row(
    *,
    sample_id: str,
    region_id: str = "grid",
    region_instance_id: str = "grid/r0/c0",
    is_control: bool = False,
    degradation: float | None,
    available: bool = True,
    metric_name: str = "margin_drop",
    perturb_op: str = "constant_fill",
    invert_mask: bool = False,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "region_id": region_id,
        "region_instance_id": region_instance_id,
        "invert_mask": invert_mask,
        "perturb_op": perturb_op,
        "is_control": is_control,
        "metric_name": metric_name,
        "degradation": degradation,
        "available": available,
    }


def _item_values(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _sample_rows(values: list[float]) -> list[dict[str, object]]:
    return [
        _item_value_row(sample_id=f"s{i}", degradation=v) for i, v in enumerate(values)
    ]


def _row_for(rows: list, region_key: str, metric: str):
    return next(row for row in rows if row.region_key == region_key and row.metric == metric)


# --- CI computation ----------------------------------------------------


def test_ci_matches_hand_computed_percentiles_with_fixed_seed() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    rows = _sample_rows(values)

    result = compute_intervals(_item_values(rows), random_seed=42, n_bootstrap=200)

    assert len(result) == 1
    row = result[0]
    assert row.region_key == "grid::grid/r0/c0"
    assert row.metric == "margin_drop"
    assert row.point_estimate == pytest.approx(float(np.mean(values)))

    expected_rng = np.random.default_rng(42)
    expected_resamples = expected_rng.choice(
        np.asarray(values), size=(200, 5), replace=True
    ).mean(axis=1)
    expected_ci_low, expected_ci_high = np.percentile(expected_resamples, [2.5, 97.5])

    assert row.ci_low == pytest.approx(expected_ci_low)
    assert row.ci_high == pytest.approx(expected_ci_high)
    assert row.ci_method == CI_METHOD
    assert row.n_bootstrap == 200


def test_all_zero_values_yield_zero_including_ci() -> None:
    rows = _sample_rows([0.0, 0.0, 0.0])

    result = compute_intervals(_item_values(rows), n_bootstrap=100)

    row = result[0]
    assert row.point_estimate == pytest.approx(0.0)
    assert row.ci_low == pytest.approx(0.0)
    assert row.ci_high == pytest.approx(0.0)
    assert row.excludes_zero is False


def test_clearly_nonzero_values_yield_excludes_zero_true() -> None:
    rows = _sample_rows([5.0, 5.0, 5.0, 5.0])

    result = compute_intervals(_item_values(rows), n_bootstrap=100)

    row = result[0]
    assert row.ci_low == pytest.approx(5.0)
    assert row.ci_high == pytest.approx(5.0)
    assert row.excludes_zero is True


def test_larger_n_bootstrap_yields_more_stable_ci_across_seeds() -> None:
    values = [1.0, 3.0, 2.0, 5.0, 4.0, 8.0, -1.0, 6.0, 0.0, 7.0, 2.5, 3.5, -2.0, 9.0, 1.5]
    rows = _sample_rows(values)
    seeds = [1, 2, 3, 4, 5]

    small_lows = [
        compute_intervals(_item_values(rows), n_bootstrap=20, random_seed=seed)[0].ci_low
        for seed in seeds
    ]
    large_lows = [
        compute_intervals(_item_values(rows), n_bootstrap=5000, random_seed=seed)[0].ci_low
        for seed in seeds
    ]

    small_range = max(small_lows) - min(small_lows)
    large_range = max(large_lows) - min(large_lows)
    assert large_range < small_range


# --- population scope: control exclusion, pooling, availability -----------


def test_control_items_excluded_from_point_estimate() -> None:
    rows = _sample_rows([10.0, 10.0]) + [
        _item_value_row(
            sample_id="s0",
            region_id="control:grid:0",
            region_instance_id="control:grid/r0/c0:0:0",
            is_control=True,
            degradation=-1000.0,
        )
    ]

    result = compute_intervals(_item_values(rows), n_bootstrap=50)

    row = _row_for(result, "grid::grid/r0/c0", "margin_drop")
    assert row.point_estimate == pytest.approx(10.0)


def test_multiple_perturb_ops_and_invert_mask_pooled_into_one_sample_value() -> None:
    rows = [
        _item_value_row(sample_id="s0", degradation=2.0, perturb_op="constant_fill", invert_mask=False),
        _item_value_row(sample_id="s0", degradation=6.0, perturb_op="blur", invert_mask=True),
    ]

    result = compute_intervals(_item_values(rows), n_bootstrap=50)

    row = result[0]
    # Both items belong to the same (sample, region, metric); stage 1 must
    # collapse them to one per-sample value (their mean) before bootstrap.
    assert row.point_estimate == pytest.approx(4.0)
    assert row.ci_low == pytest.approx(4.0)
    assert row.ci_high == pytest.approx(4.0)


def test_unavailable_items_excluded() -> None:
    rows = [
        _item_value_row(sample_id="s0", degradation=1.0),
        _item_value_row(sample_id="s0", degradation=None, available=False),
    ]

    result = compute_intervals(_item_values(rows), n_bootstrap=50)

    row = result[0]
    assert row.point_estimate == pytest.approx(1.0)


def test_no_data_for_metric_yields_no_row() -> None:
    rows = _sample_rows([1.0, 2.0])

    result = compute_intervals(_item_values(rows), metric_names=["some_other_metric"])

    assert result == []


def test_n_bootstrap_and_ci_method_recorded_on_row() -> None:
    rows = _sample_rows([1.0, 2.0, 3.0])

    result = compute_intervals(_item_values(rows), n_bootstrap=77)

    row = result[0]
    assert row.n_bootstrap == 77
    assert row.ci_method == "percentile"
