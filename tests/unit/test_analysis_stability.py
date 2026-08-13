"""Tests for A3 StabilityAnalyzer (ssat/analysis/stability.py)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ssat.analysis.stability import (
    compute_jitter_stability,
    compute_seed_stability,
    compute_strategy_stability,
)
from ssat.analysis.types import AnchorKey, ConditionKey, FlagValue
from ssat.utils.io import sha256_bytes


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


def _region_row(
    sample_id: str, region_instance_id: str, perturb_op: str, degradation: float
) -> dict[str, object]:
    return _item_value_row(
        sample_id=sample_id,
        region_instance_id=region_instance_id,
        perturb_op=perturb_op,
        degradation=degradation,
    )


_ANCHOR = AnchorKey(sample_id="s1", region_key="grid::grid/r0/c0", invert_mask=False)
_CONDITION = ConditionKey(perturb_op="constant_fill", perturb_params_hash=sha256_bytes(b"{}"))


# --- seed stability ----------------------------------------------------


def test_seed_mean_std_cv_match_hand_computed_values() -> None:
    values = [1.0, 2.0, 3.0]
    rows = [_item_value_row(sample_id="s1", degradation=v) for v in values]

    result = compute_seed_stability(_item_values(rows))

    assert len(result) == 1
    row = result[0]
    expected_mean = float(np.mean(values))
    expected_std = float(np.std(values))
    assert row.anchor_key == _ANCHOR
    assert row.condition_key == _CONDITION
    assert row.n_seeds == 3
    assert row.seed_mean == pytest.approx(expected_mean)
    assert row.seed_std == pytest.approx(expected_std)
    assert row.seed_cv == pytest.approx(expected_std / expected_mean)


def test_single_seed_yields_none_std_and_cv() -> None:
    rows = [_item_value_row(sample_id="s1", degradation=1.5)]

    result = compute_seed_stability(_item_values(rows))

    row = result[0]
    assert row.n_seeds == 1
    assert row.seed_mean == pytest.approx(1.5)
    assert row.seed_std is None
    assert row.seed_cv is None


def test_near_zero_seed_mean_yields_none_cv() -> None:
    rows = [
        _item_value_row(sample_id="s1", degradation=1e-8),
        _item_value_row(sample_id="s1", degradation=-1e-8),
    ]

    result = compute_seed_stability(_item_values(rows))

    row = result[0]
    assert row.seed_cv is None
    assert row.seed_std is not None


def test_control_anchor_included_in_seed_stability() -> None:
    rows = [
        _item_value_row(
            sample_id="s1",
            region_id="control:grid:0",
            region_instance_id="control:grid/r0/c0:0:0",
            is_control=True,
            degradation=0.4,
        ),
        _item_value_row(
            sample_id="s1",
            region_id="control:grid:0",
            region_instance_id="control:grid/r0/c0:0:0",
            is_control=True,
            degradation=0.6,
        ),
    ]

    result = compute_seed_stability(_item_values(rows))

    assert len(result) == 1
    assert result[0].n_seeds == 2
    assert result[0].anchor_key.region_key == "control:grid:0::control:grid/r0/c0:0:0"


def test_unavailable_items_excluded_from_seed_values() -> None:
    rows = [
        _item_value_row(sample_id="s1", degradation=1.0),
        _item_value_row(sample_id="s1", degradation=None, available=False),
    ]

    result = compute_seed_stability(_item_values(rows))

    assert len(result) == 1
    assert result[0].n_seeds == 1
    assert result[0].seed_mean == pytest.approx(1.0)


def test_rows_grouped_separately_per_condition_key() -> None:
    rows = [
        _item_value_row(sample_id="s1", degradation=1.0, perturb_op="constant_fill"),
        _item_value_row(sample_id="s1", degradation=2.0, perturb_op="blur"),
    ]

    result = compute_seed_stability(_item_values(rows))

    assert len(result) == 2
    ops = {row.condition_key.perturb_op for row in result}
    assert ops == {"constant_fill", "blur"}


# --- jitter stability (interface stub) ------------------------------------


def test_jitter_stability_always_unavailable() -> None:
    assert compute_jitter_stability(_item_values([])) is FlagValue.UNAVAILABLE
    non_empty = _item_values([_item_value_row(sample_id="s1", degradation=1.0)])
    assert compute_jitter_stability(non_empty) is FlagValue.UNAVAILABLE


# --- fill strategy stability: per-anchor -----------------------------------


def test_per_anchor_signs_values_and_agreement_ratio_match_hand_computed() -> None:
    rows = [
        _item_value_row(sample_id="s1", degradation=1.0, perturb_op="constant_fill"),
        _item_value_row(sample_id="s1", degradation=1.0, perturb_op="blur"),
        _item_value_row(sample_id="s1", degradation=-1.0, perturb_op="gaussian_noise"),
    ]

    strategy_rows, _ = compute_strategy_stability(_item_values(rows))

    assert len(strategy_rows) == 1
    row = strategy_rows[0]
    assert row.n_strategies == 3
    assert row.strategy_signs == {"constant_fill": 1, "blur": 1, "gaussian_noise": -1}
    assert row.strategy_values == pytest.approx(
        {"constant_fill": 1.0, "blur": 1.0, "gaussian_noise": -1.0}
    )
    assert row.sign_agreement_ratio == pytest.approx(2 / 3)


def test_control_anchors_excluded_from_per_anchor_rows() -> None:
    rows = [
        _item_value_row(
            sample_id="s1",
            region_id="control:grid:0",
            region_instance_id="control:grid/r0/c0:0:0",
            is_control=True,
            degradation=1.0,
        ),
    ]

    strategy_rows, _ = compute_strategy_stability(_item_values(rows))

    assert strategy_rows == []


def test_multiple_condition_keys_same_op_are_averaged_before_sign() -> None:
    rows = [
        _item_value_row(
            sample_id="s1",
            degradation=1.0,
            perturb_op="constant_fill",
            perturb_params_json=json.dumps({"a": 1}),
        ),
        _item_value_row(
            sample_id="s1",
            degradation=3.0,
            perturb_op="constant_fill",
            perturb_params_json=json.dumps({"a": 2}),
        ),
    ]

    strategy_rows, _ = compute_strategy_stability(_item_values(rows))

    assert len(strategy_rows) == 1
    assert strategy_rows[0].strategy_values == pytest.approx({"constant_fill": 2.0})


# --- fill strategy stability: dataset-level rank correlation ---------------


def test_rank_correlation_hand_computed_matches_pandas_rank_corr() -> None:
    regions = ["grid/r0/c0", "grid/r0/c1", "grid/r0/c2", "grid/r0/c3"]
    values_a = [1.0, 2.0, 3.0, 4.0]  # constant_fill: increasing
    values_b = [4.0, 3.0, 2.0, 1.0]  # blur: exactly reversed
    rows = [
        _region_row("s1", region, "constant_fill", value)
        for region, value in zip(regions, values_a)
    ] + [_region_row("s1", region, "blur", value) for region, value in zip(regions, values_b)]

    _, rank_rows = compute_strategy_stability(_item_values(rows))

    assert len(rank_rows) == 1
    row = rank_rows[0]
    assert row.op_a == "blur"  # alphabetical: "blur" < "constant_fill"
    assert row.op_b == "constant_fill"
    assert row.n_regions == 4
    expected = (
        pd.Series(values_b).rank().corr(pd.Series(values_a).rank())
    )  # blur (op_a) vs constant_fill (op_b), same region order
    assert row.spearman == pytest.approx(expected)
    assert row.spearman == pytest.approx(-1.0)


def test_rank_correlation_excl_top1_differs_from_full_when_dominant_region_present() -> None:
    # r0 is the dominant "patch"-like region: enormous and rank-1 in both
    # ops, mechanically pulling the full correlation toward +1. Among the
    # other three regions the ranking is exactly reversed between ops.
    rows = [
        _region_row("s1", "grid/r0/c0", "constant_fill", 1000.0),
        _region_row("s1", "grid/r0/c1", "constant_fill", 1.0),
        _region_row("s1", "grid/r0/c2", "constant_fill", 2.0),
        _region_row("s1", "grid/r0/c3", "constant_fill", 3.0),
        _region_row("s1", "grid/r0/c0", "mean_fill", 1000.0),
        _region_row("s1", "grid/r0/c1", "mean_fill", 30.0),
        _region_row("s1", "grid/r0/c2", "mean_fill", 20.0),
        _region_row("s1", "grid/r0/c3", "mean_fill", 10.0),
    ]

    _, rank_rows = compute_strategy_stability(_item_values(rows))

    assert len(rank_rows) == 1
    row = rank_rows[0]
    assert row.op_a == "constant_fill"  # alphabetical: "constant_fill" < "mean_fill"
    assert row.op_b == "mean_fill"
    assert row.n_regions == 4
    assert row.spearman == pytest.approx(0.2)
    assert row.spearman_excl_top1 == pytest.approx(-1.0)
    assert row.spearman != pytest.approx(row.spearman_excl_top1)


def test_fewer_than_two_shared_regions_yields_none_spearman() -> None:
    rows = [
        _region_row("s1", "grid/r0/c0", "constant_fill", 1.0),
        _region_row("s1", "grid/r0/c0", "blur", 2.0),
    ]

    _, rank_rows = compute_strategy_stability(_item_values(rows))

    assert len(rank_rows) == 1
    row = rank_rows[0]
    assert row.n_regions == 1
    assert row.spearman is None
    assert row.spearman_excl_top1 is None


def test_rank_correlation_pairs_are_deduplicated_and_ordered() -> None:
    rows = [
        _region_row("s1", "grid/r0/c0", op, 1.0) for op in ("constant_fill", "blur", "mean_fill")
    ]

    _, rank_rows = compute_strategy_stability(_item_values(rows))

    pairs = [(row.op_a, row.op_b) for row in rank_rows]
    assert pairs == [
        ("blur", "constant_fill"),
        ("blur", "mean_fill"),
        ("constant_fill", "mean_fill"),
    ]
