"""Tests for A4 StrategyProfiler (ssat/analysis/strategy_profile.py)."""

from __future__ import annotations

from collections import Counter

import pytest

from ssat.analysis.errors import AnalysisCorruptionError
from ssat.analysis.strategy_profile import compute_strategy_profile
from ssat.analysis.types import (
    Alignment,
    AnchorKey,
    RankCorrelationRow,
    StrategyStabilityRow,
)


def _anchor(region_key: str, sample_id: str = "s1") -> AnchorKey:
    return AnchorKey(sample_id=sample_id, region_key=region_key, invert_mask=False)


def _strategy_row(
    region_key: str, values_by_op: dict[str, float], *, metric_name: str = "margin_drop"
) -> StrategyStabilityRow:
    signs = {op: (1 if v > 0 else (-1 if v < 0 else 0)) for op, v in values_by_op.items()}
    n_strategies = len(values_by_op)
    _, top_count = Counter(signs.values()).most_common(1)[0]
    return StrategyStabilityRow(
        anchor_key=_anchor(region_key),
        metric_name=metric_name,
        strategy_signs=signs,
        strategy_values=dict(values_by_op),
        sign_agreement_ratio=top_count / n_strategies,
        n_strategies=n_strategies,
    )


def _rank_row(
    op_a: str,
    op_b: str,
    *,
    spearman: float | None,
    spearman_excl_top1: float | None = None,
    n_regions: int = 4,
    scope: str = "full_dataset",
) -> RankCorrelationRow:
    return RankCorrelationRow(
        op_a=op_a,
        op_b=op_b,
        spearman=spearman,
        n_regions=n_regions,
        spearman_excl_top1=spearman_excl_top1,
        scope=scope,
    )


def _row_for(rows: list, op: str):
    return next(row for row in rows if row.perturb_op == op)


# --- clustering --------------------------------------------------------


def test_strong_correlation_ops_cluster_together_weak_op_separate() -> None:
    strategy_rows = [
        _strategy_row("grid::r0", {"constant_fill": 1.0, "mean_fill": 1.0, "blur": 1.0}),
    ]
    rank_rows = [
        _rank_row("blur", "constant_fill", spearman=0.1, spearman_excl_top1=0.1),
        _rank_row("blur", "mean_fill", spearman=0.05, spearman_excl_top1=0.05),
        _rank_row("constant_fill", "mean_fill", spearman=0.8, spearman_excl_top1=0.8),
    ]

    result = compute_strategy_profile(strategy_rows, rank_rows)

    cf = _row_for(result, "constant_fill")
    mf = _row_for(result, "mean_fill")
    blur = _row_for(result, "blur")
    assert cf.cluster_id == mf.cluster_id
    assert blur.cluster_id != cf.cluster_id


def test_isolated_op_gets_own_cluster_with_none_within_corr() -> None:
    strategy_rows = [
        _strategy_row("grid::r0", {"constant_fill": 1.0, "mean_fill": 1.0, "blur": 1.0}),
    ]
    rank_rows = [
        _rank_row("blur", "constant_fill", spearman=0.2, spearman_excl_top1=0.2),
        _rank_row("blur", "mean_fill", spearman=0.1, spearman_excl_top1=0.1),
        _rank_row("constant_fill", "mean_fill", spearman=0.8, spearman_excl_top1=0.8),
    ]

    result = compute_strategy_profile(strategy_rows, rank_rows)

    blur = _row_for(result, "blur")
    cf = _row_for(result, "constant_fill")
    mf = _row_for(result, "mean_fill")
    assert cf.cluster_id == mf.cluster_id
    assert blur.cluster_id is not None
    assert blur.cluster_id != cf.cluster_id
    assert blur.mean_corr_within is None
    assert blur.mean_corr_across == pytest.approx((0.2 + 0.1) / 2)


def test_single_op_yields_none_cluster_and_correlations() -> None:
    strategy_rows = [_strategy_row("grid::r0", {"constant_fill": 1.0})]

    result = compute_strategy_profile(strategy_rows, [])

    row = result[0]
    assert row.cluster_id is None
    assert row.mean_corr_within is None
    assert row.mean_corr_across is None


# --- alignment -----------------------------------------------------------


def test_declared_and_empirical_fully_agree_yields_aligned() -> None:
    strategy_rows = [
        _strategy_row(
            "grid::r0",
            {"constant_fill": 1.0, "gaussian_noise": 0.5, "mean_fill": -1.0, "blur": -0.5},
        ),
    ]
    rank_rows = [
        _rank_row("blur", "constant_fill", spearman=0.1, spearman_excl_top1=0.1),
        _rank_row("blur", "gaussian_noise", spearman=0.05, spearman_excl_top1=0.05),
        _rank_row("blur", "mean_fill", spearman=0.9, spearman_excl_top1=0.9),
        _rank_row("constant_fill", "gaussian_noise", spearman=0.85, spearman_excl_top1=0.85),
        _rank_row("constant_fill", "mean_fill", spearman=0.0, spearman_excl_top1=0.0),
        _rank_row("gaussian_noise", "mean_fill", spearman=-0.2, spearman_excl_top1=-0.2),
    ]

    result = compute_strategy_profile(strategy_rows, rank_rows)

    assert all(row.alignment is Alignment.ALIGNED for row in result)


def test_declared_and_empirical_diverge_yields_divergent() -> None:
    # All three are declared preserves_statistics=True, but pairwise
    # correlations are all below threshold -- each stays its own singleton
    # empirical cluster, sharply diverging from the size-3 declared group.
    strategy_rows = [
        _strategy_row("grid::r0", {"mean_fill": 1.0, "blur": 1.0, "patch_shuffle": 1.0}),
    ]
    rank_rows = [
        _rank_row("blur", "mean_fill", spearman=0.1, spearman_excl_top1=0.1),
        _rank_row("blur", "patch_shuffle", spearman=0.05, spearman_excl_top1=0.05),
        _rank_row("mean_fill", "patch_shuffle", spearman=-0.1, spearman_excl_top1=-0.1),
    ]

    result = compute_strategy_profile(strategy_rows, rank_rows)

    assert all(row.alignment is Alignment.DIVERGENT for row in result)


def test_partial_overlap_between_declared_and_empirical_yields_partial() -> None:
    # Two declared True (mean_fill, blur), two declared False (constant_fill,
    # gaussian_noise), but all four end up in one empirical cluster -- each
    # op's declared group (size 2) is a strict subset of its empirical
    # cluster (size 4): Jaccard = 2/4 = 0.5.
    strategy_rows = [
        _strategy_row(
            "grid::r0",
            {"constant_fill": 1.0, "gaussian_noise": 1.0, "mean_fill": 1.0, "blur": 1.0},
        ),
    ]
    ops = ["blur", "constant_fill", "gaussian_noise", "mean_fill"]
    rank_rows = [
        _rank_row(ops[i], ops[j], spearman=0.9, spearman_excl_top1=0.9)
        for i in range(len(ops))
        for j in range(i + 1, len(ops))
    ]

    result = compute_strategy_profile(strategy_rows, rank_rows)

    assert all(row.alignment is Alignment.PARTIAL for row in result)


# --- sign-group summary ----------------------------------------------------


def test_mean_degradation_excl_top_and_sign_ratio_match_hand_computed() -> None:
    strategy_rows = [
        _strategy_row("grid::r0", {"constant_fill": 100.0}),
        _strategy_row("grid::r1", {"constant_fill": 1.0}),
        _strategy_row("grid::r2", {"constant_fill": 2.0}),
        _strategy_row("grid::r3", {"constant_fill": -3.0}),
    ]

    result = compute_strategy_profile(strategy_rows, [])

    row = _row_for(result, "constant_fill")
    assert row.n_anchors == 4
    assert row.sign_ratio_positive == pytest.approx(0.75)
    # Excludes the top-1 value (100.0); remaining [2.0, 1.0, -3.0] mean = 0.0.
    assert row.mean_degradation_excl_top == pytest.approx(0.0)


# --- declarative table -----------------------------------------------------


def test_declared_attribute_table_values_for_all_five_ops() -> None:
    ops = ["constant_fill", "mean_fill", "blur", "gaussian_noise", "patch_shuffle"]
    strategy_rows = [_strategy_row("grid::r0", {op: 1.0 for op in ops})]
    expected = {
        "constant_fill": (False, False, False),
        "mean_fill": (True, False, False),
        "blur": (True, False, True),
        "gaussian_noise": (False, False, False),
        "patch_shuffle": (True, True, True),
    }

    result = compute_strategy_profile(strategy_rows, [])

    assert len(result) == 5
    for row in result:
        assert (
            row.preserves_statistics,
            row.preserves_local_texture,
            row.is_global_operation,
        ) == expected[row.perturb_op]


def test_unknown_perturb_op_raises_corruption_error() -> None:
    strategy_rows = [_strategy_row("grid::r0", {"not_a_real_op": 1.0})]

    with pytest.raises(AnalysisCorruptionError):
        compute_strategy_profile(strategy_rows, [])
