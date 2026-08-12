"""Validation tests for control/stability analysis contract types."""

from __future__ import annotations

import pytest

from ssat.analysis.types import (
    Alignment,
    AnchorKey,
    AnchorRow,
    ConditionKey,
    ControlComparisonRow,
    ControlPairRow,
    CoverageReport,
    FlagValue,
    IntervalRow,
    MatchMethod,
    RankCorrelationRow,
    ReliabilityGrade,
    ReliabilityRow,
    SeedStabilityRow,
    StrategyProfileRow,
    StrategyStabilityRow,
)


def _anchor_key(**overrides: object) -> AnchorKey:
    defaults: dict[str, object] = {
        "sample_id": "sample-1",
        "region_key": "grid::0",
        "invert_mask": False,
    }
    defaults.update(overrides)
    return AnchorKey(**defaults)  # type: ignore[arg-type]


def _condition_key(**overrides: object) -> ConditionKey:
    defaults: dict[str, object] = {
        "perturb_op": "constant_fill",
        "perturb_params_hash": "a" * 64,
    }
    defaults.update(overrides)
    return ConditionKey(**defaults)  # type: ignore[arg-type]


def _anchor_row(**overrides: object) -> AnchorRow:
    defaults: dict[str, object] = {
        "anchor_key": _anchor_key(),
        "sample_id": "sample-1",
        "region_key": "grid::0",
        "invert_mask": False,
        "intended_area_px": 64,
        "effective_area_px": 60,
        "is_control": False,
        "n_conditions": 1,
        "condition_keys": (_condition_key(),),
    }
    defaults.update(overrides)
    return AnchorRow(**defaults)  # type: ignore[arg-type]


def _control_pair_row(**overrides: object) -> ControlPairRow:
    defaults: dict[str, object] = {
        "target_anchor_key": _anchor_key(),
        "control_anchor_key": _anchor_key(region_key="control::0"),
        "condition_key": _condition_key(),
        "match_method": MatchMethod.EXACT_REFERENCE,
        "area_match_ratio": 1.0,
    }
    defaults.update(overrides)
    return ControlPairRow(**defaults)  # type: ignore[arg-type]


def _control_comparison_row(**overrides: object) -> ControlComparisonRow:
    defaults: dict[str, object] = {
        "target_anchor_key": _anchor_key(),
        "condition_key": _condition_key(),
        "metric_name": "margin_drop",
        "control_available": FlagValue.TRUE,
        "area_matched": FlagValue.TRUE,
        "control_mean": 0.1,
        "control_std": 0.05,
        "n_controls": 5,
        "excess": 0.2,
        "ratio": 2.0,
        "z_vs_control": 1.5,
    }
    defaults.update(overrides)
    return ControlComparisonRow(**defaults)  # type: ignore[arg-type]


def _seed_stability_row(**overrides: object) -> SeedStabilityRow:
    defaults: dict[str, object] = {
        "anchor_key": _anchor_key(),
        "condition_key": _condition_key(),
        "metric_name": "margin_drop",
        "seed_mean": 0.1,
        "seed_std": 0.02,
        "seed_cv": 0.2,
        "n_seeds": 3,
    }
    defaults.update(overrides)
    return SeedStabilityRow(**defaults)  # type: ignore[arg-type]


def _strategy_stability_row(**overrides: object) -> StrategyStabilityRow:
    defaults: dict[str, object] = {
        "anchor_key": _anchor_key(),
        "metric_name": "margin_drop",
        "strategy_signs": {"constant_fill": 1, "blur": -1},
        "strategy_values": {"constant_fill": 0.3, "blur": -0.1},
        "sign_agreement_ratio": 0.5,
        "n_strategies": 2,
    }
    defaults.update(overrides)
    return StrategyStabilityRow(**defaults)  # type: ignore[arg-type]


def _rank_correlation_row(**overrides: object) -> RankCorrelationRow:
    defaults: dict[str, object] = {
        "op_a": "constant_fill",
        "op_b": "blur",
        "spearman": 0.5,
        "n_regions": 10,
        "spearman_excl_top1": 0.3,
        "scope": "full_dataset",
    }
    defaults.update(overrides)
    return RankCorrelationRow(**defaults)  # type: ignore[arg-type]


def _strategy_profile_row(**overrides: object) -> StrategyProfileRow:
    defaults: dict[str, object] = {
        "perturb_op": "constant_fill",
        "preserves_statistics": False,
        "preserves_local_texture": False,
        "is_global_operation": False,
        "cluster_id": 0,
        "mean_corr_within": 0.8,
        "mean_corr_across": 0.1,
        "alignment": Alignment.ALIGNED,
        "mean_degradation_excl_top": 0.2,
        "sign_ratio_positive": 0.7,
        "n_anchors": 15,
    }
    defaults.update(overrides)
    return StrategyProfileRow(**defaults)  # type: ignore[arg-type]


def _interval_row(**overrides: object) -> IntervalRow:
    defaults: dict[str, object] = {
        "region_key": "grid::0",
        "metric": "margin_drop",
        "point_estimate": 0.2,
        "ci_low": 0.1,
        "ci_high": 0.3,
        "ci_method": "percentile",
        "n_bootstrap": 1000,
        "excludes_zero": True,
    }
    defaults.update(overrides)
    return IntervalRow(**defaults)  # type: ignore[arg-type]


def _reliability_row(**overrides: object) -> ReliabilityRow:
    defaults: dict[str, object] = {
        "anchor_key": _anchor_key(),
        "metric_name": "margin_drop",
        "sign_consistent": FlagValue.TRUE,
        "exceeds_control": FlagValue.TRUE,
        "seed_stable": FlagValue.TRUE,
        "jitter_stable": FlagValue.UNAVAILABLE,
        "multi_strategy": FlagValue.TRUE,
        "ci_excludes_zero": FlagValue.TRUE,
        "area_matched": FlagValue.TRUE,
        "reliability_grade": ReliabilityGrade.HIGH,
        "reliability_reasons": ("all conditions agree",),
    }
    defaults.update(overrides)
    return ReliabilityRow(**defaults)  # type: ignore[arg-type]


def _coverage_report(**overrides: object) -> CoverageReport:
    defaults: dict[str, object] = {
        "n_anchors": 10,
        "n_conditions_insufficient": 2,
        "n_controls_unmatched": 1,
        "n_area_mismatch_warnings": 0,
    }
    defaults.update(overrides)
    return CoverageReport(**defaults)  # type: ignore[arg-type]


# --- AnchorKey -------------------------------------------------------------


def test_anchor_key_accepts_valid_fields() -> None:
    key = _anchor_key()
    assert key.sample_id == "sample-1"
    assert key.invert_mask is False


def test_anchor_key_rejects_empty_sample_id() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _anchor_key(sample_id="")


def test_anchor_key_rejects_empty_region_key() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _anchor_key(region_key="")


def test_anchor_key_is_hashable_and_supports_equality() -> None:
    a = _anchor_key()
    b = _anchor_key()
    assert hash(a) == hash(b)
    assert a == b
    assert {a: 1}[b] == 1


# --- ConditionKey ------------------------------------------------------------


def test_condition_key_accepts_valid_fields() -> None:
    key = _condition_key()
    assert key.perturb_op == "constant_fill"


def test_condition_key_rejects_empty_perturb_op() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _condition_key(perturb_op="")


def test_condition_key_rejects_empty_perturb_params_hash() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _condition_key(perturb_params_hash="")


def test_condition_key_is_hashable_and_supports_equality() -> None:
    a = _condition_key()
    b = _condition_key()
    assert hash(a) == hash(b)
    assert a == b
    assert {a: 1}[b] == 1


# --- FlagValue -----------------------------------------------------------


def test_flag_value_unavailable_is_not_falsy() -> None:
    assert bool(FlagValue.UNAVAILABLE) is True
    assert FlagValue.UNAVAILABLE is not False


def test_flag_value_false_is_also_not_falsy() -> None:
    # The whole enum is unusable with `if flag:` — FALSE must not collapse
    # to Python False either, or callers could mistake it for UNAVAILABLE.
    assert bool(FlagValue.FALSE) is True


def test_flag_value_members_are_distinct() -> None:
    assert len({FlagValue.TRUE, FlagValue.FALSE, FlagValue.UNAVAILABLE}) == 3


# --- ReliabilityGrade --------------------------------------------------------


def test_reliability_grade_has_expected_members() -> None:
    assert {grade.value for grade in ReliabilityGrade} == {
        "high",
        "moderate",
        "low",
        "unreliable",
    }


# --- AnchorRow ---------------------------------------------------------------


def test_anchor_row_accepts_valid_row() -> None:
    row = _anchor_row()
    assert row.n_conditions == 1


def test_anchor_row_rejects_non_anchor_key_type() -> None:
    with pytest.raises(TypeError, match="AnchorKey"):
        _anchor_row(anchor_key="grid::0")


def test_anchor_row_rejects_anchor_key_field_mismatch() -> None:
    with pytest.raises(ValueError, match="must match anchor_key"):
        _anchor_row(sample_id="sample-2")


def test_anchor_row_rejects_negative_intended_area() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _anchor_row(intended_area_px=-1)


def test_anchor_row_rejects_negative_n_conditions() -> None:
    with pytest.raises(ValueError, match="n_conditions must be non-negative"):
        _anchor_row(n_conditions=-1, condition_keys=())


def test_anchor_row_rejects_non_condition_key_element() -> None:
    with pytest.raises(TypeError, match="ConditionKey"):
        _anchor_row(condition_keys=("not-a-condition-key",))


def test_anchor_row_rejects_n_conditions_length_mismatch() -> None:
    with pytest.raises(ValueError, match="n_conditions must equal"):
        _anchor_row(n_conditions=2, condition_keys=(_condition_key(),))


# --- ControlPairRow ------------------------------------------------------


def test_control_pair_row_accepts_exact_reference_match() -> None:
    row = _control_pair_row(match_method=MatchMethod.EXACT_REFERENCE)
    assert row.match_method is MatchMethod.EXACT_REFERENCE


def test_control_pair_row_accepts_area_tolerance_match() -> None:
    row = _control_pair_row(match_method=MatchMethod.AREA_TOLERANCE)
    assert row.match_method is MatchMethod.AREA_TOLERANCE


def test_control_pair_row_rejects_target_equal_control() -> None:
    with pytest.raises(ValueError, match="must differ"):
        _control_pair_row(control_anchor_key=_anchor_key())


def test_control_pair_row_rejects_negative_area_match_ratio() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _control_pair_row(area_match_ratio=-0.1)


def test_control_pair_row_rejects_non_anchor_key_target() -> None:
    with pytest.raises(TypeError, match="AnchorKey"):
        _control_pair_row(target_anchor_key="sample-1::grid::0")


def test_control_pair_row_rejects_non_match_method() -> None:
    with pytest.raises(TypeError, match="MatchMethod"):
        _control_pair_row(match_method="exact_reference")


# --- ControlComparisonRow -----------------------------------------------


def test_control_comparison_row_accepts_available_row() -> None:
    row = _control_comparison_row()
    assert row.control_available is FlagValue.TRUE


def test_control_comparison_row_accepts_unavailable_row_with_null_stats() -> None:
    row = _control_comparison_row(
        control_available=FlagValue.UNAVAILABLE,
        area_matched=FlagValue.UNAVAILABLE,
        control_mean=None,
        control_std=None,
        n_controls=0,
        excess=None,
        ratio=None,
        z_vs_control=None,
    )
    assert row.control_mean is None


def test_control_comparison_row_rejects_unavailable_row_with_stats_present() -> None:
    with pytest.raises(ValueError, match="must have null control statistics"):
        _control_comparison_row(
            control_available=FlagValue.UNAVAILABLE,
            area_matched=FlagValue.UNAVAILABLE,
            n_controls=0,
        )


def test_control_comparison_row_rejects_unavailable_row_with_nonzero_n_controls() -> None:
    with pytest.raises(ValueError, match="must have null control statistics"):
        _control_comparison_row(
            control_available=FlagValue.UNAVAILABLE,
            area_matched=FlagValue.UNAVAILABLE,
            control_mean=None,
            control_std=None,
            excess=None,
            ratio=None,
            z_vs_control=None,
        )


def test_control_comparison_row_rejects_unavailable_row_with_area_matched_true() -> None:
    with pytest.raises(ValueError, match="must have null control statistics"):
        _control_comparison_row(
            control_available=FlagValue.UNAVAILABLE,
            area_matched=FlagValue.TRUE,
            control_mean=None,
            control_std=None,
            n_controls=0,
            excess=None,
            ratio=None,
            z_vs_control=None,
        )


def test_control_comparison_row_rejects_negative_n_controls() -> None:
    with pytest.raises(ValueError, match="n_controls must be non-negative"):
        _control_comparison_row(n_controls=-1)


def test_control_comparison_row_rejects_negative_control_std() -> None:
    with pytest.raises(ValueError, match="control_std must be non-negative"):
        _control_comparison_row(control_std=-0.1)


def test_control_comparison_row_rejects_empty_metric_name() -> None:
    with pytest.raises(ValueError, match="metric_name must not be empty"):
        _control_comparison_row(metric_name="")


# --- SeedStabilityRow ----------------------------------------------------


def test_seed_stability_row_accepts_valid_row() -> None:
    row = _seed_stability_row()
    assert row.n_seeds == 3


def test_seed_stability_row_rejects_negative_n_seeds() -> None:
    with pytest.raises(ValueError, match="n_seeds must be non-negative"):
        _seed_stability_row(n_seeds=-1)


def test_seed_stability_row_rejects_negative_seed_std() -> None:
    with pytest.raises(ValueError, match="seed_std must be non-negative"):
        _seed_stability_row(seed_std=-0.1)


def test_seed_stability_row_rejects_negative_seed_cv() -> None:
    with pytest.raises(ValueError, match="seed_cv must be non-negative"):
        _seed_stability_row(seed_cv=-0.1)


def test_seed_stability_row_rejects_empty_metric_name() -> None:
    with pytest.raises(ValueError, match="metric_name must not be empty"):
        _seed_stability_row(metric_name="")


# --- StrategyStabilityRow -------------------------------------------------


def test_strategy_stability_row_accepts_valid_row() -> None:
    row = _strategy_stability_row()
    assert row.n_strategies == 2


def test_strategy_stability_row_rejects_sign_agreement_ratio_out_of_range() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        _strategy_stability_row(sign_agreement_ratio=1.5)


def test_strategy_stability_row_rejects_mismatched_op_key_sets() -> None:
    with pytest.raises(ValueError, match="must share keys"):
        _strategy_stability_row(strategy_values={"constant_fill": 0.3})


def test_strategy_stability_row_rejects_n_strategies_mismatch() -> None:
    with pytest.raises(ValueError, match="n_strategies must equal"):
        _strategy_stability_row(n_strategies=3)


def test_strategy_stability_row_rejects_invalid_sign_value() -> None:
    with pytest.raises(ValueError, match="one of -1, 0, 1"):
        _strategy_stability_row(strategy_signs={"constant_fill": 2, "blur": -1})


# --- RankCorrelationRow ----------------------------------------------------


def test_rank_correlation_row_accepts_valid_row() -> None:
    row = _rank_correlation_row()
    assert row.op_a == "constant_fill"


def test_rank_correlation_row_rejects_equal_ops() -> None:
    with pytest.raises(ValueError, match="must differ"):
        _rank_correlation_row(op_b="constant_fill")


def test_rank_correlation_row_rejects_spearman_out_of_range() -> None:
    with pytest.raises(ValueError, match="between -1 and 1"):
        _rank_correlation_row(spearman=1.5)


def test_rank_correlation_row_rejects_spearman_excl_top1_out_of_range() -> None:
    with pytest.raises(ValueError, match="between -1 and 1"):
        _rank_correlation_row(spearman_excl_top1=-1.5)


def test_rank_correlation_row_rejects_negative_n_regions() -> None:
    with pytest.raises(ValueError, match="n_regions must be non-negative"):
        _rank_correlation_row(n_regions=-1)


def test_rank_correlation_row_rejects_empty_scope() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _rank_correlation_row(scope="")


# --- StrategyProfileRow ------------------------------------------------------


def test_strategy_profile_row_accepts_valid_row() -> None:
    row = _strategy_profile_row()
    assert row.alignment is Alignment.ALIGNED


def test_strategy_profile_row_rejects_empty_perturb_op() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _strategy_profile_row(perturb_op="")


def test_strategy_profile_row_rejects_sign_ratio_positive_out_of_range() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        _strategy_profile_row(sign_ratio_positive=1.5)


def test_strategy_profile_row_rejects_correlation_out_of_range() -> None:
    with pytest.raises(ValueError, match="between -1 and 1"):
        _strategy_profile_row(mean_corr_within=1.5)


def test_strategy_profile_row_rejects_negative_n_anchors() -> None:
    with pytest.raises(ValueError, match="n_anchors must be non-negative"):
        _strategy_profile_row(n_anchors=-1)


def test_strategy_profile_row_rejects_negative_cluster_id() -> None:
    with pytest.raises(ValueError, match="cluster_id must be non-negative"):
        _strategy_profile_row(cluster_id=-1)


def test_strategy_profile_row_rejects_non_alignment_type() -> None:
    with pytest.raises(TypeError, match="Alignment"):
        _strategy_profile_row(alignment="aligned")


# --- IntervalRow -------------------------------------------------------------


def test_interval_row_accepts_valid_row() -> None:
    row = _interval_row()
    assert row.excludes_zero is True


def test_interval_row_rejects_ci_low_greater_than_ci_high() -> None:
    with pytest.raises(ValueError, match="must not exceed ci_high"):
        _interval_row(ci_low=0.5, ci_high=0.1)


def test_interval_row_rejects_excludes_zero_inconsistent_with_bounds() -> None:
    with pytest.raises(ValueError, match="excludes_zero must equal"):
        _interval_row(ci_low=-0.1, ci_high=0.1, excludes_zero=True)


def test_interval_row_rejects_negative_n_bootstrap() -> None:
    with pytest.raises(ValueError, match="n_bootstrap must be non-negative"):
        _interval_row(n_bootstrap=-1)


def test_interval_row_rejects_empty_ci_method() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _interval_row(ci_method="")


# --- ReliabilityRow ------------------------------------------------------


def test_reliability_row_accepts_valid_row() -> None:
    row = _reliability_row()
    assert row.reliability_grade is ReliabilityGrade.HIGH


def test_reliability_row_rejects_non_flag_value_field() -> None:
    with pytest.raises(TypeError, match="sign_consistent must be a FlagValue"):
        _reliability_row(sign_consistent=True)


def test_reliability_row_rejects_non_reliability_grade() -> None:
    with pytest.raises(TypeError, match="ReliabilityGrade"):
        _reliability_row(reliability_grade="high")


def test_reliability_row_rejects_empty_metric_name() -> None:
    with pytest.raises(ValueError, match="metric_name must not be empty"):
        _reliability_row(metric_name="")


def test_reliability_row_rejects_non_anchor_key() -> None:
    with pytest.raises(TypeError, match="AnchorKey"):
        _reliability_row(anchor_key="sample-1::grid::0")


# --- CoverageReport ------------------------------------------------------


def test_coverage_report_accepts_valid_report() -> None:
    report = _coverage_report()
    assert report.n_anchors == 10


def test_coverage_report_rejects_negative_n_anchors() -> None:
    with pytest.raises(ValueError, match="n_anchors must be non-negative"):
        _coverage_report(n_anchors=-1)


def test_coverage_report_rejects_insufficient_exceeding_anchors() -> None:
    with pytest.raises(ValueError, match="must not exceed n_anchors"):
        _coverage_report(n_anchors=1, n_conditions_insufficient=2)


def test_coverage_report_rejects_negative_n_controls_unmatched() -> None:
    with pytest.raises(ValueError, match="n_controls_unmatched must be non-negative"):
        _coverage_report(n_controls_unmatched=-1)


def test_coverage_report_rejects_negative_n_area_mismatch_warnings() -> None:
    with pytest.raises(
        ValueError, match="n_area_mismatch_warnings must be non-negative"
    ):
        _coverage_report(n_area_mismatch_warnings=-1)
