"""Unit tests for ssat.report.assembler's pure helpers and input validation.

Complements tests/integration/test_report_synthetic_dump.py, which exercises
:class:`ReportDataAssembler` end-to-end against real store data — these
tests isolate the small, deterministic pieces (grade reduction, dataset-name
derivation, failure-rate arithmetic, constructor guards) that do not need a
dump/metrics/analysis triple at all.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from synthetic_dump_builder import (
    build_resolved_config,
    clean_record,
    compute_and_save_metrics,
    perturbed_record,
    write_dump,
)

from ssat.analysis.types import AnchorKey, FlagValue, ReliabilityGrade, ReliabilityRow
from ssat.core.config.schema import ResolvedRegionConfig
from ssat.core.types import RegionKind
from ssat.metrics.builtin_metrics.continuous import GtLogitDrop
from ssat.metrics.registry import MetricRegistry
from ssat.metrics.types import RegionGeometryRef, RegionMetrics, SpatialProfile
from ssat.report.adapters import ClassificationAdapter
from ssat.report.assembler import (
    ReportDataAssembler,
    _build_class_semantic_matrix,
    _build_semantic_concentration,
    _build_semantic_summary,
    _build_spatial_concentration,
    _dataset_name,
    _dataset_top_region_by_sample,
    _dataset_top_semantic_group_by_sample,
    _failure_rate,
    _group_report_grades,
    _is_binary_primary_metric,
    _region_id_from_region_key,
    _sample_semantic_group_degradation,
    _semantic_group_by_region_id,
    _worst_grade,
)
from ssat.report.errors import ReportDataError
from ssat.report.types import MetricCard, ReportGrade

_METRIC_NAME = "gt_logit_drop"


def _spatial_row(*, sample_id: str, region_key: str, degradation: float | None) -> SpatialProfile:
    return SpatialProfile(
        sample_id=sample_id,
        region_key=region_key,
        metric_name=_METRIC_NAME,
        region_geometry_ref=RegionGeometryRef(region_kind=RegionKind.GRID, region_params_json="{}"),
        degradation=degradation,
    )


def _reliability_row(
    *,
    sample_id: str,
    region_key: str,
    metric_name: str = _METRIC_NAME,
    grade: ReliabilityGrade,
) -> ReliabilityRow:
    return ReliabilityRow(
        anchor_key=AnchorKey(sample_id=sample_id, region_key=region_key, invert_mask=False),
        metric_name=metric_name,
        sign_consistent=FlagValue.UNAVAILABLE,
        exceeds_control=FlagValue.UNAVAILABLE,
        seed_stable=FlagValue.UNAVAILABLE,
        jitter_stable=FlagValue.UNAVAILABLE,
        multi_strategy=FlagValue.UNAVAILABLE,
        ci_excludes_zero=FlagValue.UNAVAILABLE,
        area_matched=FlagValue.UNAVAILABLE,
        reliability_grade=grade,
        reliability_reasons=(),
    )


# --- _worst_grade -----------------------------------------------------------


def test_worst_grade_returns_none_for_empty_sequence() -> None:
    assert _worst_grade(()) is None


def test_worst_grade_picks_unreliable_over_high() -> None:
    assert _worst_grade([ReportGrade.HIGH, ReportGrade.UNRELIABLE]) is ReportGrade.UNRELIABLE


def test_worst_grade_full_severity_order() -> None:
    assert _worst_grade([ReportGrade.HIGH, ReportGrade.MODERATE]) is ReportGrade.MODERATE
    assert _worst_grade([ReportGrade.MODERATE, ReportGrade.LOW]) is ReportGrade.LOW
    assert _worst_grade([ReportGrade.LOW, ReportGrade.UNRELIABLE]) is ReportGrade.UNRELIABLE


def test_worst_grade_single_element_is_itself() -> None:
    assert _worst_grade([ReportGrade.MODERATE]) is ReportGrade.MODERATE


# --- _group_report_grades ----------------------------------------------------


def test_group_report_grades_filters_by_metric_name_and_groups_by_key() -> None:
    rows = [
        _reliability_row(sample_id="s0", region_key="r0", grade=ReliabilityGrade.HIGH),
        _reliability_row(sample_id="s1", region_key="r0", grade=ReliabilityGrade.UNRELIABLE),
        _reliability_row(sample_id="s2", region_key="r1", grade=ReliabilityGrade.LOW),
        _reliability_row(
            sample_id="s3", region_key="r0", metric_name="other_metric", grade=ReliabilityGrade.HIGH
        ),
    ]

    grouped = _group_report_grades(rows, _METRIC_NAME, key=lambda row: row.anchor_key.region_key)

    assert grouped == {
        "r0": [ReportGrade.HIGH, ReportGrade.UNRELIABLE],
        "r1": [ReportGrade.LOW],
    }


def test_group_report_grades_empty_rows_yields_empty_dict() -> None:
    assert _group_report_grades([], _METRIC_NAME, key=lambda row: row.anchor_key.sample_id) == {}


# --- _dataset_top_region_by_sample -------------------------------------------


def test_dataset_top_region_by_sample_picks_max_degradation_per_sample() -> None:
    rows = [
        _spatial_row(sample_id="s0", region_key="r0", degradation=0.2),
        _spatial_row(sample_id="s0", region_key="r1", degradation=0.9),
        _spatial_row(sample_id="s1", region_key="r0", degradation=0.5),
        _spatial_row(sample_id="s1", region_key="r1", degradation=0.1),
    ]

    assert _dataset_top_region_by_sample(rows) == {"s0": "r1", "s1": "r0"}


def test_dataset_top_region_by_sample_breaks_ties_by_region_key_ascending() -> None:
    rows = [
        _spatial_row(sample_id="s0", region_key="r1", degradation=0.5),
        _spatial_row(sample_id="s0", region_key="r0", degradation=0.5),
    ]

    assert _dataset_top_region_by_sample(rows) == {"s0": "r0"}


def test_dataset_top_region_by_sample_excludes_samples_with_no_valid_degradation() -> None:
    rows = [
        _spatial_row(sample_id="s0", region_key="r0", degradation=None),
        _spatial_row(sample_id="s1", region_key="r0", degradation=0.5),
    ]

    assert _dataset_top_region_by_sample(rows) == {"s1": "r0"}


def test_dataset_top_region_by_sample_empty_input_yields_empty_dict() -> None:
    assert _dataset_top_region_by_sample([]) == {}


# --- _build_spatial_concentration ---------------------------------------------


def test_build_spatial_concentration_computes_dominant_share_and_entropy() -> None:
    # 3/4 samples top out at "r0" -- a concentrated, shortcut-like pattern.
    top_region_by_sample = {"s0": "r0", "s1": "r0", "s2": "r0", "s3": "r1"}

    concentration = _build_spatial_concentration(top_region_by_sample, ["r0", "r1"])

    assert concentration.dominant_region_key == "r0"
    assert concentration.dominant_region_share == pytest.approx(0.75)
    assert concentration.n_scored_samples == 4
    assert concentration.spatial_entropy is not None
    assert 0.0 <= concentration.spatial_entropy <= 1.0


def test_build_spatial_concentration_uniform_distribution_has_entropy_near_one() -> None:
    top_region_by_sample = {"s0": "r0", "s1": "r1", "s2": "r2", "s3": "r3"}

    concentration = _build_spatial_concentration(top_region_by_sample, ["r0", "r1", "r2", "r3"])

    assert concentration.dominant_region_share == pytest.approx(0.25)
    assert concentration.spatial_entropy == pytest.approx(1.0)


def test_build_spatial_concentration_single_dominant_region_has_zero_entropy() -> None:
    top_region_by_sample = {"s0": "r0", "s1": "r0", "s2": "r0"}

    concentration = _build_spatial_concentration(top_region_by_sample, ["r0", "r1"])

    assert concentration.dominant_region_share == pytest.approx(1.0)
    assert concentration.spatial_entropy == pytest.approx(0.0)


def test_build_spatial_concentration_empty_input_yields_all_none() -> None:
    concentration = _build_spatial_concentration({}, ["r0", "r1"])

    assert concentration.dominant_region_key is None
    assert concentration.dominant_region_share is None
    assert concentration.spatial_entropy is None
    assert concentration.n_scored_samples == 0


def test_build_spatial_concentration_single_possible_region_leaves_entropy_none() -> None:
    """Entropy is undefined (not zero) when there is only one place to spread across."""

    concentration = _build_spatial_concentration({"s0": "r0"}, ["r0"])

    assert concentration.dominant_region_share == pytest.approx(1.0)
    assert concentration.spatial_entropy is None


# --- semantic_group axis -----------------------------------------------------


class _FakeRegionFamily:
    def __init__(self, region_id: str, semantic_group: str | None) -> None:
        self.region_id = region_id
        self.semantic_group = semantic_group


class _FakeResolvedConfigWithRegions:
    def __init__(self, regions: list[_FakeRegionFamily]) -> None:
        self.regions = regions


def _region_metrics_row(
    *,
    region_key: str,
    flip_rate: float | None,
    n_valid: int = 10,
    region_kind: RegionKind = RegionKind.SKELETON_PARTS,
    metric_mean: float | None = 0.5,
    intended_area_px: int | None = 64,
    effective_area_px: int | None = 60,
) -> RegionMetrics:
    return RegionMetrics(
        region_key=region_key,
        metric_name=_METRIC_NAME,
        region_kind=region_kind,
        intended_area_px=intended_area_px,
        effective_area_px=effective_area_px,
        n_samples=n_valid,
        n_valid=n_valid,
        flip_rate=flip_rate,
        metric_mean=metric_mean,
    )


# region_id -> semantic_group fixture shared by the "foot action class"
# scenario below: two upper_limb families, two lower_limb families.
_UPPER_LOWER_LIMB_MAP = {
    "left_arm": "upper_limb",
    "right_arm": "upper_limb",
    "left_leg": "lower_limb",
    "right_leg": "lower_limb",
}


def test_region_id_from_region_key_splits_on_first_double_colon() -> None:
    assert _region_id_from_region_key("grid::0") == "grid"
    # A skeleton_parts region_key looks like "{region_id}::{region_id}/
    # {sample_id}" -- region_id itself may contain every
    # character RegionId's pattern allows (letters/digits/"_"/"."/"-", but
    # never "::"), exercised here with "." and "-".
    assert _region_id_from_region_key("left-arm.v2::left-arm.v2/sample-1") == "left-arm.v2"


def test_semantic_group_by_region_id_falls_back_to_region_id_when_unset() -> None:
    config = _FakeResolvedConfigWithRegions(
        [
            _FakeRegionFamily("left_arm", "upper_limb"),
            _FakeRegionFamily("right_arm", "upper_limb"),
            _FakeRegionFamily("grid", None),
        ]
    )

    assert _semantic_group_by_region_id(config) == {
        "left_arm": "upper_limb",
        "right_arm": "upper_limb",
        "grid": "grid",
    }


def test_is_binary_primary_metric_true_when_flip_rate_card_has_a_value() -> None:
    scorecard = (
        MetricCard(key="accuracy", label="Accuracy", value=0.9, unit="%", higher_is_better=True),
        MetricCard(key="flip_rate", label="Flip Rate", value=0.3, unit="%", higher_is_better=False),
    )
    assert _is_binary_primary_metric(scorecard) is True


def test_is_binary_primary_metric_false_when_flip_rate_card_value_is_none() -> None:
    scorecard = (
        MetricCard(
            key="flip_rate",
            label="Flip Rate",
            value=None,
            unit="%",
            higher_is_better=False,
            note="N/A: continuous metric, flip has no meaning for it.",
        ),
    )
    assert _is_binary_primary_metric(scorecard) is False


def test_sample_semantic_group_degradation_averages_within_sample_across_group() -> None:
    rows = [
        _spatial_row(sample_id="s0", region_key="left_arm::s0", degradation=0.1),
        _spatial_row(sample_id="s0", region_key="right_arm::s0", degradation=0.3),
        _spatial_row(sample_id="s0", region_key="left_leg::s0", degradation=0.8),
    ]

    result = _sample_semantic_group_degradation(rows, _UPPER_LOWER_LIMB_MAP)

    assert result == {
        ("s0", "upper_limb"): pytest.approx(0.2),  # (0.1 + 0.3) / 2
        ("s0", "lower_limb"): pytest.approx(0.8),
    }


def test_sample_semantic_group_degradation_skips_none_and_falls_back_when_ungrouped() -> None:
    rows = [
        _spatial_row(sample_id="s0", region_key="left_arm::s0", degradation=None),
        _spatial_row(sample_id="s0", region_key="grid::0", degradation=0.5),
    ]

    result = _sample_semantic_group_degradation(rows, _UPPER_LOWER_LIMB_MAP)

    # left_arm contributed no value (None); "grid" is absent from the map, so
    # it falls back to being its own semantic_group by default.
    assert result == {("s0", "grid"): pytest.approx(0.5)}


def test_dataset_top_semantic_group_by_sample_picks_max_degradation_group() -> None:
    sample_semantic_degradation = {
        ("s0", "upper_limb"): 0.1,
        ("s0", "lower_limb"): 0.85,
        ("s1", "upper_limb"): 0.85,
        ("s1", "lower_limb"): 0.15,
    }

    assert _dataset_top_semantic_group_by_sample(sample_semantic_degradation) == {
        "s0": "lower_limb",
        "s1": "upper_limb",
    }


def test_dataset_top_semantic_group_by_sample_breaks_ties_ascending() -> None:
    sample_semantic_degradation = {("s0", "upper_limb"): 0.5, ("s0", "lower_limb"): 0.5}

    assert _dataset_top_semantic_group_by_sample(sample_semantic_degradation) == {
        "s0": "lower_limb"
    }


def test_build_semantic_concentration_gate_forces_graceful_degradation_at_or_below_one_group() -> None:
    concentration = _build_semantic_concentration({"s0": "grid"}, n_semantic_groups=1)

    assert concentration.dominant_semantic_group is None
    assert concentration.dominant_semantic_group_share is None
    assert concentration.semantic_group_entropy is None
    assert concentration.n_semantic_groups == 1
    assert concentration.n_scored_samples == 0


def test_build_semantic_concentration_zero_groups_is_also_gated() -> None:
    concentration = _build_semantic_concentration({}, n_semantic_groups=0)
    assert concentration.n_semantic_groups == 0
    assert concentration.n_scored_samples == 0


def test_build_semantic_concentration_computes_dominant_share_and_entropy() -> None:
    top_semantic_group_by_sample = {
        "s0": "upper_limb",
        "s1": "upper_limb",
        "s2": "upper_limb",
        "s3": "lower_limb",
    }

    concentration = _build_semantic_concentration(top_semantic_group_by_sample, n_semantic_groups=2)

    assert concentration.dominant_semantic_group == "upper_limb"
    assert concentration.dominant_semantic_group_share == pytest.approx(0.75)
    assert concentration.n_scored_samples == 4
    assert concentration.semantic_group_entropy is not None
    assert 0.0 <= concentration.semantic_group_entropy <= 1.0


def test_build_semantic_concentration_no_scored_samples_yields_all_none() -> None:
    concentration = _build_semantic_concentration({}, n_semantic_groups=2)

    assert concentration.dominant_semantic_group is None
    assert concentration.dominant_semantic_group_share is None
    assert concentration.semantic_group_entropy is None
    assert concentration.n_scored_samples == 0


def test_build_semantic_summary_mean_degradation_and_n_samples() -> None:
    """Hand-computed "foot action class" fixture, reused by the class_semantic_matrix tests below."""

    sample_semantic_degradation = {
        ("s0", "upper_limb"): 0.1,
        ("s0", "lower_limb"): 0.85,
        ("s1", "upper_limb"): 0.1,
        ("s1", "lower_limb"): 0.65,
        ("s2", "upper_limb"): 0.85,
        ("s2", "lower_limb"): 0.15,
        ("s3", "upper_limb"): 0.65,
        ("s3", "lower_limb"): 0.05,
    }

    rows = _build_semantic_summary(
        sample_semantic_degradation,
        _UPPER_LOWER_LIMB_MAP,
        region_rows=[],
        grades_by_semantic_group={},
        is_binary_primary_metric=False,
    )

    by_group = {row.semantic_group: row for row in rows}
    assert set(by_group) == {"upper_limb", "lower_limb"}
    assert by_group["upper_limb"].region_ids == ("left_arm", "right_arm")
    assert by_group["upper_limb"].n_samples == 4
    assert by_group["upper_limb"].mean_degradation == pytest.approx((0.1 + 0.1 + 0.85 + 0.65) / 4)
    assert by_group["lower_limb"].mean_degradation == pytest.approx((0.85 + 0.65 + 0.15 + 0.05) / 4)
    # is_binary_primary_metric=False -- flip_rate must stay unavailable, not 0.
    assert by_group["upper_limb"].flip_rate is None
    assert by_group["upper_limb"].high_rate is None  # no analysis run (empty grades)


def test_build_semantic_summary_flip_rate_averages_region_metrics_when_binary() -> None:
    sample_semantic_degradation = {("s0", "upper_limb"): 0.5}
    region_rows = [
        _region_metrics_row(region_key="left_arm::s0", flip_rate=0.2),
        _region_metrics_row(region_key="right_arm::s0", flip_rate=0.6),
    ]

    rows = _build_semantic_summary(
        sample_semantic_degradation,
        _UPPER_LOWER_LIMB_MAP,
        region_rows=region_rows,
        grades_by_semantic_group={},
        is_binary_primary_metric=True,
    )

    assert rows[0].flip_rate == pytest.approx((0.2 + 0.6) / 2)


def test_build_semantic_summary_high_rate_regroups_reliability_grades() -> None:
    sample_semantic_degradation = {("s0", "upper_limb"): 0.5}
    grades_by_semantic_group = {
        "upper_limb": [ReportGrade.HIGH, ReportGrade.HIGH, ReportGrade.UNRELIABLE]
    }

    rows = _build_semantic_summary(
        sample_semantic_degradation,
        _UPPER_LOWER_LIMB_MAP,
        region_rows=[],
        grades_by_semantic_group=grades_by_semantic_group,
        is_binary_primary_metric=False,
    )

    assert rows[0].high_rate == pytest.approx(2 / 3)


def test_build_class_semantic_matrix_reproduces_foot_action_class_pattern() -> None:
    """The scenario the user asked to see: a class whose dominant body part differs.

    gt_label=0 samples degrade most under lower_limb occlusion (a "foot
    action" class); gt_label=1 samples degrade most under upper_limb
    occlusion (a "hand action" class) -- reproduced exactly here from the
    same per-(sample, semantic_group) values used above.
    """

    sample_semantic_degradation = {
        ("s0", "upper_limb"): 0.1,
        ("s0", "lower_limb"): 0.85,
        ("s1", "upper_limb"): 0.1,
        ("s1", "lower_limb"): 0.65,
        ("s2", "upper_limb"): 0.85,
        ("s2", "lower_limb"): 0.15,
        ("s3", "upper_limb"): 0.65,
        ("s3", "lower_limb"): 0.05,
    }
    gt_label_by_sample = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}

    rows, n_excluded = _build_class_semantic_matrix(sample_semantic_degradation, gt_label_by_sample)

    by_cell = {(row.gt_label, row.semantic_group): row for row in rows}
    assert by_cell[(0, "lower_limb")].mean_degradation == pytest.approx((0.85 + 0.65) / 2)
    assert by_cell[(0, "upper_limb")].mean_degradation == pytest.approx((0.1 + 0.1) / 2)
    assert by_cell[(1, "upper_limb")].mean_degradation == pytest.approx((0.85 + 0.65) / 2)
    assert by_cell[(1, "lower_limb")].mean_degradation == pytest.approx((0.15 + 0.05) / 2)
    # The core finding: for gt_label=0, lower_limb dominates; for gt_label=1, upper_limb does.
    assert by_cell[(0, "lower_limb")].mean_degradation > by_cell[(0, "upper_limb")].mean_degradation
    assert by_cell[(1, "upper_limb")].mean_degradation > by_cell[(1, "lower_limb")].mean_degradation
    assert all(row.n_samples == 2 for row in rows)
    assert all(row.flip_rate is None for row in rows)  # no (sample, semantic_group)-grain flip data
    assert n_excluded == 0


def test_build_class_semantic_matrix_excludes_samples_with_no_gt_label() -> None:
    sample_semantic_degradation = {
        ("s0", "upper_limb"): 0.5,
        ("s1", "upper_limb"): 0.5,
        ("s1", "lower_limb"): 0.2,
    }
    gt_label_by_sample = {"s0": 0, "s1": None}

    rows, n_excluded = _build_class_semantic_matrix(sample_semantic_degradation, gt_label_by_sample)

    assert {(row.gt_label, row.semantic_group) for row in rows} == {(0, "upper_limb")}
    # s1 contributed two (sample, semantic_group) pairs but is one distinct
    # sample -- the exclusion count must not double it.
    assert n_excluded == 1


def test_build_class_semantic_matrix_empty_input_yields_no_rows_no_exclusions() -> None:
    rows, n_excluded = _build_class_semantic_matrix({}, {})
    assert rows == ()
    assert n_excluded == 0


# --- _dataset_name -----------------------------------------------------------


class _FakeSourceProvenance:
    def __init__(self, manifest: Path) -> None:
        self.manifest = manifest


class _FakeResolvedConfig:
    def __init__(self, *, source_provenance: object = None, config_source: Path | None = None) -> None:
        self.source_provenance = source_provenance
        self.config_source = config_source


def test_dataset_name_from_source_provenance_manifest_parent(tmp_path: Path) -> None:
    manifest_path = tmp_path / "shortcut_A" / "manifest.json"
    config = _FakeResolvedConfig(source_provenance=_FakeSourceProvenance(manifest_path))

    assert _dataset_name(config) == "shortcut_A"


def test_dataset_name_falls_back_to_config_source_stem(tmp_path: Path) -> None:
    config = _FakeResolvedConfig(config_source=tmp_path / "my_audit.yaml")

    assert _dataset_name(config) == "my_audit"


def test_dataset_name_falls_back_to_unknown() -> None:
    assert _dataset_name(_FakeResolvedConfig()) == "unknown"


# --- _failure_rate -----------------------------------------------------------


def test_failure_rate_computes_fraction() -> None:
    summary = {"total_perturbed_items": 20, "items_excluded_perturbed_failed": 5}
    assert _failure_rate(summary) == 0.25


def test_failure_rate_none_when_no_perturbed_items() -> None:
    summary = {"total_perturbed_items": 0, "items_excluded_perturbed_failed": 0}
    assert _failure_rate(summary) is None


# --- _build_region_summary (skeleton_parts region_id aggregation) -----------


def _summary_assembler() -> ReportDataAssembler:
    """A ReportDataAssembler whose paths are never touched by ``_build_region_summary``.

    Mirrors this file's own framing (module docstring): the constructor
    stores paths without opening them, so ``_build_region_summary`` -- a
    pure function of its arguments -- can be exercised directly without a
    real dump/metrics/analysis triple on disk.
    """

    return ReportDataAssembler(
        Path("unused-dump"), Path("unused-metrics"), adapter=ClassificationAdapter(primary_metric=_METRIC_NAME)
    )


class _FakeAnalysisContext:
    """Duck-typed stand-in for ``_AnalysisContext`` (only ``.reliability_rows``/``.manifest.grade_distribution`` are read)."""

    def __init__(self, reliability_rows: list[ReliabilityRow]) -> None:
        self.reliability_rows = reliability_rows
        self.manifest = SimpleNamespace(grade_distribution={})


def test_build_region_summary_leaves_grid_rows_unaggregated() -> None:
    """Regression guard: grid's region_key is already dataset-stable and must stay per-cell."""

    region_rows = [
        _region_metrics_row(region_key="grid::r0/c0", region_kind=RegionKind.GRID, flip_rate=0.2),
        _region_metrics_row(region_key="grid::r0/c1", region_kind=RegionKind.GRID, flip_rate=0.4),
    ]

    summary = _summary_assembler()._build_region_summary(region_rows, None, _METRIC_NAME, {})

    assert [row.region_key for row in summary.rows] == ["grid::r0/c0", "grid::r0/c1"]
    assert [row.region_id for row in summary.rows] == ["grid", "grid"]


def test_build_region_summary_aggregates_skeleton_parts_rows_by_region_id() -> None:
    region_rows = [
        _region_metrics_row(
            region_key="left_arm::left_arm/s_a", n_valid=3, metric_mean=0.2, flip_rate=0.0,
            intended_area_px=100, effective_area_px=90,
        ),
        _region_metrics_row(
            region_key="left_arm::left_arm/s_b", n_valid=1, metric_mean=1.0, flip_rate=1.0,
            intended_area_px=50, effective_area_px=40,
        ),
        _region_metrics_row(
            region_key="right_arm::right_arm/s_a", n_valid=5, metric_mean=0.5, flip_rate=0.4,
            intended_area_px=80, effective_area_px=70,
        ),
    ]

    summary = _summary_assembler()._build_region_summary(region_rows, None, _METRIC_NAME, {})

    by_region_id = {row.region_id: row for row in summary.rows}
    assert set(by_region_id) == {"left_arm", "right_arm"}

    left_arm = by_region_id["left_arm"]
    # region_key must equal region_id for an aggregated row (no single
    # concrete region_key spans many samples) -- this is also what fixes
    # render_region_bar's x-axis labels.
    assert left_arm.region_key == "left_arm"
    assert left_arm.region_kind == "skeleton_parts"
    assert left_arm.n_valid == 4  # 3 + 1, summed (a count), not averaged
    # weighted by n_valid: (0.2*3 + 1.0*1) / 4
    assert left_arm.mean_degradation == pytest.approx((0.2 * 3 + 1.0 * 1) / 4)
    assert left_arm.flip_rate == pytest.approx((0.0 * 3 + 1.0 * 1) / 4)
    assert left_arm.intended_area_px == round((100 * 3 + 50 * 1) / 4)
    assert left_arm.effective_area_px == round((90 * 3 + 40 * 1) / 4)

    right_arm = by_region_id["right_arm"]
    assert right_arm.n_valid == 5
    assert right_arm.mean_degradation == pytest.approx(0.5)


def test_build_region_summary_skeleton_parts_top_region_share_uses_region_id_mapping() -> None:
    """The subtle bug: top_region_by_sample is keyed by concrete region_key, not region_id."""

    region_rows = [
        _region_metrics_row(region_key="left_arm::left_arm/s_a", n_valid=1, flip_rate=None),
        _region_metrics_row(region_key="left_arm::left_arm/s_b", n_valid=1, flip_rate=None),
        _region_metrics_row(region_key="left_arm::left_arm/s_c", n_valid=1, flip_rate=None),
    ]
    top_region_by_sample = {
        "s_a": "left_arm::left_arm/s_a",
        "s_b": "left_arm::left_arm/s_b",
        "s_c": "grid::r0/c0",  # a different sample's top region is unrelated
    }

    summary = _summary_assembler()._build_region_summary(
        region_rows, None, _METRIC_NAME, top_region_by_sample
    )

    left_arm = next(row for row in summary.rows if row.region_id == "left_arm")
    # 2 of 3 scored samples' top region reduces to region_id "left_arm".
    assert left_arm.top_region_share == pytest.approx(2 / 3)


def test_build_region_summary_skeleton_parts_merges_reliability_distribution() -> None:
    region_rows = [
        _region_metrics_row(region_key="left_arm::left_arm/s_a", n_valid=1, flip_rate=None),
        _region_metrics_row(region_key="left_arm::left_arm/s_b", n_valid=1, flip_rate=None),
    ]
    analysis = _FakeAnalysisContext(
        [
            _reliability_row(sample_id="s_a", region_key="left_arm::left_arm/s_a", grade=ReliabilityGrade.HIGH),
            _reliability_row(
                sample_id="s_b", region_key="left_arm::left_arm/s_b", grade=ReliabilityGrade.UNRELIABLE
            ),
        ]
    )

    summary = _summary_assembler()._build_region_summary(region_rows, analysis, _METRIC_NAME, {})

    left_arm = next(row for row in summary.rows if row.region_id == "left_arm")
    assert left_arm.reliability_distribution == {"high": 1, "unreliable": 1}
    assert left_arm.reliability_grade is ReportGrade.UNRELIABLE  # worst-case rollup


# --- constructor / assemble() validation -------------------------------------


def _minimal_dump_and_metrics(tmp_path: Path) -> tuple[Path, Path]:
    config = build_resolved_config(
        tmp_path,
        regions=(ResolvedRegionConfig(region_id="grid", kind=RegionKind.GRID, params={"rows": 1, "cols": 1}),),
    )
    dump_root = tmp_path / "dump"
    write_dump(
        dump_root,
        config,
        clean_records=(clean_record("s0", logits=np.array([1.0, 0.0])),),
        perturbed_records=(
            perturbed_record(
                0,
                sample_id="s0",
                region_id="grid",
                region_instance_id="grid/r0/c0",
                logits=np.array([0.5, 0.0]),
            ),
        ),
    )
    metrics_dir = tmp_path / "metrics"
    registry = MetricRegistry()
    registry.register(GtLogitDrop())
    compute_and_save_metrics(dump_root, config, metrics_dir, registry=registry, primary_metric=_METRIC_NAME)
    return dump_root, metrics_dir


@pytest.mark.parametrize("kwargs", [{"top_k": -1}, {"bottom_k": -1}, {"region_top_k": -1}])
def test_constructor_rejects_negative_selection_sizes(tmp_path: Path, kwargs: dict) -> None:
    adapter = ClassificationAdapter(primary_metric=_METRIC_NAME)
    with pytest.raises(ValueError, match="non-negative"):
        ReportDataAssembler(tmp_path / "dump", tmp_path / "metrics", adapter=adapter, **kwargs)


def test_assemble_rejects_empty_primary_metric(tmp_path: Path) -> None:
    dump_root, metrics_dir = _minimal_dump_and_metrics(tmp_path)
    adapter = ClassificationAdapter(primary_metric=_METRIC_NAME)
    assembler = ReportDataAssembler(dump_root, metrics_dir, adapter=adapter)

    with pytest.raises(ValueError, match="primary_metric"):
        assembler.assemble("")


def test_assemble_rejects_unregistered_primary_metric(tmp_path: Path) -> None:
    dump_root, metrics_dir = _minimal_dump_and_metrics(tmp_path)
    adapter = ClassificationAdapter(primary_metric=_METRIC_NAME)
    assembler = ReportDataAssembler(dump_root, metrics_dir, adapter=adapter)

    with pytest.raises(ReportDataError, match="not_a_real_metric"):
        assembler.assemble("not_a_real_metric")
