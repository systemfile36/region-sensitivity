"""Validation tests for long-form metrics-engine contract types."""

from __future__ import annotations

import pytest

from ssat.core.types import RegionKind
from ssat.metrics.types import (
    ClassMetrics,
    ExclusionReason,
    ItemMetrics,
    RegionGeometryRef,
    RegionMetrics,
    SampleMetrics,
    SpatialProfile,
)


def _geometry_ref(**overrides: object) -> RegionGeometryRef:
    defaults: dict[str, object] = {
        "region_kind": RegionKind.GRID,
        "region_params_json": '{"rows": 2, "cols": 2}',
    }
    defaults.update(overrides)
    return RegionGeometryRef(**defaults)  # type: ignore[arg-type]


# --- ItemMetrics ---------------------------------------------------------


def test_item_metrics_accepts_available_row_with_values() -> None:
    row = ItemMetrics(
        item_id="item-1",
        sample_id="sample-1",
        metric_name="margin_drop",
        clean_correct=True,
        value_clean=0.8,
        value_perturbed=0.3,
        degradation=0.5,
        available=True,
    )
    assert row.excluded_reason is None
    assert row.degradation == 0.5


def test_item_metrics_accepts_excluded_row_without_values() -> None:
    row = ItemMetrics(
        item_id="item-1",
        sample_id="sample-1",
        metric_name="margin_drop",
        clean_correct=True,
        value_clean=None,
        value_perturbed=None,
        degradation=None,
        available=False,
        excluded_reason=ExclusionReason.PERTURBED_STATUS_NOT_OK,
    )
    assert row.available is False


def test_item_metrics_rejects_empty_identity_fields() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ItemMetrics(
            item_id="",
            sample_id="sample-1",
            metric_name="margin_drop",
            clean_correct=True,
            value_clean=0.1,
            value_perturbed=0.2,
            degradation=0.1,
            available=True,
        )


def test_item_metrics_rejects_values_when_unavailable() -> None:
    with pytest.raises(ValueError, match="must not carry metric values"):
        ItemMetrics(
            item_id="item-1",
            sample_id="sample-1",
            metric_name="margin_drop",
            clean_correct=True,
            value_clean=0.1,
            value_perturbed=None,
            degradation=None,
            available=False,
        )


def test_item_metrics_rejects_values_when_excluded() -> None:
    with pytest.raises(ValueError, match="must not carry metric values"):
        ItemMetrics(
            item_id="item-1",
            sample_id="sample-1",
            metric_name="margin_drop",
            clean_correct=True,
            value_clean=None,
            value_perturbed=None,
            degradation=0.4,
            available=True,
            excluded_reason=ExclusionReason.PERTURBED_STATUS_NOT_OK,
        )


def test_item_metrics_rejects_missing_values_when_healthy() -> None:
    with pytest.raises(ValueError, match="require all metric values"):
        ItemMetrics(
            item_id="item-1",
            sample_id="sample-1",
            metric_name="margin_drop",
            clean_correct=True,
            value_clean=0.1,
            value_perturbed=None,
            degradation=0.1,
            available=True,
        )


# --- RegionGeometryRef -----------------------------------------------------


def test_region_geometry_ref_rejects_explicit_without_ref() -> None:
    with pytest.raises(ValueError, match="require ref and ref_hash"):
        _geometry_ref(region_kind=RegionKind.EXPLICIT, region_params_json="{}")


def test_region_geometry_ref_accepts_explicit_with_ref() -> None:
    ref = _geometry_ref(
        region_kind=RegionKind.EXPLICIT,
        region_params_json="{}",
        ref="masks/a.png",
        ref_hash="a" * 64,
    )
    assert ref.ref == "masks/a.png"


def test_region_geometry_ref_rejects_procedural_with_ref() -> None:
    with pytest.raises(ValueError, match="cannot define ref"):
        _geometry_ref(ref="masks/a.png")


def test_region_geometry_ref_rejects_empty_params() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _geometry_ref(region_params_json="")


def test_region_geometry_ref_rejects_non_enum_kind() -> None:
    with pytest.raises(TypeError, match="RegionKind"):
        _geometry_ref(region_kind="grid")


# --- SampleMetrics ---------------------------------------------------------


def test_sample_metrics_accepts_valid_row() -> None:
    row = SampleMetrics(
        sample_id="sample-1",
        metric_name="margin_drop",
        gt_label=3,
        clean_correct=True,
        n_items=4,
        n_valid=4,
        flip_rate=0.25,
        vulnerability_score=0.31,
        metric_mean=0.31,
        metric_max=0.55,
        metric_std=0.1,
    )
    assert row.n_valid == 4


def test_sample_metrics_rejects_n_valid_exceeding_n_items() -> None:
    with pytest.raises(ValueError, match="must not exceed n_items"):
        SampleMetrics(
            sample_id="sample-1",
            metric_name="margin_drop",
            gt_label=3,
            clean_correct=True,
            n_items=2,
            n_valid=3,
            flip_rate=None,
            vulnerability_score=None,
            metric_mean=None,
            metric_max=None,
            metric_std=None,
        )


def test_sample_metrics_rejects_flip_rate_out_of_range() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        SampleMetrics(
            sample_id="sample-1",
            metric_name="margin_drop",
            gt_label=3,
            clean_correct=True,
            n_items=4,
            n_valid=4,
            flip_rate=1.5,
            vulnerability_score=None,
            metric_mean=None,
            metric_max=None,
            metric_std=None,
        )


# --- RegionMetrics ----------------------------------------------------------


def test_region_metrics_accepts_valid_row() -> None:
    row = RegionMetrics(
        region_key="grid#0",
        metric_name="margin_drop",
        region_kind=RegionKind.GRID,
        intended_area_px=64,
        effective_area_px=60,
        n_samples=10,
        n_valid=9,
        flip_rate=0.2,
        metric_mean=0.3,
    )
    assert row.n_samples == 10


def test_region_metrics_rejects_negative_area() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RegionMetrics(
            region_key="grid#0",
            metric_name="margin_drop",
            region_kind=RegionKind.GRID,
            intended_area_px=-1,
            effective_area_px=None,
            n_samples=10,
            n_valid=9,
            flip_rate=None,
            metric_mean=None,
        )


def test_region_metrics_rejects_n_valid_exceeding_n_samples() -> None:
    with pytest.raises(ValueError, match="must not exceed n_samples"):
        RegionMetrics(
            region_key="grid#0",
            metric_name="margin_drop",
            region_kind=RegionKind.GRID,
            intended_area_px=64,
            effective_area_px=64,
            n_samples=2,
            n_valid=3,
            flip_rate=None,
            metric_mean=None,
        )


# --- ClassMetrics ------------------------------------------------------------


def test_class_metrics_accepts_valid_row() -> None:
    row = ClassMetrics(
        gt_label=3,
        metric_name="margin_drop",
        n_samples=12,
        flip_rate=0.1,
        metric_mean=0.2,
    )
    assert row.gt_label == 3


def test_class_metrics_rejects_negative_n_samples() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ClassMetrics(
            gt_label=3,
            metric_name="margin_drop",
            n_samples=-1,
            flip_rate=None,
            metric_mean=None,
        )


# --- SpatialProfile ----------------------------------------------------------


def test_spatial_profile_accepts_valid_row() -> None:
    row = SpatialProfile(
        sample_id="sample-1",
        region_key="grid#0",
        metric_name="margin_drop",
        region_geometry_ref=_geometry_ref(),
        degradation=0.4,
    )
    assert row.degradation == 0.4


def test_spatial_profile_rejects_non_geometry_ref() -> None:
    with pytest.raises(TypeError, match="RegionGeometryRef"):
        SpatialProfile(
            sample_id="sample-1",
            region_key="grid#0",
            metric_name="margin_drop",
            region_geometry_ref="grid#0",  # type: ignore[arg-type]
            degradation=0.4,
        )


def test_spatial_profile_rejects_empty_identity_field() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        SpatialProfile(
            sample_id="",
            region_key="grid#0",
            metric_name="margin_drop",
            region_geometry_ref=_geometry_ref(),
            degradation=0.4,
        )
