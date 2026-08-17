"""Tests for the metrics engine's N3 Aggregator (aggregate_item_metrics)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import pytest

from ssat.core.adapter.types import AdapterSpec
from ssat.core.config.schema import (
    DumpConfig,
    PerturbationConfig,
    ResolvedConfig,
    ResolvedRegionConfig,
    RuntimeConfig,
)
from ssat.core.types import PerturbationOp, RegionKind
from ssat.metrics.aggregate import DEFAULT_PRIMARY_METRIC, aggregate_item_metrics
from ssat.metrics.errors import MetricsCorruptionError, MetricsRegistryError
from ssat.metrics.registry import MetricRegistry
from ssat.metrics.types import ExclusionReason, ItemMetrics


class _FakeMetric:
    """A no-op metric that only carries identity/kind metadata for lookups."""

    requires: tuple[str, ...] = ()
    higher_is_better = True

    def __init__(self, name: str, kind: Literal["binary", "continuous"]) -> None:
        self.name = name
        self.kind = kind

    def available_when(self, adapter_spec: AdapterSpec) -> bool:
        return True

    def compute(self, clean, perturbed):  # pragma: no cover - unused by these tests
        raise NotImplementedError


def _registry() -> MetricRegistry:
    registry = MetricRegistry()
    registry.register(_FakeMetric("fake_binary", "binary"))
    registry.register(_FakeMetric("fake_continuous", "continuous"))
    registry.register(_FakeMetric(DEFAULT_PRIMARY_METRIC, "continuous"))
    return registry


def _resolved_config(tmp_path: Path, *, explicit: bool = False) -> ResolvedConfig:
    if explicit:
        mask_path = tmp_path / "mask.png"
        mask_path.write_bytes(b"fake-mask")
        regions = (
            ResolvedRegionConfig(
                region_id="mask", kind=RegionKind.EXPLICIT, ref=mask_path, ref_hash="d" * 64
            ),
        )
    else:
        regions = (
            ResolvedRegionConfig(
                region_id="grid", kind=RegionKind.GRID, params={"rows": 1, "cols": 2}
            ),
        )
    return ResolvedConfig(
        config_base_dir=tmp_path.resolve(),
        regions=regions,
        perturbations=(
            PerturbationConfig(op=PerturbationOp.CONSTANT_FILL, params={"value": 0.0}),
        ),
        runtime=RuntimeConfig(),
        dump=DumpConfig(),
        adapter_spec=AdapterSpec(model_id="model", deterministic=True),
    )


def _joined_row(
    item_id: str,
    *,
    sample_id: str,
    gt_label: int | None = 0,
    region_id: str = "grid",
    region_instance_id: str = "grid/r0/c0",
    region_kind: RegionKind = RegionKind.GRID,
    is_control: bool = False,
    intended_area_px: int | None = 4,
    effective_area_px: int | None = 4,
) -> dict[str, object]:
    return {
        "item_id": item_id,
        "sample_id": sample_id,
        "gt_label": gt_label,
        "region_id": region_id,
        "region_instance_id": region_instance_id,
        "region_kind": region_kind.value,
        "region_params_json": "{}",
        "is_control": is_control,
        "intended_area_px": intended_area_px,
        "effective_area_px": effective_area_px,
    }


def _joined(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _item(
    item_id: str,
    sample_id: str,
    metric_name: str,
    *,
    clean_correct: bool = True,
    value_clean: float = 1.0,
    value_perturbed: float = 0.5,
    degradation: float | None = 0.5,
    available: bool = True,
) -> ItemMetrics:
    if not available:
        return ItemMetrics(
            item_id=item_id,
            sample_id=sample_id,
            metric_name=metric_name,
            clean_correct=clean_correct,
            value_clean=None,
            value_perturbed=None,
            degradation=None,
            available=False,
            excluded_reason=ExclusionReason.PERTURBED_STATUS_NOT_OK,
        )
    return ItemMetrics(
        item_id=item_id,
        sample_id=sample_id,
        metric_name=metric_name,
        clean_correct=clean_correct,
        value_clean=value_clean,
        value_perturbed=value_perturbed,
        degradation=degradation,
        available=True,
        excluded_reason=None,
    )


def _unlabeled_item(item_id: str, sample_id: str, metric_name: str) -> ItemMetrics:
    """Build an ItemMetrics row for an item whose sample has no gt_label."""

    return ItemMetrics(
        item_id=item_id,
        sample_id=sample_id,
        metric_name=metric_name,
        clean_correct=None,
        value_clean=None,
        value_perturbed=None,
        degradation=None,
        available=False,
        excluded_reason=ExclusionReason.GT_LABEL_UNKNOWN,
    )


def test_sample_metrics_computes_mean_max_std_over_valid_items(tmp_path: Path) -> None:
    joined = _joined(
        [
            _joined_row("a" * 64, sample_id="s1"),
            _joined_row("b" * 64, sample_id="s1", region_instance_id="grid/r0/c1"),
        ]
    )
    item_metrics = [
        _item("a" * 64, "s1", "fake_continuous", degradation=1.0),
        _item("b" * 64, "s1", "fake_continuous", degradation=3.0),
    ]

    result = aggregate_item_metrics(item_metrics, joined, _registry(), _resolved_config(tmp_path))

    row = next(r for r in result.sample_metrics if r.metric_name == "fake_continuous")
    assert row.n_items == 2
    assert row.n_valid == 2
    assert row.metric_mean == pytest.approx(2.0)
    assert row.metric_max == pytest.approx(3.0)
    assert row.metric_std == pytest.approx(1.0)
    assert row.flip_rate is None


def test_flip_rate_is_populated_only_for_binary_metrics(tmp_path: Path) -> None:
    joined = _joined(
        [
            _joined_row("a" * 64, sample_id="s1"),
            _joined_row("b" * 64, sample_id="s1", region_instance_id="grid/r0/c1"),
        ]
    )
    item_metrics = [
        _item("a" * 64, "s1", "fake_binary", degradation=1.0),
        _item("b" * 64, "s1", "fake_binary", degradation=0.0),
    ]

    result = aggregate_item_metrics(item_metrics, joined, _registry(), _resolved_config(tmp_path))

    row = next(r for r in result.sample_metrics if r.metric_name == "fake_binary")
    assert row.flip_rate == pytest.approx(0.5)


def test_n_valid_excludes_unavailable_items_but_n_items_counts_them(tmp_path: Path) -> None:
    joined = _joined(
        [
            _joined_row("a" * 64, sample_id="s1"),
            _joined_row("b" * 64, sample_id="s1", region_instance_id="grid/r0/c1"),
        ]
    )
    item_metrics = [
        _item("a" * 64, "s1", "fake_continuous", degradation=2.0),
        _item("b" * 64, "s1", "fake_continuous", available=False),
    ]

    result = aggregate_item_metrics(item_metrics, joined, _registry(), _resolved_config(tmp_path))

    row = next(r for r in result.sample_metrics if r.metric_name == "fake_continuous")
    assert row.n_items == 2
    assert row.n_valid == 1
    assert row.metric_mean == pytest.approx(2.0)
    assert row.metric_std == pytest.approx(0.0)


def test_vulnerability_score_is_denormalized_across_metric_name_rows(tmp_path: Path) -> None:
    joined = _joined(
        [
            _joined_row("a" * 64, sample_id="s1"),
            _joined_row("b" * 64, sample_id="s1", region_instance_id="grid/r0/c1"),
        ]
    )
    item_metrics = [
        _item("a" * 64, "s1", DEFAULT_PRIMARY_METRIC, degradation=2.0),
        _item("b" * 64, "s1", DEFAULT_PRIMARY_METRIC, degradation=4.0),
        _item("a" * 64, "s1", "fake_continuous", degradation=99.0),
    ]

    result = aggregate_item_metrics(item_metrics, joined, _registry(), _resolved_config(tmp_path))

    by_metric = {r.metric_name: r for r in result.sample_metrics}
    assert by_metric[DEFAULT_PRIMARY_METRIC].vulnerability_score == pytest.approx(3.0)
    # Denormalized: the unrelated fake_continuous row for the same sample carries
    # the same sample-level vulnerability_score, not something derived from its
    # own degradation (99.0).
    assert by_metric["fake_continuous"].vulnerability_score == pytest.approx(3.0)


def test_vulnerability_score_is_none_when_the_primary_metric_has_no_valid_items(
    tmp_path: Path,
) -> None:
    joined = _joined([_joined_row("a" * 64, sample_id="s1")])
    item_metrics = [_item("a" * 64, "s1", DEFAULT_PRIMARY_METRIC, available=False)]

    result = aggregate_item_metrics(item_metrics, joined, _registry(), _resolved_config(tmp_path))

    assert result.sample_metrics[0].vulnerability_score is None


def test_control_items_are_excluded_from_every_aggregate(tmp_path: Path) -> None:
    joined = _joined(
        [
            _joined_row("a" * 64, sample_id="s1"),
            _joined_row("b" * 64, sample_id="s1", region_instance_id="grid/r0/c1", is_control=True),
        ]
    )
    item_metrics = [
        _item("a" * 64, "s1", "fake_continuous", degradation=1.0),
        _item("b" * 64, "s1", "fake_continuous", degradation=1000.0),
    ]

    result = aggregate_item_metrics(item_metrics, joined, _registry(), _resolved_config(tmp_path))

    row = next(r for r in result.sample_metrics if r.metric_name == "fake_continuous")
    assert row.n_items == 1
    assert row.metric_mean == pytest.approx(1.0)
    assert {r.region_key for r in result.region_metrics} == {"grid::grid/r0/c0"}


def test_control_item_with_unconfigured_region_id_does_not_raise(tmp_path: Path) -> None:
    # PlanBuilder never adds its generated control region_id
    # (f"control:{match_area_of}:{control_request_index}") to the resolved
    # config's region families -- it is a synthetic id, not a configured
    # one. A control item is always RegionKind.RANDOM_AREA_MATCH, which
    # never needs a family lookup (only EXPLICIT does), so this must not
    # raise KeyError even though "control:grid:0" is absent from
    # _resolved_config(tmp_path)'s single "grid" family.
    joined = _joined(
        [
            _joined_row("a" * 64, sample_id="s1"),
            _joined_row(
                "b" * 64,
                sample_id="s1",
                region_id="control:grid:0",
                region_instance_id="control:grid/r0/c0:0:0",
                region_kind=RegionKind.RANDOM_AREA_MATCH,
                is_control=True,
            ),
        ]
    )
    item_metrics = [
        _item("a" * 64, "s1", "fake_continuous", degradation=1.0),
        _item("b" * 64, "s1", "fake_continuous", degradation=1000.0),
    ]

    result = aggregate_item_metrics(item_metrics, joined, _registry(), _resolved_config(tmp_path))

    row = next(r for r in result.sample_metrics if r.metric_name == "fake_continuous")
    assert row.n_items == 1
    assert row.metric_mean == pytest.approx(1.0)
    assert {r.region_key for r in result.region_metrics} == {"grid::grid/r0/c0"}


def test_explicit_region_missing_family_raises(tmp_path: Path) -> None:
    joined = _joined(
        [
            _joined_row(
                "a" * 64,
                sample_id="s1",
                region_id="unknown-mask",
                region_kind=RegionKind.EXPLICIT,
            ),
        ]
    )
    item_metrics = [_item("a" * 64, "s1", "fake_continuous", degradation=1.0)]

    with pytest.raises(MetricsCorruptionError, match="not found in resolved config"):
        aggregate_item_metrics(item_metrics, joined, _registry(), _resolved_config(tmp_path))


def test_region_metrics_use_region_id_and_instance_id_as_the_key(tmp_path: Path) -> None:
    joined = _joined(
        [
            _joined_row("a" * 64, sample_id="s1", intended_area_px=4, effective_area_px=4),
            _joined_row(
                "b" * 64,
                sample_id="s1",
                region_instance_id="grid/r0/c1",
                intended_area_px=8,
                effective_area_px=8,
            ),
        ]
    )
    item_metrics = [
        _item("a" * 64, "s1", "fake_continuous", degradation=1.0),
        _item("b" * 64, "s1", "fake_continuous", degradation=2.0),
    ]

    result = aggregate_item_metrics(item_metrics, joined, _registry(), _resolved_config(tmp_path))

    by_key = {r.region_key: r for r in result.region_metrics}
    assert set(by_key) == {"grid::grid/r0/c0", "grid::grid/r0/c1"}
    assert by_key["grid::grid/r0/c0"].region_kind is RegionKind.GRID
    assert by_key["grid::grid/r0/c0"].intended_area_px == 4
    assert by_key["grid::grid/r0/c1"].intended_area_px == 8
    assert by_key["grid::grid/r0/c0"].n_samples == 1


def test_inconsistent_area_within_one_sample_raises(tmp_path: Path) -> None:
    # Two items of the same region *in the same sample* must agree: within a
    # sample the mask is resolved once, so a disagreement means the dump is
    # corrupt. (Across samples it is legitimate -- see the test below.)
    joined = _joined(
        [
            _joined_row("a" * 64, sample_id="s1", intended_area_px=4),
            _joined_row("b" * 64, sample_id="s1", intended_area_px=8),
        ]
    )
    item_metrics = [
        _item("a" * 64, "s1", "fake_continuous", degradation=1.0),
        _item("b" * 64, "s1", "fake_continuous", degradation=1.0),
    ]

    with pytest.raises(MetricsCorruptionError, match="intended_area_px"):
        aggregate_item_metrics(item_metrics, joined, _registry(), _resolved_config(tmp_path))


def test_area_may_differ_across_samples_and_is_averaged(tmp_path: Path) -> None:
    # Sample-dependent region kinds (gt_bbox, skeleton_parts) resolve to a
    # different box per sample, and a random_area_match control is re-drawn
    # per item -- so a per-region area equality check across samples is
    # simply wrong. The dataset-grain RegionMetrics area summarizes them by
    # mean rather than picking whichever item was seen first.
    joined = _joined(
        [
            _joined_row("a" * 64, sample_id="s1", intended_area_px=4, effective_area_px=40),
            _joined_row("b" * 64, sample_id="s2", intended_area_px=8, effective_area_px=60),
        ]
    )
    item_metrics = [
        _item("a" * 64, "s1", "fake_continuous", degradation=1.0),
        _item("b" * 64, "s2", "fake_continuous", degradation=1.0),
    ]

    result = aggregate_item_metrics(item_metrics, joined, _registry(), _resolved_config(tmp_path))

    row = next(r for r in result.region_metrics if r.region_key == "grid::grid/r0/c0")
    assert row.intended_area_px == 6
    assert row.effective_area_px == 50
    assert row.n_samples == 2


def test_inconsistent_region_kind_for_the_same_region_raises(tmp_path: Path) -> None:
    joined = _joined(
        [
            _joined_row("a" * 64, sample_id="s1", region_kind=RegionKind.GRID),
            _joined_row("b" * 64, sample_id="s2", region_kind=RegionKind.BBOX_PARTITION),
        ]
    )
    item_metrics = [
        _item("a" * 64, "s1", "fake_continuous", degradation=1.0),
        _item("b" * 64, "s2", "fake_continuous", degradation=1.0),
    ]

    with pytest.raises(MetricsCorruptionError, match="region_kind"):
        aggregate_item_metrics(item_metrics, joined, _registry(), _resolved_config(tmp_path))


def test_class_metrics_n_samples_counts_distinct_samples_not_items(tmp_path: Path) -> None:
    # s1 contributes two items (region c0, c1); s2 contributes one. ClassMetrics is
    # sample-grain (design confirmation: RegionMetrics/ClassMetrics's n_valid<=n_samples
    # invariant forces averaging per-sample first), so s1's per-sample mean is
    # (1.0+2.0)/2=1.5, s2's is 3.0, and the class mean is the average of those two
    # per-sample means — (1.5+3.0)/2=2.25 — not the flat item-level mean (2.0).
    joined = _joined(
        [
            _joined_row("a" * 64, sample_id="s1", gt_label=3),
            _joined_row("b" * 64, sample_id="s1", gt_label=3, region_instance_id="grid/r0/c1"),
            _joined_row("c" * 64, sample_id="s2", gt_label=3, region_instance_id="grid/r0/c1"),
        ]
    )
    item_metrics = [
        _item("a" * 64, "s1", "fake_continuous", degradation=1.0),
        _item("b" * 64, "s1", "fake_continuous", degradation=2.0),
        _item("c" * 64, "s2", "fake_continuous", degradation=3.0),
    ]

    result = aggregate_item_metrics(item_metrics, joined, _registry(), _resolved_config(tmp_path))

    row = next(r for r in result.class_metrics if r.gt_label == 3)
    assert row.n_samples == 2
    assert row.metric_mean == pytest.approx(2.25)


def test_unlabeled_sample_flows_through_sample_metrics_but_not_class_metrics(
    tmp_path: Path,
) -> None:
    """Label-free auditing (core layer's ``gt_label: int | None``) must not crash.

    A sample whose ``gt_label`` is unknown (``ExclusionReason.
    GT_LABEL_UNKNOWN``) stays visible in ``SampleMetrics`` — with
    ``gt_label``/``clean_correct`` explicitly ``None``, never silently
    dropped — but has no row to join in ``ClassMetrics``, which is
    inherently keyed by a real ground-truth class.
    """

    joined = _joined(
        [
            _joined_row("a" * 64, sample_id="s1", gt_label=None),
            _joined_row("b" * 64, sample_id="s2", gt_label=3),
        ]
    )
    item_metrics = [
        _unlabeled_item("a" * 64, "s1", "fake_continuous"),
        _item("b" * 64, "s2", "fake_continuous", degradation=2.0),
    ]

    result = aggregate_item_metrics(item_metrics, joined, _registry(), _resolved_config(tmp_path))

    unlabeled_row = next(
        r for r in result.sample_metrics if r.sample_id == "s1" and r.metric_name == "fake_continuous"
    )
    assert unlabeled_row.gt_label is None
    assert unlabeled_row.clean_correct is None
    assert unlabeled_row.n_items == 1
    assert unlabeled_row.n_valid == 0
    assert unlabeled_row.metric_mean is None

    class_gt_labels = {r.gt_label for r in result.class_metrics}
    assert None not in class_gt_labels
    assert class_gt_labels == {3}


def test_spatial_profile_averages_multiple_items_sharing_a_sample_and_region(
    tmp_path: Path,
) -> None:
    joined = _joined(
        [
            _joined_row("a" * 64, sample_id="s1"),
            _joined_row("b" * 64, sample_id="s1"),  # same region, a second perturb_op
        ]
    )
    item_metrics = [
        _item("a" * 64, "s1", "fake_continuous", degradation=2.0),
        _item("b" * 64, "s1", "fake_continuous", degradation=4.0),
    ]

    result = aggregate_item_metrics(item_metrics, joined, _registry(), _resolved_config(tmp_path))

    assert len(result.spatial_profile) == 1
    assert result.spatial_profile[0].degradation == pytest.approx(3.0)
    # RegionMetrics is sample-grain (n_valid<=n_samples is enforced by its own
    # __post_init__ — there is no n_items field to compare against): one sample
    # contributing two items to the same region must count as n_samples=1, not 2,
    # and the region's mean is the sample's own per-region mean (3.0), matching
    # the SpatialProfile row above rather than a flat mean over both raw items.
    region_row = next(r for r in result.region_metrics if r.metric_name == "fake_continuous")
    assert region_row.n_samples == 1
    assert region_row.n_valid == 1
    assert region_row.metric_mean == pytest.approx(3.0)


def test_spatial_profile_region_geometry_ref_for_explicit_and_procedural_regions(
    tmp_path: Path,
) -> None:
    procedural_joined = _joined([_joined_row("a" * 64, sample_id="s1")])
    procedural_result = aggregate_item_metrics(
        [_item("a" * 64, "s1", "fake_continuous", degradation=1.0)],
        procedural_joined,
        _registry(),
        _resolved_config(tmp_path),
    )
    procedural_ref = procedural_result.spatial_profile[0].region_geometry_ref
    assert procedural_ref.ref is None
    assert procedural_ref.ref_hash is None

    explicit_joined = _joined(
        [
            _joined_row(
                "a" * 64,
                sample_id="s1",
                region_id="mask",
                region_instance_id="mask",
                region_kind=RegionKind.EXPLICIT,
            )
        ]
    )
    explicit_result = aggregate_item_metrics(
        [_item("a" * 64, "s1", "fake_continuous", degradation=1.0)],
        explicit_joined,
        _registry(),
        _resolved_config(tmp_path, explicit=True),
    )
    explicit_ref = explicit_result.spatial_profile[0].region_geometry_ref
    assert explicit_ref.ref is not None and explicit_ref.ref.endswith("mask.png")
    assert explicit_ref.ref_hash == "d" * 64


def test_primary_metric_must_be_registered(tmp_path: Path) -> None:
    joined = _joined([_joined_row("a" * 64, sample_id="s1")])
    item_metrics = [_item("a" * 64, "s1", "fake_continuous", degradation=1.0)]

    with pytest.raises(MetricsRegistryError, match="primary_metric not registered"):
        aggregate_item_metrics(
            item_metrics,
            joined,
            _registry(),
            _resolved_config(tmp_path),
            primary_metric="not_a_real_metric",
        )
