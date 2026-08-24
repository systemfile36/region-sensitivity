"""N3 Aggregator: reduce ItemMetrics into SampleMetrics/RegionMetrics/ClassMetrics/SpatialProfile.

The four output types are all long-form, one row per ``(entity, metric_name)``
— see metrics/types.py's module docstring.

**Two aggregation grains coexist here, forced by the frozen schemas.**
``SampleMetrics`` is item-grain: its ``n_items``/``n_valid`` count
raw ``ItemMetrics`` rows within one sample, and its ``__post_init__``
enforces ``n_valid <= n_items``. ``RegionMetrics``, by contrast, has no
``n_items`` field at all, and its ``__post_init__`` enforces
``n_valid <= n_samples`` — i.e. ``n_valid`` there counts *samples*, not
items. A sample contributing several items to the same region (e.g. two
perturb_ops on one region) would otherwise let ``n_valid`` exceed
``n_samples``. So region/class aggregation is sample-grain: each sample's
items are first reduced to one per-sample value (exactly what
``SpatialProfile`` already computes for regions, and what
``SampleMetrics.metric_mean`` already computes overall), and only those
per-sample values are averaged across samples for ``RegionMetrics``/
``ClassMetrics``. This is a macro-average (average of per-sample rates) for
``flip_rate`` at those two axes, not the item-level micro-average used at
the sample axis — deliberately, so one sample with many perturbations never
dominates a region's or class's statistic.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ssat.core.config import ResolvedConfig, ResolvedRegionConfig
from ssat.core.types import RegionKind
from ssat.metrics.dump_reader import JoinedFrame, coerce_nullable_int
from ssat.metrics.errors import MetricsCorruptionError, MetricsRegistryError
from ssat.metrics.registry import MetricRegistry
from ssat.metrics.types import (
    ClassMetrics,
    ItemMetrics,
    RegionGeometryRef,
    RegionMetrics,
    SampleMetrics,
    SpatialProfile,
)
from ssat.metrics.types import region_key as format_region_key

DEFAULT_PRIMARY_METRIC = "margin_drop"


@dataclass(frozen=True, slots=True)
class AggregationResult:
    """Bundle every N3 aggregate produced from one run's ItemMetrics.

    Attributes:
        sample_metrics: One row per (sample_id, metric_name).
        region_metrics: One row per (region_key, metric_name).
        class_metrics: One row per (gt_label, metric_name).
        spatial_profile: One row per (sample_id, region_key, metric_name).
    """

    sample_metrics: list[SampleMetrics]
    region_metrics: list[RegionMetrics]
    class_metrics: list[ClassMetrics]
    spatial_profile: list[SpatialProfile]


@dataclass(frozen=True, slots=True)
class _ItemContext:
    """Per-item fields ItemMetrics itself does not carry, needed for aggregation.

    ``gt_label`` is ``None`` exactly when the item's ``ExclusionReason`` (on
    every one of its ``ItemMetrics`` rows) is ``GT_LABEL_UNKNOWN`` — the
    same "label-free auditing is a supported scenario" case
    ``coerce_nullable_int`` exists for.
    """

    sample_id: str
    gt_label: int | None
    region_key: str
    region_kind: RegionKind
    intended_area_px: int | None
    effective_area_px: int | None
    region_geometry_ref: RegionGeometryRef
    is_control: bool


def aggregate_item_metrics(
    item_metrics: Sequence[ItemMetrics],
    joined: JoinedFrame,
    registry: MetricRegistry,
    resolved_config: ResolvedConfig,
    *,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> AggregationResult:
    """Aggregate one run's ItemMetrics into every N3 sample/region/class/spatial view.

    ``item_metrics`` must have been produced by ``registry`` against this same
    ``joined`` frame. Items whose ``is_control`` flag is set are excluded
    before any aggregation — control-group analysis is a later module's
    responsibility.

    ``RegionMetrics``/``ClassMetrics`` intentionally do not stratify by
    clean_correct: that schema (metrics/types.py) has no
    clean_correct column at those axes, only SampleMetrics does (where each
    row already carries its own sample's clean_correct without needing to
    split rows). Region/class statistics therefore pool every non-control
    item regardless of clean_correct.

    Samples with an unknown ``gt_label`` (label-free auditing,
    ``ExclusionReason.GT_LABEL_UNKNOWN``) remain in ``SampleMetrics``,
    ``RegionMetrics``, and ``SpatialProfile`` — their per-item values are
    simply unavailable, the same as any other excluded item — but are
    skipped from ``ClassMetrics`` entirely, since that axis has no
    ``gt_label=None`` row to belong to (see ``_aggregate_class_metrics``).

    Args:
        item_metrics: Long-form item-level rows from ``registry.compute_item_metrics``.
        joined: The same JoinedFrame that produced ``item_metrics``, used to
            recover per-item sample/region/class context ItemMetrics itself
            omits.
        registry: The registry that computed ``item_metrics`` — consulted for
            each metric's ``kind`` (drives ``flip_rate``) and to validate
            ``primary_metric``.
        resolved_config: The source run's resolved config, used to look up
            each region family's explicit-mask ``ref``/``ref_hash`` (not
            stored per-item in the dump).
        primary_metric: Registered metric name whose degradation defines
            ``vulnerability_score`` (default ``margin_drop``).

    Raises:
        MetricsRegistryError: If ``primary_metric`` is not registered.
        MetricsCorruptionError: If the same concrete region reports
            inconsistent geometry (region_kind or area) across items.
    """

    if primary_metric not in registry.names:
        raise MetricsRegistryError(f"primary_metric not registered: {primary_metric}")

    region_families = {region.region_id: region for region in resolved_config.regions}
    context = _build_context(joined, region_families)
    rows = [row for row in item_metrics if not context[row.item_id].is_control]

    sample_metrics = _aggregate_sample_metrics(rows, context, registry, primary_metric)
    spatial_profile = _aggregate_spatial_profile(rows, context)
    region_metrics = _aggregate_region_metrics(spatial_profile, context, registry)
    class_metrics = _aggregate_class_metrics(sample_metrics, registry)

    return AggregationResult(sample_metrics, region_metrics, class_metrics, spatial_profile)


def _build_context(
    joined: JoinedFrame, region_families: dict[str, ResolvedRegionConfig]
) -> dict[str, _ItemContext]:
    """Build a per-item lookup and reject inconsistent per-region geometry.

    Geometry consistency is a property of one region *within one sample*,
    not of a region across the dataset: sample-dependent kinds (GT_BBOX,
    SKELETON_PARTS) legitimately resolve to a different box per sample, and
    a RANDOM_AREA_MATCH control is re-drawn per item by construction. The
    area maps are therefore keyed on ``(sample_id, region_key)``.

    Raises:
        MetricsCorruptionError: If a concrete region reports a different
            ``region_kind``, or a conflicting non-null area within the same
            sample, or if an EXPLICIT-kind row's ``region_id`` has no
            matching family in ``region_families``.
    """

    context: dict[str, _ItemContext] = {}
    kind_by_region: dict[str, RegionKind] = {}
    area_by_region: dict[tuple[str, str], tuple[int | None, int | None]] = {}

    for row in joined.itertuples(index=False):
        region_key = format_region_key(row.region_id, row.region_instance_id)
        region_kind = RegionKind(row.region_kind)
        # A missing area arrives from parquet as float NaN, not None, and
        # NaN != NaN would read as a genuine conflict below. Normalize once
        # here so everything downstream sees the documented ``int | None``.
        intended_area_px = _area_or_none(row.intended_area_px)
        effective_area_px = _area_or_none(row.effective_area_px)
        if region_kind is RegionKind.EXPLICIT:
            family = region_families.get(row.region_id)
            if family is None:
                raise MetricsCorruptionError(
                    f"region_id {row.region_id!r} not found in resolved "
                    "config's region families"
                )
            ref = family.ref.as_posix() if family.ref is not None else None
            ref_hash = family.ref_hash
        else:
            # Non-EXPLICIT regions (GRID, BBOX_PARTITION, SKELETON_PARTS,
            # RANDOM_AREA_MATCH -- including every control item, whose
            # region_id is a PlanBuilder-generated synthetic id that is
            # never itself a configured family) never carry ref/ref_hash
            # (RegionGeometryRef.__post_init__ forbids it), so no family
            # lookup is needed or performed here.
            ref = None
            ref_hash = None
        geometry_ref = RegionGeometryRef(
            region_kind=region_kind,
            region_params_json=row.region_params_json,
            ref=ref,
            ref_hash=ref_hash,
        )

        _check_region_geometry(
            kind_by_region,
            area_by_region,
            sample_id=row.sample_id,
            region_key=region_key,
            region_kind=region_kind,
            intended_area_px=intended_area_px,
            effective_area_px=effective_area_px,
        )

        context[row.item_id] = _ItemContext(
            sample_id=row.sample_id,
            gt_label=coerce_nullable_int(row.gt_label),
            region_key=region_key,
            region_kind=region_kind,
            intended_area_px=intended_area_px,
            effective_area_px=effective_area_px,
            region_geometry_ref=geometry_ref,
            is_control=bool(row.is_control),
        )

    return context


def _area_or_none(value: Any) -> int | None:
    """Coerce one parquet area cell to ``int``, mapping null/NaN to ``None``."""

    if value is None or value != value:  # NaN is the only value unequal to itself.
        return None
    return int(value)


def _check_region_geometry(
    kind_by_region: dict[str, RegionKind],
    area_by_region: dict[tuple[str, str], tuple[int | None, int | None]],
    *,
    sample_id: str,
    region_key: str,
    region_kind: RegionKind,
    intended_area_px: int | None,
    effective_area_px: int | None,
) -> None:
    """Track one region's geometry and reject a later, conflicting report.

    ``None`` areas (from perturbation failures) never conflict; they simply
    defer to whatever non-null value is already known.

    ``region_kind`` is checked dataset-wide (one region_key names one kind
    everywhere), but areas are checked only within a sample — see
    ``_build_context``. RANDOM_AREA_MATCH is exempt from the area check
    entirely: its mask is re-drawn from an item-local rng, so two seed
    salts of the same (sample, control slot) legitimately differ. Its real
    contract — matching the target's area — is enforced where the pairing
    is known (``ssat.analysis.indexer``), not here.

    Raises:
        MetricsCorruptionError: If ``region_kind`` differs from a previously
            recorded value for the same ``region_key``, or a non-null area
            differs from a previous value for the same
            ``(sample_id, region_key)``.
    """

    existing_kind = kind_by_region.setdefault(region_key, region_kind)
    if existing_kind is not region_kind:
        raise MetricsCorruptionError(f"region {region_key!r} reports inconsistent region_kind")

    if region_kind is RegionKind.RANDOM_AREA_MATCH:
        return

    new_areas = (intended_area_px, effective_area_px)
    geometry_key = (sample_id, region_key)
    existing_areas = area_by_region.get(geometry_key)
    if existing_areas is None:
        area_by_region[geometry_key] = new_areas
        return

    merged: list[int | None] = []
    for existing, new, field_name in zip(
        existing_areas, new_areas, ("intended_area_px", "effective_area_px")
    ):
        if existing is not None and new is not None and existing != new:
            raise MetricsCorruptionError(
                f"region {region_key!r} reports inconsistent {field_name} "
                f"within sample {sample_id!r}"
            )
        merged.append(existing if existing is not None else new)
    area_by_region[geometry_key] = (merged[0], merged[1])


def _aggregate_sample_metrics(
    rows: Sequence[ItemMetrics],
    context: dict[str, _ItemContext],
    registry: MetricRegistry,
    primary_metric: str,
) -> list[SampleMetrics]:
    """Item-grain: one row per (sample_id, metric_name), counting raw items."""

    groups: dict[tuple[str, str], list[ItemMetrics]] = defaultdict(list)
    for item in rows:
        groups[(item.sample_id, item.metric_name)].append(item)

    vulnerability_by_sample = _compute_vulnerability_scores(rows, primary_metric)

    result: list[SampleMetrics] = []
    for (sample_id, metric_name), items in groups.items():
        valid = [item.degradation for item in items if item.available]
        n_valid = len(valid)
        mean = float(np.mean(valid)) if n_valid else None
        kind = registry.get(metric_name).kind
        result.append(
            SampleMetrics(
                sample_id=sample_id,
                metric_name=metric_name,
                gt_label=context[items[0].item_id].gt_label,
                clean_correct=items[0].clean_correct,
                n_items=len(items),
                n_valid=n_valid,
                flip_rate=mean if kind == "binary" and n_valid else None,
                vulnerability_score=vulnerability_by_sample.get(sample_id),
                metric_mean=mean,
                metric_max=float(np.max(valid)) if n_valid else None,
                metric_std=float(np.std(valid)) if n_valid else None,
            )
        )
    return result


def _compute_vulnerability_scores(
    rows: Sequence[ItemMetrics], primary_metric: str
) -> dict[str, float]:
    """Average the primary metric's degradation per sample.

    Samples with zero valid primary-metric items are simply absent from the
    result, so callers naturally get ``None`` via ``dict.get``.
    """

    by_sample: dict[str, list[float]] = defaultdict(list)
    for item in rows:
        if item.metric_name == primary_metric and item.available:
            by_sample[item.sample_id].append(item.degradation)  # type: ignore[arg-type]
    return {sample_id: float(np.mean(values)) for sample_id, values in by_sample.items()}


def _aggregate_spatial_profile(
    rows: Sequence[ItemMetrics], context: dict[str, _ItemContext]
) -> list[SpatialProfile]:
    """Item-grain: one row per (sample_id, region_key, metric_name)."""

    groups: dict[tuple[str, str, str], list[ItemMetrics]] = defaultdict(list)
    for item in rows:
        groups[(item.sample_id, context[item.item_id].region_key, item.metric_name)].append(item)

    result: list[SpatialProfile] = []
    for (sample_id, region_key, metric_name), items in groups.items():
        valid = [item.degradation for item in items if item.available]
        result.append(
            SpatialProfile(
                sample_id=sample_id,
                region_key=region_key,
                metric_name=metric_name,
                region_geometry_ref=context[items[0].item_id].region_geometry_ref,
                degradation=float(np.mean(valid)) if valid else None,
            )
        )
    return result


def _aggregate_region_metrics(
    spatial_profile: Sequence[SpatialProfile],
    context: dict[str, _ItemContext],
    registry: MetricRegistry,
) -> list[RegionMetrics]:
    """Sample-grain: each (sample, region, metric) SpatialProfile row is one datum."""

    # Dataset-grain areas are a summary across samples, so they are averaged
    # rather than sampled from whichever item happened to come first: a
    # region whose geometry varies per sample (GT_BBOX, SKELETON_PARTS, or a
    # RANDOM_AREA_MATCH control re-drawn per item) has no single "the" area,
    # and an arbitrary representative would silently misreport it. For a
    # region with constant geometry the mean is exactly that constant.
    kind_by_region: dict[str, RegionKind] = {}
    intended_by_region: dict[str, list[int]] = defaultdict(list)
    effective_by_region: dict[str, list[int]] = defaultdict(list)
    for ctx in context.values():
        kind_by_region.setdefault(ctx.region_key, ctx.region_kind)
        if ctx.intended_area_px is not None:
            intended_by_region[ctx.region_key].append(ctx.intended_area_px)
        if ctx.effective_area_px is not None:
            effective_by_region[ctx.region_key].append(ctx.effective_area_px)

    def _mean_area(areas: list[int]) -> int | None:
        return int(round(sum(areas) / len(areas))) if areas else None

    groups: dict[tuple[str, str], list[SpatialProfile]] = defaultdict(list)
    for row in spatial_profile:
        groups[(row.region_key, row.metric_name)].append(row)

    result: list[RegionMetrics] = []
    for (region_key, metric_name), per_sample_rows in groups.items():
        region_kind = kind_by_region[region_key]
        intended_area_px = _mean_area(intended_by_region[region_key])
        effective_area_px = _mean_area(effective_by_region[region_key])
        valid = [row.degradation for row in per_sample_rows if row.degradation is not None]
        n_valid = len(valid)
        mean = float(np.mean(valid)) if n_valid else None
        kind = registry.get(metric_name).kind
        result.append(
            RegionMetrics(
                region_key=region_key,
                metric_name=metric_name,
                region_kind=region_kind,
                intended_area_px=intended_area_px,
                effective_area_px=effective_area_px,
                n_samples=len(per_sample_rows),
                n_valid=n_valid,
                flip_rate=mean if kind == "binary" and n_valid else None,
                metric_mean=mean,
            )
        )
    return result


def _aggregate_class_metrics(
    sample_metrics: Sequence[SampleMetrics], registry: MetricRegistry
) -> list[ClassMetrics]:
    """Sample-grain: each (sample, metric) SampleMetrics row is one datum.

    Samples whose ``gt_label`` is unknown (``None`` — label-free auditing,
    ``ExclusionReason.GT_LABEL_UNKNOWN``) are skipped here: ``ClassMetrics``
    is inherently keyed by a real ground-truth class, so there is no
    ``gt_label=None`` row for it to belong to. Such samples remain fully
    visible in ``SampleMetrics`` (``gt_label=None`` there is an explicit,
    typed value, not an omission) and in ``ItemMetrics`` (via
    ``excluded_reason``) — this is a narrowing of *this* axis only, not a
    silent drop of the underlying data.
    """

    groups: dict[tuple[int, str], list[SampleMetrics]] = defaultdict(list)
    for row in sample_metrics:
        if row.gt_label is None:
            continue
        groups[(row.gt_label, row.metric_name)].append(row)

    result: list[ClassMetrics] = []
    for (gt_label, metric_name), per_sample_rows in groups.items():
        valid = [row.metric_mean for row in per_sample_rows if row.metric_mean is not None]
        n_valid = len(valid)
        mean = float(np.mean(valid)) if n_valid else None
        kind = registry.get(metric_name).kind
        result.append(
            ClassMetrics(
                gt_label=gt_label,
                metric_name=metric_name,
                n_samples=len(per_sample_rows),
                flip_rate=mean if kind == "binary" and n_valid else None,
                metric_mean=mean,
            )
        )
    return result
