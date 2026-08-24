"""Spatial-sensitivity heatmap view (V2).

Overlays one metric's per-region degradation on the original image so a
developer can see whether sensitive regions sit on the object or scatter
across the background.

Unlike V1 (``mask_check.py``), this view's data input is ``SpatialProfile``
(the N3 Aggregator's output, loaded via ``ssat.metrics.store.load_metrics``)
rather than the raw dump — ``SpatialProfile``'s own docstring states every
row "already carries enough region geometry to re-resolve a mask without a
second dump join". A ``dump_root`` is still needed, but only for the
``metrics.viz -> core.source`` exception that lets this module load the
original image referenced by ``ResolvedConfig.source_provenance``.

Region geometry is re-resolved via ``core.region.RegionResolver`` from each
row's ``RegionGeometryRef`` (the same explicit dependency exception V1
relies on). Two region kinds cannot be reproduced here and raise
``DebugVizError``:

- ``random_area_match``: its mask generator requires an item-local RNG
  seeded from the dump's ``seed_used`` (ssat/core/region/mask_generators.py),
  but ``SpatialProfile`` rows are aggregated across items and carry no
  seed — there is no single seed to replay (a hard data constraint, the same
  shape as V1's explicit-region gap, just a different region kind).
- ``bbox_partition``/``skeleton_parts``/``gt_bbox``: reserved kinds the v1
  core has not implemented yet (``ssat/core/region/mask_factory.py``);
  ``RegionResolver.resolve`` itself raises for these, and this module wraps
  that failure as ``DebugVizError``.

``explicit`` regions, notably, *are* supported here even though V1 rejects
them: ``RegionGeometryRef.ref``/``ref_hash`` are filled in by the Aggregator
from the run's ``ResolvedConfig.regions`` (region-family config), not from
the raw dump columns V1 reads — so the reference is actually available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ssat.core.region import RegionResolutionError, RegionResolver
from ssat.core.region.skeleton_store import SkeletonBBoxStore
from ssat.core.region.types import RegionSpec
from ssat.core.source import SampleSource
from ssat.core.source.types import LoadError
from ssat.core.types import RegionKind
from ssat.metrics.aggregate import DEFAULT_PRIMARY_METRIC
from ssat.metrics.dump_reader import DumpHandle
from ssat.metrics.errors import DebugVizError
from ssat.metrics.store import load_metrics, verify_source_dump
from ssat.metrics.types import SpatialProfile
from ssat.metrics.viz._shared import decanonicalize, open_image_source, open_skeleton_store

__all__ = [
    "HeatmapView",
    "render_heatmap_panel",
    "resolve_heatmap_view",
    "save_heatmap_views",
    "select_spatial_profile_rows",
]


@dataclass(frozen=True, slots=True)
class HeatmapView:
    """One sample's per-pixel degradation intensity, reconstructed from SpatialProfile rows.

    Attributes:
        sample_id: Sample this view belongs to.
        metric_name: Registered Metric name the intensity values come from.
        original: Source pixels in ``(H, W, C)`` uint8 layout -- the
            ``frame_index``-th frame of a video source, or the sole frame of
            an image source.
        intensity: Degradation per pixel in ``(H, W)`` layout; ``NaN`` where
            no reconstructed region covers that pixel. For a per-frame
            ``(T, H, W)`` region mask, only its ``frame_index``-th slice is
            painted in.
        region_keys: Region keys actually painted into ``intensity``, in the
            order they were applied (``select_spatial_profile_rows``' sort
            order).
        frame_index: Which frame of the source clip ``original``/``intensity``
            were reconstructed from. Always ``0`` for an image source.
        num_frames: Total frames in the source clip ``frame_index`` was
            chosen from. Always ``1`` for an image source.
    """

    sample_id: str
    metric_name: str
    original: NDArray[np.uint8]
    intensity: NDArray[np.float64]
    region_keys: tuple[str, ...]
    frame_index: int = 0
    num_frames: int = 1


def select_spatial_profile_rows(
    spatial_profile: Sequence[SpatialProfile],
    *,
    metric_name: str,
    n_samples: int = 5,
    sample_ids: Sequence[str] | None = None,
) -> dict[str, list[SpatialProfile]]:
    """Group rows with an available degradation value by sample, for one metric.

    Args:
        spatial_profile: Rows from ``AggregationResult.spatial_profile``
            (typically reloaded via ``ssat.metrics.store.load_metrics``).
        metric_name: Registered metric name to select rows for.
        n_samples: Maximum number of samples to select when ``sample_ids``
            is not given. Ignored otherwise.
        sample_ids: Explicit samples to select, in the given order. When
            omitted, the first ``n_samples`` sample_ids in sorted order that
            have at least one eligible row are used.

    Returns:
        A mapping from sample_id to its rows (``metric_name`` matches and
        ``degradation is not None``), sorted by ``region_key``, in selection
        order.

    Raises:
        DebugVizError: If an explicitly requested sample has no eligible row.
    """

    eligible = [
        row
        for row in spatial_profile
        if row.metric_name == metric_name and row.degradation is not None
    ]
    by_sample: dict[str, list[SpatialProfile]] = {}
    for row in eligible:
        by_sample.setdefault(row.sample_id, []).append(row)
    for rows in by_sample.values():
        rows.sort(key=lambda row: row.region_key)

    if sample_ids is None:
        chosen = sorted(by_sample)[:n_samples]
    else:
        chosen = list(sample_ids)

    result: dict[str, list[SpatialProfile]] = {}
    for sample_id in chosen:
        rows = by_sample.get(sample_id)
        if not rows:
            raise DebugVizError(
                f"no available metric_name={metric_name!r} spatial_profile "
                f"row found for sample_id={sample_id!r}"
            )
        result[sample_id] = rows
    return result


def resolve_heatmap_view(
    sample_id: str,
    rows: Sequence[SpatialProfile],
    *,
    source: SampleSource,
    skeleton_store: SkeletonBBoxStore | None = None,
) -> HeatmapView:
    """Reconstruct one sample's per-pixel degradation heatmap.

    For a video source, the representative frame is the clip's middle frame
    (``num_frames // 2``), falling back to the nearest frame (by ascending
    ``|i - middle|``, ties toward the lower index) that has at least one
    tracked pixel in one of ``rows``' resolved masks -- this keeps the
    rendered overlay from landing on a frame where every evaluated region
    happens to be untracked. An image source (``num_frames == 1``) always
    uses frame 0, exactly as before this parameter existed.

    Args:
        sample_id: Sample the rows belong to.
        rows: This sample's ``SpatialProfile`` rows (see
            ``select_spatial_profile_rows``), all sharing one metric_name.
        source: Sample source already built from the dump's
            ``source_provenance`` (see ``save_heatmap_views``) -- an image or
            video source.
        skeleton_store: Pre-computed per-frame body-part bounding boxes,
            required only when ``rows`` includes a ``skeleton_parts`` region.

    Returns:
        The reconstructed view.

    Raises:
        DebugVizError: If ``rows`` is empty or mixes metric_name values, the
            source sample fails to load, a region is ``random_area_match``
            (not reproducible without an item-level seed), a
            ``skeleton_parts`` region is present without ``skeleton_store``,
            or region resolution otherwise fails.
    """

    if not rows:
        raise DebugVizError(f"no spatial_profile rows given for sample_id={sample_id!r}")
    metric_names = {row.metric_name for row in rows}
    if len(metric_names) != 1:
        raise DebugVizError(
            f"spatial_profile rows for sample_id={sample_id!r} mix metric_name values: "
            f"{sorted(metric_names)}"
        )

    loaded = source.load(sample_id)
    if isinstance(loaded, LoadError):
        raise DebugVizError(
            f"failed to load source image for sample_id={sample_id!r}: "
            f"{loaded.error_type}: {loaded.message}"
        )

    num_frames, height, width = loaded.original_shape[0], loaded.original_shape[1], loaded.original_shape[2]
    resolved = [
        (row, _resolve_region_mask(row, loaded.original_shape, skeleton_store=skeleton_store))
        for row in rows
    ]
    frame_index = _select_representative_frame(num_frames, [mask for _, mask in resolved])

    intensity = np.full((height, width), np.nan, dtype=np.float64)
    painted: list[str] = []
    for row, mask in resolved:
        frame_mask = mask[frame_index] if mask.ndim == 3 else mask
        intensity[frame_mask] = row.degradation
        painted.append(row.region_key)

    return HeatmapView(
        sample_id=sample_id,
        metric_name=rows[0].metric_name,
        original=loaded.array[frame_index],
        intensity=intensity,
        region_keys=tuple(painted),
        frame_index=frame_index,
        num_frames=num_frames,
    )


def _select_representative_frame(
    num_frames: int, masks: Sequence[NDArray[np.bool_]]
) -> int:
    """Pick the frame a video sample's heatmap overlay/thumbnail should show.

    Middle frame by default, falling back to the nearest frame (ascending
    ``|i - middle|``, ties toward the lower index) that has at least one
    ``True`` pixel in some per-frame ``(T, H, W)`` mask in ``masks`` --
    avoids landing on a frame where every evaluated region is untracked.
    Falls back to the middle frame itself if no per-frame mask has a
    tracked pixel on any frame (e.g. every mask is 2D, or the sample is
    untracked throughout).

    Args:
        num_frames: Total frames in the source clip.
        masks: Every row's resolved mask for this sample, ``(H, W)`` or
            ``(T, H, W)``.

    Returns:
        A frame index in ``[0, num_frames)``.
    """

    middle = num_frames // 2
    if num_frames <= 1:
        return 0

    per_frame_masks = [mask for mask in masks if mask.ndim == 3]
    if not per_frame_masks:
        return middle

    candidates = sorted(range(num_frames), key=lambda i: (abs(i - middle), i))
    for candidate in candidates:
        if any(mask[candidate].any() for mask in per_frame_masks):
            return candidate
    return middle


def _resolve_region_mask(
    row: SpatialProfile,
    shape: tuple[int, int, int, int],
    *,
    skeleton_store: SkeletonBBoxStore | None = None,
) -> NDArray[np.bool_]:
    """Re-resolve one SpatialProfile row's mask via RegionResolver.

    Args:
        row: The row whose mask to re-resolve.
        shape: Source sample shape in ``(T, H, W, C)`` layout.
        skeleton_store: Pre-computed per-frame body-part bounding boxes,
            required only for a ``skeleton_parts`` region.

    Raises:
        DebugVizError: If the region is ``random_area_match``, or resolution
            otherwise fails (e.g. a ``skeleton_parts`` region without
            ``skeleton_store``).
    """

    geometry = row.region_geometry_ref
    if geometry.region_kind is RegionKind.RANDOM_AREA_MATCH:
        raise DebugVizError(
            "heatmap cannot reproduce random_area_match regions: no item-level "
            f"seed is retained past aggregation (region_key={row.region_key!r})"
        )

    region_id, _, region_instance_id = row.region_key.partition("::")
    if geometry.region_kind is RegionKind.EXPLICIT:
        spec = RegionSpec(
            region_id=region_id,
            region_instance_id=region_instance_id,
            kind=geometry.region_kind,
            ref=geometry.ref,
            ref_hash=geometry.ref_hash,
        )
    else:
        spec = RegionSpec(
            region_id=region_id,
            region_instance_id=region_instance_id,
            kind=geometry.region_kind,
            params=decanonicalize(json.loads(geometry.region_params_json)),
        )

    try:
        mask, _ = RegionResolver(skeleton_store=skeleton_store).resolve(shape, spec)
    except RegionResolutionError as error:
        raise DebugVizError(
            f"failed to resolve region_key={row.region_key!r}: {error}"
        ) from error
    return mask


def render_heatmap_panel(axes: tuple[Axes, Axes], view: HeatmapView) -> None:
    """Draw one view's original/heatmap-overlay panels onto two given axes.

    Public so ``ranking.py`` can reuse it for its top/bottom composite views.

    Args:
        axes: ``(original_axis, overlay_axis)`` to draw into.
        view: Reconstructed view to render.
    """

    original_axis, overlay_axis = axes

    original_axis.imshow(view.original)
    original_axis.set_title("original")

    valid = ~np.isnan(view.intensity)
    normalized = np.zeros_like(view.intensity)
    if valid.any():
        vmin = float(np.nanmin(view.intensity))
        vmax = float(np.nanmax(view.intensity))
        span = vmax - vmin
        normalized[valid] = (view.intensity[valid] - vmin) / span if span > 0 else 0.5

    colored = colormaps["inferno"](normalized)
    colored[..., 3] = np.where(valid, 0.6, 0.0)

    overlay_axis.imshow(view.original)
    overlay_axis.imshow(colored)
    overlay_axis.set_title(f"{view.metric_name} heatmap")

    for axis in axes:
        axis.axis("off")


def save_heatmap_views(
    dump_root: str | Path,
    metrics_dir: str | Path,
    output_dir: str | Path,
    *,
    metric_name: str = DEFAULT_PRIMARY_METRIC,
    n_samples: int = 5,
    sample_ids: Sequence[str] | None = None,
) -> tuple[Path, ...]:
    """Render and save one 2-panel PNG per selected sample.

    Panels, left to right: the original image, and the original image with
    a per-pixel degradation heatmap overlaid.

    Args:
        dump_root: Root directory of the raw audit dump these metrics were
            computed from.
        metrics_dir: Directory holding a stored aggregation run (see
            ``ssat.metrics.store.save_metrics``).
        output_dir: Directory PNGs are written into (created if missing).
        metric_name: Registered metric to visualize (default matches
            ``AggregationResult``'s ``vulnerability_score`` source).
        n_samples: See ``select_spatial_profile_rows``.
        sample_ids: See ``select_spatial_profile_rows``.

    Returns:
        The saved PNG paths, one per selected sample.

    Raises:
        DebugVizError: If the dump has no ``source_provenance``, or a
            selected sample cannot be reconstructed.
        MetricsCorruptionError: If ``metrics_dir`` no longer matches
            ``dump_root`` (see ``ssat.metrics.store.verify_source_dump``).
    """

    source = open_image_source(dump_root)
    skeleton_store = open_skeleton_store(dump_root)
    _, result, manifest = load_metrics(Path(metrics_dir))
    verify_source_dump(manifest, DumpHandle(dump_root).manifest_path)

    rows_by_sample = select_spatial_profile_rows(
        result.spatial_profile,
        metric_name=metric_name,
        n_samples=n_samples,
        sample_ids=sample_ids,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for sample_id, rows in rows_by_sample.items():
        view = resolve_heatmap_view(sample_id, rows, source=source, skeleton_store=skeleton_store)
        png_path = output_path / f"sample_{sample_id}.png"
        _save_view_png(view, png_path)
        saved.append(png_path)
    return tuple(saved)


def _save_view_png(view: HeatmapView, path: Path) -> None:
    """Draw and save one view's 2-panel figure.

    Args:
        view: Reconstructed view to render.
        path: Destination PNG path.
    """

    figure = Figure(figsize=(8, 4))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    render_heatmap_panel((axes[0], axes[1]), view)
    figure.suptitle(_view_caption(view))
    figure.savefig(path)


def _view_caption(view: HeatmapView) -> str:
    """Build a figure caption, noting the selected frame for a video sample."""

    caption = f"sample_id={view.sample_id}"
    if view.num_frames > 1:
        caption += f" frame={view.frame_index}/{view.num_frames}"
    return caption
