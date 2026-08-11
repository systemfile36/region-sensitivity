"""Mask-verification panel view (V1).

Renders, for a handful of samples, a 3-panel PNG (original image / mask
overlay / actual perturbed image) so a developer can catch coordinate-system
bugs (flipped axes, wrong region-index mapping, inverted ``invert_mask``,
unexpected crop) that a numeric table would never surface (design
METRIC_ENGINE_DESIGN_v1.md §N5 "왜 필요한가").

This module needs only the raw dump and ``core.region.RegionResolver`` —
``SpatialProfile``/``MetricRegistry`` are irrelevant here (design §N5 "V1").
Reproducing the *actual* perturbed image byte-for-byte additionally requires
``core.source`` (to load the original image referenced by
``ResolvedConfig.source_provenance``) and ``core.perturb`` (to re-apply the
exact perturbation). These two imports, together with ``core.region``, are
the explicit dependency-direction exceptions granted to ``metrics.viz``
(IMPLE_PLAN_METRIC_DESIGN_v1.md §3.3, extended for this stage — see
docs/IMPLE_PLAN_METRIC_DESIGN_v1.md 단계 7 plan notes).

Reproduction is exact because it replays the same RNG consumption order the
core runtime uses (ssat/core/runtime/processors.py): one ``default_rng``
seeded from the dump's already-derived ``seed_used`` is consumed first by
``RegionResolver.resolve`` (mask materialization), then, after any
``invert_mask`` flip, by ``Perturbator.apply`` (perturbation). Because
``seed_used`` is the final derived seed (not the run's global seed), no other
context is needed to replay this sequence deterministically.

Explicit regions are not supported: their ``ref``/``ref_hash`` are consumed
during config resolution and never written to the raw dump or ``JoinedFrame``
(design §N0), so there is no way to locate the original mask file from a
dump alone. Rows with ``region_kind == "explicit"`` raise ``DebugVizError``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ssat.core.perturb import Perturbator
from ssat.core.region import RegionResolver
from ssat.core.region.types import RegionSpec
from ssat.core.source import ImageFolderSource
from ssat.core.source.types import LoadError
from ssat.core.types import ItemStatus, PerturbationOp, RegionKind
from ssat.metrics.dump_reader import DumpHandle, JoinedFrame
from ssat.metrics.errors import DebugVizError
from ssat.metrics.viz._shared import decanonicalize, open_image_source

__all__ = [
    "MaskCheckView",
    "resolve_mask_check_view",
    "save_mask_check_views",
    "select_mask_check_rows",
]


@dataclass(frozen=True, slots=True)
class MaskCheckView:
    """One reconstructed (original, mask, perturbed) triple for one item.

    Attributes:
        sample_id: Owning clean sample identifier.
        item_id: Perturbed item identifier this view reconstructs.
        region_key: ``f"{region_id}::{region_instance_id}"``.
        original: Source pixels in ``(H, W, C)`` uint8 layout.
        mask: Boolean selection in ``(H, W)`` layout, after ``invert_mask``.
        perturbed: Reconstructed perturbed pixels in ``(H, W, C)`` layout.
    """

    sample_id: str
    item_id: str
    region_key: str
    original: NDArray[np.uint8]
    mask: NDArray[np.bool_]
    perturbed: NDArray[np.uint8]


def select_mask_check_rows(
    joined: JoinedFrame,
    *,
    n_samples: int = 5,
    sample_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Pick one successfully-perturbed row per selected sample.

    Args:
        joined: Item-grain frame produced by ``DumpHandle.joined()``.
        n_samples: Maximum number of samples to select when ``sample_ids``
            is not given. Ignored otherwise.
        sample_ids: Explicit samples to select, in the given order. When
            omitted, the first ``n_samples`` sample_ids in sorted order that
            have at least one row with ``status_perturbed == "ok"`` are used.

    Returns:
        One row per selected sample (the first eligible row, sorted by
        ``region_id``/``region_instance_id``), indexed 0..n-1.
    """

    eligible = joined[joined["status_perturbed"] == ItemStatus.OK.value]
    if sample_ids is None:
        chosen = sorted(eligible["sample_id"].unique())[:n_samples]
    else:
        chosen = list(sample_ids)

    rows = []
    for sample_id in chosen:
        candidates = eligible[eligible["sample_id"] == sample_id].sort_values(
            ["region_id", "region_instance_id"]
        )
        if candidates.empty:
            raise DebugVizError(
                f"no successfully-perturbed item found for sample_id={sample_id!r}"
            )
        rows.append(candidates.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def resolve_mask_check_view(
    row: Mapping[str, object],
    *,
    source: ImageFolderSource,
) -> MaskCheckView:
    """Reconstruct one view: load the image, resolve the mask, reproduce the perturbation.

    Args:
        row: One row of a ``JoinedFrame`` (or an equivalent mapping) with the
            columns documented in ``ssat.metrics.dump_reader.JoinedFrame``.
        source: Image source already built from the dump's
            ``source_provenance`` (see ``save_mask_check_views``).

    Returns:
        The reconstructed view.

    Raises:
        DebugVizError: If the row's region is ``explicit``, the source image
            fails to load, or region/perturbation reproduction fails.
    """

    region_kind = RegionKind(row["region_kind"])
    if region_kind is RegionKind.EXPLICIT:
        raise DebugVizError(
            "mask_check cannot reproduce explicit regions: ref/ref_hash are "
            f"not preserved in the dump (sample_id={row['sample_id']!r}, "
            f"item_id={row['item_id']!r})"
        )

    loaded = source.load(str(row["sample_id"]))
    if isinstance(loaded, LoadError):
        raise DebugVizError(
            f"failed to load source image for sample_id={row['sample_id']!r}: "
            f"{loaded.error_type}: {loaded.message}"
        )

    spec = RegionSpec(
        region_id=str(row["region_id"]),
        region_instance_id=str(row["region_instance_id"]),
        kind=region_kind,
        params=decanonicalize(json.loads(str(row["region_params_json"]))),
    )
    # seed_used is persisted as a fixed-width hex string (ssat.core.dump.types.seed_to_hex),
    # not a decimal string, so it must be parsed with base 16.
    rng = np.random.default_rng(int(row["seed_used"], 16))
    mask, _ = RegionResolver().resolve(loaded.original_shape, spec, rng)
    if bool(row["invert_mask"]):
        mask = np.logical_not(mask)

    perturb_params = decanonicalize(json.loads(str(row["perturb_params_json"])))
    perturbed = Perturbator().apply(
        loaded.array,
        mask,
        PerturbationOp(row["perturb_op"]),
        perturb_params,
        rng,
    )

    return MaskCheckView(
        sample_id=str(row["sample_id"]),
        item_id=str(row["item_id"]),
        region_key=f"{row['region_id']}::{row['region_instance_id']}",
        original=loaded.array[0],
        mask=mask,
        perturbed=perturbed[0],
    )


def save_mask_check_views(
    dump_root: str | Path,
    output_dir: str | Path,
    *,
    n_samples: int = 5,
    sample_ids: Sequence[str] | None = None,
) -> tuple[Path, ...]:
    """Render and save one 3-panel PNG per selected sample.

    Panels, left to right: the original image, the original image with the
    resolved mask overlaid, and the reconstructed perturbed image.

    Args:
        dump_root: Root directory of a raw audit dump.
        output_dir: Directory PNGs are written into (created if missing).
        n_samples: See ``select_mask_check_rows``.
        sample_ids: See ``select_mask_check_rows``.

    Returns:
        The saved PNG paths, one per selected sample, in selection order.

    Raises:
        DebugVizError: If the dump has no ``source_provenance``, or a
            selected row cannot be reconstructed (see
            ``resolve_mask_check_view``).
    """

    handle = DumpHandle(dump_root)
    source = open_image_source(dump_root)

    rows = select_mask_check_rows(handle.joined(), n_samples=n_samples, sample_ids=sample_ids)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for _, row in rows.iterrows():
        view = resolve_mask_check_view(row, source=source)
        png_path = output_path / f"sample_{view.sample_id}.png"
        _save_panel_png(view, png_path)
        saved.append(png_path)
    return tuple(saved)


def _save_panel_png(view: MaskCheckView, path: Path) -> None:
    """Draw and save the 3-panel figure for one view.

    Args:
        view: Reconstructed view to render.
        path: Destination PNG path.
    """

    figure = Figure(figsize=(12, 4))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)

    axes[0].imshow(view.original)
    axes[0].set_title("original")

    overlay = view.original.astype(np.float64).copy()
    overlay[view.mask] = 0.5 * overlay[view.mask] + 0.5 * np.array([255.0, 0.0, 0.0])
    axes[1].imshow(overlay.astype(np.uint8))
    axes[1].set_title(f"mask overlay ({view.region_key})")

    axes[2].imshow(view.perturbed)
    axes[2].set_title("perturbed")

    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"sample_id={view.sample_id} item_id={view.item_id}")
    figure.savefig(path)
