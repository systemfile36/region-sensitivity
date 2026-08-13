"""Shared constants and small helpers for the L3 synthetic-shortcut experiment.

Design reference: docs/STAGE9_SYNTHETIC_SHORTCUT_DESIGN_v1.md. Every numeric
choice here (grid size, patch geometry, class-color palette) implements that
document's section 3. These are treated as pre-registered experiment
parameters, not tunables -- changing them after seeing Q1-Q5 results would
defeat the point of pre-registration (design doc section 5/9), so this
module intentionally exposes them as module-level constants rather than CLI
flags that could be nudged run-to-run.
"""

from __future__ import annotations

import colorsys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    # Deferred so scripts that only need the data-prep constants (prepare_data.py,
    # train.py) don't pull in ssat.analysis at import time -- only
    # build_item_values (used by the analysis-side scripts) needs the type.
    from ssat.analysis import AnalysisReader

__all__ = [
    "CELL_SIZE",
    "CROP_FREE_PREPROCESSING_OPS",
    "GRID_COLS",
    "GRID_ROWS",
    "IMAGE_SIZE",
    "ManifestSample",
    "NUM_CLASSES",
    "PATCH_COL",
    "PATCH_COLORS",
    "PATCH_REGION_ID",
    "PATCH_REGION_INSTANCE_ID",
    "PATCH_REGION_KEY",
    "PATCH_ROW",
    "build_crop_free_transform",
    "build_item_values",
    "draw_patch",
    "write_manifest",
]

NUM_CLASSES = 10
IMAGE_SIZE = 32  # CIFAR-10's native resolution; kept as-is per design section 1.

# A 4x4 grid over a 32x32 image gives 8x8-pixel cells (design section 3):
# small enough that the patch's area is a clear minority of the image (this
# avoids the "biggest region always wins" pitfall the design doc explicitly
# warns about), while still being one full cell rather than a tiny dot -- a
# dot-sized patch would be too fragile to survive the model adapter's forced
# Resize+CenterCrop preprocessing (see prepare_data.py's module docstring).
GRID_ROWS = 4
GRID_COLS = 4
CELL_SIZE = IMAGE_SIZE // GRID_ROWS  # 8

# The patch always sits in the top-left cell (row=0, col=0), matching the
# design doc's example placement. The region_key format below mirrors the
# f"{region_id}::{region_instance_id}" convention used throughout
# ssat.metrics (see ssat/metrics/aggregate.py), so evaluate.py can look this
# exact key up in RegionMetrics rows without re-deriving it.
PATCH_REGION_ID = "grid"
PATCH_ROW = 0
PATCH_COL = 0
PATCH_REGION_INSTANCE_ID = f"grid/r{PATCH_ROW}/c{PATCH_COL}"
PATCH_REGION_KEY = f"{PATCH_REGION_ID}::{PATCH_REGION_INSTANCE_ID}"

# A crop-free alternative to squeezenet1_0's ImageNet preset
# (Resize(256)->CenterCrop(224)), used to remove the CenterCrop-induced
# model-space area confound documented in
# docs/CONTROL_STABILITY_DESIGN_v1.md section 0's addendum: under the
# preset, nominally-equal-area grid cells land at different effective
# pixel counts depending on position (corner/edge/center), which turned
# out to correlate with the fill-strategy sign split the design doc
# originally took as a substantive finding. Resizing directly to 224x224
# (no crop) makes every cell's effective area identical by construction.
# mean/std match the preset's values (see SqueezeNet1_0_Weights.DEFAULT
# .transforms()) so the only thing that changes is the crop step, not the
# normalization statistics the model was implicitly designed around.
CROP_FREE_PREPROCESSING_OPS: tuple[dict, ...] = (
    {"op": "resize", "size": [224, 224]},
    {"op": "to_float"},
    {"op": "normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    {"op": "channels_first"},
    {"op": "squeeze_time"},
)


def _class_color_palette() -> tuple[tuple[int, int, int], ...]:
    """Build one fully-saturated RGB color per class, evenly spaced by hue.

    Spacing hues evenly around the color wheel (design section 3: "클래스별로
    다른 색") maximizes the perceptual distance between any two classes'
    colors. This is a closed-form computation with no randomness involved,
    so the palette is bit-identical across every run without needing to
    persist or replay a seed for it.
    """

    colors = []
    for class_index in range(NUM_CLASSES):
        hue = class_index / NUM_CLASSES
        red, green, blue = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append((round(red * 255), round(green * 255), round(blue * 255)))
    return tuple(colors)


PATCH_COLORS: tuple[tuple[int, int, int], ...] = _class_color_palette()


def draw_patch(
    image: NDArray[np.uint8],
    class_index: int,
    *,
    row: int,
    col: int,
) -> NDArray[np.uint8]:
    """Return a copy of ``image`` with one grid cell filled by a solid class color.

    Args:
        image: Source image, shape (H, W, 3), dtype uint8.
        class_index: Index into PATCH_COLORS selecting the fill color.
        row: Grid row of the cell to fill (0-indexed).
        col: Grid column of the cell to fill (0-indexed).

    Returns:
        A new array; ``image`` itself is left untouched so callers can reuse
        the same source array to render multiple variants (A/B/C) of one
        photo without re-reading it from disk.
    """

    patched = image.copy()
    row_start, row_end = row * CELL_SIZE, (row + 1) * CELL_SIZE
    col_start, col_end = col * CELL_SIZE, (col + 1) * CELL_SIZE
    patched[row_start:row_end, col_start:col_end, :] = PATCH_COLORS[class_index]
    return patched


@dataclass(frozen=True, slots=True)
class ManifestSample:
    """One row of an ssat image_manifest source.

    Field names deliberately match ssat.application.config._ManifestSample
    (sample_id/path/gt_label) so write_manifest's output is a direct,
    schema-compatible input to ssat's image_manifest source config.
    """

    sample_id: str
    path: Path
    gt_label: int


def write_manifest(path: Path, samples: Sequence[ManifestSample]) -> None:
    """Write an image_manifest JSON document ssat's source loader can read.

    Image paths are stored as resolved absolute strings so the manifest
    stays valid regardless of which directory a downstream script (train.py,
    run_audit.py, ...) happens to be invoked from.
    """

    document = {
        "samples": [
            {
                "sample_id": sample.sample_id,
                "path": str(sample.path.resolve()),
                "gt_label": sample.gt_label,
            }
            for sample in samples
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def build_crop_free_transform():
    """Build a PIL-image-to-tensor transform using ssat's own preprocessing code.

    Returns a callable equivalent in spirit to
    ``SqueezeNet1_0_Weights.DEFAULT.transforms()`` (the object train.py used
    to fetch to keep train/audit preprocessing identical), but for the
    crop-free pipeline. Rather than hand-writing a second implementation of
    ``CROP_FREE_PREPROCESSING_OPS`` in torchvision-transform terms -- which
    would reintroduce exactly the kind of "two independent
    implementations that are supposed to agree" risk this experiment is
    trying to eliminate -- this drives
    ``ssat.core.adapter.preprocessing.DeclarativePreprocessor`` directly,
    the same class ``TorchvisionAdapter(preprocessing_ops=...)`` uses at
    audit time. Train-time and audit-time preprocessing are therefore the
    same code path, not just numerically similar outputs.

    Local imports (torch, DeclarativePreprocessor) keep this out of
    prepare_data.py's / other lightweight consumers' import graph, matching
    train.py's existing pattern for its own torchvision import.
    """

    import torch
    from ssat.core.adapter.preprocessing import DeclarativePreprocessor

    preprocessor = DeclarativePreprocessor(CROP_FREE_PREPROCESSING_OPS)

    def transform(image) -> "torch.Tensor":
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        batch = array[None, None, :, :, :]  # (B=1, T=1, H, W, C)
        transformed = preprocessor.transform_batch(batch)  # (1, C, H, W)
        return torch.from_numpy(np.ascontiguousarray(transformed[0]))

    return transform


def build_item_values(reader: "AnalysisReader") -> pd.DataFrame:
    """Join one AnalysisReader's item_context() identity columns with its item_metrics.

    Thin wrapper kept for the existing call sites (analyze_control_stability.py,
    validate_reliability_thresholds.py) -- the join itself now lives on
    ``AnalysisReader.item_values()`` (promoted there once the Application
    layer's ``analyze()`` became a second production caller needing the same
    frame; see that method's docstring).
    """

    return reader.item_values()
