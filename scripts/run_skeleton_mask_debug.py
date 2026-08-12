#!/usr/bin/env python3
"""Manually rasterize one skeleton body-part track and save its per-frame masks.

This is a developer convenience, not a test: pytest already covers
SkeletonPartsMaskGenerator's rasterization math (``tests/unit/
test_skeleton_mask_generator.py``) and SkeletonRegionProvider's expansion
(``tests/unit/test_skeleton_provider.py``). This script exists so a
developer can point it at a real (or hand-crafted) skeleton bbox JSON file
and look at the resulting per-frame PNGs with their own eyes, run inside the
``region-sensitivity-workspace`` container.

Example:
    python3 scripts/run_skeleton_mask_debug.py \\
        --skeleton-json /path/to/skeleton_frame_bbox.json \\
        --sample-id clip_001 --body-part left_arm \\
        --output-dir /tmp/debug_viz/skeleton_mask
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ssat.core.region.mask_base import ExplicitMaskCache, MaskResolutionContext
from ssat.core.region.skeleton_mask_generator import SkeletonPartsMaskGenerator
from ssat.core.region.skeleton_store import SkeletonDataError, load_skeleton_bbox_store
from ssat.core.region.types import RegionSpec
from ssat.core.types import RegionKind


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one rasterization run."""

    parser = argparse.ArgumentParser(
        description=(
            "Rasterize one sample's tracked skeleton body part and save its "
            "per-frame boolean masks as PNGs for visual review."
        )
    )
    parser.add_argument(
        "--skeleton-json", type=Path, required=True, help="skeleton bbox JSON file"
    )
    parser.add_argument("--sample-id", required=True, help="sample_id to rasterize")
    parser.add_argument("--body-part", required=True, help="body part name to rasterize")
    parser.add_argument(
        "--bbox-scale",
        type=float,
        default=1.0,
        help="scale factor applied to each bbox about its center (default: 1.0)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="destination for per-frame PNGs"
    )
    return parser.parse_args()


def main() -> None:
    """Load one skeleton store, rasterize one track, and save mask PNGs."""

    args = parse_args()
    try:
        store = load_skeleton_bbox_store(args.skeleton_json)
    except SkeletonDataError as error:
        raise SystemExit(f"failed to load skeleton data: {error}") from error

    frame_size = store.frame_size(args.sample_id)
    if frame_size is None:
        raise SystemExit(f"no skeleton data for sample_id={args.sample_id!r}")
    width, height = frame_size

    context = MaskResolutionContext(
        explicit_cache=ExplicitMaskCache(1),
        resolve_target=lambda h, w, spec: np.zeros((h, w), dtype=np.bool_),
        skeleton_store=store,
    )
    generator = SkeletonPartsMaskGenerator(context)
    spec = RegionSpec(
        region_id="skeleton_debug",
        region_instance_id=f"skeleton_debug/{args.sample_id}",
        kind=RegionKind.SKELETON_PARTS,
        params={
            "sample_id": args.sample_id,
            "body_part": args.body_part,
            "bbox_scale": args.bbox_scale,
        },
    )
    mask = generator.get_mask(height, width, spec)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for frame_index in range(mask.shape[0]):
        image = Image.fromarray(mask[frame_index].astype(np.uint8) * 255, mode="L")
        destination = args.output_dir / f"frame_{frame_index:04d}.png"
        image.save(destination)

    print(
        f"saved {mask.shape[0]} frame mask(s) "
        f"({width}x{height}) to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
