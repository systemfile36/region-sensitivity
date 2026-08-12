"""Tests for SkeletonPartsMaskGenerator bounding-box rasterization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ssat.core.region.mask_base import (
    ExplicitMaskCache,
    MaskResolutionContext,
    RegionResolutionError,
)
from ssat.core.region.skeleton_mask_generator import SkeletonPartsMaskGenerator
from ssat.core.region.skeleton_store import load_skeleton_bbox_store
from ssat.core.region.types import RegionSpec
from ssat.core.types import RegionKind


def _spec(**params: object) -> RegionSpec:
    """Build a concrete skeleton_parts recipe for one sample and part."""

    base = {"sample_id": "clip", "body_part": "left_arm", "bbox_scale": 1.0}
    base.update(params)
    return RegionSpec(
        region_id="region",
        region_instance_id="region/clip",
        kind=RegionKind.SKELETON_PARTS,
        params=base,
    )


def _store(tmp_path: Path, payload: object):
    path = tmp_path / "skeleton.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_skeleton_bbox_store(path)


def _context(store=None) -> MaskResolutionContext:
    return MaskResolutionContext(
        explicit_cache=ExplicitMaskCache(1),
        resolve_target=lambda h, w, s: np.zeros((h, w), dtype=np.bool_),
        skeleton_store=store,
    )


def test_supports_only_skeleton_parts_kind() -> None:
    generator = SkeletonPartsMaskGenerator(_context())

    assert generator.supports(_spec())
    assert not generator.supports(
        RegionSpec(
            region_id="r",
            region_instance_id="r/0",
            kind=RegionKind.GRID,
            params={"rows": 1, "cols": 1, "row_index": 0, "col_index": 0},
        )
    )


def test_rasterizes_exact_bbox_per_frame(tmp_path: Path) -> None:
    """With bbox_scale=1.0 the mask exactly covers each frame's bbox."""

    store = _store(
        tmp_path,
        {
            "clip": {
                "frame_size": [10, 8],
                "parts": {"left_arm": [[2.0, 1.0, 5.0, 4.0], None]},
            }
        },
    )
    generator = SkeletonPartsMaskGenerator(_context(store))

    mask = generator.get_mask(8, 10, _spec())

    assert mask.shape == (2, 8, 10)
    expected_frame0 = np.zeros((8, 10), dtype=np.bool_)
    expected_frame0[1:4, 2:5] = True
    assert np.array_equal(mask[0], expected_frame0)
    assert not mask[1].any()  # untracked (null) frame stays empty


def test_bbox_scale_expands_and_clips_to_frame(tmp_path: Path) -> None:
    """A large bbox_scale expands about the box center and clips at edges."""

    store = _store(
        tmp_path,
        {
            "clip": {
                "frame_size": [10, 10],
                "parts": {"left_arm": [[0.0, 0.0, 2.0, 2.0]]},
            }
        },
    )
    generator = SkeletonPartsMaskGenerator(_context(store))

    mask = generator.get_mask(10, 10, _spec(bbox_scale=100.0))

    assert mask.shape == (1, 10, 10)
    assert mask[0].all()  # scaled box covers and clips to the whole frame


def test_missing_store_raises() -> None:
    generator = SkeletonPartsMaskGenerator(_context(None))

    with pytest.raises(RegionResolutionError, match="SkeletonBBoxStore"):
        generator.get_mask(8, 10, _spec())


@pytest.mark.parametrize(
    "params",
    [
        {"sample_id": "clip", "body_part": "left_arm"},  # missing bbox_scale
        {"sample_id": "", "body_part": "left_arm", "bbox_scale": 1.0},
        {"sample_id": "clip", "body_part": "", "bbox_scale": 1.0},
        {"sample_id": "clip", "body_part": "left_arm", "bbox_scale": 0.0},
        {"sample_id": "clip", "body_part": "left_arm", "bbox_scale": -1.0},
        {"sample_id": "clip", "body_part": "left_arm", "bbox_scale": True},
    ],
)
def test_invalid_params_are_rejected(tmp_path: Path, params: dict[str, object]) -> None:
    store = _store(
        tmp_path,
        {
            "clip": {
                "frame_size": [10, 8],
                "parts": {"left_arm": [[2.0, 1.0, 5.0, 4.0]]},
            }
        },
    )
    generator = SkeletonPartsMaskGenerator(_context(store))
    spec = RegionSpec(
        region_id="region",
        region_instance_id="region/clip",
        kind=RegionKind.SKELETON_PARTS,
        params=params,
    )

    with pytest.raises(RegionResolutionError):
        generator.get_mask(8, 10, spec)


def test_missing_sample_raises(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        {"other": {"frame_size": [10, 8], "parts": {"left_arm": [[0.0, 0.0, 1.0, 1.0]]}}},
    )
    generator = SkeletonPartsMaskGenerator(_context(store))

    with pytest.raises(RegionResolutionError, match="no skeleton bbox data"):
        generator.get_mask(8, 10, _spec())


def test_missing_part_raises(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        {"clip": {"frame_size": [10, 8], "parts": {"right_arm": [[0.0, 0.0, 1.0, 1.0]]}}},
    )
    generator = SkeletonPartsMaskGenerator(_context(store))

    with pytest.raises(RegionResolutionError, match="no skeleton bbox data"):
        generator.get_mask(8, 10, _spec())


def test_mismatched_frame_size_raises(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        {"clip": {"frame_size": [10, 8], "parts": {"left_arm": [[0.0, 0.0, 1.0, 1.0]]}}},
    )
    generator = SkeletonPartsMaskGenerator(_context(store))

    with pytest.raises(RegionResolutionError, match="frame_size"):
        generator.get_mask(20, 20, _spec())
