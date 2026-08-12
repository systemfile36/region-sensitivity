"""Tests for SkeletonRegionProvider sample-dependent region expansion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ssat.core.config.schema import ResolvedRegionConfig
from ssat.core.plan.expansion_base import RegionExpansionError
from ssat.core.region.skeleton_provider import SkeletonRegionProvider
from ssat.core.region.skeleton_store import load_skeleton_bbox_store
from ssat.core.source.types import SampleMeta
from ssat.core.types import RegionKind


def _store(tmp_path: Path):
    path = tmp_path / "skeleton.json"
    path.write_text(
        json.dumps(
            {
                "clip": {
                    "frame_size": [10, 8],
                    "parts": {"left_arm": [[2.0, 1.0, 5.0, 4.0]]},
                }
            }
        ),
        encoding="utf-8",
    )
    return load_skeleton_bbox_store(path)


def _sample(sample_id: str = "clip") -> SampleMeta:
    return SampleMeta(sample_id=sample_id, path=Path("clip.mp4"))


def _family(**params: object) -> ResolvedRegionConfig:
    return ResolvedRegionConfig(
        region_id="occlude_left_arm",
        kind=RegionKind.SKELETON_PARTS,
        params=params,
    )


def test_expands_to_one_region_with_default_bbox_scale(tmp_path: Path) -> None:
    provider = SkeletonRegionProvider(_store(tmp_path))

    instances = provider.expand(_sample(), _family(body_part="left_arm"))

    assert len(instances) == 1
    instance = instances[0]
    assert instance.region_id == "occlude_left_arm"
    assert instance.region_instance_id == "occlude_left_arm/clip"
    assert instance.kind is RegionKind.SKELETON_PARTS
    assert instance.params == {
        "sample_id": "clip",
        "body_part": "left_arm",
        "bbox_scale": 1.0,
    }


def test_expands_with_custom_bbox_scale(tmp_path: Path) -> None:
    provider = SkeletonRegionProvider(_store(tmp_path))

    instances = provider.expand(
        _sample(), _family(body_part="left_arm", bbox_scale=1.15)
    )

    assert instances[0].params["bbox_scale"] == 1.15


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"body_part": ""},
        {"body_part": "left_arm", "bbox_scale": 0.0},
        {"body_part": "left_arm", "bbox_scale": -1.0},
        {"body_part": "left_arm", "bbox_scale": True},
        {"body_part": "left_arm", "unexpected": 1},
    ],
)
def test_invalid_params_are_rejected(
    tmp_path: Path, params: dict[str, object]
) -> None:
    provider = SkeletonRegionProvider(_store(tmp_path))

    with pytest.raises(RegionExpansionError):
        provider.expand(_sample(), _family(**params))


def test_missing_skeleton_data_raises(tmp_path: Path) -> None:
    provider = SkeletonRegionProvider(_store(tmp_path))

    with pytest.raises(RegionExpansionError, match="no skeleton bbox data"):
        provider.expand(_sample("other_clip"), _family(body_part="left_arm"))

    with pytest.raises(RegionExpansionError, match="no skeleton bbox data"):
        provider.expand(_sample(), _family(body_part="right_arm"))
