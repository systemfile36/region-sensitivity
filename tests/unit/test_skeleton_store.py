"""Tests for the pre-computed skeleton body-part bbox data loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ssat.core.region.skeleton_store import SkeletonDataError, load_skeleton_bbox_store
from ssat.utils.io import sha256_file

_VALID_PAYLOAD = {
    "clip_001": {
        "frame_size": [64, 48],
        "parts": {
            "left_arm": [[10.0, 10.0, 20.0, 20.0], None, [12.0, 12.0, 22.0, 22.0]],
            "right_arm": [[30.0, 10.0, 40.0, 20.0], [31.0, 11.0, 41.0, 21.0], None],
        },
    }
}


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "skeleton.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_valid_payload_and_exposes_frame_size_and_boxes(tmp_path: Path) -> None:
    """A well-formed file loads and its data is queryable by sample and part."""

    path = _write(tmp_path, _VALID_PAYLOAD)

    store = load_skeleton_bbox_store(path)

    assert store.content_hash == sha256_file(path)
    assert store.frame_size("clip_001") == (64, 48)
    assert store.get("clip_001", "left_arm") == (
        (10.0, 10.0, 20.0, 20.0),
        None,
        (12.0, 12.0, 22.0, 22.0),
    )
    assert store.frame_size("missing") is None
    assert store.get("clip_001", "missing_part") is None
    assert store.get("missing", "left_arm") is None


def test_expected_hash_is_verified(tmp_path: Path) -> None:
    """A mismatched expected_hash is rejected before parsing."""

    path = _write(tmp_path, _VALID_PAYLOAD)

    load_skeleton_bbox_store(path, expected_hash=sha256_file(path))

    with pytest.raises(SkeletonDataError, match="ref_hash mismatch"):
        load_skeleton_bbox_store(path, expected_hash="a" * 64)


def test_missing_file_raises() -> None:
    """A nonexistent path raises a clear SkeletonDataError."""

    with pytest.raises(SkeletonDataError, match="cannot read"):
        load_skeleton_bbox_store("/nonexistent/skeleton.json")


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"clip": {"frame_size": [1, 1]}},
        {"clip": {"frame_size": [1, 1], "parts": {}, "extra": 1}},
        {"": {"frame_size": [1, 1], "parts": {"a": [None]}}},
    ],
)
def test_malformed_top_level_shape_is_rejected(tmp_path: Path, payload: object) -> None:
    """Payloads violating the sample-entry shape are rejected."""

    path = _write(tmp_path, payload)

    with pytest.raises(SkeletonDataError):
        load_skeleton_bbox_store(path)


@pytest.mark.parametrize(
    "frame_size",
    [[1], [1, 1, 1], [0, 1], [1, -1], ["a", 1], [1.5, 1]],
)
def test_invalid_frame_size_is_rejected(tmp_path: Path, frame_size: object) -> None:
    """frame_size must be exactly two positive integers."""

    payload = {
        "clip": {"frame_size": frame_size, "parts": {"a": [[0.0, 0.0, 1.0, 1.0]]}}
    }
    path = _write(tmp_path, payload)

    with pytest.raises(SkeletonDataError, match="frame_size"):
        load_skeleton_bbox_store(path)


def test_parts_must_be_non_empty(tmp_path: Path) -> None:
    """A sample with an empty parts mapping is rejected."""

    payload = {"clip": {"frame_size": [1, 1], "parts": {}}}
    path = _write(tmp_path, payload)

    with pytest.raises(SkeletonDataError, match="parts"):
        load_skeleton_bbox_store(path)


def test_parts_must_share_one_frame_count(tmp_path: Path) -> None:
    """Every body part in a sample must have the same number of frames."""

    payload = {
        "clip": {
            "frame_size": [10, 10],
            "parts": {
                "a": [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                "b": [[0.0, 0.0, 1.0, 1.0]],
            },
        }
    }
    path = _write(tmp_path, payload)

    with pytest.raises(SkeletonDataError, match="frame count"):
        load_skeleton_bbox_store(path)


@pytest.mark.parametrize(
    "bbox",
    [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0],
        ["a", 0.0, 1.0, 1.0],
        [float("nan"), 0.0, 1.0, 1.0],
        [float("inf"), 0.0, 1.0, 1.0],
        [5.0, 0.0, 1.0, 1.0],
        [0.0, 5.0, 1.0, 1.0],
        [-1.0, 0.0, 1.0, 1.0],
    ],
)
def test_invalid_bbox_entries_are_rejected(tmp_path: Path, bbox: object) -> None:
    """Malformed or degenerate bounding boxes are rejected."""

    payload = {"clip": {"frame_size": [10, 10], "parts": {"a": [bbox]}}}
    path = _write(tmp_path, payload)

    with pytest.raises(SkeletonDataError):
        load_skeleton_bbox_store(path)


def test_null_bbox_entries_are_allowed(tmp_path: Path) -> None:
    """A null bbox marks an untracked frame without failing validation."""

    payload = {"clip": {"frame_size": [10, 10], "parts": {"a": [None, None]}}}
    path = _write(tmp_path, payload)

    store = load_skeleton_bbox_store(path)

    assert store.get("clip", "a") == (None, None)
