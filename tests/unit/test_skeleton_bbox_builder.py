from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from ssat.core.region.skeleton_bbox_builder import (
    joints_to_part_bboxes,
    write_skeleton_bbox_json,
)
from ssat.core.region.skeleton_store import load_skeleton_bbox_store


# --- joints_to_part_bboxes: margin + clamp --------------------------------------


def test_two_joint_part_gets_a_margined_union_bbox() -> None:
    """Hand-calculated: margin_ratio scales each axis's own span."""

    joints_xy = np.array([[[10.0, 20.0], [30.0, 50.0]]])  # (T=1, J=2, 2)
    joint_valid = np.array([[True, True]])

    result = joints_to_part_bboxes(
        joints_xy,
        joint_valid,
        {"limb": [0, 1]},
        frame_size=(100, 100),
        margin_ratio=0.1,
    )

    # width=20 -> margin_x=2.0, height=30 -> margin_y=3.0
    assert result["limb"] == [(8.0, 17.0, 32.0, 53.0)]


def test_single_valid_joint_falls_back_to_a_one_pixel_margin() -> None:
    """A zero-span box (one point) would otherwise collapse to x1==x2."""

    joints_xy = np.array([[[2.0, 2.0]]])  # (T=1, J=1, 2)
    joint_valid = np.array([[True]])

    result = joints_to_part_bboxes(
        joints_xy,
        joint_valid,
        {"point": [0]},
        frame_size=(50, 50),
        margin_ratio=0.1,
    )

    assert result["point"] == [(1.0, 1.0, 3.0, 3.0)]


def test_margin_is_clamped_to_frame_bounds() -> None:
    """A joint near the frame edge must not push the box outside [0, frame_size]."""

    joints_xy = np.array([[[49.0, 49.0]]])
    joint_valid = np.array([[True]])

    result = joints_to_part_bboxes(
        joints_xy,
        joint_valid,
        {"point": [0]},
        frame_size=(50, 50),
        margin_ratio=0.1,
    )

    assert result["point"] == [(48.0, 48.0, 50.0, 50.0)]


# --- min_valid_joints filtering --------------------------------------------------


def test_frame_below_min_valid_joints_is_recorded_as_none() -> None:
    joints_xy = np.array(
        [
            [[10.0, 10.0], [20.0, 20.0]],  # frame 0: both valid
            [[10.0, 10.0], [20.0, 20.0]],  # frame 1: only joint 0 valid
        ]
    )
    joint_valid = np.array([[True, True], [True, False]])

    result = joints_to_part_bboxes(
        joints_xy,
        joint_valid,
        {"limb": [0, 1]},
        frame_size=(100, 100),
        min_valid_joints=2,
    )

    assert result["limb"][0] is not None
    assert result["limb"][1] is None


# --- genericity: body_parts is not tied to any one joint set --------------------


def test_works_with_a_non_ntu_five_joint_toy_skeleton() -> None:
    joints_xy = np.zeros((2, 5, 2), dtype=np.float64)
    joints_xy[:, :, 0] = np.arange(5) * 10.0
    joints_xy[:, :, 1] = np.arange(5) * 5.0
    joint_valid = np.ones((2, 5), dtype=bool)

    result = joints_to_part_bboxes(
        joints_xy,
        joint_valid,
        {"left": [0, 1], "right": [3, 4]},
        frame_size=(200, 200),
    )

    assert set(result) == {"left", "right"}
    assert all(box is not None for box in result["left"])
    assert all(box is not None for box in result["right"])


# --- input validation -------------------------------------------------------------


def test_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="joints_xy"):
        joints_to_part_bboxes(
            np.zeros((1, 2, 3)),
            np.ones((1, 2), dtype=bool),
            {"p": [0]},
            frame_size=(10, 10),
        )
    with pytest.raises(ValueError, match="joint_valid"):
        joints_to_part_bboxes(
            np.zeros((1, 2, 2)),
            np.ones((1, 3), dtype=bool),
            {"p": [0]},
            frame_size=(10, 10),
        )


def test_rejects_out_of_range_joint_index() -> None:
    with pytest.raises(ValueError, match="out-of-range"):
        joints_to_part_bboxes(
            np.zeros((1, 2, 2)),
            np.ones((1, 2), dtype=bool),
            {"p": [5]},
            frame_size=(10, 10),
        )


def test_rejects_non_positive_frame_size() -> None:
    with pytest.raises(ValueError, match="frame_size"):
        joints_to_part_bboxes(
            np.zeros((1, 1, 2)),
            np.ones((1, 1), dtype=bool),
            {"p": [0]},
            frame_size=(0, 10),
        )


# --- write_skeleton_bbox_json: round-trip through the real runtime loader -------


def test_write_skeleton_bbox_json_round_trips_through_the_runtime_loader(
    tmp_path: Path,
) -> None:
    joints_xy = np.array([[[10.0, 20.0], [30.0, 50.0]]])
    joint_valid = np.array([[True, True]])
    part_bboxes = joints_to_part_bboxes(
        joints_xy, joint_valid, {"limb": [0, 1]}, frame_size=(100, 100), margin_ratio=0.1
    )

    output_path = tmp_path / "skeleton_bbox.json"
    digest = write_skeleton_bbox_json(
        {"sample-1": part_bboxes},
        {"sample-1": (100, 100)},
        output_path,
    )

    store = load_skeleton_bbox_store(output_path)
    assert store.content_hash == digest
    assert store.frame_size("sample-1") == (100, 100)
    assert store.get("sample-1", "limb") == ((8.0, 17.0, 32.0, 53.0),)


def test_write_skeleton_bbox_json_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        write_skeleton_bbox_json({}, {}, tmp_path / "out.json")


def test_write_skeleton_bbox_json_rejects_missing_frame_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frame_sizes"):
        write_skeleton_bbox_json(
            {"sample-1": {"limb": [(0.0, 0.0, 1.0, 1.0)]}},
            {},
            tmp_path / "out.json",
        )


# --- dependency boundary: never imported from a runtime audit module -----------


@pytest.mark.parametrize(
    "runtime_module",
    [
        "resolver.py",
        "mask_generators.py",
        "mask_base.py",
        "skeleton_provider.py",
    ],
)
def test_runtime_region_modules_do_not_import_the_offline_builder(
    runtime_module: str,
) -> None:
    """skeleton_bbox_builder is an offline generation tool, not part of the audit path.

    Statically enforces that no runtime component may import it, so this
    check parses source instead of importing the module.
    """

    source_path = Path(__file__).resolve().parents[2] / "ssat" / "core" / "region" / runtime_module
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert "skeleton_bbox_builder" not in module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert "skeleton_bbox_builder" not in alias.name
