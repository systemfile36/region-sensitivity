"""Build the pre-computed skeleton bounding-box JSON that skeleton_store reads.

This module is the offline counterpart to :mod:`ssat.core.region.skeleton_store`:
it turns per-frame joint coordinates into the exact per-body-part bounding-box
JSON that store's module docstring describes as "not implemented in this
package". It is dataset-agnostic -- callers supply ``body_parts`` (a joint-set
name -> joint-index mapping), so the same functions work for NTU-25, COCO-17,
or any other skeleton convention.

No runtime audit component imports this module: it is meant to run once,
offline, ahead of an audit, producing a file consumed later through the
existing ``skeleton_source.bbox_data`` config contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ssat.core.region.skeleton_store import load_skeleton_bbox_store
from ssat.utils.io import write_json_atomic

FrameBBox: TypeAlias = tuple[float, float, float, float]


def joints_to_part_bboxes(
    joints_xy: NDArray[np.floating],
    joint_valid: NDArray[np.bool_],
    body_parts: Mapping[str, Sequence[int]],
    *,
    frame_size: tuple[int, int],
    margin_ratio: float = 0.15,
    min_valid_joints: int = 1,
) -> dict[str, list[FrameBBox | None]]:
    """Compute per-frame, per-body-part bounding boxes from joint tracks.

    For each part and frame, the box is the union of that part's valid
    joints, expanded on each axis by ``margin_ratio`` of that axis's own
    span (falling back to a 1-pixel margin when the span is zero, e.g. a
    single valid joint), then clamped to ``[0, frame_size]``. A frame whose
    part has fewer than ``min_valid_joints`` valid joints -- or whose
    clamped box collapses to zero width or height -- is recorded as
    ``None``, matching the untracked-frame convention that
    ``skeleton_store.py`` already reads.

    Args:
        joints_xy: Per-frame joint pixel coordinates, shape ``(T, J, 2)``.
        joint_valid: Per-frame joint validity flags, shape ``(T, J)``.
        body_parts: Maps a part name to the joint indices that compose it.
            Dataset-agnostic -- pass NTU-25, COCO-17, or any other table.
        frame_size: ``(width, height)`` used as the clamp bound.
        margin_ratio: Non-negative fraction of each axis's own span added
            as a margin on both sides.
        min_valid_joints: Minimum number of valid joints a part needs in a
            frame to produce a box.

    Returns:
        A mapping from part name to one bounding box (or ``None``) per
        frame, in ``joints_xy``'s frame order.

    Raises:
        ValueError: If the input shapes are inconsistent, ``body_parts``
            references an out-of-range or empty joint index list,
            ``frame_size`` is not positive, ``margin_ratio`` is negative,
            or ``min_valid_joints`` is less than 1.
    """

    joints_xy = np.asarray(joints_xy)
    joint_valid = np.asarray(joint_valid)
    if joints_xy.ndim != 3 or joints_xy.shape[2] != 2:
        raise ValueError("joints_xy must have shape (T, J, 2)")
    if joint_valid.shape != joints_xy.shape[:2]:
        raise ValueError("joint_valid must have shape (T, J) matching joints_xy")
    if margin_ratio < 0:
        raise ValueError("margin_ratio must be non-negative")
    if min_valid_joints < 1:
        raise ValueError("min_valid_joints must be at least 1")
    frame_width, frame_height = frame_size
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_size must be a positive (width, height) pair")

    num_frames, num_joints, _ = joints_xy.shape
    result: dict[str, list[FrameBBox | None]] = {}
    for part_name, joint_indices in body_parts.items():
        indices = list(joint_indices)
        if not indices:
            raise ValueError(f"body_parts[{part_name!r}] must not be empty")
        for index in indices:
            if index < 0 or index >= num_joints:
                raise ValueError(
                    f"body_parts[{part_name!r}] references out-of-range joint index {index}"
                )
        frames: list[FrameBBox | None] = []
        for frame_index in range(num_frames):
            valid = joint_valid[frame_index, indices]
            if int(np.count_nonzero(valid)) < min_valid_joints:
                frames.append(None)
                continue
            points = joints_xy[frame_index, indices][valid]
            frames.append(
                _clamped_part_bbox(
                    points,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    margin_ratio=margin_ratio,
                )
            )
        result[part_name] = frames
    return result


def _clamped_part_bbox(
    points: NDArray[np.floating],
    *,
    frame_width: int,
    frame_height: int,
    margin_ratio: float,
) -> FrameBBox | None:
    """Union one frame's valid part points into a margined, clamped box."""

    min_x = float(np.min(points[:, 0]))
    max_x = float(np.max(points[:, 0]))
    min_y = float(np.min(points[:, 1]))
    max_y = float(np.max(points[:, 1]))

    margin_x = (max_x - min_x) * margin_ratio or 1.0
    margin_y = (max_y - min_y) * margin_ratio or 1.0

    x1 = max(0.0, min_x - margin_x)
    x2 = min(float(frame_width), max_x + margin_x)
    y1 = max(0.0, min_y - margin_y)
    y2 = min(float(frame_height), max_y + margin_y)

    if x1 >= x2 or y1 >= y2:
        return None
    return (x1, y1, x2, y2)


def write_skeleton_bbox_json(
    sample_bboxes: Mapping[str, Mapping[str, Sequence[FrameBBox | None]]],
    frame_sizes: Mapping[str, tuple[int, int]],
    output_path: str | Path,
) -> str:
    """Write the exact JSON ``skeleton_store.py`` reads and verify it round-trips.

    Args:
        sample_bboxes: Maps sample_id to that sample's part-name -> per-frame
            bounding-box list, typically one :func:`joints_to_part_bboxes`
            result per sample.
        frame_sizes: Maps sample_id to the ``(width, height)`` recorded as
            that sample's ``frame_size``. Must match the source's actual
            decoded resolution -- a mismatch silently mispositions masks
            without failing any schema check.
        output_path: Destination JSON file.

    Returns:
        The written file's SHA-256 digest, as computed by re-reading it
        through :func:`~ssat.core.region.skeleton_store.load_skeleton_bbox_store`.

    Raises:
        ValueError: If ``sample_bboxes`` is empty or ``frame_sizes`` is
            missing an entry for one of its sample IDs.
        SkeletonDataError: If the written content fails round-trip
            validation (surfaces a schema mistake at write time instead of
            inside a later audit run).
    """

    if not sample_bboxes:
        raise ValueError("sample_bboxes must not be empty")
    missing = sorted(set(sample_bboxes) - set(frame_sizes))
    if missing:
        raise ValueError(f"frame_sizes is missing entries for sample_id(s): {missing}")

    payload = {
        sample_id: {
            "frame_size": [int(frame_sizes[sample_id][0]), int(frame_sizes[sample_id][1])],
            "parts": {
                part_name: [
                    None if box is None else [float(coordinate) for coordinate in box]
                    for box in boxes
                ]
                for part_name, boxes in parts.items()
            },
        }
        for sample_id, parts in sample_bboxes.items()
    }

    resolved_path = Path(output_path)
    write_json_atomic(resolved_path, payload)
    store = load_skeleton_bbox_store(resolved_path)
    return store.content_hash
