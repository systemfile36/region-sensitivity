"""Load and query pre-computed per-frame skeleton body-part bounding boxes.

The store holds the output of an offline skeleton-to-bbox conversion (not
implemented in this package): for every sample, a per-body-part sequence of
axis-aligned bounding boxes, one entry per source frame, in the same pixel
coordinate space as the decoded video. Runtime components only read this
pre-computed data; no skeleton/joint parsing happens here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, TypeAlias

from ssat.utils.io import load_json, sha256_file

FrameBBox: TypeAlias = tuple[float, float, float, float]


class SkeletonDataError(ValueError):
    """Indicate that skeleton bounding-box data is malformed or unverifiable."""


@dataclass(frozen=True, slots=True)
class _SampleEntry:
    """Hold one sample's frame size and per-body-part bounding boxes."""

    frame_size: tuple[int, int]
    parts: Mapping[str, tuple[FrameBBox | None, ...]]


class SkeletonBBoxStore:
    """Provide in-memory lookup of pre-computed per-frame body-part bboxes.

    Instances are built by :func:`load_skeleton_bbox_store`, not constructed
    directly.

    Attributes:
        content_hash: SHA-256 digest of the source file this store was
            loaded from.
    """

    def __init__(
        self,
        samples: Mapping[str, _SampleEntry],
        content_hash: str,
    ) -> None:
        self._samples = samples
        self.content_hash = content_hash

    def frame_size(self, sample_id: str) -> tuple[int, int] | None:
        """Return a sample's ``(width, height)`` annotation frame size.

        Args:
            sample_id: Dataset-stable sample identifier.

        Returns:
            The annotated frame size, or ``None`` if the sample is absent.
        """

        entry = self._samples.get(sample_id)
        return entry.frame_size if entry is not None else None

    def get(
        self,
        sample_id: str,
        part_name: str,
    ) -> tuple[FrameBBox | None, ...] | None:
        """Return one sample's per-frame bounding boxes for one body part.

        Args:
            sample_id: Dataset-stable sample identifier.
            part_name: Body-part name as recorded in the source file.

        Returns:
            One bounding box (or ``None`` for an untracked frame) per source
            frame, in stable frame order. ``None`` if the sample or part is
            not present in the store.
        """

        entry = self._samples.get(sample_id)
        if entry is None:
            return None
        return entry.parts.get(part_name)


def load_skeleton_bbox_store(
    path: str | Path,
    *,
    expected_hash: str | None = None,
) -> SkeletonBBoxStore:
    """Load and validate a pre-computed skeleton bounding-box JSON file.

    The expected JSON shape is::

        {
          "<sample_id>": {
            "frame_size": [width, height],
            "parts": {
              "<part_name>": [[x1, y1, x2, y2] | null, ...]
            }
          }
        }

    All part lists for one sample must share the same length (the sample's
    frame count); a ``null`` entry marks a frame with no tracked box for
    that part.

    Args:
        path: JSON file to read.
        expected_hash: Optional SHA-256 digest the file content must match.

    Returns:
        A queryable store over the validated content.

    Raises:
        SkeletonDataError: If the file cannot be read, its hash does not
            match ``expected_hash``, or its content is malformed.
    """

    resolved = Path(path)
    try:
        actual_hash = sha256_file(resolved)
    except OSError as error:
        raise SkeletonDataError(f"cannot read skeleton bbox file: {resolved}") from error
    if expected_hash is not None and expected_hash.lower() != actual_hash:
        raise SkeletonDataError("skeleton bbox file ref_hash mismatch")

    try:
        raw = load_json(resolved)
    except Exception as error:
        raise SkeletonDataError(f"cannot decode skeleton bbox file: {resolved}") from error

    samples = _parse_payload(raw)
    return SkeletonBBoxStore(samples, actual_hash)


def _parse_payload(raw: Any) -> dict[str, _SampleEntry]:
    """Validate and convert a decoded JSON payload into sample entries."""

    if not isinstance(raw, Mapping):
        raise SkeletonDataError("skeleton bbox data must be a JSON object")

    samples: dict[str, _SampleEntry] = {}
    for sample_id, entry in raw.items():
        if not isinstance(sample_id, str) or not sample_id:
            raise SkeletonDataError("skeleton bbox sample_id keys must be non-empty strings")
        if not isinstance(entry, Mapping) or set(entry) != {"frame_size", "parts"}:
            raise SkeletonDataError(
                f"sample_id={sample_id!r} must contain exactly frame_size and parts"
            )
        frame_size = _parse_frame_size(entry["frame_size"], sample_id)
        parts = _parse_parts(entry["parts"], sample_id)
        samples[sample_id] = _SampleEntry(frame_size=frame_size, parts=parts)
    return samples


def _parse_frame_size(value: Any, sample_id: str) -> tuple[int, int]:
    """Validate one sample's ``[width, height]`` annotation frame size."""

    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise SkeletonDataError(
            f"sample_id={sample_id!r} frame_size must be a [width, height] pair"
        )
    width, height = value
    for name, dimension in (("width", width), ("height", height)):
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise SkeletonDataError(
                f"sample_id={sample_id!r} frame_size.{name} must be a positive integer"
            )
    return int(width), int(height)


def _parse_parts(
    value: Any,
    sample_id: str,
) -> dict[str, tuple[FrameBBox | None, ...]]:
    """Validate one sample's per-body-part per-frame bounding box lists."""

    if not isinstance(value, Mapping) or not value:
        raise SkeletonDataError(f"sample_id={sample_id!r} parts must be a non-empty object")

    parts: dict[str, tuple[FrameBBox | None, ...]] = {}
    frame_count: int | None = None
    for part_name, boxes in value.items():
        if not isinstance(part_name, str) or not part_name:
            raise SkeletonDataError(
                f"sample_id={sample_id!r} part names must be non-empty strings"
            )
        if (
            not isinstance(boxes, Sequence)
            or isinstance(boxes, (str, bytes))
            or not boxes
        ):
            raise SkeletonDataError(
                f"sample_id={sample_id!r} part={part_name!r} must be a non-empty list"
            )
        if frame_count is None:
            frame_count = len(boxes)
        elif len(boxes) != frame_count:
            raise SkeletonDataError(
                f"sample_id={sample_id!r} parts must share one frame count"
            )
        parts[part_name] = tuple(
            _parse_bbox(box, sample_id, part_name, frame_index)
            for frame_index, box in enumerate(boxes)
        )
    return parts


def _parse_bbox(
    value: Any,
    sample_id: str,
    part_name: str,
    frame_index: int,
) -> FrameBBox | None:
    """Validate one frame's bounding box, allowing an untracked ``null``."""

    if value is None:
        return None
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        raise SkeletonDataError(
            f"sample_id={sample_id!r} part={part_name!r} frame={frame_index} "
            "bbox must be null or [x1, y1, x2, y2]"
        )
    try:
        x1, y1, x2, y2 = (float(coordinate) for coordinate in value)
    except (TypeError, ValueError) as error:
        raise SkeletonDataError(
            f"sample_id={sample_id!r} part={part_name!r} frame={frame_index} "
            "bbox coordinates must be numbers"
        ) from error
    if not all(isfinite(coordinate) for coordinate in (x1, y1, x2, y2)):
        raise SkeletonDataError(
            f"sample_id={sample_id!r} part={part_name!r} frame={frame_index} "
            "bbox coordinates must be finite"
        )
    if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2:
        raise SkeletonDataError(
            f"sample_id={sample_id!r} part={part_name!r} frame={frame_index} "
            "bbox must satisfy 0 <= x1 < x2 and 0 <= y1 < y2"
        )
    return x1, y1, x2, y2
