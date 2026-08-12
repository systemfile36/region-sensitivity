"""Tests for the decord-backed explicit video catalog source."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from ssat.core.source import LoadError, LoadedSample, SampleMeta, VideoFolderSource
from ssat.core.source.video_folder import uniform_frame_indices
from ssat.utils.io import sha256_file

# 48 triggers a known decord/mp4v get_batch size-mismatch quirk; keep clips tiny
# but avoid that resolution.
_SIZE = 40


def _write_video(path: Path, frame_count: int, *, size: int = _SIZE) -> None:
    """Write a small mp4v clip whose frames each hold a distinct flat color."""

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (size, size)
    )
    for index in range(frame_count):
        frame = np.full((size, size, 3), index * 20 % 256, dtype=np.uint8)
        writer.write(frame[:, :, ::-1])  # RGB -> BGR for cv2's writer
    writer.release()


def test_uniform_frame_indices_spacing_and_validation() -> None:
    """Sampling is evenly spaced and repeats indices for short clips."""

    assert list(uniform_frame_indices(10, 5)) == [0, 2, 4, 7, 9]
    assert list(uniform_frame_indices(3, 5)) == [0, 0, 1, 2, 2]
    assert list(uniform_frame_indices(4, 1)) == [0]

    with pytest.raises(ValueError, match="frame_count"):
        uniform_frame_indices(0, 4)
    with pytest.raises(ValueError, match="num_frames"):
        uniform_frame_indices(4, 0)


def test_lists_metadata_without_loading_and_rejects_duplicates(tmp_path: Path) -> None:
    """The source preserves its explicit catalog and enforces unique IDs."""

    sample = SampleMeta("clip", tmp_path / "clip.mp4", gt_label=1)
    source = VideoFolderSource((sample,), num_frames=4)

    assert source.list_samples() == [sample]
    with pytest.raises(ValueError, match="duplicate"):
        VideoFolderSource((sample, sample), num_frames=4)
    with pytest.raises(TypeError, match="SampleMeta"):
        VideoFolderSource((sample, object()), num_frames=4)  # type: ignore[arg-type]


@pytest.mark.parametrize("num_frames", [0, -1, True])
def test_num_frames_must_be_a_positive_integer(tmp_path: Path, num_frames: object) -> None:
    """A non-positive or non-integer num_frames is rejected eagerly."""

    sample = SampleMeta("clip", tmp_path / "clip.mp4")
    with pytest.raises(ValueError, match="num_frames"):
        VideoFolderSource((sample,), num_frames=num_frames)  # type: ignore[arg-type]


def test_loads_clips_as_thwc_rgb_with_provenance_and_temporal_variation(
    tmp_path: Path,
) -> None:
    """Decoded clips use RGB uint8 layout, retain labels/hashes, and vary over T."""

    path = tmp_path / "clip.mp4"
    _write_video(path, frame_count=12)
    source = VideoFolderSource((SampleMeta("clip", path, gt_label=2),), num_frames=4)

    loaded = source.load("clip")

    assert isinstance(loaded, LoadedSample)
    assert loaded.array.shape == (4, _SIZE, _SIZE, 3)
    assert loaded.original_shape == loaded.array.shape
    assert loaded.array.dtype == np.uint8
    assert loaded.content_hash == sha256_file(path)
    assert loaded.gt_label == 2
    # Uniformly sampled frames from a 12-frame clip must not collapse to one value.
    assert not np.array_equal(loaded.array[0], loaded.array[-1])


def test_load_returns_recoverable_errors(tmp_path: Path) -> None:
    """Unknown, missing, and corrupt samples are represented as values."""

    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a video")
    source = VideoFolderSource(
        (
            SampleMeta("missing", tmp_path / "missing.mp4"),
            SampleMeta("corrupt", corrupt),
        ),
        num_frames=2,
    )

    unknown = source.load("unknown")
    missing = source.load("missing")
    broken = source.load("corrupt")

    assert isinstance(unknown, LoadError)
    assert unknown.error_type == "sample_not_found"
    assert isinstance(missing, LoadError)
    assert missing.error_type == "file_not_found"
    assert isinstance(broken, LoadError)
    assert broken.error_type == "decode_error"
