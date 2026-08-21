"""Video source backed by explicit sample metadata and decord decoding."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import decord
import numpy as np
from numpy.typing import NDArray

from ssat.core.source.types import LoadError, LoadedSample, SampleMeta
from ssat.utils.io import sha256_file

decord.bridge.set_bridge("native")


def uniform_frame_indices(frame_count: int, num_frames: int) -> NDArray[np.int64]:
    """Select ``num_frames`` indices evenly spaced across a video.

    Args:
        frame_count: Positive total number of decodable frames.
        num_frames: Positive number of indices to return.

    Returns:
        Ascending frame indices in ``[0, frame_count)``. Shorter videos repeat
        indices rather than raising, keeping sampling deterministic for any
        video length.

    Raises:
        ValueError: If ``frame_count`` or ``num_frames`` is not positive.
    """

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    positions = np.linspace(0, frame_count - 1, num=num_frames)
    return np.round(positions).astype(np.int64)


def segment_center_frame_indices(
    frame_count: int, num_frames: int
) -> NDArray[np.int64]:
    """Select deterministic MMAction2-style test-time segment centers.

    This is the ``clip_len=1``, ``frame_interval=1``, ``test_mode=True`` case
    used by the eight-segment TSM validation pipeline.  MMAction2 computes
    floating segment centers and converts them to integers by truncation,
    which is equivalent to ``floor((i + 0.5) * frame_count / num_frames)``.
    """

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    positions = (np.arange(num_frames, dtype=np.float64) + 0.5) * (
        frame_count / num_frames
    )
    return np.minimum(positions.astype(np.int64), frame_count - 1)


FrameSampling = Literal["uniform", "segment_center"]
_FRAME_SAMPLERS: dict[str, Callable[[int, int], NDArray[np.int64]]] = {
    "uniform": uniform_frame_indices,
    "segment_center": segment_center_frame_indices,
}


class VideoFolderSource:
    """Load RGB video clips referenced by an explicit sample catalog.

    Every clip is decoded with decord and deterministically sampled to a fixed
    ``num_frames`` length, matching the ``(T, H, W, C)`` source contract used
    by image sources (``T=1``) so that region, perturbation, and adapter
    layers require no code changes for video input.

    Args:
        samples: Lightweight metadata containing stable IDs and video paths.
        num_frames: Positive number of frames sampled per clip.
        sampling: Registered deterministic frame-index strategy.

    Raises:
        TypeError: If the catalog contains a value other than ``SampleMeta``.
        ValueError: If the catalog has duplicate sample IDs, ``num_frames``
            is not positive, or ``sampling`` is unknown.
    """

    def __init__(
        self,
        samples: Sequence[SampleMeta],
        *,
        num_frames: int,
        sampling: FrameSampling = "uniform",
    ) -> None:
        catalog = tuple(samples)
        if any(not isinstance(sample, SampleMeta) for sample in catalog):
            raise TypeError("samples must contain only SampleMeta values")

        sample_ids = tuple(sample.sample_id for sample in catalog)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("samples must not contain duplicate sample_id values")
        if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
            raise ValueError("num_frames must be a positive integer")
        if sampling not in _FRAME_SAMPLERS:
            known = ", ".join(_FRAME_SAMPLERS)
            raise ValueError(f"sampling must be one of: {known}")

        self._samples = catalog
        self._sample_by_id = {sample.sample_id: sample for sample in catalog}
        self._num_frames = num_frames
        self._sampling = sampling

    def list_samples(self) -> list[SampleMeta]:
        """Return a fresh metadata list without decoding any video.

        Returns:
            Sample metadata in the order supplied to the constructor.
        """

        return list(self._samples)

    def load(self, sample_id: str) -> LoadedSample | LoadError:
        """Decode one catalog video into the source pixel-space contract.

        Args:
            sample_id: Stable identifier of the video to load.

        Returns:
            An RGB ``(T,H,W,3)`` ``LoadedSample`` or a recoverable
            ``LoadError`` value.
        """

        sample = self._sample_by_id.get(sample_id)
        if sample is None:
            return LoadError(
                sample_id=sample_id,
                error_type="sample_not_found",
                message=f"unknown sample_id: {sample_id!r}",
            )
        if not sample.path.is_file():
            return self._load_error(
                sample_id, "file_not_found", FileNotFoundError(str(sample.path))
            )

        try:
            content_hash = sha256_file(sample.path)
            reader = decord.VideoReader(str(sample.path))
            frame_count = len(reader)
            if frame_count <= 0:
                raise ValueError("video contains no decodable frames")
            indices = _FRAME_SAMPLERS[self._sampling](frame_count, self._num_frames)
            array = np.asarray(reader.get_batch(indices).asnumpy(), dtype=np.uint8)
        except Exception as error:  # decord raises broad/backend-specific errors
            return self._load_error(sample_id, "decode_error", error)

        return LoadedSample(
            array=array,
            sample_id=sample.sample_id,
            original_shape=tuple(array.shape),
            content_hash=content_hash,
            gt_label=sample.gt_label,
        )

    @staticmethod
    def _load_error(sample_id: str, error_type: str, error: Exception) -> LoadError:
        """Convert a video loading exception into a recoverable value.

        Args:
            sample_id: Identifier requested by the caller.
            error_type: Stable machine-readable failure category.
            error: Original exception raised while loading the video.

        Returns:
            A ``LoadError`` preserving the exception type and message.
        """

        detail = str(error) or error.__class__.__name__
        return LoadError(
            sample_id=sample_id,
            error_type=error_type,
            message=f"{error.__class__.__name__}: {detail}",
        )
