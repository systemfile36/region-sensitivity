"""Framework-independent sample metadata and load results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SampleMeta:
    """Identify a source sample without loading its pixel data.

    Attributes:
        sample_id: Dataset-stable identifier used throughout the audit.
        path: Location consumed by a concrete SampleSource.
        gt_label: Optional integer class label for clean-output validation.
    """

    sample_id: str
    path: Path
    gt_label: int | None = None

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True, slots=True)
class LoadedSample:
    """Carry decoded source pixels from a worker-side SampleSource.

    Attributes:
        array: Original uint8 pixels in ``(T, H, W, C)`` layout.
        sample_id: Identifier matching the requested sample metadata.
        original_shape: Shape captured before adapter preprocessing.
        content_hash: SHA-256 digest used for data provenance.
        gt_label: Optional integer class label.
    """

    array: NDArray[np.uint8]
    sample_id: str
    original_shape: tuple[int, int, int, int]
    content_hash: str
    gt_label: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.array, np.ndarray):
            raise TypeError("array must be a numpy ndarray")
        if self.array.dtype != np.uint8:
            raise TypeError("array must have dtype uint8")
        if self.array.ndim != 4:
            raise ValueError("array must use (T, H, W, C) layout")
        if tuple(self.array.shape) != self.original_shape:
            raise ValueError("original_shape must match array.shape")
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if len(self.content_hash) != 64 or any(
            c not in "0123456789abcdef" for c in self.content_hash
        ):
            raise ValueError("content_hash must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class LoadError:
    """Represent a recoverable sample-loading failure as a value.

    Attributes:
        sample_id: Identifier of the sample that could not be loaded.
        error_type: Short machine-readable failure category.
        message: Human-readable diagnostic detail.
    """

    sample_id: str
    error_type: str
    message: str

    def __post_init__(self) -> None:
        if not self.sample_id or not self.error_type:
            raise ValueError("sample_id and error_type must not be empty")
