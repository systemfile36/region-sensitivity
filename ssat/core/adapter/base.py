"""Framework-neutral model adapter contracts and validation helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ssat.core.adapter.preprocessor import validate_mask
from ssat.core.adapter.types import AdapterSpec, RawOutput


class AdapterError(RuntimeError):
    """Indicate that an adapter input, invocation, or output is invalid."""


class AdapterOutOfMemoryError(AdapterError):
    """Represent recoverable model-device OOM without framework coupling."""


class SupportsDescribe(Protocol):
    """Provide deterministic adapter metadata without running inference."""

    def describe(self) -> AdapterSpec:
        """Return the adapter contract inspected by ConfigResolver."""


class ModelAdapter(ABC):
    """Define the complete v1 inference adapter contract.

    Implementations receive batches in ``(B, T, H, W, C)`` uint8 layout and
    return exactly one one-dimensional floating logits vector per input.
    ``transform_mask`` is deliberately optional: returning ``None`` tells the
    runtime that effective model-space area is unavailable.
    """

    @abstractmethod
    def describe(self) -> AdapterSpec:
        """Return stable, manifest-ready adapter metadata."""

    @abstractmethod
    def predict(self, batch: NDArray[np.uint8]) -> list[RawOutput]:
        """Run inference for a validated THWC uint8 batch."""

    def transform_mask(
        self,
        mask: NDArray[np.bool_],
    ) -> NDArray[np.bool_] | None:
        """Apply model-space geometry, or report that it is unavailable."""

        self._validate_mask(mask)
        return None

    @staticmethod
    def _validate_batch(
        batch: NDArray[np.uint8],
        spec: AdapterSpec,
    ) -> None:
        """Validate the shared adapter input and declared batch ceiling."""

        if not isinstance(batch, np.ndarray):
            raise TypeError("batch must be a numpy ndarray")
        if batch.dtype != np.uint8:
            raise TypeError("batch must have dtype uint8")
        if batch.ndim != 5:
            raise ValueError("batch must use (B, T, H, W, C) layout")
        if any(size <= 0 for size in batch.shape[1:]):
            raise ValueError("batch frame, height, width, and channel sizes must be positive")
        if spec.max_batch_size is not None and batch.shape[0] > spec.max_batch_size:
            raise AdapterError(
                f"batch size {batch.shape[0]} exceeds max_batch_size "
                f"{spec.max_batch_size}"
            )

    @staticmethod
    def _validate_mask(mask: NDArray[np.bool_]) -> None:
        """Validate a source-space binary mask."""

        validate_mask(mask)
