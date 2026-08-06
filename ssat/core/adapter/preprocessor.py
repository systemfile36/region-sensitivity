"""Framework-neutral preprocessing contracts and callable implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ssat.core.adapter.types import PreprocessingSpec


class Preprocessor(ABC):
    """Transform source-space batches and matching masks through one contract."""

    @abstractmethod
    def describe(self) -> PreprocessingSpec:
        """Return stable preprocessing metadata."""

    @abstractmethod
    def transform_batch(self, batch: NDArray[np.uint8]) -> Any:
        """Transform a validated THWC uint8 batch into model input."""

    @abstractmethod
    def transform_mask(
        self,
        mask: NDArray[np.bool_],
    ) -> NDArray[np.bool_] | None:
        """Apply matching geometry, or report that it is unavailable."""


class IdentityPreprocessor(Preprocessor):
    """Pass model pixels through unchanged without claiming mask geometry."""

    _SPEC = PreprocessingSpec(
        kind="identity",
        deterministic=True,
        description="identity",
        fingerprint=None,
        mask_transform_available=False,
    )

    def describe(self) -> PreprocessingSpec:
        """Return the immutable identity preprocessing metadata."""

        return self._SPEC

    def transform_batch(self, batch: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Return the original batch to preserve callable compatibility."""

        return batch

    def transform_mask(self, mask: NDArray[np.bool_]) -> None:
        """Report unavailable model-space mask geometry."""

        validate_mask(mask)
        return None


class CallablePreprocessor(Preprocessor):
    """Wrap optional user batch and mask transformation callables."""

    def __init__(
        self,
        *,
        transform_batch_fn: Callable[[NDArray[np.uint8]], Any] | None = None,
        transform_mask_fn: (
            Callable[[NDArray[np.bool_]], NDArray[np.bool_]] | None
        ) = None,
        deterministic: bool = True,
        description: str = "caller-managed",
        fingerprint: str | None = None,
    ) -> None:
        if transform_batch_fn is not None and not callable(transform_batch_fn):
            raise TypeError("transform_batch_fn must be callable when provided")
        if transform_mask_fn is not None and not callable(transform_mask_fn):
            raise TypeError("transform_mask_fn must be callable when provided")
        self._transform_batch_fn = transform_batch_fn
        self._transform_mask_fn = transform_mask_fn
        self._spec = PreprocessingSpec(
            kind="callable",
            deterministic=deterministic,
            description=description,
            fingerprint=fingerprint,
            mask_transform_available=transform_mask_fn is not None,
        )

    def describe(self) -> PreprocessingSpec:
        """Return caller-supplied preprocessing metadata."""

        return self._spec

    def transform_batch(self, batch: NDArray[np.uint8]) -> Any:
        """Invoke the optional batch transform or pass through unchanged."""

        if self._transform_batch_fn is None:
            return batch
        return self._transform_batch_fn(batch)

    def transform_mask(
        self,
        mask: NDArray[np.bool_],
    ) -> NDArray[np.bool_] | None:
        """Invoke and validate the optional user mask transform."""

        validate_mask(mask)
        if self._transform_mask_fn is None:
            return None
        transformed = self._transform_mask_fn(mask.copy())
        validate_mask(transformed)
        return np.array(transformed, copy=True)


def validate_mask(mask: NDArray[np.bool_]) -> None:
    """Validate the shared source/model-space binary mask contract."""

    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy ndarray")
    if mask.dtype != np.bool_:
        raise TypeError("mask must have dtype bool")
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if any(size <= 0 for size in mask.shape):
        raise ValueError("mask height and width must be positive")


def fingerprint_payload(value: Any) -> str:
    """Hash a JSON-compatible preprocessing description deterministically."""

    normalized = _normalize_fingerprint_value(value)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_fingerprint_value(value: Any) -> Any:
    """Normalize common preprocessing metadata into canonical JSON values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("preprocessing fingerprint rejects NaN and infinity")
        return format(0.0 if value == 0.0 else value, ".12f")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("preprocessing fingerprint mappings require string keys")
        return {
            key: _normalize_fingerprint_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_normalize_fingerprint_value(item) for item in value]
    if hasattr(value, "value"):
        return _normalize_fingerprint_value(value.value)
    raise TypeError(
        f"unsupported preprocessing fingerprint value: {type(value).__name__}"
    )
