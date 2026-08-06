"""Declarative, deterministic numpy preprocessing operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from ssat.core.adapter.preprocessor import (
    Preprocessor,
    fingerprint_payload,
    validate_mask,
)
from ssat.core.adapter.types import PreprocessingSpec


Size: TypeAlias = int | tuple[int, int]


def _validate_size(size: Size) -> None:
    """Require a positive short-edge or exact ``(height, width)`` size."""

    if isinstance(size, bool):
        raise TypeError("size must be an int or a (height, width) pair")
    if isinstance(size, int):
        if size <= 0:
            raise ValueError("size must be positive")
        return
    if not isinstance(size, tuple):
        raise TypeError("size must be an int or a (height, width) tuple")
    if len(size) != 2 or any(isinstance(value, bool) or value <= 0 for value in size):
        raise ValueError("size pair must contain positive height and width")


@dataclass(frozen=True, slots=True)
class Resize:
    """Resize the short edge, or resize to an exact height and width."""

    size: Size

    def __post_init__(self) -> None:
        _validate_size(self.size)


@dataclass(frozen=True, slots=True)
class CenterCrop:
    """Crop around the image center, zero-padding undersized inputs."""

    size: Size

    def __post_init__(self) -> None:
        _validate_size(self.size)


@dataclass(frozen=True, slots=True)
class ToFloat:
    """Convert pixels to float32 and multiply by ``scale``."""

    scale: float = 1.0 / 255.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("ToFloat scale must be finite and positive")


@dataclass(frozen=True, slots=True)
class Normalize:
    """Apply channel-wise ``(value - mean) / std`` in channel-last layout."""

    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.mean or len(self.mean) != len(self.std):
            raise ValueError("Normalize mean and std must be non-empty and aligned")
        if any(not np.isfinite(value) for value in (*self.mean, *self.std)):
            raise ValueError("Normalize values must be finite")
        if any(value <= 0 for value in self.std):
            raise ValueError("Normalize std values must be positive")


@dataclass(frozen=True, slots=True)
class ChannelsFirst:
    """Move the channel axis from last to ``(B, T, C, H, W)``."""


@dataclass(frozen=True, slots=True)
class SqueezeTime:
    """Remove a singleton time dimension for image classifiers."""


PreprocessingOp: TypeAlias = (
    Resize | CenterCrop | ToFloat | Normalize | ChannelsFirst | SqueezeTime
)
OpInput: TypeAlias = PreprocessingOp | Mapping[str, Any]


class DeclarativePreprocessor(Preprocessor):
    """Apply one declared deterministic pipeline to both pixels and masks."""

    def __init__(self, ops: Sequence[OpInput]) -> None:
        self._ops = parse_preprocessing_ops(ops)
        self._spec = PreprocessingSpec(
            kind="declarative",
            deterministic=True,
            description=describe_preprocessing(self._ops),
            fingerprint=fingerprint_payload(_ops_payload(self._ops)),
            mask_transform_available=True,
        )

    @property
    def ops(self) -> tuple[PreprocessingOp, ...]:
        """Expose the immutable resolved operation list."""

        return self._ops

    def describe(self) -> PreprocessingSpec:
        """Return the canonical declarative pipeline identity."""

        return self._spec

    def transform_batch(self, batch: NDArray[np.uint8]) -> NDArray[Any]:
        """Apply all declared operations to model pixels."""

        return apply_preprocessing(batch, self._ops)

    def transform_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Apply only declared geometry with nearest-neighbor interpolation."""

        validate_mask(mask)
        return transform_mask_geometry(mask, self._ops)


def _coerce_size(value: Any) -> Size:
    """Convert JSON-style size lists into the immutable size contract."""

    if isinstance(value, bool):
        raise TypeError("size must be an int or a (height, width) pair")
    if isinstance(value, int):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            raise TypeError("size pair entries must be integers")
        return (value[0], value[1])
    raise TypeError("size must be an int or a (height, width) pair")


def parse_preprocessing_ops(ops: Sequence[OpInput]) -> tuple[PreprocessingOp, ...]:
    """Parse typed operations or JSON-compatible ``{"op": ...}`` mappings."""

    parsed: list[PreprocessingOp] = []
    known_types = (Resize, CenterCrop, ToFloat, Normalize, ChannelsFirst, SqueezeTime)
    for index, value in enumerate(ops):
        if isinstance(value, known_types):
            parsed.append(value)
            continue
        if not isinstance(value, Mapping):
            raise TypeError(f"preprocessing op at index {index} is invalid")
        raw = dict(value)
        name = raw.pop("op", None)
        try:
            if name == "resize":
                parsed.append(Resize(_coerce_size(raw.pop("size"))))
            elif name == "center_crop":
                parsed.append(CenterCrop(_coerce_size(raw.pop("size"))))
            elif name == "to_float":
                parsed.append(ToFloat(float(raw.pop("scale", 1.0 / 255.0))))
            elif name == "normalize":
                parsed.append(
                    Normalize(
                        tuple(float(item) for item in raw.pop("mean")),
                        tuple(float(item) for item in raw.pop("std")),
                    )
                )
            elif name == "channels_first":
                parsed.append(ChannelsFirst())
            elif name == "squeeze_time":
                parsed.append(SqueezeTime())
            else:
                raise ValueError(f"unknown preprocessing op {name!r}")
        except KeyError as error:
            raise ValueError(
                f"preprocessing op {name!r} is missing {error.args[0]!r}"
            ) from error
        if raw:
            names = ", ".join(sorted(raw))
            raise ValueError(f"preprocessing op {name!r} has unknown fields: {names}")

    _validate_order(parsed)
    return tuple(parsed)


def _validate_order(ops: Sequence[PreprocessingOp]) -> None:
    """Keep geometry and channel-last photometry before layout conversion."""

    layout_converted = False
    time_squeezed = False
    for op in ops:
        if isinstance(op, ChannelsFirst):
            if layout_converted or time_squeezed:
                raise ValueError("ChannelsFirst may appear only once")
            layout_converted = True
        elif isinstance(op, SqueezeTime):
            if time_squeezed:
                raise ValueError("SqueezeTime may appear only once")
            time_squeezed = True
        elif layout_converted or time_squeezed:
            raise ValueError("resize, crop, and photometric ops must precede layout ops")


def describe_preprocessing(ops: Sequence[PreprocessingOp]) -> str:
    """Create a concise stable description for ``AdapterSpec``."""

    return " -> ".join(repr(op) for op in ops) if ops else "identity"


def _ops_payload(ops: Sequence[PreprocessingOp]) -> list[dict[str, Any]]:
    """Convert typed operations into their canonical fingerprint payload."""

    payload: list[dict[str, Any]] = []
    for op in ops:
        if isinstance(op, Resize):
            payload.append({"op": "resize", "size": op.size})
        elif isinstance(op, CenterCrop):
            payload.append({"op": "center_crop", "size": op.size})
        elif isinstance(op, ToFloat):
            payload.append({"op": "to_float", "scale": op.scale})
        elif isinstance(op, Normalize):
            payload.append({"op": "normalize", "mean": op.mean, "std": op.std})
        elif isinstance(op, ChannelsFirst):
            payload.append({"op": "channels_first"})
        elif isinstance(op, SqueezeTime):
            payload.append({"op": "squeeze_time"})
    return payload


def apply_preprocessing(batch: NDArray[Any], ops: Sequence[PreprocessingOp]) -> NDArray[Any]:
    """Apply a validated declarative pipeline to a THWC batch."""

    result = batch
    for op in ops:
        if isinstance(op, Resize):
            result = _resize_batch(result, op.size, interpolation=cv2.INTER_LINEAR)
        elif isinstance(op, CenterCrop):
            result = _center_crop_batch(result, op.size)
        elif isinstance(op, ToFloat):
            result = result.astype(np.float32) * np.float32(op.scale)
        elif isinstance(op, Normalize):
            if result.shape[-1] != len(op.mean):
                raise ValueError("Normalize channel count does not match the batch")
            mean = np.asarray(op.mean, dtype=np.float32)
            std = np.asarray(op.std, dtype=np.float32)
            result = (result.astype(np.float32) - mean) / std
        elif isinstance(op, ChannelsFirst):
            result = np.transpose(result, (0, 1, 4, 2, 3))
        elif isinstance(op, SqueezeTime):
            if result.shape[1] != 1:
                raise ValueError("SqueezeTime requires T=1")
            result = np.squeeze(result, axis=1)
    return np.ascontiguousarray(result)


def transform_mask_geometry(
    mask: NDArray[np.bool_],
    ops: Sequence[PreprocessingOp],
) -> NDArray[np.bool_]:
    """Apply only resize/crop operations with nearest-neighbor interpolation."""

    result: NDArray[Any] = mask[None, None, :, :, None].astype(np.uint8)
    for op in ops:
        if isinstance(op, Resize):
            result = _resize_batch(result, op.size, interpolation=cv2.INTER_NEAREST)
        elif isinstance(op, CenterCrop):
            result = _center_crop_batch(result, op.size)
    return np.asarray(result[0, 0, :, :, 0], dtype=np.bool_)


def _output_size(height: int, width: int, size: Size) -> tuple[int, int]:
    """Resolve torchvision-style integer short-edge resize semantics."""

    if isinstance(size, tuple):
        return size
    if height <= width:
        return size, max(1, int(size * width / height))
    return max(1, int(size * height / width)), size


def _resize_batch(
    batch: NDArray[Any],
    size: Size,
    *,
    interpolation: int,
) -> NDArray[Any]:
    """Resize every frame while retaining all B/T/channel axes."""

    height, width = batch.shape[2:4]
    out_height, out_width = _output_size(height, width, size)
    frames: list[NDArray[Any]] = []
    for frame in batch.reshape((-1, height, width, batch.shape[-1])):
        resized = cv2.resize(frame, (out_width, out_height), interpolation=interpolation)
        if resized.ndim == 2:
            resized = resized[:, :, None]
        frames.append(resized)
    return np.stack(frames).reshape(
        (batch.shape[0], batch.shape[1], out_height, out_width, batch.shape[-1])
    )


def _center_crop_batch(batch: NDArray[Any], size: Size) -> NDArray[Any]:
    """Center crop every frame after symmetric constant padding if needed."""

    target_height, target_width = (size, size) if isinstance(size, int) else size
    height, width = batch.shape[2:4]
    pad_height = max(target_height - height, 0)
    pad_width = max(target_width - width, 0)
    if pad_height or pad_width:
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
        batch = np.pad(
            batch,
            ((0, 0), (0, 0), (top, bottom), (left, right), (0, 0)),
            mode="constant",
        )
        height, width = batch.shape[2:4]
    top = (height - target_height) // 2
    left = (width - target_width) // 2
    return batch[:, :, top : top + target_height, left : left + target_width, :]
