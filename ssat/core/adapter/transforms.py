"""Deterministic MMAction2-style built-in transforms and pipeline preprocessor.

Each transform here delegates its actual pixel/mask math to
:mod:`ssat.core.adapter.preprocessing` wherever an equivalent op already
exists there -- this module only adds the registry-addressable names
(``SampleFrames``, ``Resize``, ``CenterCrop``, ``TenCrop``, ``ToFloat``,
``Normalize``, ``FormatShape``) plus the handful of new ops
(``SampleFrames``/``TenCrop``/``FormatShape``) that ``preprocessing.py`` has
no equivalent for. Augmentation-style random transforms (RandomCrop,
RandomFlip, ColorJitter, ...) are intentionally not provided -- they conflict
with this tool's audit-reproducibility purpose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from ssat.core.adapter import preprocessing as _decl
from ssat.core.adapter.preprocessor import Preprocessor, fingerprint_payload, validate_mask
from ssat.core.adapter.transform_registry import (
    BaseTransform,
    TransformError,
    TransformRegistry,
    build_pipeline,
)
from ssat.core.adapter.types import PreprocessingSpec


@dataclass(frozen=True, slots=True)
class SampleFrames(BaseTransform):
    """Select ``clip_len`` frames centered on the batch's existing time axis.

    v1 supports only the deterministic center-clip case (no random shift, no
    multi-clip TSN-style sampling). Frame decode/uniform sampling already
    happened at the Source stage (``VideoFolderSource.load()``); this only
    re-slices the already-decoded ``(B, T, H, W, C)`` batch's T axis.
    """

    type_name: ClassVar[str] = "SampleFrames"
    clip_len: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.clip_len, bool)
            or not isinstance(self.clip_len, int)
            or self.clip_len <= 0
        ):
            raise ValueError("clip_len must be a positive int")

    def apply_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
        return _center_slice(batch, self.clip_len, axis=1)

    def apply_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        if mask.ndim != 3:  # (H, W) broadcasts across every frame -- no T axis to slice
            return mask
        return _center_slice(mask, self.clip_len, axis=0)


def _center_slice(array: NDArray[Any], clip_len: int, *, axis: int) -> NDArray[Any]:
    """Slice ``clip_len`` entries centered on ``axis``, rounding down on ties."""

    total = array.shape[axis]
    if clip_len > total:
        raise TransformError(f"clip_len={clip_len} exceeds available frames={total}")
    start = (total - clip_len) // 2
    index: list[slice] = [slice(None)] * array.ndim
    index[axis] = slice(start, start + clip_len)
    return array[tuple(index)]


@dataclass(frozen=True, slots=True)
class Resize(BaseTransform):
    """MMAction2-style ``scale=[w, h]`` (one axis may be ``-1``) resize.

    mmcv/mmaction2 convention is ``(width, height)``; the underlying
    ``preprocessing.Resize.size`` convention is ``(height, width)``.
    ``_coerce_scale`` converts explicitly -- getting this axis order wrong is
    the single highest-risk correctness bug in this module (see the
    implementation plan's risk table), so it is covered by a dedicated
    non-square regression test.
    """

    type_name: ClassVar[str] = "Resize"
    scale: int | tuple[int, int]
    _op: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_op", _decl.Resize(_coerce_scale(self.scale)))

    def apply_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
        return _decl.apply_preprocessing(batch, (self._op,))

    def apply_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        return _decl.transform_mask_geometry(mask, (self._op,))


def _coerce_scale(scale: int | Sequence[int]) -> _decl.Size:
    """Convert an mmaction ``(w, h)`` pair, or a bare short-edge int, to ``decl.Size``.

    Either axis may be ``-1`` to request short-edge resizing with the other
    axis as the target length (mirrors ``preprocessing.Resize``'s bare-int
    form); if neither is ``-1`` the pair is an exact ``(width, height)``
    target and is returned axis-swapped as ``(height, width)`` for the
    underlying engine. Setting both to ``-1`` is rejected as undefined.
    """

    if isinstance(scale, int) and not isinstance(scale, bool):
        return scale
    if isinstance(scale, bool) or not isinstance(scale, (tuple, list)) or len(scale) != 2:
        raise ValueError("Resize.scale must be an int or a (w, h) pair")
    w, h = scale
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (w, h)):
        raise ValueError("Resize.scale entries must be integers")
    if w == -1 and h == -1:
        raise ValueError("Resize.scale cannot set both (w, h) to -1")
    return h if w == -1 else w if h == -1 else (h, w)


@dataclass(frozen=True, slots=True)
class CenterCrop(BaseTransform):
    """Square center crop, zero-padding undersized inputs.

    v1 accepts only a square ``crop_size: int``. A rectangular ``(h, w)``
    crop would reopen the same axis-order ambiguity as ``Resize.scale``
    (mmaction2's rectangular crop_size is also ``(w, h)``) and is not needed
    by the built-in op set yet.
    """

    type_name: ClassVar[str] = "CenterCrop"
    crop_size: int
    _op: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.crop_size, bool)
            or not isinstance(self.crop_size, int)
            or self.crop_size <= 0
        ):
            raise ValueError("crop_size must be a positive int")
        object.__setattr__(self, "_op", _decl.CenterCrop(self.crop_size))

    def apply_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
        return _decl.apply_preprocessing(batch, (self._op,))

    def apply_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        return _decl.transform_mask_geometry(mask, (self._op,))


@dataclass(frozen=True, slots=True)
class ToFloat(BaseTransform):
    """Convert pixels to float32 and multiply by ``scale``."""

    type_name: ClassVar[str] = "ToFloat"
    scale: float = 1.0 / 255.0
    _op: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_op", _decl.ToFloat(self.scale))

    def apply_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
        return _decl.apply_preprocessing(batch, (self._op,))

    # apply_mask: default identity is correct -- ToFloat is photometric-only.


@dataclass(frozen=True, slots=True)
class Normalize(BaseTransform):
    """Apply channel-wise ``(value - mean) / std`` in channel-last layout."""

    type_name: ClassVar[str] = "Normalize"
    mean: tuple[float, ...]
    std: tuple[float, ...]
    _op: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_op", _decl.Normalize(tuple(self.mean), tuple(self.std)))

    def apply_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
        return _decl.apply_preprocessing(batch, (self._op,))

    # apply_mask: default identity is correct -- Normalize is photometric-only.


@dataclass(frozen=True, slots=True)
class TenCrop(BaseTransform):
    """Corners + center crop, each mirrored horizontally -- 10 deterministic views.

    Not augmentation (no randomness), but does expand the batch axis 10x:
    ``(B, T, H, W, C) -> (B*10, T, H, W, C)``, with the crop index varying
    fastest within each original sample's block (sample 0's 10 crops, then
    sample 1's 10 crops, ...). Averaging the resulting 10x predictions back
    down to one score per sample is the caller's/adapter's responsibility --
    it is out of scope here.
    """

    type_name: ClassVar[str] = "TenCrop"
    mask_supported: ClassVar[bool] = False  # batch expansion breaks the 1:1 mask contract
    crop_size: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.crop_size, bool)
            or not isinstance(self.crop_size, int)
            or self.crop_size <= 0
        ):
            raise ValueError("crop_size must be a positive int")

    def apply_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
        return _ten_crop_batch(batch, self.crop_size)

    def apply_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        raise TransformError("TenCrop does not support mask geometry (mask_supported=False)")


def _ten_crop_batch(batch: NDArray[Any], crop_size: int) -> NDArray[Any]:
    """Take 5 fixed crops (corners + center) plus their horizontal mirrors.

    Pads like ``preprocessing._center_crop_batch`` when the input is smaller
    than ``crop_size``, then merges the new 10-view axis into the batch axis.
    """

    height, width = batch.shape[2:4]
    pad_height = max(crop_size - height, 0)
    pad_width = max(crop_size - width, 0)
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

    origins = (
        (0, 0),
        (0, width - crop_size),
        (height - crop_size, 0),
        (height - crop_size, width - crop_size),
        ((height - crop_size) // 2, (width - crop_size) // 2),
    )
    views: list[NDArray[Any]] = []
    for top, left in origins:
        crop = batch[:, :, top : top + crop_size, left : left + crop_size, :]
        views.append(crop)
        views.append(crop[:, :, :, ::-1, :])  # horizontal flip (W axis)
    stacked = np.stack(views, axis=1)  # (B, 10, T, crop_size, crop_size, C)
    batch_size = batch.shape[0]
    merged = stacked.reshape((batch_size * 10, *stacked.shape[2:]))
    return np.ascontiguousarray(merged)


@dataclass(frozen=True, slots=True)
class FormatShape(BaseTransform):
    """Final layout conversion -- delegates entirely to ChannelsFirst/SqueezeTime.

    Only two ``input_format`` values are supported in v1, chosen to exactly
    match what the two existing adapters already consume, not the full
    mmaction2 set:

    - ``"NCHW"``: channels-first, then squeeze T (requires ``T == 1``, e.g.
      after ``SampleFrames(clip_len=1)``) -> ``(B, C, H, W)``. Matches
      ``TorchvisionAdapter``/``DeclarativePreprocessor``'s existing image
      path.
    - ``"NTCHW"``: channels-first, T kept -> ``(B, T, C, H, W)``. Matches
      what ``TorchvisionVideoAdapter.predict()`` already expects before its
      own hardcoded ``.permute(0, 2, 1, 3, 4)`` to ``(B, C, T, H, W)``.

    A literal mmaction ``"NCTHW"`` (channel-before-time, no adapter-side
    permute needed) is deliberately not offered in v1.
    """

    type_name: ClassVar[str] = "FormatShape"
    input_format: Literal["NCHW", "NTCHW"]

    def __post_init__(self) -> None:
        if self.input_format not in ("NCHW", "NTCHW"):
            raise ValueError('FormatShape.input_format must be "NCHW" or "NTCHW"')

    def apply_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
        ops: tuple[Any, ...] = (_decl.ChannelsFirst(),)
        if self.input_format == "NCHW":
            ops += (_decl.SqueezeTime(),)  # raises ValueError if T != 1 -- existing behavior
        return _decl.apply_preprocessing(batch, ops)

    # apply_mask: default identity is correct -- transform_mask_geometry already
    # ignores ChannelsFirst/SqueezeTime (channel/time reshuffling doesn't move
    # pixels), so FormatShape doesn't need to override apply_mask.


def default_transform_registry() -> TransformRegistry:
    """Return a fresh registry containing only the v1 built-in 7 transforms."""

    registry = TransformRegistry()
    built_ins = (SampleFrames, Resize, CenterCrop, TenCrop, ToFloat, Normalize, FormatShape)
    for transform_cls in built_ins:
        registry.register(transform_cls)
    return registry


class PipelinePreprocessor(Preprocessor):
    """Bridge a registry-built ``Pipeline`` into the existing ``Preprocessor`` ABC.

    Sits alongside ``DeclarativePreprocessor``/``TorchvisionPreprocessor``/
    ``TimmPreprocessor`` as a fourth ``Preprocessor`` implementation -- no
    change to the ABC or its three existing implementations.
    """

    def __init__(
        self,
        config: Sequence[Mapping[str, Any]],
        *,
        registry: TransformRegistry | None = None,
    ) -> None:
        self._pipeline = build_pipeline(config, registry or default_transform_registry())
        self._spec = PreprocessingSpec(
            kind="pipeline",
            deterministic=True,
            description=self._pipeline.describe(),
            fingerprint=fingerprint_payload(list(self._pipeline.config)),
            mask_transform_available=self._pipeline.mask_supported,
        )

    def describe(self) -> PreprocessingSpec:
        """Return the canonical registry-pipeline identity."""

        return self._spec

    def transform_batch(self, batch: NDArray[np.uint8]) -> NDArray[Any]:
        """Apply every registered step to model pixels."""

        return self._pipeline(batch)

    def transform_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_] | None:
        """Apply matching geometry, or report it is unavailable.

        Accepts and returns ``(H, W)`` or ``(T, H, W)`` masks. Returns
        ``None`` when any step in the pipeline (e.g. ``TenCrop``) does not
        support mask geometry, matching ``IdentityPreprocessor``'s contract.
        """

        validate_mask(mask)
        if not self._pipeline.mask_supported:
            return None
        return self._pipeline.apply_mask(mask)
