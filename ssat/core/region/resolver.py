"""Validated facade for concrete region mask generators."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ssat.core.region.mask_base import (
    ExplicitMaskCache,
    MaskResolutionContext,
    RegionMaskGenerator,
    RegionResolutionError,
    mean_frame_area,
)
from ssat.core.region.mask_dispatch import dispatch_mask_generator
from ssat.core.region.mask_factory import build_mask_generators
from ssat.core.region.skeleton_store import SkeletonBBoxStore
from ssat.core.region.types import RegionMeta, RegionSpec

REGION_GENERATOR_VERSION = "1.0.0"


class RegionResolver:
    """Validate region inputs and dispatch concrete mask generation.

    Args:
        explicit_cache_size: Maximum decoded explicit masks retained per
            resolver instance.
        mask_generators: Optional generators in dispatch-priority order.
        skeleton_store: Optional pre-computed skeleton body-part bbox data,
            shared with the default ``SkeletonPartsMaskGenerator``. Ignored
            when ``mask_generators`` is supplied explicitly, since the
            caller then owns each generator's context directly.

    Raises:
        TypeError: If a supplied value is not a mask generator.
        ValueError: If cache size or the generator collection is invalid.
    """

    def __init__(
        self,
        *,
        explicit_cache_size: int = 128,
        mask_generators: Sequence[RegionMaskGenerator] | None = None,
        skeleton_store: SkeletonBBoxStore | None = None,
    ) -> None:
        cache = ExplicitMaskCache(explicit_cache_size)
        self._mask_generators: tuple[RegionMaskGenerator, ...] = ()
        if mask_generators is None:
            context = MaskResolutionContext(
                explicit_cache=cache,
                resolve_target=self._resolve_target,
                skeleton_store=skeleton_store,
            )
            resolved = tuple(build_mask_generators(context))
        else:
            resolved = tuple(mask_generators)
        if not resolved:
            raise ValueError("mask_generators must not be empty")
        if any(
            not isinstance(generator, RegionMaskGenerator)
            for generator in resolved
        ):
            raise TypeError(
                "mask_generators must contain RegionMaskGenerator values"
            )
        self._mask_generators = resolved

    def resolve(
        self,
        shape: tuple[int, int, int, int],
        spec: RegionSpec,
        rng: Generator | None = None,
    ) -> tuple[NDArray[np.bool_], RegionMeta]:
        """Materialize a concrete recipe for one source sample shape.

        Args:
            shape: Positive source shape in ``(T, H, W, C)`` layout.
            spec: Concrete region recipe produced during planning.
            rng: Item-local generator required by random controls.

        Returns:
            A ``(H, W)`` or ``(T, H, W)`` boolean mask and its measured
            metadata. ``intended_area_px``/``intended_area_ratio`` report the
            mean per-frame nonzero count for a ``(T, H, W)`` mask, so they
            stay comparable to a broadcast ``(H, W)`` mask's exact count.

        Raises:
            RegionResolutionError: If the shape, recipe, dispatch, or returned
                mask violates the region contract.
        """

        try:
            time, height, width, _ = self._validate_shape(shape)
            if not isinstance(spec, RegionSpec):
                raise RegionResolutionError("spec must be a RegionSpec")
            mask = self._get_mask(height, width, spec, rng)
            if mask.ndim == 3 and mask.shape[0] != time:
                raise RegionResolutionError(
                    "mask frame count does not match the source sample"
                )
        except RegionResolutionError:
            raise
        except Exception as error:
            raise RegionResolutionError(
                "failed to resolve "
                f"region_instance_id={getattr(spec, 'region_instance_id', None)!r}"
            ) from error

        area = mean_frame_area(mask)
        return mask, RegionMeta(
            intended_area_px=int(round(area)),
            intended_area_ratio=area / (height * width),
            generator_kind=spec.kind.value,
            generator_version=REGION_GENERATOR_VERSION,
            confidence=None,
        )

    def _get_mask(
        self,
        height: int,
        width: int,
        spec: RegionSpec,
        rng: Generator | None,
    ) -> NDArray[np.bool_]:
        """Dispatch and validate one generated mask.

        Args:
            height: Source image height.
            width: Source image width.
            spec: Concrete region recipe.
            rng: Optional item-local generator.

        Returns:
            Validated boolean mask in ``(H, W)`` or ``(T, H, W)`` layout.

        Raises:
            RegionResolutionError: If dispatch fails or output is invalid.
        """

        mask = dispatch_mask_generator(
            self._mask_generators,
            height,
            width,
            spec,
            rng,
        )
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
            raise RegionResolutionError(
                "mask generator produced an invalid output mask"
            )
        if mask.ndim == 2:
            if mask.shape != (height, width):
                raise RegionResolutionError(
                    "mask generator produced an invalid output mask"
                )
        elif mask.ndim == 3:
            if mask.shape[0] <= 0 or mask.shape[1:] != (height, width):
                raise RegionResolutionError(
                    "mask generator produced an invalid output mask"
                )
        else:
            raise RegionResolutionError(
                "mask generator produced an invalid output mask"
            )
        return mask

    def _resolve_target(
        self,
        height: int,
        width: int,
        spec: RegionSpec,
    ) -> NDArray[np.bool_]:
        """Resolve an embedded deterministic target through the same registry.

        Args:
            height: Source image height.
            width: Source image width.
            spec: Embedded target recipe.

        Returns:
            Validated ``(H, W)`` target mask.

        Raises:
            RegionResolutionError: If the target resolves to a per-frame
                ``(T, H, W)`` mask, which area-matched controls do not
                support yet.
        """

        mask = self._get_mask(height, width, spec, None)
        if mask.ndim == 3:
            raise RegionResolutionError(
                "random_area_match does not support per-frame (T, H, W) targets"
            )
        return mask

    @staticmethod
    def _validate_shape(
        shape: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """Validate and normalize the source array shape.

        Args:
            shape: Candidate ``(T, H, W, C)`` shape.

        Returns:
            A tuple containing native positive integers.

        Raises:
            RegionResolutionError: If the shape violates the source contract.
        """

        if not isinstance(shape, tuple) or len(shape) != 4:
            raise RegionResolutionError("shape must be a (T, H, W, C) tuple")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value <= 0
            for value in shape
        ):
            raise RegionResolutionError("shape dimensions must be positive integers")
        return tuple(int(value) for value in shape)  # type: ignore[return-value]
