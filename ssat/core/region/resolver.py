"""Materialize concrete region recipes into source-space masks."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from PIL import Image

from ssat.core.region.types import RegionMeta, RegionSpec
from ssat.core.types import RegionKind, thaw_json_value
from ssat.utils.io import sha256_file

REGION_GENERATOR_VERSION = "1.0.0"


class RegionResolutionError(ValueError):
    """Indicate that a concrete region cannot be materialized safely."""


class RegionResolver:
    """Convert one concrete ``RegionSpec`` into one boolean image mask.

    Args:
        explicit_cache_size: Maximum decoded explicit masks retained per
            resolver instance.

    Raises:
        ValueError: If ``explicit_cache_size`` is not a positive integer.
    """

    def __init__(self, *, explicit_cache_size: int = 128) -> None:
        if (
            isinstance(explicit_cache_size, bool)
            or not isinstance(explicit_cache_size, int)
            or explicit_cache_size <= 0
        ):
            raise ValueError("explicit_cache_size must be a positive integer")
        self._explicit_cache_size = explicit_cache_size
        self._explicit_cache: OrderedDict[str, NDArray[np.bool_]] = OrderedDict()

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
            A ``(H, W)`` boolean mask and its measured metadata.

        Raises:
            RegionResolutionError: If the shape or recipe is invalid, a
                referenced mask changed, or the region kind is unsupported.
        """

        try:
            _, height, width, _ = self._validate_shape(shape)
            if not isinstance(spec, RegionSpec):
                raise RegionResolutionError("spec must be a RegionSpec")

            if spec.kind is RegionKind.GRID:
                mask = self._resolve_grid(height, width, spec)
            elif spec.kind is RegionKind.EXPLICIT:
                mask = self._resolve_explicit(height, width, spec)
            elif spec.kind is RegionKind.RANDOM_AREA_MATCH:
                mask = self._resolve_random_area_match(height, width, spec, rng)
            else:
                raise RegionResolutionError(
                    f"region kind {spec.kind.value!r} is not implemented"
                )
        except RegionResolutionError:
            raise
        except Exception as error:
            raise RegionResolutionError(
                f"failed to resolve region_instance_id={getattr(spec, 'region_instance_id', None)!r}"
            ) from error

        area = int(np.count_nonzero(mask))
        return mask, RegionMeta(
            intended_area_px=area,
            intended_area_ratio=area / (height * width),
            generator_kind=spec.kind.value,
            generator_version=REGION_GENERATOR_VERSION,
            confidence=None,
        )

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

    @staticmethod
    def _resolve_grid(
        height: int,
        width: int,
        spec: RegionSpec,
    ) -> NDArray[np.bool_]:
        """Materialize one grid cell using exact integer boundaries.

        Args:
            height: Source image height.
            width: Source image width.
            spec: Concrete grid-cell recipe.

        Returns:
            Boolean mask selecting exactly one grid cell.

        Raises:
            RegionResolutionError: If the grid recipe is incomplete or invalid.
        """

        expected = {"rows", "cols", "row_index", "col_index"}
        if set(spec.params) != expected:
            raise RegionResolutionError(
                "grid params must contain rows, cols, row_index, and col_index"
            )
        rows = RegionResolver._positive_int(spec.params["rows"], "grid.rows")
        cols = RegionResolver._positive_int(spec.params["cols"], "grid.cols")
        row_index = RegionResolver._nonnegative_int(
            spec.params["row_index"], "grid.row_index"
        )
        col_index = RegionResolver._nonnegative_int(
            spec.params["col_index"], "grid.col_index"
        )
        if row_index >= rows or col_index >= cols:
            raise RegionResolutionError("grid cell index is outside its grid")

        row_start = row_index * height // rows
        row_end = (row_index + 1) * height // rows
        col_start = col_index * width // cols
        col_end = (col_index + 1) * width // cols
        mask = np.zeros((height, width), dtype=np.bool_)
        mask[row_start:row_end, col_start:col_end] = True
        return mask

    def _resolve_explicit(
        self,
        height: int,
        width: int,
        spec: RegionSpec,
    ) -> NDArray[np.bool_]:
        """Verify, decode, and cache a single-channel explicit mask.

        Args:
            height: Expected source image height.
            width: Expected source image width.
            spec: Explicit mask recipe with an absolute reference and hash.

        Returns:
            A caller-owned boolean mask matching the source dimensions.

        Raises:
            RegionResolutionError: If the reference, digest, mode, or shape is
                invalid.
        """

        if spec.params:
            raise RegionResolutionError("explicit regions do not accept params")
        if spec.ref is None or spec.ref_hash is None:
            raise RegionResolutionError("explicit regions require ref and ref_hash")

        path = Path(spec.ref)
        try:
            actual_hash = sha256_file(path)
        except OSError as error:
            raise RegionResolutionError(f"cannot read explicit mask: {path}") from error
        if actual_hash != spec.ref_hash:
            raise RegionResolutionError("explicit mask ref_hash mismatch")

        cached = self._explicit_cache.get(actual_hash)
        if cached is None:
            try:
                with Image.open(path) as image:
                    if getattr(image, "n_frames", 1) != 1:
                        raise RegionResolutionError(
                            "explicit mask must contain exactly one frame"
                        )
                    decoded = np.asarray(image)
            except RegionResolutionError:
                raise
            except Exception as error:
                raise RegionResolutionError(
                    f"cannot decode explicit mask: {path}"
                ) from error
            if decoded.ndim != 2:
                raise RegionResolutionError(
                    "explicit mask must be a single-channel bitmap"
                )
            cached = np.asarray(decoded != 0, dtype=np.bool_)
            cached.setflags(write=False)
            self._explicit_cache[actual_hash] = cached
            self._explicit_cache.move_to_end(actual_hash)
            while len(self._explicit_cache) > self._explicit_cache_size:
                self._explicit_cache.popitem(last=False)
        else:
            self._explicit_cache.move_to_end(actual_hash)

        if cached.shape != (height, width):
            raise RegionResolutionError(
                f"explicit mask shape {cached.shape} does not match {(height, width)}"
            )
        return cached.copy()

    def _resolve_random_area_match(
        self,
        height: int,
        width: int,
        spec: RegionSpec,
        rng: Generator | None,
    ) -> NDArray[np.bool_]:
        """Sample a uniform mask with the area of an embedded target recipe.

        Args:
            height: Source image height.
            width: Source image width.
            spec: Internal area-matched control recipe.
            rng: Item-local random generator.

        Returns:
            A boolean mask with exactly the target mask's pixel count.

        Raises:
            RegionResolutionError: If the recipe or generator is invalid.
        """

        if rng is None or not isinstance(rng, Generator):
            raise RegionResolutionError("random_area_match requires a numpy Generator")
        expected = {"target_region", "control_request_index", "control_index"}
        if set(spec.params) != expected:
            raise RegionResolutionError(
                "random_area_match params must contain target_region and control indices"
            )
        self._nonnegative_int(
            spec.params["control_request_index"], "control_request_index"
        )
        self._nonnegative_int(spec.params["control_index"], "control_index")
        target = self._target_region_spec(spec.params["target_region"])
        if target.kind is RegionKind.RANDOM_AREA_MATCH:
            raise RegionResolutionError("nested random_area_match targets are not allowed")

        target_mask, _ = self.resolve((1, height, width, 1), target)
        target_area = int(np.count_nonzero(target_mask))
        selected = rng.choice(height * width, size=target_area, replace=False)
        mask = np.zeros(height * width, dtype=np.bool_)
        mask[selected] = True
        return mask.reshape(height, width)

    @staticmethod
    def _target_region_spec(value: Any) -> RegionSpec:
        """Reconstruct a target ``RegionSpec`` embedded by ``PlanBuilder``.

        Args:
            value: Frozen JSON recipe stored in control parameters.

        Returns:
            A validated concrete target region.

        Raises:
            RegionResolutionError: If the target recipe is malformed.
        """

        if not isinstance(value, Mapping):
            raise RegionResolutionError("target_region must be a mapping")
        expected = {
            "region_id",
            "region_instance_id",
            "kind",
            "params",
            "ref",
            "ref_hash",
        }
        if set(value) != expected:
            raise RegionResolutionError("target_region recipe has invalid fields")
        params = thaw_json_value(value["params"])
        if not isinstance(params, dict):
            raise RegionResolutionError("target_region.params must be a mapping")
        try:
            kind = RegionKind(value["kind"])
            return RegionSpec(
                region_id=value["region_id"],
                region_instance_id=value["region_instance_id"],
                kind=kind,
                params=params,
                ref=value["ref"],
                ref_hash=value["ref_hash"],
            )
        except (TypeError, ValueError) as error:
            raise RegionResolutionError("target_region recipe is invalid") from error

    @staticmethod
    def _positive_int(value: Any, field_name: str) -> int:
        """Validate a positive integer recipe field.

        Args:
            value: Candidate field value.
            field_name: Logical name included in an error.

        Returns:
            The validated integer.

        Raises:
            RegionResolutionError: If the value is not a positive integer.
        """

        integer = RegionResolver._nonnegative_int(value, field_name)
        if integer == 0:
            raise RegionResolutionError(f"{field_name} must be positive")
        return integer

    @staticmethod
    def _nonnegative_int(value: Any, field_name: str) -> int:
        """Validate a non-negative integer recipe field.

        Args:
            value: Candidate field value.
            field_name: Logical name included in an error.

        Returns:
            The validated integer.

        Raises:
            RegionResolutionError: If the value is not a non-negative integer.
        """

        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise RegionResolutionError(
                f"{field_name} must be a non-negative integer"
            )
        return int(value)
