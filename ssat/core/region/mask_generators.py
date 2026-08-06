"""Built-in concrete region mask generators."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from PIL import Image

from ssat.core.region.mask_base import RegionMaskGenerator, RegionResolutionError
from ssat.core.region.types import RegionSpec
from ssat.core.types import RegionKind, thaw_json_value
from ssat.utils.io import sha256_file


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
        raise RegionResolutionError(f"{field_name} must be a non-negative integer")
    return int(value)


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

    integer = _nonnegative_int(value, field_name)
    if integer == 0:
        raise RegionResolutionError(f"{field_name} must be positive")
    return integer


def _target_region_spec(value: Any) -> RegionSpec:
    """Reconstruct a target recipe embedded by ``PlanBuilder``.

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
        return RegionSpec(
            region_id=value["region_id"],
            region_instance_id=value["region_instance_id"],
            kind=RegionKind(value["kind"]),
            params=params,
            ref=value["ref"],
            ref_hash=value["ref_hash"],
        )
    except (TypeError, ValueError) as error:
        raise RegionResolutionError("target_region recipe is invalid") from error


class GridMaskGenerator(RegionMaskGenerator):
    """Materialize one concrete cell from a resolved grid recipe."""

    def supports(self, spec: RegionSpec) -> bool:
        """Return whether the recipe is a concrete grid cell.

        Args:
            spec: Concrete region recipe.

        Returns:
            ``True`` only for the grid kind.
        """

        return spec.kind is RegionKind.GRID

    def get_mask(
        self,
        height: int,
        width: int,
        spec: RegionSpec,
        rng: Generator | None = None,
    ) -> NDArray[np.bool_]:
        """Materialize a grid cell using exact integer boundaries.

        Args:
            height: Source image height.
            width: Source image width.
            spec: Concrete grid-cell recipe.
            rng: Unused item-local generator.

        Returns:
            Boolean mask selecting exactly one grid cell.

        Raises:
            RegionResolutionError: If the grid recipe is invalid.
        """

        expected = {"rows", "cols", "row_index", "col_index"}
        if set(spec.params) != expected:
            raise RegionResolutionError(
                "grid params must contain rows, cols, row_index, and col_index"
            )
        rows = _positive_int(spec.params["rows"], "grid.rows")
        cols = _positive_int(spec.params["cols"], "grid.cols")
        row_index = _nonnegative_int(spec.params["row_index"], "grid.row_index")
        col_index = _nonnegative_int(spec.params["col_index"], "grid.col_index")
        if row_index >= rows or col_index >= cols:
            raise RegionResolutionError("grid cell index is outside its grid")

        row_start = row_index * height // rows
        row_end = (row_index + 1) * height // rows
        col_start = col_index * width // cols
        col_end = (col_index + 1) * width // cols
        mask = np.zeros((height, width), dtype=np.bool_)
        mask[row_start:row_end, col_start:col_end] = True
        return mask


class ExplicitMaskGenerator(RegionMaskGenerator):
    """Verify, decode, and cache single-channel explicit masks."""

    def supports(self, spec: RegionSpec) -> bool:
        """Return whether the recipe references an explicit bitmap.

        Args:
            spec: Concrete region recipe.

        Returns:
            ``True`` only for the explicit kind.
        """

        return spec.kind is RegionKind.EXPLICIT

    def get_mask(
        self,
        height: int,
        width: int,
        spec: RegionSpec,
        rng: Generator | None = None,
    ) -> NDArray[np.bool_]:
        """Verify and materialize one explicit bitmap mask.

        Args:
            height: Expected source image height.
            width: Expected source image width.
            spec: Explicit mask recipe with a reference and hash.
            rng: Unused item-local generator.

        Returns:
            A caller-owned boolean mask matching the source dimensions.

        Raises:
            RegionResolutionError: If the reference, hash, mode, or shape is
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

        cached = self._context.explicit_cache.get(actual_hash)
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
            self._context.explicit_cache.put(actual_hash, cached)

        if cached.shape != (height, width):
            raise RegionResolutionError(
                f"explicit mask shape {cached.shape} does not match {(height, width)}"
            )
        return cached.copy()


class RandomAreaMatchMaskGenerator(RegionMaskGenerator):
    """Sample a uniform mask matching an embedded target's pixel count."""

    def supports(self, spec: RegionSpec) -> bool:
        """Return whether the recipe is an area-matched random control.

        Args:
            spec: Concrete region recipe.

        Returns:
            ``True`` only for the random-area-match kind.
        """

        return spec.kind is RegionKind.RANDOM_AREA_MATCH

    def get_mask(
        self,
        height: int,
        width: int,
        spec: RegionSpec,
        rng: Generator | None = None,
    ) -> NDArray[np.bool_]:
        """Select pixels uniformly without replacement at the target area.

        Args:
            height: Source image height.
            width: Source image width.
            spec: Internal area-matched control recipe.
            rng: Required item-local NumPy generator.

        Returns:
            Boolean mask with exactly the target mask's pixel count.

        Raises:
            RegionResolutionError: If the recipe or generator is invalid.
        """

        if rng is None or not isinstance(rng, Generator):
            raise RegionResolutionError(
                "random_area_match requires a numpy Generator"
            )
        expected = {"target_region", "control_request_index", "control_index"}
        if set(spec.params) != expected:
            raise RegionResolutionError(
                "random_area_match params must contain target_region and control indices"
            )
        _nonnegative_int(
            spec.params["control_request_index"], "control_request_index"
        )
        _nonnegative_int(spec.params["control_index"], "control_index")
        target = _target_region_spec(spec.params["target_region"])
        if target.kind is RegionKind.RANDOM_AREA_MATCH:
            raise RegionResolutionError(
                "nested random_area_match targets are not allowed"
            )

        target_mask = self._context.resolve_target(height, width, target)
        target_area = int(np.count_nonzero(target_mask))
        selected = rng.choice(height * width, size=target_area, replace=False)
        mask = np.zeros(height * width, dtype=np.bool_)
        mask[selected] = True
        return mask.reshape(height, width)
