"""Built-in source-space perturbation operator implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ssat.core.perturb._helpers import (
    composite,
    fill_value,
    positive_real,
    require_keys,
    require_rng,
    validate_config_fill_value,
)
from ssat.core.perturb.base import PerturbationError, PerturbationOperator
from ssat.core.types import PerturbationOp


def _apply_fill(
    array: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    params: Mapping[str, Any],
    op_name: str,
) -> NDArray[np.uint8]:
    """Apply shared scalar or per-channel fill behavior.

    Args:
        array: Validated source pixels.
        mask: Validated ``(H, W)`` or ``(T, H, W)`` selection mask.
        params: Mapping containing exactly ``value``.
        op_name: Concrete fill operation name used in errors.

    Returns:
        A copied array containing the resolved fill value.
    """

    require_keys(params, {"value"}, op_name)
    fill = fill_value(params["value"], array.shape[-1], op_name)
    result = array.copy()
    if mask.ndim == 2:
        result[:, mask, :] = fill
    else:
        result[mask] = fill
    return result


class ConstantFillOperator(PerturbationOperator):
    """Fill selected pixels with a configured constant value."""

    def supports(self, op: PerturbationOp) -> bool:
        """Return whether the requested operation is constant fill.

        Args:
            op: Perturbation operation requested by a work item.

        Returns:
            ``True`` only for ``constant_fill``.
        """

        return op is PerturbationOp.CONSTANT_FILL

    def validate_config(self, params: Mapping[str, Any]) -> None:
        """Validate constant-fill user parameters.

        Args:
            params: Mapping containing exactly one scalar or channel-list value.

        Raises:
            PerturbationError: If the parameter contract is invalid.
        """

        op_name = PerturbationOp.CONSTANT_FILL.value
        require_keys(params, {"value"}, op_name)
        validate_config_fill_value(params["value"], op_name)

    def apply(
        self,
        array: NDArray[np.uint8],
        mask: NDArray[np.bool_],
        params: Mapping[str, Any],
        rng: Generator | None = None,
    ) -> NDArray[np.uint8]:
        """Fill selected pixels and ignore an optional generator.

        Args:
            array: Validated source pixels.
            mask: Validated selection mask.
            params: Mapping containing exactly ``value``.
            rng: Unused item-local generator.

        Returns:
            A copied array containing the configured fill value.
        """

        return _apply_fill(array, mask, params, PerturbationOp.CONSTANT_FILL.value)


class MeanFillOperator(PerturbationOperator):
    """Fill selected pixels with the resolved dataset channel mean."""

    def supports(self, op: PerturbationOp) -> bool:
        """Return whether the requested operation is dataset-mean fill.

        Args:
            op: Perturbation operation requested by a work item.

        Returns:
            ``True`` only for ``mean_fill``.
        """

        return op is PerturbationOp.MEAN_FILL

    def validate_config(self, params: Mapping[str, Any]) -> None:
        """Require mean-fill user parameters to remain empty.

        Args:
            params: User-supplied mean-fill parameter mapping.

        Raises:
            PerturbationError: If any user parameter is present.
        """

        require_keys(params, set(), PerturbationOp.MEAN_FILL.value)

    def requires_dataset_stats(self) -> bool:
        """Declare the channel mean required for runtime resolution.

        Returns:
            Always ``True`` for dataset-mean fill.
        """

        return True

    def resolve_config_params(
        self,
        params: Mapping[str, Any],
        channel_mean: tuple[float, ...] | None = None,
    ) -> dict[str, Any]:
        """Replace empty user params with the resolved channel mean.

        Args:
            params: Validated empty user parameter mapping.
            channel_mean: Required dataset channel mean.

        Returns:
            Runtime params containing channel means under ``value``.

        Raises:
            PerturbationError: If dataset statistics are unavailable.
        """

        if channel_mean is None:
            raise PerturbationError(
                "mean_fill requires resolved dataset statistics"
            )
        return {"value": list(channel_mean)}

    def apply(
        self,
        array: NDArray[np.uint8],
        mask: NDArray[np.bool_],
        params: Mapping[str, Any],
        rng: Generator | None = None,
    ) -> NDArray[np.uint8]:
        """Fill selected pixels and ignore an optional generator.

        Args:
            array: Validated source pixels.
            mask: Validated selection mask.
            params: Mapping containing the resolved channel mean as ``value``.
            rng: Unused item-local generator.

        Returns:
            A copied array containing the resolved channel mean.
        """

        return _apply_fill(array, mask, params, PerturbationOp.MEAN_FILL.value)


class BlurOperator(PerturbationOperator):
    """Composite a full-frame Gaussian blur inside selected pixels."""

    def supports(self, op: PerturbationOp) -> bool:
        """Return whether the requested operation is Gaussian blur.

        Args:
            op: Perturbation operation requested by a work item.

        Returns:
            ``True`` only for ``blur``.
        """

        return op is PerturbationOp.BLUR

    def validate_config(self, params: Mapping[str, Any]) -> None:
        """Validate Gaussian-blur user parameters.

        Args:
            params: Mapping containing a positive ``sigma``.

        Raises:
            PerturbationError: If the parameter contract is invalid.
        """

        require_keys(params, {"sigma"}, PerturbationOp.BLUR.value)
        positive_real(params["sigma"], "blur.sigma")

    def apply(
        self,
        array: NDArray[np.uint8],
        mask: NDArray[np.bool_],
        params: Mapping[str, Any],
        rng: Generator | None = None,
    ) -> NDArray[np.uint8]:
        """Blur every frame and composite only selected pixels.

        Args:
            array: Validated source pixels.
            mask: Validated selection mask.
            params: Mapping containing a positive ``sigma``.
            rng: Unused item-local generator.

        Returns:
            A copied array with blurred values inside the mask.
        """

        require_keys(params, {"sigma"}, PerturbationOp.BLUR.value)
        sigma = positive_real(params["sigma"], "blur.sigma")
        blurred_frames: list[NDArray[np.uint8]] = []
        for frame in array:
            blurred = cv2.GaussianBlur(
                frame,
                ksize=(0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REFLECT_101,
            )
            if blurred.ndim == 2:
                blurred = blurred[..., np.newaxis]
            blurred_frames.append(np.asarray(blurred, dtype=np.uint8))
        candidate = np.stack(blurred_frames, axis=0)
        return composite(array, candidate, mask)


class GaussianNoiseOperator(PerturbationOperator):
    """Add item-local Gaussian noise inside selected pixels."""

    def supports(self, op: PerturbationOp) -> bool:
        """Return whether the requested operation is Gaussian noise.

        Args:
            op: Perturbation operation requested by a work item.

        Returns:
            ``True`` only for ``gaussian_noise``.
        """

        return op is PerturbationOp.GAUSSIAN_NOISE

    def validate_config(self, params: Mapping[str, Any]) -> None:
        """Validate Gaussian-noise user parameters.

        Args:
            params: Mapping containing a positive ``sigma``.

        Raises:
            PerturbationError: If the parameter contract is invalid.
        """

        require_keys(params, {"sigma"}, PerturbationOp.GAUSSIAN_NOISE.value)
        positive_real(params["sigma"], "gaussian_noise.sigma")

    def apply(
        self,
        array: NDArray[np.uint8],
        mask: NDArray[np.bool_],
        params: Mapping[str, Any],
        rng: Generator | None = None,
    ) -> NDArray[np.uint8]:
        """Generate clipped Gaussian noise with an item-local generator.

        Args:
            array: Validated source pixels.
            mask: Validated selection mask.
            params: Mapping containing a positive ``sigma``.
            rng: Required item-local NumPy generator.

        Returns:
            A copied array with clipped noisy values inside the mask.

        Raises:
            PerturbationError: If the operation has no valid generator.
        """

        require_keys(params, {"sigma"}, PerturbationOp.GAUSSIAN_NOISE.value)
        sigma = positive_real(params["sigma"], "gaussian_noise.sigma")
        generator = require_rng(rng, PerturbationOp.GAUSSIAN_NOISE.value)
        noise = generator.normal(0.0, sigma, size=array.shape)
        candidate = np.clip(
            np.rint(array.astype(np.float64) + noise), 0, 255
        ).astype(np.uint8)
        return composite(array, candidate, mask)


class PatchShuffleOperator(PerturbationOperator):
    """Shuffle complete spatial tiles and preserve partial edge tiles."""

    def supports(self, op: PerturbationOp) -> bool:
        """Return whether the requested operation is patch shuffle.

        Args:
            op: Perturbation operation requested by a work item.

        Returns:
            ``True`` only for ``patch_shuffle``.
        """

        return op is PerturbationOp.PATCH_SHUFFLE

    def validate_config(self, params: Mapping[str, Any]) -> None:
        """Validate patch-shuffle user parameters.

        Args:
            params: Mapping containing a positive integer ``patch_size``.

        Raises:
            PerturbationError: If the parameter contract is invalid.
        """

        require_keys(params, {"patch_size"}, PerturbationOp.PATCH_SHUFFLE.value)
        patch_size = params["patch_size"]
        if isinstance(patch_size, bool) or not isinstance(patch_size, int):
            raise PerturbationError(
                "patch_shuffle.patch_size must be an integer"
            )
        if patch_size <= 0:
            raise PerturbationError(
                "patch_shuffle.patch_size must be positive"
            )

    def apply(
        self,
        array: NDArray[np.uint8],
        mask: NDArray[np.bool_],
        params: Mapping[str, Any],
        rng: Generator | None = None,
    ) -> NDArray[np.uint8]:
        """Shuffle tile positions with one permutation shared by all frames.

        Args:
            array: Validated source pixels.
            mask: Validated selection mask.
            params: Mapping containing a positive integer ``patch_size``.
            rng: Required item-local NumPy generator.

        Returns:
            A copied array with shuffled tile values inside the mask.

        Raises:
            PerturbationError: If the patch size or generator is invalid.
        """

        require_keys(params, {"patch_size"}, PerturbationOp.PATCH_SHUFFLE.value)
        patch_size = params["patch_size"]
        if (
            isinstance(patch_size, bool)
            or not isinstance(patch_size, int)
            or patch_size <= 0
        ):
            raise PerturbationError(
                "patch_shuffle.patch_size must be a positive integer"
            )
        generator = require_rng(rng, PerturbationOp.PATCH_SHUFFLE.value)
        height, width = array.shape[1:3]
        tile_rows = height // patch_size
        tile_cols = width // patch_size
        tile_count = tile_rows * tile_cols
        candidate = array.copy()
        if tile_count > 1:
            permutation = generator.permutation(tile_count)
            for destination, source in enumerate(permutation):
                destination_row, destination_col = divmod(destination, tile_cols)
                source_row, source_col = divmod(int(source), tile_cols)
                destination_rows = slice(
                    destination_row * patch_size,
                    (destination_row + 1) * patch_size,
                )
                destination_cols = slice(
                    destination_col * patch_size,
                    (destination_col + 1) * patch_size,
                )
                source_rows = slice(
                    source_row * patch_size,
                    (source_row + 1) * patch_size,
                )
                source_cols = slice(
                    source_col * patch_size,
                    (source_col + 1) * patch_size,
                )
                candidate[:, destination_rows, destination_cols, :] = array[
                    :, source_rows, source_cols, :
                ]
        return composite(array, candidate, mask)
