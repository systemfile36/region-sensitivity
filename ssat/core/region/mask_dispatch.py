"""Ordered dispatch for concrete region mask generators."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ssat.core.region.mask_base import RegionMaskGenerator, RegionResolutionError
from ssat.core.region.types import RegionSpec


def dispatch_mask_generator(
    generators: Sequence[RegionMaskGenerator],
    height: int,
    width: int,
    spec: RegionSpec,
    rng: Generator | None = None,
) -> NDArray[np.bool_]:
    """Execute the first generator supporting a concrete region recipe.

    Args:
        generators: Generators in explicit dispatch-priority order.
        height: Source image height.
        width: Source image width.
        spec: Concrete region recipe.
        rng: Optional item-local generator.

    Returns:
        The first supporting generator's mask.

    Raises:
        RegionResolutionError: If dispatch or generation fails, or no
            generator supports ``spec``.
    """

    for generator in generators:
        try:
            supported = generator.supports(spec)
        except RegionResolutionError:
            raise
        except Exception as error:
            raise RegionResolutionError(
                "mask generator support check failed "
                f"kind={spec.kind.value} generator={generator.__class__.__name__}"
            ) from error
        if not isinstance(supported, bool):
            raise RegionResolutionError(
                "mask generator supports() must return bool: "
                f"{generator.__class__.__name__}"
            )
        if not supported:
            continue
        try:
            return generator.get_mask(height, width, spec, rng)
        except RegionResolutionError:
            raise
        except Exception as error:
            raise RegionResolutionError(
                "mask generator execution failed "
                f"kind={spec.kind.value} generator={generator.__class__.__name__}"
            ) from error
    raise RegionResolutionError(
        f"region kind {spec.kind.value!r} is not implemented"
    )
