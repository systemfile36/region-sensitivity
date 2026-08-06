"""Region contracts."""

from ssat.core.region.mask_base import (
    MaskResolutionContext,
    RegionMaskGenerator,
    RegionResolutionError,
)
from ssat.core.region.mask_factory import (
    RegionMaskGeneratorFactory,
    build_mask_generators,
)
from ssat.core.region.resolver import RegionResolver
from ssat.core.region.types import RegionMeta, RegionSpec

__all__ = [
    "MaskResolutionContext",
    "RegionMaskGenerator",
    "RegionMaskGeneratorFactory",
    "RegionMeta",
    "RegionResolutionError",
    "RegionResolver",
    "RegionSpec",
    "build_mask_generators",
]
