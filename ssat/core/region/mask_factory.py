"""Factory for deterministic region mask generator construction."""

from __future__ import annotations

from collections.abc import Sequence

from ssat.core.region.mask_base import MaskResolutionContext, RegionMaskGenerator
from ssat.core.region.mask_generators import (
    ExplicitMaskGenerator,
    GridMaskGenerator,
    RandomAreaMatchMaskGenerator,
)

MaskGeneratorType = type[RegionMaskGenerator]

_DEFAULT_GENERATOR_TYPES: tuple[MaskGeneratorType, ...] = (
    GridMaskGenerator,
    ExplicitMaskGenerator,
    RandomAreaMatchMaskGenerator,
)


class RegionMaskGeneratorFactory:
    """Register mask generator classes in stable dispatch order.

    Args:
        generator_types: Optional generator classes registered at construction.
    """

    def __init__(self, generator_types: Sequence[MaskGeneratorType] = ()) -> None:
        self._generator_types: list[MaskGeneratorType] = []
        for generator_type in generator_types:
            self.register(generator_type)

    def register(self, generator_type: MaskGeneratorType) -> None:
        """Append one mask generator class to the factory.

        Args:
            generator_type: ``RegionMaskGenerator`` subclass to instantiate.

        Raises:
            TypeError: If the class does not implement the generator contract.
            ValueError: If the same class is already registered.
        """

        if not isinstance(generator_type, type) or not issubclass(
            generator_type, RegionMaskGenerator
        ):
            raise TypeError("generator_type must be a RegionMaskGenerator subclass")
        if generator_type in self._generator_types:
            raise ValueError(
                f"mask generator type already registered: {generator_type.__name__}"
            )
        self._generator_types.append(generator_type)

    def build(
        self,
        context: MaskResolutionContext,
    ) -> list[RegionMaskGenerator]:
        """Construct registered generators with shared resolver context.

        Args:
            context: Cache and recursive resolution services to inject.

        Returns:
            Fresh generators in deterministic dispatch order.
        """

        return [generator_type(context) for generator_type in self._generator_types]


def build_mask_generators(
    context: MaskResolutionContext,
) -> list[RegionMaskGenerator]:
    """Build a fresh list containing all built-in mask generators.

    Args:
        context: Cache and recursive resolution services to inject.

    Returns:
        Built-in generators in deterministic dispatch order.
    """

    return RegionMaskGeneratorFactory(_DEFAULT_GENERATOR_TYPES).build(context)
