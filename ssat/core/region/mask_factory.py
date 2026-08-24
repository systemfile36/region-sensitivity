"""Factory for deterministic region mask generator construction."""

from __future__ import annotations

from collections.abc import Sequence

from ssat.core._strategy_registry import StrategyRegistry
from ssat.core.region.mask_base import MaskResolutionContext, RegionMaskGenerator
from ssat.core.region.mask_generators import (
    ExplicitMaskGenerator,
    GridMaskGenerator,
    RandomAreaMatchMaskGenerator,
)
from ssat.core.region.skeleton_mask_generator import SkeletonPartsMaskGenerator

MaskGeneratorType = type[RegionMaskGenerator]

_DEFAULT_GENERATOR_TYPES: tuple[MaskGeneratorType, ...] = (
    GridMaskGenerator,
    ExplicitMaskGenerator,
    SkeletonPartsMaskGenerator,
    RandomAreaMatchMaskGenerator,
)


class RegionMaskGeneratorFactory:
    """Register mask generator classes in stable dispatch order.

    Args:
        generator_types: Optional generator classes registered at construction.
    """

    def __init__(self, generator_types: Sequence[MaskGeneratorType] = ()) -> None:
        self._registry: StrategyRegistry[RegionMaskGenerator] = StrategyRegistry(
            RegionMaskGenerator,
            type_label="generator_type",
            item_label="mask generator type",
            strategy_types=generator_types,
        )

    def register(self, generator_type: MaskGeneratorType) -> None:
        """Append one mask generator class to the factory.

        Args:
            generator_type: ``RegionMaskGenerator`` subclass to instantiate.

        Raises:
            TypeError: If the class does not implement the generator contract.
            ValueError: If the same class is already registered.
        """

        self._registry.register(generator_type)

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

        return [
            generator_type(context)
            for generator_type in self._registry.registered_types
        ]


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
