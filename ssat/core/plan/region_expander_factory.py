"""Factory for deterministic region-family expander construction."""

from __future__ import annotations

from collections.abc import Sequence

from ssat.core._strategy_registry import StrategyRegistry
from ssat.core.plan.expansion_base import (
    RegionExpansionContext,
    RegionFamilyExpander,
)
from ssat.core.plan.region_expanders import (
    ExplicitRegionExpander,
    GridRegionExpander,
    RandomAreaMatchRegionExpander,
    SampleDependentRegionExpander,
)

FamilyExpanderType = type[RegionFamilyExpander]

_DEFAULT_EXPANDER_TYPES: tuple[FamilyExpanderType, ...] = (
    GridRegionExpander,
    ExplicitRegionExpander,
    SampleDependentRegionExpander,
    RandomAreaMatchRegionExpander,
)


class RegionFamilyExpanderFactory:
    """Register family expander classes in stable dispatch order.

    Args:
        expander_types: Optional expander classes registered at construction.
    """

    def __init__(self, expander_types: Sequence[FamilyExpanderType] = ()) -> None:
        self._registry: StrategyRegistry[RegionFamilyExpander] = StrategyRegistry(
            RegionFamilyExpander,
            type_label="expander_type",
            item_label="family expander type",
            strategy_types=expander_types,
        )

    def register(self, expander_type: FamilyExpanderType) -> None:
        """Append one family expander class to the factory.

        Args:
            expander_type: ``RegionFamilyExpander`` subclass to instantiate.

        Raises:
            TypeError: If the class does not implement the expander contract.
            ValueError: If the same class is already registered.
        """

        self._registry.register(expander_type)

    def build(
        self,
        context: RegionExpansionContext,
    ) -> list[RegionFamilyExpander]:
        """Construct registered expanders with shared planning context.

        Args:
            context: Optional provider services to inject.

        Returns:
            Fresh expanders in deterministic dispatch order.
        """

        return [
            expander_type(context) for expander_type in self._registry.registered_types
        ]


def build_family_expanders(
    context: RegionExpansionContext,
) -> list[RegionFamilyExpander]:
    """Build a fresh list containing all built-in family expanders.

    Args:
        context: Optional provider services to inject.

    Returns:
        Built-in expanders in deterministic dispatch order.
    """

    return RegionFamilyExpanderFactory(_DEFAULT_EXPANDER_TYPES).build(context)
