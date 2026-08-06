"""Factory for deterministic region-family expander construction."""

from __future__ import annotations

from collections.abc import Sequence

from ssat.core.plan.expansion_base import (
    RegionExpansionContext,
    RegionFamilyExpander,
)
from ssat.core.plan.region_expanders import (
    ExplicitRegionExpander,
    GridRegionExpander,
    SampleDependentRegionExpander,
)

FamilyExpanderType = type[RegionFamilyExpander]

_DEFAULT_EXPANDER_TYPES: tuple[FamilyExpanderType, ...] = (
    GridRegionExpander,
    ExplicitRegionExpander,
    SampleDependentRegionExpander,
)


class RegionFamilyExpanderFactory:
    """Register family expander classes in stable dispatch order.

    Args:
        expander_types: Optional expander classes registered at construction.
    """

    def __init__(self, expander_types: Sequence[FamilyExpanderType] = ()) -> None:
        self._expander_types: list[FamilyExpanderType] = []
        for expander_type in expander_types:
            self.register(expander_type)

    def register(self, expander_type: FamilyExpanderType) -> None:
        """Append one family expander class to the factory.

        Args:
            expander_type: ``RegionFamilyExpander`` subclass to instantiate.

        Raises:
            TypeError: If the class does not implement the expander contract.
            ValueError: If the same class is already registered.
        """

        if not isinstance(expander_type, type) or not issubclass(
            expander_type, RegionFamilyExpander
        ):
            raise TypeError("expander_type must be a RegionFamilyExpander subclass")
        if expander_type in self._expander_types:
            raise ValueError(
                f"family expander type already registered: {expander_type.__name__}"
            )
        self._expander_types.append(expander_type)

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

        return [expander_type(context) for expander_type in self._expander_types]


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
