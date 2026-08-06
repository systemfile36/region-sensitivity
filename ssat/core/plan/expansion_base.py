"""Abstract contracts for deterministic region-family expansion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from ssat.core.config.schema import RegionConfig, ResolvedRegionConfig
from ssat.core.region.types import RegionSpec
from ssat.core.source.types import SampleMeta


class RegionExpansionError(ValueError):
    """Indicate that a region family cannot produce concrete instances."""


RegionFamilyConfig: TypeAlias = RegionConfig | ResolvedRegionConfig


class SampleRegionProvider(Protocol):
    """Provide sample-dependent concrete regions without loading pixels."""

    def expand(
        self,
        sample: SampleMeta,
        family: ResolvedRegionConfig,
    ) -> Sequence[RegionSpec]:
        """Return deterministic concrete regions for one sample and family.

        Args:
            sample: Lightweight metadata for the source sample.
            family: Resolved sample-dependent region-family recipe.

        Returns:
            Concrete region specifications in stable semantic order.
        """


@dataclass(frozen=True, slots=True)
class RegionExpansionContext:
    """Provide optional annotation-backed expansion services.

    Attributes:
        sample_region_provider: Provider for future sample-dependent families.
    """

    sample_region_provider: SampleRegionProvider | None = None


class RegionFamilyExpander(ABC):
    """Define support detection and expansion for one region-family kind.

    Args:
        context: Shared planning services supplied by the facade.
    """

    def __init__(self, context: RegionExpansionContext) -> None:
        self._context = context

    @abstractmethod
    def supports(self, family: RegionFamilyConfig) -> bool:
        """Return whether this expander supports a region family.

        Args:
            family: User or resolved region-family recipe.

        Returns:
            ``True`` when this expander owns ``family`` validation and expansion.
        """

    @abstractmethod
    def validate_config(self, family: RegionConfig) -> None:
        """Validate one user-configured region family.

        Args:
            family: Unresolved region-family recipe from user configuration.

        Raises:
            RegionExpansionError: If the family configuration is invalid or
                not implemented.
        """

    @abstractmethod
    def expand(
        self,
        sample: SampleMeta,
        family: ResolvedRegionConfig,
    ) -> Sequence[RegionSpec]:
        """Expand one family into deterministic concrete regions.

        Args:
            sample: Lightweight source metadata without loaded pixels.
            family: Resolved region-family recipe.

        Returns:
            Concrete region specifications in stable semantic order.

        Raises:
            RegionExpansionError: If expansion cannot complete safely.
        """
