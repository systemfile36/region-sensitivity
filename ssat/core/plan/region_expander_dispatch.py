"""Ordered dispatch for deterministic region-family expanders."""

from __future__ import annotations

from collections.abc import Sequence

from ssat.core.config.schema import ResolvedRegionConfig
from ssat.core.plan.expansion_base import (
    RegionExpansionError,
    RegionFamilyConfig,
    RegionFamilyExpander,
)
from ssat.core.region.types import RegionSpec
from ssat.core.source.types import SampleMeta


def find_family_expander(
    expanders: Sequence[RegionFamilyExpander],
    family: RegionFamilyConfig,
) -> RegionFamilyExpander:
    """Find the first registered expander supporting a region family.

    Args:
        expanders: Expanders in explicit dispatch-priority order.
        family: User or resolved region-family recipe.

    Returns:
        The first supporting family expander.

    Raises:
        RegionExpansionError: If support discovery fails or no expander matches.
    """

    for expander in expanders:
        try:
            supported = expander.supports(family)
        except RegionExpansionError:
            raise
        except Exception as error:
            raise RegionExpansionError(
                "family expander support check failed "
                f"kind={family.kind.value} expander={expander.__class__.__name__}"
            ) from error
        if not isinstance(supported, bool):
            raise RegionExpansionError(
                "family expander supports() must return bool: "
                f"{expander.__class__.__name__}"
            )
        if not supported:
            continue
        return expander
    raise RegionExpansionError(
        f"region kind {family.kind.value!r} cannot be expanded from config"
    )


def dispatch_family_expander(
    expanders: Sequence[RegionFamilyExpander],
    sample: SampleMeta,
    family: ResolvedRegionConfig,
) -> Sequence[RegionSpec]:
    """Execute the first expander supporting a resolved family.

    Args:
        expanders: Expanders in explicit dispatch-priority order.
        sample: Lightweight source metadata.
        family: Resolved region-family recipe.

    Returns:
        The first supporting expander's concrete regions.

    Raises:
        RegionExpansionError: If dispatch or expansion fails.
    """

    expander = find_family_expander(expanders, family)
    try:
        return expander.expand(sample, family)
    except RegionExpansionError:
        raise
    except Exception as error:
        raise RegionExpansionError(
            "family expander execution failed "
            f"kind={family.kind.value} expander={expander.__class__.__name__}"
        ) from error
