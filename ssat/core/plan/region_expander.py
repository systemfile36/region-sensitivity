"""Validated facade for deterministic region-family expanders."""

from __future__ import annotations

from collections.abc import Sequence

from ssat.core.config.schema import ResolvedRegionConfig
from ssat.core.plan.expansion_base import (
    RegionExpansionContext,
    RegionExpansionError,
    RegionFamilyExpander,
    SampleRegionProvider,
)
from ssat.core.plan.region_expander_dispatch import dispatch_family_expander
from ssat.core.plan.region_expander_factory import build_family_expanders
from ssat.core.region.types import RegionSpec
from ssat.core.source.types import SampleMeta


class RegionExpander:
    """Validate and dispatch region-family expansion before planning.

    Args:
        sample_region_provider: Optional annotation-backed provider.
        family_expanders: Optional expanders in dispatch-priority order.

    Raises:
        TypeError: If a supplied value is not a family expander.
        ValueError: If the expander collection is empty.
    """

    def __init__(
        self,
        sample_region_provider: SampleRegionProvider | None = None,
        *,
        family_expanders: Sequence[RegionFamilyExpander] | None = None,
    ) -> None:
        if family_expanders is None:
            context = RegionExpansionContext(sample_region_provider)
            resolved = tuple(build_family_expanders(context))
        else:
            resolved = tuple(family_expanders)
        if not resolved:
            raise ValueError("family_expanders must not be empty")
        if any(
            not isinstance(expander, RegionFamilyExpander)
            for expander in resolved
        ):
            raise TypeError(
                "family_expanders must contain RegionFamilyExpander values"
            )
        self._family_expanders = resolved

    def expand(
        self,
        sample: SampleMeta,
        family: ResolvedRegionConfig,
    ) -> tuple[RegionSpec, ...]:
        """Expand one family into validated concrete region specifications.

        Args:
            sample: Lightweight metadata for the source sample.
            family: Resolved region-family recipe.

        Returns:
            Concrete region specifications in planning order.

        Raises:
            RegionExpansionError: If dispatch, expansion, or output validation
                fails.
        """

        instances = tuple(
            dispatch_family_expander(
                self._family_expanders,
                sample,
                family,
            )
        )
        self._validate_instances(family, instances)
        return instances

    @staticmethod
    def _validate_instances(
        family: ResolvedRegionConfig,
        instances: tuple[RegionSpec, ...],
    ) -> None:
        """Validate concrete instances returned by one family expander.

        Args:
            family: Resolved source family.
            instances: Concrete instances returned by dispatch.

        Raises:
            RegionExpansionError: If output violates the planning contract.
        """

        if not instances:
            raise RegionExpansionError(
                f"region family {family.region_id!r} produced no instances"
            )
        if any(not isinstance(instance, RegionSpec) for instance in instances):
            raise RegionExpansionError(
                "region expansion must return RegionSpec values"
            )
        if any(instance.region_id != family.region_id for instance in instances):
            raise RegionExpansionError(
                "expanded RegionSpec.region_id must match its family region_id"
            )
        instance_ids = tuple(instance.region_instance_id for instance in instances)
        if len(set(instance_ids)) != len(instance_ids):
            raise RegionExpansionError(
                f"region family {family.region_id!r} produced duplicate instance IDs"
            )
