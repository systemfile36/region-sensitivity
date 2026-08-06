"""Deterministic expansion of configured region families for work planning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ssat.core.config.schema import ResolvedRegionConfig
from ssat.core.region.types import RegionSpec
from ssat.core.source.types import SampleMeta
from ssat.core.types import RegionKind


class RegionExpansionError(ValueError):
    """Indicate that a region family cannot produce concrete instances."""


class SampleRegionProvider(Protocol):
    """Provide sample-dependent concrete regions without loading pixels.

    Future skeleton and ground-truth bounding-box integrations implement this
    contract using lightweight annotation metadata available during planning.
    """

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


class RegionExpander:
    """Expand static region families before WorkItem enumeration.

    Args:
        sample_region_provider: Optional provider reserved for future
            sample-dependent region kinds.
    """

    _SAMPLE_DEPENDENT_KINDS = {
        RegionKind.BBOX_PARTITION,
        RegionKind.SKELETON_PARTS,
        RegionKind.GT_BBOX,
    }

    def __init__(
        self,
        sample_region_provider: SampleRegionProvider | None = None,
    ) -> None:
        self._sample_region_provider = sample_region_provider

    def expand(
        self,
        sample: SampleMeta,
        family: ResolvedRegionConfig,
    ) -> tuple[RegionSpec, ...]:
        """Expand one family into deterministic concrete RegionSpecs.

        Args:
            sample: Lightweight metadata for the source sample.
            family: Resolved region-family recipe to expand.

        Returns:
            Concrete region specifications in planning order.

        Raises:
            RegionExpansionError: If the recipe is invalid, unsupported, or
                produces invalid concrete instances.
        """

        if family.kind is RegionKind.GRID:
            instances = self._expand_grid(family)
        elif family.kind is RegionKind.EXPLICIT:
            instances = (self._expand_explicit(family),)
        elif family.kind in self._SAMPLE_DEPENDENT_KINDS:
            instances = self._expand_sample_dependent(sample, family)
        else:
            raise RegionExpansionError(
                f"region kind {family.kind.value!r} cannot be expanded from config"
            )

        self._validate_instances(family, instances)
        return instances

    @staticmethod
    def _expand_grid(family: ResolvedRegionConfig) -> tuple[RegionSpec, ...]:
        """Create row-major concrete cells for one grid family."""

        params = family.params
        expected = {"rows", "cols"}
        if set(params) != expected:
            raise RegionExpansionError(
                "grid params must contain exactly ['cols', 'rows']"
            )
        rows = params["rows"]
        cols = params["cols"]
        for field_name, value in (("rows", rows), ("cols", cols)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RegionExpansionError(
                    f"grid.{field_name} must be a positive integer"
                )

        return tuple(
            RegionSpec(
                region_id=family.region_id,
                region_instance_id=(
                    f"{family.region_id}/r{row_index}/c{col_index}"
                ),
                kind=family.kind,
                params={
                    "rows": rows,
                    "cols": cols,
                    "row_index": row_index,
                    "col_index": col_index,
                },
            )
            for row_index in range(rows)
            for col_index in range(cols)
        )

    @staticmethod
    def _expand_explicit(family: ResolvedRegionConfig) -> RegionSpec:
        """Create the single concrete instance of an explicit family."""

        return RegionSpec(
            region_id=family.region_id,
            region_instance_id=family.region_id,
            kind=family.kind,
            params=family.params,
            ref=family.ref.as_posix() if family.ref is not None else None,
            ref_hash=family.ref_hash,
        )

    def _expand_sample_dependent(
        self,
        sample: SampleMeta,
        family: ResolvedRegionConfig,
    ) -> tuple[RegionSpec, ...]:
        """Delegate a reserved sample-dependent family to its provider."""

        if self._sample_region_provider is None:
            raise RegionExpansionError(
                f"region kind {family.kind.value!r} is not implemented; "
                "provide a SampleRegionProvider"
            )
        try:
            return tuple(self._sample_region_provider.expand(sample, family))
        except RegionExpansionError:
            raise
        except Exception as error:
            raise RegionExpansionError(
                f"sample region provider failed for region_id={family.region_id!r}"
            ) from error

    @staticmethod
    def _validate_instances(
        family: ResolvedRegionConfig,
        instances: tuple[RegionSpec, ...],
    ) -> None:
        """Validate provider and built-in expansion output."""

        if not instances:
            raise RegionExpansionError(
                f"region family {family.region_id!r} produced no instances"
            )
        if any(not isinstance(instance, RegionSpec) for instance in instances):
            raise RegionExpansionError("region expansion must return RegionSpec values")
        if any(instance.region_id != family.region_id for instance in instances):
            raise RegionExpansionError(
                "expanded RegionSpec.region_id must match its family region_id"
            )
        instance_ids = tuple(instance.region_instance_id for instance in instances)
        if len(set(instance_ids)) != len(instance_ids):
            raise RegionExpansionError(
                f"region family {family.region_id!r} produced duplicate instance IDs"
            )
