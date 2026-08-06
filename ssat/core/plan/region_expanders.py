"""Built-in deterministic region-family expanders."""

from __future__ import annotations

from collections.abc import Sequence

from ssat.core.config.schema import ResolvedRegionConfig
from ssat.core.plan.expansion_base import (
    RegionExpansionError,
    RegionFamilyExpander,
)
from ssat.core.region.types import RegionSpec
from ssat.core.source.types import SampleMeta
from ssat.core.types import RegionKind


class GridRegionExpander(RegionFamilyExpander):
    """Expand a grid family into row-major concrete cells."""

    def supports(self, family: ResolvedRegionConfig) -> bool:
        """Return whether the family is a grid recipe.

        Args:
            family: Resolved region-family recipe.

        Returns:
            ``True`` only for the grid kind.
        """

        return family.kind is RegionKind.GRID

    def expand(
        self,
        sample: SampleMeta,
        family: ResolvedRegionConfig,
    ) -> Sequence[RegionSpec]:
        """Create all grid cells in deterministic row-major order.

        Args:
            sample: Unused lightweight source metadata.
            family: Resolved grid-family recipe.

        Returns:
            Concrete grid cell specifications.

        Raises:
            RegionExpansionError: If rows or columns are invalid.
        """

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


class ExplicitRegionExpander(RegionFamilyExpander):
    """Expand an explicit family into its single concrete mask reference."""

    def supports(self, family: ResolvedRegionConfig) -> bool:
        """Return whether the family references an explicit mask.

        Args:
            family: Resolved region-family recipe.

        Returns:
            ``True`` only for the explicit kind.
        """

        return family.kind is RegionKind.EXPLICIT

    def expand(
        self,
        sample: SampleMeta,
        family: ResolvedRegionConfig,
    ) -> Sequence[RegionSpec]:
        """Create the single concrete explicit region.

        Args:
            sample: Unused lightweight source metadata.
            family: Resolved explicit-family recipe.

        Returns:
            A one-item sequence containing the explicit region.
        """

        return (
            RegionSpec(
                region_id=family.region_id,
                region_instance_id=family.region_id,
                kind=family.kind,
                params=family.params,
                ref=family.ref.as_posix() if family.ref is not None else None,
                ref_hash=family.ref_hash,
            ),
        )


class SampleDependentRegionExpander(RegionFamilyExpander):
    """Delegate reserved annotation-backed families to a provider."""

    _SUPPORTED_KINDS = {
        RegionKind.BBOX_PARTITION,
        RegionKind.SKELETON_PARTS,
        RegionKind.GT_BBOX,
    }

    def supports(self, family: ResolvedRegionConfig) -> bool:
        """Return whether the family requires sample annotations.

        Args:
            family: Resolved region-family recipe.

        Returns:
            ``True`` for reserved sample-dependent kinds.
        """

        return family.kind in self._SUPPORTED_KINDS

    def expand(
        self,
        sample: SampleMeta,
        family: ResolvedRegionConfig,
    ) -> Sequence[RegionSpec]:
        """Delegate concrete instance creation to the configured provider.

        Args:
            sample: Lightweight source metadata.
            family: Resolved sample-dependent family.

        Returns:
            Provider-supplied concrete regions.

        Raises:
            RegionExpansionError: If no provider exists or it fails.
        """

        provider = self._context.sample_region_provider
        if provider is None:
            raise RegionExpansionError(
                f"region kind {family.kind.value!r} is not implemented; "
                "provide a SampleRegionProvider"
            )
        try:
            return tuple(provider.expand(sample, family))
        except RegionExpansionError:
            raise
        except Exception as error:
            raise RegionExpansionError(
                f"sample region provider failed for region_id={family.region_id!r}"
            ) from error
