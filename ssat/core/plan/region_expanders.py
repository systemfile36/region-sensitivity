"""Built-in deterministic region-family expanders."""

from __future__ import annotations

from collections.abc import Sequence

from ssat.core.config.schema import RegionConfig, ResolvedRegionConfig
from ssat.core.plan.expansion_base import (
    RegionExpansionError,
    RegionFamilyConfig,
    RegionFamilyExpander,
)
from ssat.core.region.skeleton_provider import validate_skeleton_family_params
from ssat.core.region.types import RegionSpec
from ssat.core.source.types import SampleMeta
from ssat.core.types import RegionKind


class GridRegionExpander(RegionFamilyExpander):
    """Expand a grid family into row-major concrete cells."""

    def supports(self, family: RegionFamilyConfig) -> bool:
        """Return whether the family is a grid recipe.

        Args:
            family: Resolved region-family recipe.

        Returns:
            ``True`` only for the grid kind.
        """

        return family.kind is RegionKind.GRID

    def validate_config(self, family: RegionConfig) -> None:
        """Validate grid dimensions before reference resolution.

        Args:
            family: User-configured grid family.

        Raises:
            RegionExpansionError: If rows or columns are missing or invalid.
        """

        self._validate_params(family.params)

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
        self._validate_params(params)
        rows = params["rows"]
        cols = params["cols"]

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
    def _validate_params(params: dict[str, object]) -> None:
        """Validate the parameter mapping shared by config and planning.

        Args:
            params: Grid family parameters.

        Raises:
            RegionExpansionError: If the grid recipe is not valid.
        """

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


class ExplicitRegionExpander(RegionFamilyExpander):
    """Expand an explicit family into its single concrete mask reference."""

    def supports(self, family: RegionFamilyConfig) -> bool:
        """Return whether the family references an explicit mask.

        Args:
            family: Resolved region-family recipe.

        Returns:
            ``True`` only for the explicit kind.
        """

        return family.kind is RegionKind.EXPLICIT

    def validate_config(self, family: RegionConfig) -> None:
        """Accept an explicit family validated by the config schema.

        Args:
            family: User-configured explicit family.
        """

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

    def supports(self, family: RegionFamilyConfig) -> bool:
        """Return whether the family requires sample annotations.

        Args:
            family: Resolved region-family recipe.

        Returns:
            ``True`` for reserved sample-dependent kinds.
        """

        return family.kind in self._SUPPORTED_KINDS

    def validate_config(self, family: RegionConfig) -> None:
        """Validate a sample-dependent family, or reject it if unimplemented.

        ``skeleton_parts`` has a reference provider/generator (see
        ``ssat.core.region.skeleton_provider``/``skeleton_mask_generator``)
        and is validated structurally here; ``bbox_partition``/``gt_bbox``
        remain reserved and are always rejected.

        Args:
            family: User-configured sample-dependent family.

        Raises:
            RegionExpansionError: If ``skeleton_parts`` params are invalid,
                or the kind is still reserved and unimplemented.
        """

        if family.kind is RegionKind.SKELETON_PARTS:
            validate_skeleton_family_params(family.params)
            return
        raise RegionExpansionError(
            f"region kind {family.kind.value!r} is not implemented"
        )

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


class RandomAreaMatchRegionExpander(RegionFamilyExpander):
    """Reserve random area matching for internally generated controls."""

    def supports(self, family: RegionFamilyConfig) -> bool:
        """Return whether the family is an area-matched control recipe.

        Args:
            family: User or resolved region-family recipe.

        Returns:
            ``True`` only for random area matching.
        """

        return family.kind is RegionKind.RANDOM_AREA_MATCH

    def validate_config(self, family: RegionConfig) -> None:
        """Reject direct configuration of an internal control family.

        Args:
            family: User-configured random-area family.

        Raises:
            RegionExpansionError: Always, because controls create this recipe.
        """

        raise RegionExpansionError(
            "random_area_match is internal to area-matched controls and "
            "cannot be configured as a region family"
        )

    def expand(
        self,
        sample: SampleMeta,
        family: ResolvedRegionConfig,
    ) -> Sequence[RegionSpec]:
        """Reject planning expansion of an internal control family.

        Args:
            sample: Unused lightweight source metadata.
            family: Resolved random-area family.

        Raises:
            RegionExpansionError: Always, because PlanBuilder creates controls.
        """

        raise RegionExpansionError(
            "random_area_match is generated internally by area-matched controls"
        )
