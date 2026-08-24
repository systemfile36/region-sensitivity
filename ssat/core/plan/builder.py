"""Deterministic enumeration and rematerialization of audit work chunks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ssat.core.config.schema import ResolvedConfig
from ssat.core.plan.hashing import compute_chunk_id, compute_item_id
from ssat.core.plan.region_expander import RegionExpander, RegionExpansionError
from ssat.core.plan.types import (
    WorkChunk,
    WorkChunkMeta,
    WorkItem,
    _item_identity_payload,
)
from ssat.core.region.types import RegionSpec
from ssat.core.source.base import SampleSource
from ssat.core.source.types import SampleMeta
from ssat.core.types import PerturbationOp, RegionKind, thaw_json_value
from ssat.utils.logger_factory import get_logger


class PlanBuildError(ValueError):
    """Indicate that deterministic work enumeration could not complete."""


@dataclass(frozen=True, slots=True)
class _ChunkLocator:
    """Locate a chunk without retaining its fully materialized WorkItems."""

    sample_id: str
    chunk_ordinal: int
    meta: WorkChunkMeta


class PlanBuilder:
    """Enumerate lightweight chunk metadata and rematerialize worker inputs.

    Args:
        resolved_config: Fully resolved configuration shared by main and workers.
        sample_source: Source used only to list deterministic sample metadata.
        logger: Optional package logger for planning lifecycle events.
        region_expander: Optional deterministic region-family expander.
    """

    def __init__(
        self,
        resolved_config: ResolvedConfig,
        sample_source: SampleSource,
        logger: logging.Logger | None = None,
        *,
        region_expander: RegionExpander | None = None,
    ) -> None:
        self._config = resolved_config
        self._sample_source = sample_source
        self._logger = logger or get_logger(__name__)
        self._region_expander = region_expander or RegionExpander()
        self._samples: tuple[SampleMeta, ...] | None = None
        self._sample_by_id: dict[str, SampleMeta] | None = None
        self._chunk_metas: tuple[WorkChunkMeta, ...] | None = None
        self._chunk_locators: dict[str, _ChunkLocator] | None = None

    def enumerate(self) -> tuple[WorkChunkMeta, ...]:
        """Return ordered, lightweight metadata for all perturbed chunks."""

        self._ensure_plan()
        if self._chunk_metas is None:  # pragma: no cover - defensive invariant
            raise PlanBuildError("chunk metadata cache was not initialized")
        return self._chunk_metas

    def enumerate_clean(self) -> tuple[SampleMeta, ...]:
        """Return clean samples independently of perturbation enumeration."""

        return self._ensure_samples()

    def materialize(self, chunk_id: str) -> WorkChunk:
        """Recompute and return the complete WorkChunk for one chunk ID.

        Args:
            chunk_id: Identifier previously returned by :meth:`enumerate`.

        Returns:
            A complete chunk whose items match the cached metadata exactly.

        Raises:
            PlanBuildError: If the ID is unknown or rematerialization diverges.
        """

        self._ensure_plan()
        if self._chunk_locators is None:  # pragma: no cover - defensive invariant
            raise PlanBuildError("chunk locator cache was not initialized")
        locator = self._chunk_locators.get(chunk_id)
        if locator is None:
            raise PlanBuildError(f"unknown chunk_id: {chunk_id}")

        all_items = self._enumerate_items_for_sample(locator.sample_id)
        chunk_size = self._config.runtime.variants_per_chunk
        start = locator.chunk_ordinal * chunk_size
        items = all_items[start : start + chunk_size]
        item_ids = tuple(item.item_id for item in items)
        if item_ids != locator.meta.item_ids:
            raise PlanBuildError(
                f"materialized item_ids do not match metadata for chunk_id={chunk_id}"
            )
        recomputed_chunk_id = compute_chunk_id(
            locator.sample_id,
            locator.chunk_ordinal,
            item_ids,
            schema_version=self._config.schema_version,
        )
        if recomputed_chunk_id != chunk_id:
            raise PlanBuildError(
                f"materialized chunk_id does not match requested chunk_id={chunk_id}"
            )

        chunk = WorkChunk(
            chunk_id=chunk_id,
            sample_id=locator.sample_id,
            items=items,
        )
        self._logger.debug(
            "plan.chunk_materialized chunk_id=%s sample_id=%s items=%d",
            chunk.chunk_id,
            chunk.sample_id,
            len(chunk.items),
        )
        return chunk

    def _ensure_samples(self) -> tuple[SampleMeta, ...]:
        """Load, validate, sort, and cache lightweight sample metadata."""

        if self._samples is not None:
            return self._samples
        try:
            samples = tuple(self._sample_source.list_samples())
        except Exception as error:
            raise PlanBuildError("sample_source.list_samples() failed") from error
        if not samples:
            raise PlanBuildError("work planning requires at least one sample")
        if any(not isinstance(sample, SampleMeta) for sample in samples):
            raise PlanBuildError("sample_source.list_samples() must return SampleMeta values")

        sorted_samples = tuple(sorted(samples, key=lambda sample: sample.sample_id))
        sample_ids = tuple(sample.sample_id for sample in sorted_samples)
        if len(set(sample_ids)) != len(sample_ids):
            raise PlanBuildError("sample_source returned duplicate sample_id values")

        self._samples = sorted_samples
        self._sample_by_id = {sample.sample_id: sample for sample in sorted_samples}
        return sorted_samples

    def _ensure_plan(self) -> None:
        """Build and cache immutable chunk metadata and lookup locators."""

        if self._chunk_metas is not None and self._chunk_locators is not None:
            return

        chunk_metas: list[WorkChunkMeta] = []
        chunk_locators: dict[str, _ChunkLocator] = {}
        total_items = 0
        control_items = 0
        chunk_size = self._config.runtime.variants_per_chunk
        samples = self._ensure_samples()
        self._logger.info(
            "plan.enumeration_started samples=%d variants_per_chunk=%d",
            len(samples),
            chunk_size,
        )

        for sample in samples:
            items = self._enumerate_items_for_sample(sample.sample_id)
            total_items += len(items)
            control_items += sum(item.is_control for item in items)
            for chunk_ordinal, start in enumerate(range(0, len(items), chunk_size)):
                chunk_items = items[start : start + chunk_size]
                item_ids = tuple(item.item_id for item in chunk_items)
                chunk_id = compute_chunk_id(
                    sample.sample_id,
                    chunk_ordinal,
                    item_ids,
                    schema_version=self._config.schema_version,
                )
                meta = WorkChunkMeta(
                    chunk_id=chunk_id,
                    sample_id=sample.sample_id,
                    item_ids=item_ids,
                )
                if chunk_id in chunk_locators:
                    raise PlanBuildError(f"duplicate chunk_id generated: {chunk_id}")
                locator = _ChunkLocator(
                    sample_id=sample.sample_id,
                    chunk_ordinal=chunk_ordinal,
                    meta=meta,
                )
                chunk_metas.append(meta)
                chunk_locators[chunk_id] = locator

        self._chunk_metas = tuple(chunk_metas)
        self._chunk_locators = chunk_locators
        self._logger.info(
            "plan.enumerated samples=%d items=%d controls=%d chunks=%d",
            len(samples),
            total_items,
            control_items,
            len(chunk_metas),
        )

    def _enumerate_items_for_sample(self, sample_id: str) -> tuple[WorkItem, ...]:
        """Create the canonical ordered WorkItems for one sample."""

        if self._sample_by_id is None:
            self._ensure_samples()
        if self._sample_by_id is None or sample_id not in self._sample_by_id:
            raise PlanBuildError(f"unknown sample_id: {sample_id}")

        sample = self._sample_by_id[sample_id]
        resolved_regions = tuple(self._config.regions)
        expanded_by_family: dict[str, tuple[RegionSpec, ...]] = {}
        runtime_regions: list[RegionSpec] = []
        for region in resolved_regions:
            try:
                instances = self._region_expander.expand(sample, region)
            except RegionExpansionError as error:
                raise PlanBuildError(
                    f"region expansion failed for region_id={region.region_id!r}: {error}"
                ) from error
            expanded_by_family[region.region_id] = instances
            runtime_regions.extend(instances)
        items: list[WorkItem] = []

        for region_spec in runtime_regions:
            for perturbation in self._config.perturbations:
                for seed_salt in perturbation.seed_salts:
                    items.append(
                        self._make_item(
                            sample_id=sample_id,
                            region_spec=region_spec,
                            perturb_op=perturbation.op,
                            perturb_params=perturbation.params,
                            invert_mask=perturbation.invert_mask,
                            seed_salt=seed_salt,
                            is_control=False,
                        )
                    )

        if len(expanded_by_family) != len(resolved_regions):
            raise PlanBuildError("resolved config contains duplicate region_id values")
        configured_region_ids = set(expanded_by_family)
        for control_request_index, control in enumerate(self._config.controls):
            target_regions = expanded_by_family.get(control.match_area_of)
            if target_regions is None:
                raise PlanBuildError(
                    f"control references unknown region_id={control.match_area_of!r}"
                )
            control_region_id = (
                f"control:{control.match_area_of}:{control_request_index}"
            )
            if control_region_id in configured_region_ids:
                raise PlanBuildError(
                    f"generated control region_id collides with config: {control_region_id}"
                )
            for target_region in target_regions:
                target_recipe = self._region_recipe(target_region)
                for control_index in range(control.n_samples):
                    control_instance_id = (
                        f"control:{target_region.region_instance_id}:"
                        f"{control_request_index}:{control_index}"
                    )
                    control_region = RegionSpec(
                        region_id=control_region_id,
                        region_instance_id=control_instance_id,
                        kind=RegionKind.RANDOM_AREA_MATCH,
                        params={
                            "target_region": target_recipe,
                            "control_request_index": control_request_index,
                            "control_index": control_index,
                        },
                    )
                    for perturbation in self._config.perturbations:
                        for seed_salt in perturbation.seed_salts:
                            items.append(
                                self._make_item(
                                    sample_id=sample_id,
                                    region_spec=control_region,
                                    perturb_op=perturbation.op,
                                    perturb_params=perturbation.params,
                                    invert_mask=perturbation.invert_mask,
                                    seed_salt=seed_salt,
                                    is_control=True,
                                )
                            )

        item_ids = tuple(item.item_id for item in items)
        if len(set(item_ids)) != len(item_ids):
            raise PlanBuildError(
                f"duplicate item_id generated for sample_id={sample_id!r}"
            )
        return tuple(items)

    def _make_item(
        self,
        *,
        sample_id: str,
        region_spec: RegionSpec,
        perturb_op: PerturbationOp,
        perturb_params: dict[str, Any],
        invert_mask: bool,
        seed_salt: int,
        is_control: bool,
    ) -> WorkItem:
        """Create a WorkItem and derive its identity from the complete payload."""

        payload = _item_identity_payload(
            sample_id=sample_id,
            region_spec=region_spec,
            perturb_op=perturb_op,
            perturb_params=perturb_params,
            invert_mask=invert_mask,
            seed_salt=seed_salt,
            is_control=is_control,
        )
        return WorkItem(
            item_id=compute_item_id(
                payload,
                schema_version=self._config.schema_version,
            ),
            **payload,
        )

    @staticmethod
    def _region_recipe(region: RegionSpec) -> dict[str, Any]:
        """Create the self-contained JSON recipe embedded in control regions."""

        return {
            "region_id": region.region_id,
            "region_instance_id": region.region_instance_id,
            "kind": region.kind.value,
            "params": thaw_json_value(region.params),
            "ref": region.ref,
            "ref_hash": region.ref_hash,
        }
