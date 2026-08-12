"""Worker-side clean loading and perturbation preparation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ssat.core.perturb.perturbator import Perturbator
from ssat.core.perturb.rng import derive
from ssat.core.plan.builder import PlanBuilder
from ssat.core.plan.types import WorkChunkMeta
from ssat.core.region.mask_base import mean_frame_area
from ssat.core.region.resolver import RegionResolver
from ssat.core.region.types import RegionMeta
from ssat.core.runtime.errors import RuntimeContractError
from ssat.core.runtime.types import FailedChunk, ItemMeta, PreparedChunk
from ssat.core.source.base import SampleSource
from ssat.core.source.types import LoadError, LoadedSample, SampleMeta
from ssat.core.types import ItemStatus


@dataclass(slots=True)
class CleanProcessor:
    """DataLoader dataset that loads one clean sample per index."""

    samples: tuple[SampleMeta, ...]
    sample_source: SampleSource

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> LoadedSample | LoadError:
        sample = self.samples[index]
        try:
            result = self.sample_source.load(sample.sample_id)
        except Exception as error:
            return LoadError(
                sample_id=sample.sample_id,
                error_type="load_error",
                message=f"{error.__class__.__name__}: {error}",
            )
        if not isinstance(result, (LoadedSample, LoadError)):
            raise RuntimeContractError(
                "sample_source.load() must return LoadedSample or LoadError"
            )
        if result.sample_id != sample.sample_id:
            raise RuntimeContractError("sample_source returned a mismatched sample_id")
        return result


@dataclass(slots=True)
class ChunkProcessor:
    """DataLoader dataset that prepares every item in one work chunk."""

    chunks: tuple[WorkChunkMeta, ...]
    plan_builder: PlanBuilder
    sample_source: SampleSource
    global_seed: int
    fail_fast: bool = False
    region_resolver: RegionResolver | None = None
    perturbator: Perturbator | None = None

    def __post_init__(self) -> None:
        if self.region_resolver is None:
            self.region_resolver = RegionResolver()
        if self.perturbator is None:
            self.perturbator = Perturbator()

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> PreparedChunk | FailedChunk:
        chunk_meta = self.chunks[index]
        chunk = self.plan_builder.materialize(chunk_meta.chunk_id)
        if (
            chunk.sample_id != chunk_meta.sample_id
            or tuple(item.item_id for item in chunk.items) != chunk_meta.item_ids
        ):
            raise RuntimeContractError("materialized chunk does not match its metadata")

        try:
            loaded = self.sample_source.load(chunk.sample_id)
        except Exception:
            loaded = LoadError(
                sample_id=chunk.sample_id,
                error_type="load_error",
                message="sample_source.load() raised",
            )
        if not isinstance(loaded, (LoadedSample, LoadError)):
            raise RuntimeContractError(
                "sample_source.load() must return LoadedSample or LoadError"
            )
        if loaded.sample_id != chunk.sample_id:
            raise RuntimeContractError("sample_source returned a mismatched sample_id")
        if isinstance(loaded, LoadError):
            return FailedChunk(
                chunk_id=chunk.chunk_id,
                reason=ItemStatus.LOAD_FAILED,
                item_ids=tuple(item.item_id for item in chunk.items),
            )

        arrays: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        successful: list[ItemMeta] = []
        failed: list[ItemMeta] = []
        resolver = self.region_resolver
        perturbator = self.perturbator
        if resolver is None or perturbator is None:  # pragma: no cover - post-init
            raise RuntimeContractError("chunk processor was not initialized")

        for item in chunk.items:
            region_meta: RegionMeta | None = None
            try:
                seed = derive(self.global_seed, item.item_id, item.seed_salt)
                rng = np.random.default_rng(seed)
                mask, region_meta = resolver.resolve(
                    loaded.original_shape,
                    item.region_spec,
                    rng,
                )
                if item.invert_mask:
                    mask = np.logical_not(mask)
                    height, width = mask.shape[-2:]
                    area = mean_frame_area(mask)
                    region_meta = RegionMeta(
                        intended_area_px=int(round(area)),
                        intended_area_ratio=area / (height * width),
                        generator_kind=region_meta.generator_kind,
                        generator_version=region_meta.generator_version,
                        confidence=region_meta.confidence,
                    )
                array = perturbator.apply(
                    loaded.array,
                    mask,
                    item.perturb_op,
                    item.perturb_params,
                    rng,
                )
            except Exception:
                failed.append(
                    ItemMeta(
                        item_id=item.item_id,
                        region_meta=region_meta,
                        status=ItemStatus.PREPARE_FAILED,
                    )
                )
                if self.fail_fast:
                    break
                continue

            arrays.append(array)
            masks.append(mask)
            successful.append(ItemMeta(item_id=item.item_id, region_meta=region_meta))

        if arrays:
            stacked_arrays = np.stack(arrays)
        else:
            stacked_arrays = np.empty((0, *loaded.array.shape), dtype=np.uint8)
        return PreparedChunk(
            chunk_id=chunk.chunk_id,
            arrays=stacked_arrays,
            # Kept as a tuple, not stacked: items in one chunk may mix
            # broadcast (H, W) masks with per-frame (T, H, W) masks.
            masks=tuple(masks),
            item_metas=tuple(successful),
            failed_items=tuple(failed),
        )
