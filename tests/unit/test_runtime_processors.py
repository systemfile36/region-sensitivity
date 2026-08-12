"""Tests for worker-side clean and chunk processing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ssat.core.adapter.types import AdapterSpec
from ssat.core.config.schema import (
    DumpConfig,
    PerturbationConfig,
    ResolvedConfig,
    ResolvedRegionConfig,
    RuntimeConfig,
)
from ssat.core.perturb import Perturbator
from ssat.core.plan import PlanBuilder
from ssat.core.region import MaskResolutionContext, RegionMaskGenerator, RegionResolver
from ssat.core.region.mask_base import ExplicitMaskCache
from ssat.core.region.types import RegionSpec
from ssat.core.runtime import ChunkProcessor, FailedChunk, PreparedChunk
from ssat.core.source.types import LoadError, LoadedSample, SampleMeta
from ssat.core.types import ItemStatus, PerturbationOp, RegionKind


class MemorySource:
    def __init__(self, *, fail: bool = False) -> None:
        self.meta = SampleMeta("sample", "unused")
        self.fail = fail
        self.load_calls = 0

    def list_samples(self) -> list[SampleMeta]:
        return [self.meta]

    def load(self, sample_id: str) -> LoadedSample | LoadError:
        self.load_calls += 1
        if self.fail:
            return LoadError(sample_id, "test", "failed")
        array = np.full((1, 4, 4, 3), 255, dtype=np.uint8)
        return LoadedSample(array, sample_id, array.shape, "a" * 64)


class AlwaysFailPerturbator(Perturbator):
    def __init__(self) -> None:
        pass

    def apply(self, *args: object, **kwargs: object) -> np.ndarray:
        raise RuntimeError("prepare failed")


def _config(tmp_path: Path, *, fail_fast: bool = False) -> ResolvedConfig:
    return ResolvedConfig(
        config_base_dir=tmp_path.resolve(),
        regions=(
            ResolvedRegionConfig(
                region_id="grid",
                kind=RegionKind.GRID,
                params={"rows": 1, "cols": 4},
            ),
        ),
        perturbations=(
            PerturbationConfig(
                op=PerturbationOp.CONSTANT_FILL,
                params={"value": 0},
                invert_mask=True,
            ),
        ),
        runtime=RuntimeConfig(variants_per_chunk=4, fail_fast=fail_fast),
        dump=DumpConfig(),
        adapter_spec=AdapterSpec(model_id="model", deterministic=True),
    )


def test_chunk_processor_loads_once_and_transfers_actual_inverted_masks(
    tmp_path: Path,
) -> None:
    source = MemorySource()
    config = _config(tmp_path)
    builder = PlanBuilder(config, source)
    chunks = builder.enumerate()
    processor = ChunkProcessor(chunks, builder, source, config.runtime.global_seed)

    first = processor[0]
    second = processor[0]

    assert isinstance(first, PreparedChunk)
    assert isinstance(second, PreparedChunk)
    assert source.load_calls == 2
    assert np.array_equal(first.arrays, second.arrays)
    assert len(first.masks) == len(second.masks) == 4
    assert all(np.array_equal(a, b) for a, b in zip(first.masks, second.masks))
    assert all(mask.shape == (4, 4) for mask in first.masks)
    assert [int(mask.sum()) for mask in first.masks] == [12, 12, 12, 12]
    assert [meta.region_meta.intended_area_px for meta in first.item_metas] == [
        12,
        12,
        12,
        12,
    ]


def test_chunk_processor_returns_chunk_wide_load_failure(tmp_path: Path) -> None:
    source = MemorySource(fail=True)
    config = _config(tmp_path)
    builder = PlanBuilder(config, source)
    chunks = builder.enumerate()

    result = ChunkProcessor(chunks, builder, source, 0)[0]

    assert isinstance(result, FailedChunk)
    assert result.reason is ItemStatus.LOAD_FAILED
    assert result.item_ids == chunks[0].item_ids
    assert source.load_calls == 1


def test_chunk_processor_separates_prepare_failures_and_stops_in_fail_fast(
    tmp_path: Path,
) -> None:
    source = MemorySource()
    config = _config(tmp_path)
    builder = PlanBuilder(config, source)
    chunks = builder.enumerate()

    regular = ChunkProcessor(
        chunks,
        builder,
        source,
        0,
        perturbator=AlwaysFailPerturbator(),
    )[0]
    fast = ChunkProcessor(
        chunks,
        builder,
        source,
        0,
        fail_fast=True,
        perturbator=AlwaysFailPerturbator(),
    )[0]

    assert isinstance(regular, PreparedChunk)
    assert len(regular.failed_items) == 4
    assert all(meta.status is ItemStatus.PREPARE_FAILED for meta in regular.failed_items)
    assert isinstance(fast, PreparedChunk)
    assert len(fast.failed_items) == 1


class MultiFrameSource(MemorySource):
    """Load a fixed three-frame clip instead of a single-frame image."""

    def load(self, sample_id: str) -> LoadedSample | LoadError:
        self.load_calls += 1
        array = np.full((3, 4, 4, 3), 255, dtype=np.uint8)
        return LoadedSample(array, sample_id, array.shape, "a" * 64)


class PerFrameGridGenerator(RegionMaskGenerator):
    """Return a fixed (T, H, W) mask regardless of the requested grid cell."""

    def supports(self, spec: RegionSpec) -> bool:
        return spec.kind is RegionKind.GRID

    def get_mask(
        self,
        height: int,
        width: int,
        spec: RegionSpec,
        rng: Generator | None = None,
    ) -> NDArray[np.bool_]:
        mask = np.zeros((3, height, width), dtype=np.bool_)
        mask[0] = True  # full frame
        mask[1, :2, :] = True  # half frame
        # frame 2 stays empty
        return mask


def test_chunk_processor_stacks_per_frame_masks_as_a_tuple(tmp_path: Path) -> None:
    """(T, H, W) masks are kept as a tuple, and inverted area is a frame mean."""

    source = MultiFrameSource()
    config = _config(tmp_path)  # grid 1x4, constant_fill with invert_mask=True
    builder = PlanBuilder(config, source)
    chunks = builder.enumerate()
    def _unused_resolve_target(*_args: object, **_kwargs: object) -> NDArray[np.bool_]:
        raise NotImplementedError("this test never resolves an embedded target")

    context = MaskResolutionContext(
        explicit_cache=ExplicitMaskCache(1),
        resolve_target=_unused_resolve_target,
    )
    resolver = RegionResolver(mask_generators=(PerFrameGridGenerator(context),))
    processor = ChunkProcessor(
        chunks, builder, source, config.runtime.global_seed, region_resolver=resolver
    )

    result = processor[0]

    assert isinstance(result, PreparedChunk)
    assert isinstance(result.masks, tuple)
    assert all(mask.shape == (3, 4, 4) for mask in result.masks)
    # invert_mask negates: frame0 empty, frame1 half, frame2 full -> mean 8/16 px.
    assert [meta.region_meta.intended_area_px for meta in result.item_metas] == [8] * 4
