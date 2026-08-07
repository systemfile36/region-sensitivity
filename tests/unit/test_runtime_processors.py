"""Tests for worker-side clean and chunk processing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
    assert np.array_equal(first.masks, second.masks)
    assert first.masks.shape == (4, 4, 4)
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
