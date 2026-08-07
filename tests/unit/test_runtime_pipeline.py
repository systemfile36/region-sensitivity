"""Tests for shared runtime preparation and inference flows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ssat.core.adapter import CallableAdapter
from ssat.core.adapter.base import AdapterOutOfMemoryError
from ssat.core.config.schema import (
    DumpConfig,
    PerturbationConfig,
    ResolvedConfig,
    ResolvedRegionConfig,
    RuntimeConfig,
)
from ssat.core.perturb.rng import derive
from ssat.core.plan import PlanBuilder
from ssat.core.runtime.errors import RuntimeContractError
from ssat.core.runtime.pipeline import (
    iter_clean_preparation_results,
    iter_prediction_batches,
    iter_prepared_work_chunks,
)
from ssat.core.runtime.types import BatchSizeState, CleanInferenceItem
from ssat.core.source.types import LoadError, LoadedSample, SampleMeta
from ssat.core.types import ItemStatus, PerturbationOp, RegionKind


class MemorySource:
    """Provide one in-memory sample with optional load failure."""

    def __init__(self, *, fail: bool = False, invalid: bool = False) -> None:
        self.sample = SampleMeta("sample", "unused")
        self.fail = fail
        self.invalid = invalid

    def list_samples(self) -> list[SampleMeta]:
        """Return the single sample catalog."""

        return [self.sample]

    def load(self, sample_id: str) -> LoadedSample | LoadError | object:
        """Return a loaded sample, recoverable failure, or invalid test value."""

        if self.invalid:
            return object()
        if self.fail:
            return LoadError(sample_id, "injected", "load failed")
        array = np.full((1, 4, 4, 3), 255, dtype=np.uint8)
        return LoadedSample(array, sample_id, array.shape, "a" * 64)


def _adapter(
    predict_fn=lambda batch: np.zeros((len(batch), 2), dtype=np.float32),
    *,
    transform_mask_fn=lambda mask: mask,
) -> CallableAdapter:
    """Build a test adapter with injectable prediction and mask behavior."""

    return CallableAdapter(
        predict_fn,
        model_id="pipeline",
        transform_mask_fn=transform_mask_fn,
    )


def _config(tmp_path: Path, adapter: CallableAdapter) -> ResolvedConfig:
    """Build a two-item deterministic perturbation plan."""

    return ResolvedConfig(
        config_base_dir=tmp_path.resolve(),
        regions=(
            ResolvedRegionConfig(
                region_id="grid",
                kind=RegionKind.GRID,
                params={"rows": 1, "cols": 2},
            ),
        ),
        perturbations=(
            PerturbationConfig(
                op=PerturbationOp.CONSTANT_FILL,
                params={"value": 0},
            ),
        ),
        runtime=RuntimeConfig(variants_per_chunk=2, target_batch_size=4),
        dump=DumpConfig(),
        adapter_spec=adapter.describe(),
    )


def test_clean_pipeline_normalizes_load_results_and_rejects_contracts() -> None:
    """Clean preparation should emit values and reject unsupported outputs."""

    loaded_source = MemorySource()
    loaded = list(
        iter_clean_preparation_results(
            (loaded_source.sample,),
            loaded_source,
            num_workers=0,
        )
    )
    assert isinstance(loaded[0], CleanInferenceItem)

    failed_source = MemorySource(fail=True)
    failed = list(
        iter_clean_preparation_results(
            (failed_source.sample,),
            failed_source,
            num_workers=0,
        )
    )
    assert isinstance(failed[0], LoadError)

    invalid_source = MemorySource(invalid=True)
    with pytest.raises(RuntimeContractError):
        list(
            iter_clean_preparation_results(
                (invalid_source.sample,),
                invalid_source,
                num_workers=0,
            )
        )


def test_perturbed_pipeline_normalizes_items_failures_and_memory(
    tmp_path: Path,
) -> None:
    """Perturbed preparation should retain order, seeds, failures, and bytes."""

    adapter = _adapter()
    source = MemorySource()
    config = _config(tmp_path, adapter)
    builder = PlanBuilder(config, source)
    chunk_meta = builder.enumerate()[0]
    prepared = list(
        iter_prepared_work_chunks(
            (chunk_meta,),
            builder,
            source,
            adapter,
            global_seed=config.runtime.global_seed,
            num_workers=0,
            fail_fast=False,
        )
    )[0]

    assert [item.work_item.item_id for item in prepared.items] == list(
        chunk_meta.item_ids
    )
    assert prepared.failures == ()
    assert prepared.prepared_bytes == 128
    assert prepared.max_item_bytes == 64
    assert prepared.items[0].seed_used == derive(
        config.runtime.global_seed,
        prepared.items[0].work_item.item_id,
        prepared.items[0].work_item.seed_salt,
    )

    failed_source = MemorySource(fail=True)
    failed_builder = PlanBuilder(config, failed_source)
    failed_meta = failed_builder.enumerate()[0]
    failed = list(
        iter_prepared_work_chunks(
            (failed_meta,),
            failed_builder,
            failed_source,
            adapter,
            global_seed=0,
            num_workers=0,
            fail_fast=False,
        )
    )[0]
    assert failed.items == ()
    assert [item.status for item in failed.failures] == [
        ItemStatus.LOAD_FAILED,
        ItemStatus.LOAD_FAILED,
    ]


def test_perturbed_pipeline_converts_mask_transform_failures(
    tmp_path: Path,
) -> None:
    """Mask transform exceptions should become terminal preparation failures."""

    def fail_mask(mask: np.ndarray) -> np.ndarray:
        raise RuntimeError("mask transform failed")

    adapter = _adapter(transform_mask_fn=fail_mask)
    source = MemorySource()
    config = _config(tmp_path, adapter)
    builder = PlanBuilder(config, source)
    prepared = list(
        iter_prepared_work_chunks(
            builder.enumerate(),
            builder,
            source,
            adapter,
            global_seed=0,
            num_workers=0,
            fail_fast=False,
        )
    )[0]

    assert prepared.items == ()
    assert len(prepared.failures) == 2
    assert all(
        failure.status is ItemStatus.PREPARE_FAILED
        for failure in prepared.failures
    )
    assert prepared.prepared_bytes == 128


def test_prediction_pipeline_preserves_recursive_oom_behavior() -> None:
    """Shared inference should preserve ordered recursive OOM recovery."""

    calls: list[int] = []

    def predict(batch: np.ndarray) -> np.ndarray:
        calls.append(len(batch))
        if len(batch) > 1:
            raise AdapterOutOfMemoryError("injected")
        return np.zeros((len(batch), 2), dtype=np.float32)

    adapter = _adapter(predict)
    source = MemorySource()
    clean = list(
        iter_clean_preparation_results(
            (source.sample, SampleMeta("other", "unused")),
            source,
            num_workers=0,
        )
    )
    state = BatchSizeState(4)
    outcome = list(iter_prediction_batches(clean, adapter, state))[0]

    assert calls == [2, 1, 1]
    assert outcome.error is None
    assert [result.status for result in outcome.results] == [
        ItemStatus.OK,
        ItemStatus.OK,
    ]
    assert state.current_size == 1


def test_prediction_pipeline_exposes_unexpected_splitter_failures() -> None:
    """Unexpected splitter errors should remain policy-neutral outcomes."""

    def predict(batch: np.ndarray) -> np.ndarray:
        raise AdapterOutOfMemoryError("injected")

    def fail_cleanup() -> None:
        raise RuntimeError("cleanup failed")

    adapter = CallableAdapter(
        predict,
        model_id="pipeline",
        transform_mask_fn=lambda mask: mask,
        cleanup_after_oom_fn=fail_cleanup,
    )
    source = MemorySource()
    clean = list(
        iter_clean_preparation_results(
            (source.sample,),
            source,
            num_workers=0,
        )
    )
    outcome = list(
        iter_prediction_batches(clean, adapter, BatchSizeState(1))
    )[0]

    assert outcome.results == ()
    assert isinstance(outcome.error, RuntimeError)
