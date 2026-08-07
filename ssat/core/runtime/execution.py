"""Public execution loop connecting planning, workers, inference, and dumps."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
import logging

import numpy as np

from ssat.core.adapter.base import ModelAdapter
from ssat.core.config.schema import ResolvedConfig
from ssat.core.dump.types import CleanDumpRecord, PerturbedDumpRecord
from ssat.core.dump.writer import DumpWriter
from ssat.core.perturb.perturbator import Perturbator
from ssat.core.perturb.rng import derive
from ssat.core.plan.builder import PlanBuilder
from ssat.core.plan.types import WorkChunk, WorkChunkMeta, WorkItem
from ssat.core.region.resolver import RegionResolver
from ssat.core.region.types import RegionMeta
from ssat.core.resume.index import ResumeIndex
from ssat.core.runtime.batching import BatchSplitter, Rebatcher
from ssat.core.runtime.errors import RuntimeContractError, RuntimeExecutionError
from ssat.core.runtime.loader import iter_worker_results
from ssat.core.runtime.processors import ChunkProcessor, CleanProcessor
from ssat.core.runtime.types import (
    BatchSizeState,
    CleanInferenceItem,
    ExecutionSummary,
    FailedChunk,
    InferenceBatch,
    InferenceItem,
    PerturbedInferenceItem,
    PredictionResult,
    PreparedChunk,
)
from ssat.core.source.base import SampleSource
from ssat.core.source.types import LoadError, LoadedSample, SampleMeta
from ssat.core.types import ItemStatus
from ssat.utils.logger_factory import get_logger

class _RunState:
    """Accumulate records and status counts for one run_audit invocation."""

    def __init__(self, writer: DumpWriter, model_id: str) -> None:
        self.writer = writer
        self.model_id = model_id
        self.counts: Counter[ItemStatus] = Counter()
        self.clean_records = 0
        self.perturbed_records = 0

    def write_clean(self, record: CleanDumpRecord) -> None:
        self.writer.write_clean(record)
        self.clean_records += 1
        self.counts[record.status] += 1

    def write_perturbed(self, record: PerturbedDumpRecord) -> None:
        self.writer.write_perturbed(record)
        self.perturbed_records += 1
        self.counts[record.status] += 1


def run_audit(
    config: ResolvedConfig,
    plan_builder: PlanBuilder,
    sample_source: SampleSource,
    adapter: ModelAdapter,
    writer: DumpWriter,
    resume_index: ResumeIndex,
    *,
    region_resolver: RegionResolver | None = None,
    perturbator: Perturbator | None = None,
    logger: logging.Logger | None = None,
) -> ExecutionSummary:
    """Run pending clean inference and perturbation chunks in deterministic order.

    The caller owns writer creation and closure. A successful invocation flushes
    durable fragments but deliberately leaves the writer open.
    """

    if not isinstance(config, ResolvedConfig):
        raise TypeError("config must be a ResolvedConfig")
    if not isinstance(adapter, ModelAdapter):
        raise TypeError("adapter must be a ModelAdapter")
    adapter_spec = adapter.describe()
    if adapter_spec != config.adapter_spec:
        raise RuntimeExecutionError("adapter spec does not match resolved config")
    if writer.manifest.resolved_config != config:
        raise RuntimeExecutionError("writer config does not match resolved config")
    if writer.manifest.adapter_spec != adapter_spec:
        raise RuntimeExecutionError("writer adapter spec does not match adapter")

    runtime = config.runtime
    resolved_logger = logger or get_logger(__name__)
    clean_samples = resume_index.pending_clean_samples(
        plan_builder.enumerate_clean(),
        retry_failed=runtime.retry_failed,
    )
    chunks = resume_index.pending_chunks(
        plan_builder.enumerate(),
        retry_failed=runtime.retry_failed,
    )
    adapter_cap = adapter_spec.max_batch_size
    initial_size = (
        runtime.target_batch_size
        if adapter_cap is None
        else min(runtime.target_batch_size, adapter_cap)
    )
    batch_state = BatchSizeState(initial_size)
    splitter = BatchSplitter(adapter, batch_state)
    state = _RunState(writer, adapter_spec.model_id)

    resolved_logger.info(
        "runtime.started clean=%d chunks=%d batch_size=%d workers=%d",
        len(clean_samples),
        len(chunks),
        initial_size,
        runtime.num_workers,
    )

    clean_items = _iter_clean_items(
        clean_samples,
        sample_source,
        state,
        num_workers=runtime.num_workers,
        fail_fast=runtime.fail_fast,
    )
    _run_batches(
        Rebatcher(clean_items, batch_state),
        splitter,
        state,
        fail_fast=runtime.fail_fast,
    )

    perturbed_items = _iter_perturbed_items(
        chunks,
        plan_builder,
        sample_source,
        adapter,
        state,
        global_seed=runtime.global_seed,
        num_workers=runtime.num_workers,
        fail_fast=runtime.fail_fast,
        region_resolver=region_resolver,
        perturbator=perturbator,
    )
    _run_batches(
        Rebatcher(perturbed_items, batch_state),
        splitter,
        state,
        fail_fast=runtime.fail_fast,
    )
    writer.flush()

    summary = ExecutionSummary(
        clean_pending=len(clean_samples),
        chunks_pending=len(chunks),
        clean_records=state.clean_records,
        perturbed_records=state.perturbed_records,
        counts_by_status={status: state.counts[status] for status in ItemStatus},
        oom_events=batch_state.oom_events,
        initial_batch_size=batch_state.initial_size,
        final_batch_size=batch_state.current_size,
    )
    resolved_logger.info(
        "runtime.finished clean_records=%d perturbed_records=%d oom_events=%d",
        summary.clean_records,
        summary.perturbed_records,
        summary.oom_events,
    )
    return summary


def _iter_clean_items(
    samples: tuple[SampleMeta, ...],
    sample_source: SampleSource,
    state: _RunState,
    *,
    num_workers: int,
    fail_fast: bool,
) -> Iterator[CleanInferenceItem]:
    processor = CleanProcessor(samples, sample_source)
    try:
        for result in iter_worker_results(processor, num_workers=num_workers):
            if isinstance(result, LoadError):
                state.write_clean(
                    CleanDumpRecord(
                        sample_id=result.sample_id,
                        status=ItemStatus.LOAD_FAILED,
                        model_id=state.model_id,
                    )
                )
                _raise_if_fail_fast(
                    fail_fast,
                    state,
                    f"clean sample load failed: {result.sample_id}",
                )
                continue
            if not isinstance(result, LoadedSample):
                raise RuntimeContractError("clean worker returned an unsupported value")
            yield CleanInferenceItem(result)
    except RuntimeExecutionError:
        raise
    except Exception as error:
        raise RuntimeContractError("clean worker contract failed") from error


def _iter_perturbed_items(
    chunks: tuple[WorkChunkMeta, ...],
    plan_builder: PlanBuilder,
    sample_source: SampleSource,
    adapter: ModelAdapter,
    state: _RunState,
    *,
    global_seed: int,
    num_workers: int,
    fail_fast: bool,
    region_resolver: RegionResolver | None,
    perturbator: Perturbator | None,
) -> Iterator[PerturbedInferenceItem]:
    processor = ChunkProcessor(
        chunks,
        plan_builder,
        sample_source,
        global_seed,
        fail_fast=fail_fast,
        region_resolver=region_resolver,
        perturbator=perturbator,
    )
    try:
        for result in iter_worker_results(processor, num_workers=num_workers):
            if not isinstance(result, (PreparedChunk, FailedChunk)):
                raise RuntimeContractError("chunk worker returned an unsupported value")
            chunk = plan_builder.materialize(result.chunk_id)
            if isinstance(result, FailedChunk):
                _validate_failed_chunk(result, chunk)
                for item in chunk.items:
                    state.write_perturbed(
                        _perturbed_failure(
                            item,
                            result.reason,
                            global_seed=global_seed,
                        )
                    )
                _raise_if_fail_fast(
                    fail_fast,
                    state,
                    f"chunk load failed: {result.chunk_id}",
                )
                continue

            work_items = _validate_prepared_chunk(result, chunk, fail_fast=fail_fast)
            for failed in result.failed_items:
                item = work_items[failed.item_id]
                state.write_perturbed(
                    _perturbed_failure(
                        item,
                        failed.status,
                        global_seed=global_seed,
                        region_meta=failed.region_meta,
                    )
                )
            if result.failed_items:
                _raise_if_fail_fast(
                    fail_fast,
                    state,
                    f"chunk preparation failed: {result.chunk_id}",
                )

            for index, meta in enumerate(result.item_metas):
                item = work_items[meta.item_id]
                mask = result.masks[index]
                try:
                    transformed = adapter.transform_mask(mask)
                    if transformed is not None and (
                        not isinstance(transformed, np.ndarray)
                        or transformed.dtype != np.bool_
                        or transformed.ndim != 2
                    ):
                        raise TypeError("adapter mask transform must return a 2D bool array")
                    effective_area = (
                        None
                        if transformed is None
                        else int(np.count_nonzero(transformed))
                    )
                except Exception:
                    state.write_perturbed(
                        _perturbed_failure(
                            item,
                            ItemStatus.PREPARE_FAILED,
                            global_seed=global_seed,
                            region_meta=meta.region_meta,
                        )
                    )
                    _raise_if_fail_fast(
                        fail_fast,
                        state,
                        f"adapter mask transform failed: {item.item_id}",
                    )
                    continue
                if meta.region_meta is None:  # protected by ItemMeta validation
                    raise RuntimeContractError("prepared item lacks region metadata")
                yield PerturbedInferenceItem(
                    work_item=item,
                    array=result.arrays[index],
                    mask=mask,
                    region_meta=meta.region_meta,
                    seed_used=derive(global_seed, item.item_id, item.seed_salt),
                    effective_area_px=effective_area,
                )
    except RuntimeExecutionError:
        raise
    except Exception as error:
        raise RuntimeContractError("chunk worker contract failed") from error


def _run_batches(
    batches: Iterable[InferenceBatch],
    splitter: BatchSplitter,
    state: _RunState,
    *,
    fail_fast: bool,
) -> None:
    try:
        for batch in batches:
            try:
                results = splitter.predict(batch)
            except RuntimeContractError:
                raise
            except Exception:
                for item in batch.items:
                    _write_prediction_failure(item, state)
                _raise_if_fail_fast(
                    fail_fast,
                    state,
                    "adapter prediction failed",
                )
                continue

            for result in results:
                _write_prediction_result(result, state)
            if any(result.status is not ItemStatus.OK for result in results):
                message = (
                    "singleton inference ran out of memory"
                    if any(
                        result.status is ItemStatus.SKIPPED_OOM
                        for result in results
                    )
                    else "adapter prediction failed"
                )
                _raise_if_fail_fast(
                    fail_fast,
                    state,
                    message,
                )
    except RuntimeExecutionError:
        raise
    except Exception as error:
        raise RuntimeContractError("runtime batch stream failed") from error


def _write_prediction_result(result: PredictionResult, state: _RunState) -> None:
    if isinstance(result.item, CleanInferenceItem):
        sample = result.item.sample
        state.write_clean(
            CleanDumpRecord(
                sample_id=sample.sample_id,
                status=result.status,
                model_id=state.model_id,
                logits=result.output,
                content_hash=sample.content_hash,
                gt_label=sample.gt_label,
                original_shape=sample.original_shape,
            )
        )
        return
    item = result.item
    state.write_perturbed(
        PerturbedDumpRecord(
            work_item=item.work_item,
            status=result.status,
            seed_used=item.seed_used,
            logits=result.output,
            region_meta=item.region_meta,
            effective_area_px=item.effective_area_px,
        )
    )


def _write_prediction_failure(item: InferenceItem, state: _RunState) -> None:
    if isinstance(item, CleanInferenceItem):
        sample = item.sample
        state.write_clean(
            CleanDumpRecord(
                sample_id=sample.sample_id,
                status=ItemStatus.PREDICT_FAILED,
                model_id=state.model_id,
                content_hash=sample.content_hash,
                gt_label=sample.gt_label,
                original_shape=sample.original_shape,
            )
        )
        return
    state.write_perturbed(
        PerturbedDumpRecord(
            work_item=item.work_item,
            status=ItemStatus.PREDICT_FAILED,
            seed_used=item.seed_used,
            region_meta=item.region_meta,
            effective_area_px=item.effective_area_px,
        )
    )


def _perturbed_failure(
    item: WorkItem,
    status: ItemStatus,
    *,
    global_seed: int,
    region_meta: RegionMeta | None = None,
) -> PerturbedDumpRecord:
    return PerturbedDumpRecord(
        work_item=item,
        status=status,
        seed_used=derive(global_seed, item.item_id, item.seed_salt),
        region_meta=region_meta,
    )


def _validate_failed_chunk(result: FailedChunk, chunk: WorkChunk) -> None:
    expected = tuple(item.item_id for item in chunk.items)
    if result.reason is not ItemStatus.LOAD_FAILED or result.item_ids != expected:
        raise RuntimeContractError("failed chunk does not match materialized work")


def _validate_prepared_chunk(
    result: PreparedChunk,
    chunk: WorkChunk,
    *,
    fail_fast: bool,
) -> dict[str, WorkItem]:
    work_items = {item.item_id: item for item in chunk.items}
    returned = tuple(meta.item_id for meta in result.item_metas + result.failed_items)
    if len(set(returned)) != len(returned) or any(
        item_id not in work_items for item_id in returned
    ):
        raise RuntimeContractError("prepared chunk contains unknown or duplicate item IDs")
    expected = tuple(item.item_id for item in chunk.items)
    if fail_fast and result.failed_items:
        if len(result.failed_items) != 1 or returned != expected[: len(returned)]:
            raise RuntimeContractError("fail-fast prepared chunk must return one failed prefix")
    else:
        failed_ids = {meta.item_id for meta in result.failed_items}
        expected_success = tuple(
            item_id for item_id in expected if item_id not in failed_ids
        )
        expected_failed = tuple(
            item_id for item_id in expected if item_id in failed_ids
        )
        actual_success = tuple(meta.item_id for meta in result.item_metas)
        actual_failed = tuple(meta.item_id for meta in result.failed_items)
        if actual_success != expected_success or actual_failed != expected_failed:
            raise RuntimeContractError(
                "prepared chunk omitted or reordered materialized work items"
            )
    return work_items


def _raise_if_fail_fast(
    fail_fast: bool,
    state: _RunState,
    message: str,
) -> None:
    if fail_fast:
        state.writer.flush()
        raise RuntimeExecutionError(message)
