"""Shared preparation and inference flows for runtime consumers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from ssat.core.adapter.base import ModelAdapter
from ssat.core.adapter.types import AdapterSpec
from ssat.core.perturb.perturbator import Perturbator
from ssat.core.perturb.rng import derive
from ssat.core.plan.builder import PlanBuilder
from ssat.core.plan.types import WorkChunk, WorkChunkMeta, WorkItem
from ssat.core.region.mask_base import mean_frame_area
from ssat.core.region.resolver import RegionResolver
from ssat.core.region.types import RegionMeta
from ssat.core.runtime.batching import BatchSplitter, Rebatcher
from ssat.core.runtime.errors import RuntimeContractError
from ssat.core.runtime.loader import iter_worker_results
from ssat.core.runtime.processors import ChunkProcessor, CleanProcessor
from ssat.core.runtime.types import (
    BatchSizeState,
    CleanInferenceItem,
    FailedChunk,
    InferenceItem,
    PerturbedInferenceItem,
    PredictionResult,
    PreparedChunk,
)
from ssat.core.source.base import SampleSource
from ssat.core.source.types import LoadError, LoadedSample, SampleMeta
from ssat.core.types import ItemStatus


CleanPreparationResult = CleanInferenceItem | LoadError


@dataclass(frozen=True, slots=True)
class PreparationFailure:
    """Describe one work item that could not reach model inference.

    Attributes:
        work_item: Planned item associated with the failure.
        status: Terminal preparation status for the item.
        seed_used: Deterministic item seed recorded by downstream consumers.
        region_meta: Optional region measurements produced before the failure.
    """

    work_item: WorkItem
    status: ItemStatus
    seed_used: int
    region_meta: RegionMeta | None = None


@dataclass(frozen=True, slots=True)
class PreparedWorkChunk:
    """Carry normalized preparation results for one planned chunk.

    Attributes:
        chunk_id: Stable identifier of the materialized chunk.
        items: Successful items ready for model inference.
        failures: Items that reached a terminal preparation status.
        prepared_bytes: Total bytes held by prepared arrays and masks.
        max_item_bytes: Largest prepared array-and-mask pair in the chunk.
        failure_message: Fail-fast message matching the first failure stage.
    """

    chunk_id: str
    items: tuple[PerturbedInferenceItem, ...]
    failures: tuple[PreparationFailure, ...]
    prepared_bytes: int
    max_item_bytes: int
    failure_message: str | None = None


@dataclass(frozen=True, slots=True)
class PredictionBatchOutcome:
    """Carry one prediction batch or a policy-neutral execution failure.

    Attributes:
        items: Ordered inference items submitted as one batch.
        results: Ordered terminal prediction results when execution completed.
        error: Unexpected splitter failure left for the consumer to interpret.
    """

    items: tuple[InferenceItem, ...]
    results: tuple[PredictionResult, ...]
    error: Exception | None = None


def initial_batch_size(target_batch_size: int, spec: AdapterSpec) -> int:
    """Resolve the initial batch cap shared by execution and estimation.

    Args:
        target_batch_size: Batch size requested by runtime configuration.
        spec: Adapter metadata containing an optional hard batch limit.

    Returns:
        The requested size limited by the adapter capability when present.
    """

    return (
        target_batch_size
        if spec.max_batch_size is None
        else min(target_batch_size, spec.max_batch_size)
    )


def iter_clean_preparation_results(
    samples: tuple[SampleMeta, ...],
    sample_source: SampleSource,
    *,
    num_workers: int,
) -> Iterator[CleanPreparationResult]:
    """Yield validated clean worker results in deterministic input order.

    Args:
        samples: Ordered sample metadata to load.
        sample_source: Source used by clean workers.
        num_workers: Number of data-loading workers.

    Yields:
        A clean inference item or a recoverable load error for each sample.

    Raises:
        RuntimeContractError: If a worker violates the clean-result contract.
    """

    processor = CleanProcessor(samples, sample_source)
    try:
        for result in iter_worker_results(processor, num_workers=num_workers):
            if isinstance(result, LoadError):
                yield result
            elif isinstance(result, LoadedSample):
                yield CleanInferenceItem(result)
            else:
                raise RuntimeContractError(
                    "clean worker returned an unsupported value"
                )
    except RuntimeContractError:
        raise
    except Exception as error:
        raise RuntimeContractError("clean worker contract failed") from error


def iter_prepared_work_chunks(
    chunks: tuple[WorkChunkMeta, ...],
    plan_builder: PlanBuilder,
    sample_source: SampleSource,
    adapter: ModelAdapter,
    *,
    global_seed: int,
    num_workers: int,
    fail_fast: bool,
    region_resolver: RegionResolver | None = None,
    perturbator: Perturbator | None = None,
) -> Iterator[PreparedWorkChunk]:
    """Normalize worker chunks into inference items and terminal failures.

    Args:
        chunks: Ordered metadata for chunks that require preparation.
        plan_builder: Planner used to rematerialize complete work items.
        sample_source: Source loaded by chunk workers.
        adapter: Adapter used to transform source-space masks.
        global_seed: Global deterministic seed from runtime configuration.
        num_workers: Number of chunk-preparation workers.
        fail_fast: Whether workers stop after their first preparation failure.
        region_resolver: Optional region resolver injected into workers.
        perturbator: Optional perturbation facade injected into workers.

    Yields:
        Normalized results for each worker chunk in deterministic order.

    Raises:
        RuntimeContractError: If worker output does not match the planned work.
    """

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
                raise RuntimeContractError(
                    "chunk worker returned an unsupported value"
                )
            chunk = plan_builder.materialize(result.chunk_id)
            if isinstance(result, FailedChunk):
                _validate_failed_chunk(result, chunk)
                failures = tuple(
                    _preparation_failure(
                        item,
                        result.reason,
                        global_seed=global_seed,
                    )
                    for item in chunk.items
                )
                yield PreparedWorkChunk(
                    chunk_id=chunk.chunk_id,
                    items=(),
                    failures=failures,
                    prepared_bytes=0,
                    max_item_bytes=0,
                    failure_message=f"chunk load failed: {chunk.chunk_id}",
                )
                continue

            work_items = _validate_prepared_chunk(
                result,
                chunk,
                fail_fast=fail_fast,
            )
            prepared_bytes = int(
                result.arrays.nbytes + sum(mask.nbytes for mask in result.masks)
            )
            max_item_bytes = max(
                (
                    int(result.arrays[index].nbytes + result.masks[index].nbytes)
                    for index in range(len(result.item_metas))
                ),
                default=0,
            )
            failures = [
                _preparation_failure(
                    work_items[failed.item_id],
                    failed.status,
                    global_seed=global_seed,
                    region_meta=failed.region_meta,
                )
                for failed in result.failed_items
            ]
            if result.failed_items and fail_fast:
                yield PreparedWorkChunk(
                    chunk_id=chunk.chunk_id,
                    items=(),
                    failures=tuple(failures),
                    prepared_bytes=prepared_bytes,
                    max_item_bytes=max_item_bytes,
                    failure_message=f"chunk preparation failed: {chunk.chunk_id}",
                )
                continue

            prepared_items: list[PerturbedInferenceItem] = []
            failure_message = (
                f"chunk preparation failed: {chunk.chunk_id}"
                if failures
                else None
            )
            for index, meta in enumerate(result.item_metas):
                item = work_items[meta.item_id]
                mask = result.masks[index]
                try:
                    effective_area = _effective_mask_area(adapter, mask)
                except Exception:
                    failures.append(
                        _preparation_failure(
                            item,
                            ItemStatus.PREPARE_FAILED,
                            global_seed=global_seed,
                            region_meta=meta.region_meta,
                        )
                    )
                    if failure_message is None:
                        failure_message = (
                            f"adapter mask transform failed: {item.item_id}"
                        )
                    if fail_fast:
                        break
                    continue
                if meta.region_meta is None:  # Enforced by ItemMeta validation.
                    raise RuntimeContractError(
                        "prepared item lacks region metadata"
                    )
                prepared_items.append(
                    PerturbedInferenceItem(
                        work_item=item,
                        array=result.arrays[index],
                        mask=mask,
                        region_meta=meta.region_meta,
                        seed_used=derive(
                            global_seed,
                            item.item_id,
                            item.seed_salt,
                        ),
                        effective_area_px=effective_area,
                    )
                )
            yield PreparedWorkChunk(
                chunk_id=chunk.chunk_id,
                items=tuple(prepared_items),
                failures=tuple(failures),
                prepared_bytes=prepared_bytes,
                max_item_bytes=max_item_bytes,
                failure_message=failure_message,
            )
    except RuntimeContractError:
        raise
    except Exception as error:
        raise RuntimeContractError("chunk worker contract failed") from error


def iter_prediction_batches(
    items: Iterable[InferenceItem],
    adapter: ModelAdapter,
    batch_size_state: BatchSizeState,
) -> Iterator[PredictionBatchOutcome]:
    """Run shape-aware inference and yield one ordered result tuple per batch.

    Args:
        items: Prepared clean or perturbed inference items.
        adapter: Model adapter invoked by the OOM-aware splitter.
        batch_size_state: Mutable cap shared across all emitted batches.

    Yields:
        Policy-neutral outcomes for each dynamically formed batch.

    Raises:
        RuntimeContractError: If batching or adapter output violates a contract.
    """

    splitter = BatchSplitter(adapter, batch_size_state)
    for batch in Rebatcher(items, batch_size_state):
        try:
            results = splitter.predict(batch)
        except RuntimeContractError:
            raise
        except Exception as error:
            yield PredictionBatchOutcome(batch.items, (), error)
        else:
            yield PredictionBatchOutcome(batch.items, results)


def _effective_mask_area(
    adapter: ModelAdapter,
    mask: np.ndarray,
) -> int | None:
    """Transform a mask and return its model-space area when available.

    Args:
        adapter: Adapter owning the model-space geometry transform.
        mask: Source-space boolean mask.

    Returns:
        The transformed nonzero area (mean per-frame count for a ``(T, H,
        W)`` mask, matching the ``RegionMeta`` convention), or ``None`` when
        unavailable.

    Raises:
        TypeError: If the adapter returns an invalid transformed mask.
    """

    transformed = adapter.transform_mask(mask)
    if transformed is None:
        return None
    if (
        not isinstance(transformed, np.ndarray)
        or transformed.dtype != np.bool_
        or transformed.ndim not in (2, 3)
    ):
        raise TypeError(
            "adapter mask transform must return a (H, W) or (T, H, W) bool array"
        )
    return int(round(mean_frame_area(transformed)))


def _preparation_failure(
    item: WorkItem,
    status: ItemStatus,
    *,
    global_seed: int,
    region_meta: RegionMeta | None = None,
) -> PreparationFailure:
    """Build one normalized preparation failure.

    Args:
        item: Work item that failed before inference.
        status: Terminal preparation status.
        global_seed: Global deterministic seed from runtime configuration.
        region_meta: Optional measurements created before the failure.

    Returns:
        A normalized failure containing the deterministic item seed.
    """

    return PreparationFailure(
        work_item=item,
        status=status,
        seed_used=derive(global_seed, item.item_id, item.seed_salt),
        region_meta=region_meta,
    )


def _validate_failed_chunk(result: FailedChunk, chunk: WorkChunk) -> None:
    """Validate a chunk-wide load failure against planned work.

    Args:
        result: Failure returned by a chunk worker.
        chunk: Authoritative materialized chunk.

    Raises:
        RuntimeContractError: If identifiers, ordering, or status diverge.
    """

    expected = tuple(item.item_id for item in chunk.items)
    if result.reason is not ItemStatus.LOAD_FAILED or result.item_ids != expected:
        raise RuntimeContractError("failed chunk does not match materialized work")


def _validate_prepared_chunk(
    result: PreparedChunk,
    chunk: WorkChunk,
    *,
    fail_fast: bool,
) -> dict[str, WorkItem]:
    """Validate prepared successes and failures against planned item order.

    Args:
        result: Prepared output returned by a chunk worker.
        chunk: Authoritative materialized chunk.
        fail_fast: Whether the worker is expected to return a failed prefix.

    Returns:
        A mapping from item identifiers to authoritative work items.

    Raises:
        RuntimeContractError: If the worker omits, duplicates, or reorders work.
    """

    work_items = {item.item_id: item for item in chunk.items}
    returned = tuple(meta.item_id for meta in result.item_metas + result.failed_items)
    if len(set(returned)) != len(returned) or any(
        item_id not in work_items for item_id in returned
    ):
        raise RuntimeContractError(
            "prepared chunk contains unknown or duplicate item IDs"
        )
    expected = tuple(item.item_id for item in chunk.items)
    if fail_fast and result.failed_items:
        if len(result.failed_items) != 1 or returned != expected[: len(returned)]:
            raise RuntimeContractError(
                "fail-fast prepared chunk must return one failed prefix"
            )
        return work_items

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
