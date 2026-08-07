"""Execution-aware cost estimation and clean accuracy sanity checks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import logging
import math
import time
from typing import TypeVar

import numpy as np

from ssat.core.adapter.base import ModelAdapter
from ssat.core.adapter.types import AdapterSpec, RawOutput
from ssat.core.config.schema import ResolvedConfig
from ssat.core.estimate.errors import EstimationError
from ssat.core.estimate.types import (
    Advisory,
    AdvisoryCode,
    DumpSizeAssumptions,
    EstimateOptions,
    EstimateReport,
    EstimationLimits,
    LimitExceedance,
    LimitKind,
    ProfileResult,
    SanityCheckResult,
)
from ssat.core.perturb.perturbator import Perturbator
from ssat.core.perturb.rng import derive
from ssat.core.plan.builder import PlanBuilder
from ssat.core.plan.types import WorkChunk, WorkChunkMeta, WorkItem
from ssat.core.region.resolver import RegionResolver
from ssat.core.resume.index import ResumeIndex
from ssat.core.runtime.batching import BatchSplitter, Rebatcher
from ssat.core.runtime.loader import iter_worker_results
from ssat.core.runtime.processors import ChunkProcessor, CleanProcessor
from ssat.core.runtime.types import (
    BatchSizeState,
    CleanInferenceItem,
    FailedChunk,
    InferenceItem,
    PerturbedInferenceItem,
    PreparedChunk,
)
from ssat.core.source.base import SampleSource
from ssat.core.source.types import LoadError, LoadedSample, SampleMeta
from ssat.core.types import ItemStatus
from ssat.utils.logger_factory import get_logger

T = TypeVar("T")


class _CountingAdapter(ModelAdapter):
    """Count every adapter invocation, including recursive OOM retries."""

    def __init__(self, adapter: ModelAdapter) -> None:
        self.adapter = adapter
        self.inference_calls = 0

    def describe(self) -> AdapterSpec:
        return self.adapter.describe()

    def predict(self, batch: np.ndarray) -> list[RawOutput]:
        self.inference_calls += 1
        return self.adapter.predict(batch)

    def transform_mask(self, mask: np.ndarray) -> np.ndarray | None:
        return self.adapter.transform_mask(mask)

    def cleanup_after_oom(self) -> None:
        self.adapter.cleanup_after_oom()


@dataclass(slots=True)
class _ProfileAccumulator:
    status_counts: Counter[ItemStatus]
    terminal_items: int = 0
    max_prepared_item_bytes: int = 0
    max_prepared_chunk_bytes: int = 0


class SanityCheck:
    """Measure clean throughput and optional labeled top-1 accuracy."""

    def __init__(
        self,
        *,
        max_samples: int = 20,
        minimum_accuracy: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
        logger: logging.Logger | None = None,
    ) -> None:
        if (
            isinstance(max_samples, bool)
            or not isinstance(max_samples, int)
            or max_samples <= 0
        ):
            raise ValueError("max_samples must be a positive integer")
        if minimum_accuracy is not None and (
            isinstance(minimum_accuracy, bool)
            or not isinstance(minimum_accuracy, (int, float))
            or not math.isfinite(minimum_accuracy)
            or not 0.0 <= minimum_accuracy <= 1.0
        ):
            raise ValueError("minimum_accuracy must be between 0 and 1")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._max_samples = max_samples
        self._minimum_accuracy = minimum_accuracy
        self._clock = clock
        self._logger = logger or get_logger(__name__)

    def run(
        self,
        config: ResolvedConfig,
        samples: Sequence[SampleMeta],
        sample_source: SampleSource,
        adapter: ModelAdapter,
        *,
        batch_size_state: BatchSizeState | None = None,
    ) -> SanityCheckResult:
        """Run a bounded clean pass without writing dump records."""

        _validate_provenance(config, adapter)
        selected = _select_evenly(tuple(samples), self._max_samples)
        if not selected:
            raise EstimationError("sanity check requires at least one sample")

        status_counts: Counter[ItemStatus] = Counter()
        terminal = 0
        successful = 0
        labeled = 0
        correct = 0
        unlabeled = 0
        invalid_labels = 0
        invalid_logits = 0
        class_count: int | None = None
        state = batch_size_state or BatchSizeState(
            _initial_batch_size(config, adapter.describe())
        )
        starting_batch_size = state.current_size
        starting_oom_events = state.oom_events
        counting_adapter = _CountingAdapter(adapter)
        splitter = BatchSplitter(counting_adapter, state)

        def clean_items() -> Iterable[CleanInferenceItem]:
            nonlocal terminal
            processor = CleanProcessor(selected, sample_source)
            for result in iter_worker_results(
                processor,
                num_workers=config.runtime.num_workers,
            ):
                if isinstance(result, LoadError):
                    terminal += 1
                    status_counts[ItemStatus.LOAD_FAILED] += 1
                    continue
                if not isinstance(result, LoadedSample):
                    raise EstimationError("clean worker returned an unsupported value")
                yield CleanInferenceItem(result)

        started = self._clock()
        try:
            for batch in Rebatcher(clean_items(), state):
                results = splitter.predict(batch)
                for result in results:
                    terminal += 1
                    status_counts[result.status] += 1
                    if result.status is not ItemStatus.OK or result.output is None:
                        continue
                    output = result.output.logits
                    if output.size == 0 or not np.all(np.isfinite(output)):
                        invalid_logits += 1
                        continue
                    class_count = _merge_class_count(class_count, int(output.size))
                    successful += 1
                    sample = result.item.sample
                    if sample.gt_label is None:
                        unlabeled += 1
                    elif not 0 <= sample.gt_label < output.size:
                        invalid_labels += 1
                    else:
                        labeled += 1
                        correct += int(np.argmax(output) == sample.gt_label)
        except EstimationError:
            raise
        except Exception as error:
            raise EstimationError("clean sanity execution failed") from error
        elapsed = _elapsed(self._clock() - started)

        accuracy = None if labeled == 0 else correct / labeled
        passed = (
            None
            if self._minimum_accuracy is None
            else accuracy is not None and accuracy >= self._minimum_accuracy
        )
        advisories: list[Advisory] = []
        failures = terminal - status_counts[ItemStatus.OK]
        if failures or invalid_labels or invalid_logits:
            advisories.append(
                Advisory(
                    AdvisoryCode.SANITY_PARTIAL_FAILURES,
                    "Clean sanity check contained failed or invalid outputs.",
                )
            )
        if accuracy is None:
            advisories.append(
                Advisory(
                    AdvisoryCode.SANITY_NO_LABELED_OUTPUTS,
                    "No successful labeled output was available for top-1 accuracy.",
                )
            )
        if passed is False:
            advisories.append(
                Advisory(
                    AdvisoryCode.SANITY_ACCURACY_BELOW_MINIMUM,
                    "Clean top-1 accuracy is unavailable or below the requested minimum.",
                )
            )
            self._logger.warning("estimate.sanity_accuracy_below_minimum")

        return SanityCheckResult(
            selected_samples=len(selected),
            terminal_samples=terminal,
            successful_predictions=successful,
            labeled_predictions=labeled,
            correct_predictions=correct,
            unlabeled_predictions=unlabeled,
            invalid_label_predictions=invalid_labels,
            invalid_logit_predictions=invalid_logits,
            status_counts=_complete_counts(status_counts),
            elapsed_seconds=elapsed,
            items_per_second=terminal / elapsed,
            inference_calls=counting_adapter.inference_calls,
            oom_events=state.oom_events - starting_oom_events,
            initial_batch_size=starting_batch_size,
            final_batch_size=state.current_size,
            class_count=class_count,
            accuracy=accuracy,
            minimum_accuracy=self._minimum_accuracy,
            passed=passed,
            advisories=tuple(advisories),
        )


class CostEstimator:
    """Build a bounded, resume-aware preflight estimate without dump writes."""

    def __init__(
        self,
        options: EstimateOptions | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        logger: logging.Logger | None = None,
    ) -> None:
        self._options = options or EstimateOptions()
        if not isinstance(self._options, EstimateOptions):
            raise TypeError("options must be an EstimateOptions")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._logger = logger or get_logger(__name__)

    def estimate(
        self,
        config: ResolvedConfig,
        plan_builder: PlanBuilder,
        sample_source: SampleSource,
        adapter: ModelAdapter,
        *,
        resume_index: ResumeIndex | None = None,
        region_resolver: RegionResolver | None = None,
        perturbator: Perturbator | None = None,
    ) -> EstimateReport:
        """Measure bounded work and extrapolate the remaining audit cost."""

        _validate_provenance(config, adapter)
        all_samples = tuple(plan_builder.enumerate_clean())
        all_chunks = tuple(plan_builder.enumerate())
        if resume_index is None:
            pending_samples = all_samples
            pending_chunks = all_chunks
        else:
            pending_samples = resume_index.pending_clean_samples(
                all_samples,
                retry_failed=config.runtime.retry_failed,
            )
            pending_chunks = resume_index.pending_chunks(
                all_chunks,
                retry_failed=config.runtime.retry_failed,
            )

        total_items = sum(chunk.n_items for chunk in all_chunks)
        pending_items = sum(chunk.n_items for chunk in pending_chunks)
        no_pending_work = not pending_samples and not pending_chunks
        sanity: SanityCheckResult | None = None
        profile: ProfileResult | None = None
        shared_batch_state = BatchSizeState(
            _initial_batch_size(config, adapter.describe())
        )

        if not no_pending_work:
            sanity = SanityCheck(
                max_samples=self._options.max_sanity_samples,
                minimum_accuracy=self._options.minimum_accuracy,
                clock=self._clock,
                logger=self._logger,
            ).run(
                config,
                all_samples,
                sample_source,
                adapter,
                batch_size_state=shared_batch_state,
            )
            if pending_chunks:
                profile = self._profile_perturbed(
                    config,
                    plan_builder,
                    sample_source,
                    adapter,
                    pending_chunks,
                    batch_size_state=shared_batch_state,
                    region_resolver=region_resolver,
                    perturbator=perturbator,
                )

        class_count = _resolve_class_count(adapter.describe(), sanity, profile)
        if not no_pending_work and class_count is None:
            raise EstimationError(
                "class count is unavailable from AdapterSpec and successful outputs"
            )

        estimated_seconds = 0.0
        inference_calls = 0
        if pending_samples:
            if sanity is None or sanity.terminal_samples == 0:
                raise EstimationError("clean throughput is unavailable")
            estimated_seconds += len(pending_samples) / sanity.items_per_second
            inference_calls += math.ceil(
                sanity.inference_calls
                * len(pending_samples)
                / sanity.terminal_samples
            )
        if pending_chunks:
            if profile is None or profile.terminal_items == 0:
                raise EstimationError("perturbed throughput is unavailable")
            estimated_seconds += pending_items / profile.items_per_second
            inference_calls += math.ceil(
                profile.inference_calls * pending_items / profile.terminal_items
            )

        total_dump = (
            None
            if class_count is None
            else _estimate_dump_bytes(
                len(all_samples),
                total_items,
                class_count,
                self._options.dump_size,
                include_manifest=True,
            )
        )
        remaining_dump = (
            0
            if no_pending_work
            else _estimate_dump_bytes(
                len(pending_samples),
                pending_items,
                class_count,
                self._options.dump_size,
                include_manifest=False,
            )
        )
        exceedances = _find_exceedances(
            pending_items,
            estimated_seconds,
            remaining_dump,
            self._options.limits,
        )
        advisories = list(sanity.advisories if sanity is not None else ())
        if profile is not None:
            advisories.extend(profile.advisories)
        advisories.extend(
            Advisory(
                AdvisoryCode.LIMIT_EXCEEDED,
                f"Estimated {item.kind.value} exceeds its configured limit; "
                "reduce the sample fraction, region/control count, or seed_salts.",
            )
            for item in exceedances
        )

        recommended_variants = _recommended_variants_per_chunk(
            config.runtime.variants_per_chunk,
            profile,
            self._options.prepared_chunk_budget_bytes,
        )
        if profile is not None and (
            profile.max_prepared_chunk_bytes
            > self._options.prepared_chunk_budget_bytes
        ):
            advisories.append(
                Advisory(
                    AdvisoryCode.PREPARED_CHUNK_MEMORY_HIGH,
                    "Observed prepared chunk memory exceeds the configured budget.",
                )
            )
        sample_fraction = (
            None
            if not exceedances
            else max(0.0, min(1.0, *(item.allowed_fraction for item in exceedances)))
        )
        if exceedances:
            self._logger.warning(
                "estimate.confirmation_required limits=%s",
                ",".join(item.kind.value for item in exceedances),
            )

        sanity_requires_confirmation = (
            sanity is not None
            and self._options.minimum_accuracy is not None
            and sanity.passed is False
        )
        recommendations: list[str] = []
        if sample_fraction is not None:
            recommendations.extend(
                (
                    f"Use at most {sample_fraction:.6f} of the pending samples.",
                    "Reduce region instances, controls, or perturbation seed_salts.",
                )
            )
        if recommended_variants is not None:
            recommendations.append(
                f"Set variants_per_chunk to at most {recommended_variants}."
            )
        return EstimateReport(
            total_clean_samples=len(all_samples),
            pending_clean_samples=len(pending_samples),
            total_chunks=len(all_chunks),
            pending_chunks=len(pending_chunks),
            total_perturbed_items=total_items,
            pending_perturbed_items=pending_items,
            class_count=class_count,
            estimated_inference_calls=inference_calls,
            estimated_remaining_seconds=estimated_seconds,
            estimated_total_dump_bytes=total_dump,
            estimated_remaining_dump_bytes=remaining_dump,
            profile=profile,
            sanity=sanity,
            options=self._options,
            limits=self._options.limits,
            dump_size_assumptions=self._options.dump_size,
            exceedances=exceedances,
            advisories=tuple(advisories),
            confirmation_required=bool(exceedances) or sanity_requires_confirmation,
            recommended_sample_fraction=sample_fraction,
            recommended_variants_per_chunk=recommended_variants,
            recommendations=tuple(recommendations),
        )

    def _profile_perturbed(
        self,
        config: ResolvedConfig,
        plan_builder: PlanBuilder,
        sample_source: SampleSource,
        adapter: ModelAdapter,
        pending_chunks: tuple[WorkChunkMeta, ...],
        *,
        batch_size_state: BatchSizeState,
        region_resolver: RegionResolver | None,
        perturbator: Perturbator | None,
    ) -> ProfileResult:
        selected = _select_evenly(
            pending_chunks,
            self._options.max_profile_chunks,
        )
        selected_items = sum(chunk.n_items for chunk in selected)
        accumulator = _ProfileAccumulator(Counter())
        batch_state = batch_size_state
        starting_batch_size = batch_state.current_size
        starting_oom_events = batch_state.oom_events
        counting_adapter = _CountingAdapter(adapter)
        splitter = BatchSplitter(counting_adapter, batch_state)
        class_count: int | None = None
        successful = 0

        started = self._clock()
        try:
            items = _iter_profile_items(
                selected,
                config,
                plan_builder,
                sample_source,
                adapter,
                accumulator,
                region_resolver=region_resolver,
                perturbator=perturbator,
            )
            for batch in Rebatcher(items, batch_state):
                for result in splitter.predict(batch):
                    accumulator.terminal_items += 1
                    accumulator.status_counts[result.status] += 1
                    if result.status is ItemStatus.OK and result.output is not None:
                        class_count = _merge_class_count(
                            class_count,
                            int(result.output.logits.size),
                        )
                        successful += 1
        except EstimationError:
            raise
        except Exception as error:
            raise EstimationError("perturbed profile execution failed") from error
        elapsed = _elapsed(self._clock() - started)
        if accumulator.terminal_items != selected_items:
            raise EstimationError("profile did not produce one terminal result per item")
        if successful == 0 or class_count is None:
            raise EstimationError("perturbed profile produced no successful inference")

        advisories: tuple[Advisory, ...] = ()
        if successful != accumulator.terminal_items:
            advisories = (
                Advisory(
                    AdvisoryCode.PROFILE_PARTIAL_FAILURES,
                    "Perturbed profile contained terminal item failures.",
                ),
            )
        return ProfileResult(
            selected_chunks=len(selected),
            selected_items=selected_items,
            terminal_items=accumulator.terminal_items,
            successful_predictions=successful,
            status_counts=_complete_counts(accumulator.status_counts),
            elapsed_seconds=elapsed,
            items_per_second=accumulator.terminal_items / elapsed,
            inference_calls=counting_adapter.inference_calls,
            class_count=class_count,
            oom_events=batch_state.oom_events - starting_oom_events,
            initial_batch_size=starting_batch_size,
            final_batch_size=batch_state.current_size,
            max_prepared_item_bytes=accumulator.max_prepared_item_bytes,
            max_prepared_chunk_bytes=accumulator.max_prepared_chunk_bytes,
            advisories=advisories,
        )


def _iter_profile_items(
    chunks: tuple[WorkChunkMeta, ...],
    config: ResolvedConfig,
    plan_builder: PlanBuilder,
    sample_source: SampleSource,
    adapter: ModelAdapter,
    accumulator: _ProfileAccumulator,
    *,
    region_resolver: RegionResolver | None,
    perturbator: Perturbator | None,
) -> Iterable[PerturbedInferenceItem]:
    processor = ChunkProcessor(
        chunks,
        plan_builder,
        sample_source,
        config.runtime.global_seed,
        fail_fast=False,
        region_resolver=region_resolver,
        perturbator=perturbator,
    )
    for result in iter_worker_results(
        processor,
        num_workers=config.runtime.num_workers,
    ):
        if not isinstance(result, (PreparedChunk, FailedChunk)):
            raise EstimationError("chunk worker returned an unsupported value")
        chunk = plan_builder.materialize(result.chunk_id)
        if isinstance(result, FailedChunk):
            expected = tuple(item.item_id for item in chunk.items)
            if result.item_ids != expected or result.reason is not ItemStatus.LOAD_FAILED:
                raise EstimationError("failed chunk does not match materialized work")
            accumulator.terminal_items += len(result.item_ids)
            accumulator.status_counts[result.reason] += len(result.item_ids)
            continue

        work_items = _validate_prepared(result, chunk)
        accumulator.terminal_items += len(result.failed_items)
        for failed in result.failed_items:
            accumulator.status_counts[failed.status] += 1
        chunk_bytes = int(result.arrays.nbytes + result.masks.nbytes)
        accumulator.max_prepared_chunk_bytes = max(
            accumulator.max_prepared_chunk_bytes,
            chunk_bytes,
        )
        for index, meta in enumerate(result.item_metas):
            item = work_items[meta.item_id]
            array = result.arrays[index]
            mask = result.masks[index]
            item_bytes = int(array.nbytes + mask.nbytes)
            accumulator.max_prepared_item_bytes = max(
                accumulator.max_prepared_item_bytes,
                item_bytes,
            )
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
                accumulator.terminal_items += 1
                accumulator.status_counts[ItemStatus.PREPARE_FAILED] += 1
                continue
            if meta.region_meta is None:
                raise EstimationError("prepared item lacks region metadata")
            yield PerturbedInferenceItem(
                work_item=item,
                array=array,
                mask=mask,
                region_meta=meta.region_meta,
                seed_used=derive(
                    config.runtime.global_seed,
                    item.item_id,
                    item.seed_salt,
                ),
                effective_area_px=effective_area,
            )


def _validate_prepared(
    result: PreparedChunk,
    chunk: WorkChunk,
) -> dict[str, WorkItem]:
    work_items = {item.item_id: item for item in chunk.items}
    failed_ids = {meta.item_id for meta in result.failed_items}
    expected = tuple(item.item_id for item in chunk.items)
    successful = tuple(meta.item_id for meta in result.item_metas)
    failed = tuple(meta.item_id for meta in result.failed_items)
    if len(failed_ids) != len(failed) or any(item_id not in work_items for item_id in failed):
        raise EstimationError("prepared chunk contains duplicate or unknown failures")
    if successful != tuple(item_id for item_id in expected if item_id not in failed_ids):
        raise EstimationError("prepared chunk successful item order is invalid")
    if failed != tuple(item_id for item_id in expected if item_id in failed_ids):
        raise EstimationError("prepared chunk failed item order is invalid")
    return work_items


def _select_evenly(values: Sequence[T], limit: int) -> tuple[T, ...]:
    """Select deterministic representatives spanning the full sequence."""

    items = tuple(values)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if len(items) <= limit:
        return items
    if limit == 1:
        return (items[(len(items) - 1) // 2],)
    indices = [round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)]
    return tuple(items[index] for index in indices)


def _initial_batch_size(config: ResolvedConfig, spec: AdapterSpec) -> int:
    return (
        config.runtime.target_batch_size
        if spec.max_batch_size is None
        else min(config.runtime.target_batch_size, spec.max_batch_size)
    )


def _validate_provenance(config: ResolvedConfig, adapter: ModelAdapter) -> None:
    if not isinstance(config, ResolvedConfig):
        raise TypeError("config must be a ResolvedConfig")
    if not isinstance(adapter, ModelAdapter):
        raise TypeError("adapter must be a ModelAdapter")
    if adapter.describe() != config.adapter_spec:
        raise EstimationError("adapter spec does not match resolved config")


def _merge_class_count(current: int | None, observed: int) -> int:
    if observed <= 0:
        raise EstimationError("successful logits must contain at least one class")
    if current is not None and current != observed:
        raise EstimationError("successful logits class dimensions are inconsistent")
    return observed


def _resolve_class_count(
    spec: AdapterSpec,
    sanity: SanityCheckResult | None,
    profile: ProfileResult | None,
) -> int | None:
    values = []
    if spec.class_names is not None:
        values.append(len(spec.class_names))
    if sanity is not None and sanity.class_count is not None:
        values.append(sanity.class_count)
    if profile is not None:
        values.append(profile.class_count)
    if not values:
        return None
    if len(set(values)) != 1:
        raise EstimationError("AdapterSpec and observed logits class counts disagree")
    return values[0]


def _elapsed(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise EstimationError("profile clock must report positive finite elapsed time")
    return value


def _complete_counts(counts: Counter[ItemStatus]) -> dict[ItemStatus, int]:
    return {status: counts[status] for status in ItemStatus}


def _estimate_dump_bytes(
    clean_rows: int,
    perturbed_rows: int,
    class_count: int | None,
    assumptions: DumpSizeAssumptions,
    *,
    include_manifest: bool,
) -> int:
    if class_count is None:
        raise EstimationError("class count is required for dump size estimation")
    logits = assumptions.float_bytes * class_count
    raw_bytes = (
        clean_rows * (logits + assumptions.clean_row_overhead_bytes)
        + perturbed_rows
        * (
            logits
            + assumptions.perturbed_row_overhead_bytes
            + assumptions.index_row_overhead_bytes
        )
    )
    compressed = math.ceil(assumptions.compression_ratio * raw_bytes)
    return compressed + (assumptions.manifest_bytes if include_manifest else 0)


def _find_exceedances(
    pending_items: int,
    estimated_seconds: float,
    remaining_dump_bytes: int,
    limits: EstimationLimits,
) -> tuple[LimitExceedance, ...]:
    candidates = (
        (LimitKind.PENDING_ITEMS, float(pending_items), limits.max_pending_items),
        (
            LimitKind.ESTIMATED_SECONDS,
            estimated_seconds,
            limits.max_estimated_seconds,
        ),
        (
            LimitKind.REMAINING_DUMP_BYTES,
            float(remaining_dump_bytes),
            limits.max_remaining_dump_bytes,
        ),
    )
    return tuple(
        LimitExceedance(kind, estimated, float(limit))
        for kind, estimated, limit in candidates
        if limit is not None and estimated > limit
    )


def _recommended_variants_per_chunk(
    current: int,
    profile: ProfileResult | None,
    budget_bytes: int,
) -> int | None:
    if (
        profile is None
        or profile.max_prepared_chunk_bytes <= budget_bytes
        or profile.max_prepared_item_bytes <= 0
    ):
        return None
    capacity = max(1, budget_bytes // profile.max_prepared_item_bytes)
    recommended = min(current, capacity)
    return recommended if recommended < current else None
