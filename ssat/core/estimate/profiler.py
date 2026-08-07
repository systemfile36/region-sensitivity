"""Bounded perturbed-work profiling for cost estimation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ssat.core.adapter.base import ModelAdapter
from ssat.core.config.schema import ResolvedConfig
from ssat.core.estimate.errors import EstimationError
from ssat.core.estimate.measurement import (
    _CountingAdapter,
    _complete_counts,
    _elapsed,
    _merge_class_count,
    _select_evenly,
)
from ssat.core.estimate.types import Advisory, AdvisoryCode, ProfileResult
from ssat.core.perturb.perturbator import Perturbator
from ssat.core.plan.builder import PlanBuilder
from ssat.core.plan.types import WorkChunkMeta
from ssat.core.region.resolver import RegionResolver
from ssat.core.runtime.pipeline import (
    iter_prediction_batches,
    iter_prepared_work_chunks,
)
from ssat.core.runtime.types import BatchSizeState, PerturbedInferenceItem
from ssat.core.source.base import SampleSource
from ssat.core.types import ItemStatus


@dataclass(slots=True)
class _ProfileAccumulator:
    """Accumulate terminal statuses and preparation memory observations.

    Attributes:
        status_counts: Sparse terminal-status counter.
        terminal_items: Number of items with terminal outcomes.
        max_prepared_item_bytes: Largest observed prepared item allocation.
        max_prepared_chunk_bytes: Largest observed prepared chunk allocation.
    """

    status_counts: Counter[ItemStatus]
    terminal_items: int = 0
    max_prepared_item_bytes: int = 0
    max_prepared_chunk_bytes: int = 0


class PerturbedProfiler:
    """Measure bounded perturbed execution without writing dump records.

    Args:
        max_chunks: Positive maximum number of chunks to profile.
        clock: Monotonic callable used to measure elapsed time.
    """

    def __init__(
        self,
        *,
        max_chunks: int,
        clock: Callable[[], float],
    ) -> None:
        """Initialize a bounded perturbed profiler.

        Args:
            max_chunks: Positive maximum number of chunks to profile.
            clock: Monotonic callable used for elapsed time.
        """

        self._max_chunks = max_chunks
        self._clock = clock

    def run(
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
        """Profile representative pending chunks through shared runtime flows.

        Args:
            config: Fully resolved audit configuration.
            plan_builder: Planner used to rematerialize selected chunks.
            sample_source: Source loaded by chunk workers.
            adapter: Model adapter measured by the profile.
            pending_chunks: Resume-filtered chunks eligible for profiling.
            batch_size_state: Adaptive state shared with the sanity pass.
            region_resolver: Optional resolver injected into chunk workers.
            perturbator: Optional perturbation facade injected into workers.

        Returns:
            Perturbed throughput, status, class-count, and memory observations.

        Raises:
            EstimationError: If execution fails or produces incomplete results.
        """

        selected = _select_evenly(pending_chunks, self._max_chunks)
        selected_items = sum(chunk.n_items for chunk in selected)
        accumulator = _ProfileAccumulator(Counter())
        starting_batch_size = batch_size_state.current_size
        starting_oom_events = batch_size_state.oom_events
        counting_adapter = _CountingAdapter(adapter)
        class_count: int | None = None
        successful = 0

        def prepared_items() -> Iterable[PerturbedInferenceItem]:
            """Collect preparation metrics and yield successful items.

            Yields:
                Perturbed items that reached model inference.
            """

            for prepared in iter_prepared_work_chunks(
                selected,
                plan_builder,
                sample_source,
                adapter,
                global_seed=config.runtime.global_seed,
                num_workers=config.runtime.num_workers,
                fail_fast=False,
                region_resolver=region_resolver,
                perturbator=perturbator,
            ):
                accumulator.max_prepared_chunk_bytes = max(
                    accumulator.max_prepared_chunk_bytes,
                    prepared.prepared_bytes,
                )
                accumulator.max_prepared_item_bytes = max(
                    accumulator.max_prepared_item_bytes,
                    prepared.max_item_bytes,
                )
                for failure in prepared.failures:
                    accumulator.terminal_items += 1
                    accumulator.status_counts[failure.status] += 1
                yield from prepared.items

        # Include loading, preparation, and inference in the measured duration.
        started = self._clock()
        try:
            for outcome in iter_prediction_batches(
                prepared_items(),
                counting_adapter,
                batch_size_state,
            ):
                if outcome.error is not None:
                    raise outcome.error
                for result in outcome.results:
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
            raise EstimationError(
                "profile did not produce one terminal result per item"
            )
        if successful == 0 or class_count is None:
            raise EstimationError(
                "perturbed profile produced no successful inference"
            )

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
            oom_events=batch_size_state.oom_events - starting_oom_events,
            initial_batch_size=starting_batch_size,
            final_batch_size=batch_size_state.current_size,
            max_prepared_item_bytes=accumulator.max_prepared_item_bytes,
            max_prepared_chunk_bytes=accumulator.max_prepared_chunk_bytes,
            advisories=advisories,
        )
