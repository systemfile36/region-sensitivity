"""Public execution loop connecting planning, workers, inference, and dumps."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
import logging

from ssat.core.adapter.base import ModelAdapter
from ssat.core.config.schema import ResolvedConfig
from ssat.core.dump.types import CleanDumpRecord, PerturbedDumpRecord
from ssat.core.dump.writer import DumpWriter
from ssat.core.perturb.perturbator import Perturbator
from ssat.core.plan.builder import PlanBuilder
from ssat.core.plan.types import WorkChunkMeta
from ssat.core.region.resolver import RegionResolver
from ssat.core.resume.index import ResumeIndex
from ssat.core.runtime.errors import RuntimeContractError, RuntimeExecutionError
from ssat.core.runtime.pipeline import (
    PreparationFailure,
    initial_batch_size,
    iter_clean_preparation_results,
    iter_prediction_batches,
    iter_prepared_work_chunks,
)
from ssat.core.runtime.types import (
    BatchSizeState,
    CleanInferenceItem,
    ExecutionSummary,
    InferenceItem,
    PerturbedInferenceItem,
    PredictionResult,
)
from ssat.core.source.base import SampleSource
from ssat.core.source.types import LoadError, SampleMeta
from ssat.core.types import ItemStatus
from ssat.utils.logger_factory import get_logger


class _RunState:
    """Accumulate records and status counts for one audit invocation.

    Args:
        writer: Dump writer that owns durable record persistence.
        model_id: Stable model identifier written to clean records.
    """

    def __init__(self, writer: DumpWriter, model_id: str) -> None:
        """Initialize mutable counters around an open dump writer.

        Args:
            writer: Dump writer that owns durable record persistence.
            model_id: Stable model identifier written to clean records.
        """

        self.writer = writer
        self.model_id = model_id
        self.counts: Counter[ItemStatus] = Counter()
        self.clean_records = 0
        self.perturbed_records = 0

    def write_clean(self, record: CleanDumpRecord) -> None:
        """Write and count one clean record.

        Args:
            record: Validated clean dump record.
        """

        self.writer.write_clean(record)
        self.clean_records += 1
        self.counts[record.status] += 1

    def write_perturbed(self, record: PerturbedDumpRecord) -> None:
        """Write and count one perturbed record.

        Args:
            record: Validated perturbed dump record.
        """

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
    """Run pending clean inference and perturbation work deterministically.

    The caller owns writer creation and closure. A successful invocation flushes
    durable fragments but deliberately leaves the writer open.

    Args:
        config: Fully resolved audit configuration.
        plan_builder: Planner for clean samples and perturbation chunks.
        sample_source: Source used by worker-side loaders.
        adapter: Model adapter used for mask transforms and prediction.
        writer: Open dump writer matching the resolved configuration.
        resume_index: Status index used to select pending work.
        region_resolver: Optional resolver injected into chunk workers.
        perturbator: Optional perturbation facade injected into chunk workers.
        logger: Optional runtime event logger.

    Returns:
        A summary of pending work, emitted records, and adaptive batching.

    Raises:
        TypeError: If the configuration or adapter has an invalid type.
        RuntimeExecutionError: If provenance validation or fail-fast execution
            fails.
        RuntimeContractError: If a worker, batch, or adapter violates a runtime
            contract.
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

    # Resume filtering defines the exact work set before any model call occurs.
    clean_samples = resume_index.pending_clean_samples(
        plan_builder.enumerate_clean(),
        retry_failed=runtime.retry_failed,
    )
    chunks = resume_index.pending_chunks(
        plan_builder.enumerate(),
        retry_failed=runtime.retry_failed,
    )
    batch_state = BatchSizeState(
        initial_batch_size(runtime.target_batch_size, adapter_spec)
    )
    state = _RunState(writer, adapter_spec.model_id)

    resolved_logger.info(
        "runtime.started clean=%d chunks=%d batch_size=%d workers=%d",
        len(clean_samples),
        len(chunks),
        batch_state.initial_size,
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
        clean_items,
        adapter,
        batch_state,
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
        perturbed_items,
        adapter,
        batch_state,
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
    """Convert shared clean preparation results into runtime policy actions.

    Args:
        samples: Ordered pending clean samples.
        sample_source: Source used by clean workers.
        state: Mutable runtime dump state.
        num_workers: Number of data-loading workers.
        fail_fast: Whether the first load failure aborts execution.

    Yields:
        Clean items ready for model inference.

    Raises:
        RuntimeExecutionError: If fail-fast handling aborts the run.
        RuntimeContractError: If the shared worker stream fails.
    """

    try:
        for result in iter_clean_preparation_results(
            samples,
            sample_source,
            num_workers=num_workers,
        ):
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
            yield result
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
    """Apply runtime failure policy to shared perturbed preparation results.

    Args:
        chunks: Ordered pending chunk metadata.
        plan_builder: Planner used to materialize complete chunks.
        sample_source: Source loaded by chunk workers.
        adapter: Adapter used for source-to-model mask transforms.
        state: Mutable runtime dump state.
        global_seed: Global deterministic seed.
        num_workers: Number of chunk-preparation workers.
        fail_fast: Whether the first preparation failure aborts execution.
        region_resolver: Optional resolver injected into chunk workers.
        perturbator: Optional perturbation facade injected into chunk workers.

    Yields:
        Perturbed items ready for model inference.

    Raises:
        RuntimeExecutionError: If fail-fast handling aborts the run.
        RuntimeContractError: If the shared worker stream fails.
    """

    try:
        for prepared in iter_prepared_work_chunks(
            chunks,
            plan_builder,
            sample_source,
            adapter,
            global_seed=global_seed,
            num_workers=num_workers,
            fail_fast=fail_fast,
            region_resolver=region_resolver,
            perturbator=perturbator,
        ):
            for failure in prepared.failures:
                _write_preparation_failure(failure, state)
            if prepared.failures:
                _raise_if_fail_fast(
                    fail_fast,
                    state,
                    prepared.failure_message or "chunk preparation failed",
                )
            yield from prepared.items
    except RuntimeExecutionError:
        raise
    except Exception as error:
        raise RuntimeContractError("chunk worker contract failed") from error


def _run_batches(
    items: Iterable[InferenceItem],
    adapter: ModelAdapter,
    batch_size_state: BatchSizeState,
    state: _RunState,
    *,
    fail_fast: bool,
) -> None:
    """Write prediction results while applying runtime fail-fast policy.

    Args:
        items: Clean or perturbed items ready for prediction.
        adapter: Model adapter used by the shared inference pipeline.
        batch_size_state: Mutable OOM-aware batch-size state.
        state: Mutable runtime dump state.
        fail_fast: Whether any prediction failure aborts execution.

    Raises:
        RuntimeExecutionError: If fail-fast handling aborts the run.
        RuntimeContractError: If the batch stream violates a contract.
    """

    try:
        for outcome in iter_prediction_batches(items, adapter, batch_size_state):
            if outcome.error is not None:
                for item in outcome.items:
                    _write_prediction_failure(item, state)
                _raise_if_fail_fast(
                    fail_fast,
                    state,
                    "adapter prediction failed",
                )
                continue
            for result in outcome.results:
                _write_prediction_result(result, state)
            if any(
                result.status is not ItemStatus.OK
                for result in outcome.results
            ):
                message = (
                    "singleton inference ran out of memory"
                    if any(
                        result.status is ItemStatus.SKIPPED_OOM
                        for result in outcome.results
                    )
                    else "adapter prediction failed"
                )
                _raise_if_fail_fast(fail_fast, state, message)
    except RuntimeExecutionError:
        raise
    except Exception as error:
        raise RuntimeContractError("runtime batch stream failed") from error


def _write_prediction_result(result: PredictionResult, state: _RunState) -> None:
    """Convert one shared prediction result into its dump representation.

    Args:
        result: Ordered prediction result from the shared pipeline.
        state: Mutable runtime dump state.
    """

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


def _write_preparation_failure(
    failure: PreparationFailure,
    state: _RunState,
) -> None:
    """Write one normalized preparation failure.

    Args:
        failure: Failure produced by the shared preparation pipeline.
        state: Mutable runtime dump state.
    """

    state.write_perturbed(
        PerturbedDumpRecord(
            work_item=failure.work_item,
            status=failure.status,
            seed_used=failure.seed_used,
            region_meta=failure.region_meta,
        )
    )


def _write_prediction_failure(item: InferenceItem, state: _RunState) -> None:
    """Write a failed prediction when a batch cannot produce results.

    Args:
        item: Clean or perturbed inference item submitted in the failed batch.
        state: Mutable runtime dump state.
    """

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


def _raise_if_fail_fast(
    fail_fast: bool,
    state: _RunState,
    message: str,
) -> None:
    """Flush durable state and raise when fail-fast mode is enabled.

    Args:
        fail_fast: Whether failure should abort the current run.
        state: Mutable runtime dump state.
        message: Stable execution error message.

    Raises:
        RuntimeExecutionError: If ``fail_fast`` is enabled.
    """

    if fail_fast:
        state.writer.flush()
        raise RuntimeExecutionError(message)
