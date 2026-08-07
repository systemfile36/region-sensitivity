"""Resume-aware orchestration for bounded execution cost estimation."""

from __future__ import annotations

from collections.abc import Callable
import logging
import time

from ssat.core.adapter.base import ModelAdapter
from ssat.core.config.schema import ResolvedConfig
from ssat.core.estimate.cost_model import (
    EstimateInputs,
    build_estimate_report,
)
from ssat.core.estimate.measurement import (
    _new_batch_size_state,
    _validate_provenance,
)
from ssat.core.estimate.profiler import PerturbedProfiler
from ssat.core.estimate.sanity import SanityCheck
from ssat.core.estimate.types import (
    EstimateOptions,
    EstimateReport,
    ProfileResult,
    SanityCheckResult,
)
from ssat.core.perturb.perturbator import Perturbator
from ssat.core.plan.builder import PlanBuilder
from ssat.core.region.resolver import RegionResolver
from ssat.core.resume.index import ResumeIndex
from ssat.core.source.base import SampleSource
from ssat.utils.logger_factory import get_logger

__all__ = ["CostEstimator", "SanityCheck"]


class CostEstimator:
    """Build a bounded, resume-aware preflight estimate without dump writes.

    Args:
        options: Optional limits and profiling bounds.
        clock: Monotonic callable shared by sanity and profile measurements.
        logger: Optional event logger for warnings.

    Raises:
        TypeError: If options or clock has an invalid type.
    """

    def __init__(
        self,
        options: EstimateOptions | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize a resume-aware cost estimator.

        Args:
            options: Optional limits and profiling bounds.
            clock: Monotonic callable shared by all measurements.
            logger: Optional event logger for warnings.

        Raises:
            TypeError: If options or clock has an invalid type.
        """

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
        """Measure bounded work and extrapolate the remaining audit cost.

        Args:
            config: Fully resolved audit configuration.
            plan_builder: Planner for clean samples and perturbation chunks.
            sample_source: Source used by measurement workers.
            adapter: Model adapter measured by sanity and profiling passes.
            resume_index: Optional index used to exclude completed work.
            region_resolver: Optional resolver injected into chunk workers.
            perturbator: Optional perturbation facade injected into workers.

        Returns:
            A complete cost, capacity, and confirmation report.

        Raises:
            TypeError: If provenance inputs have invalid types.
            EstimationError: If execution or analytical inputs are incomplete.
        """

        _validate_provenance(config, adapter)
        all_samples = tuple(plan_builder.enumerate_clean())
        all_chunks = tuple(plan_builder.enumerate())

        # Resolve pending work before measuring so completed runs avoid model calls.
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

        if not no_pending_work:
            shared_batch_state = _new_batch_size_state(
                config,
                adapter.describe(),
            )
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
                profile = PerturbedProfiler(
                    max_chunks=self._options.max_profile_chunks,
                    clock=self._clock,
                ).run(
                    config,
                    plan_builder,
                    sample_source,
                    adapter,
                    pending_chunks,
                    batch_size_state=shared_batch_state,
                    region_resolver=region_resolver,
                    perturbator=perturbator,
                )

        report = build_estimate_report(
            EstimateInputs(
                total_clean_samples=len(all_samples),
                pending_clean_samples=len(pending_samples),
                total_chunks=len(all_chunks),
                pending_chunks=len(pending_chunks),
                total_perturbed_items=total_items,
                pending_perturbed_items=pending_items,
                adapter_spec=adapter.describe(),
                sanity=sanity,
                profile=profile,
                options=self._options,
                variants_per_chunk=config.runtime.variants_per_chunk,
            )
        )
        if report.exceedances:
            self._logger.warning(
                "estimate.confirmation_required limits=%s",
                ",".join(item.kind.value for item in report.exceedances),
            )
        return report
