"""Pure analytical cost and limit calculations for estimate reports."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ssat.core.adapter.types import AdapterSpec
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


@dataclass(frozen=True, slots=True)
class EstimateInputs:
    """Collect measured facts required to build an estimate report.

    Attributes:
        total_clean_samples: Number of clean samples in the complete plan.
        pending_clean_samples: Number of resume-filtered clean samples.
        total_chunks: Number of chunks in the complete perturbation plan.
        pending_chunks: Number of resume-filtered perturbation chunks.
        total_perturbed_items: Number of items in the complete plan.
        pending_perturbed_items: Number of resume-filtered perturbation items.
        adapter_spec: Adapter metadata used to resolve class count.
        sanity: Optional clean measurement result.
        profile: Optional perturbed measurement result.
        options: User-selected estimation options.
        variants_per_chunk: Current runtime chunk-size setting.
    """

    total_clean_samples: int
    pending_clean_samples: int
    total_chunks: int
    pending_chunks: int
    total_perturbed_items: int
    pending_perturbed_items: int
    adapter_spec: AdapterSpec
    sanity: SanityCheckResult | None
    profile: ProfileResult | None
    options: EstimateOptions
    variants_per_chunk: int


def build_estimate_report(inputs: EstimateInputs) -> EstimateReport:
    """Combine plan counts and measurements into a final report.

    Args:
        inputs: Immutable counts, measurements, and analytical options.

    Returns:
        A complete resume-aware estimate report.

    Raises:
        EstimationError: If required throughput or class-count data is missing.
    """

    no_pending_work = (
        inputs.pending_clean_samples == 0 and inputs.pending_chunks == 0
    )
    class_count = _resolve_class_count(
        inputs.adapter_spec,
        inputs.sanity,
        inputs.profile,
    )
    if not no_pending_work and class_count is None:
        raise EstimationError(
            "class count is unavailable from AdapterSpec and successful outputs"
        )

    # Extrapolate each measured stream only over its resume-filtered work.
    estimated_seconds = 0.0
    inference_calls = 0
    if inputs.pending_clean_samples:
        sanity = inputs.sanity
        if sanity is None or sanity.terminal_samples == 0:
            raise EstimationError("clean throughput is unavailable")
        estimated_seconds += (
            inputs.pending_clean_samples / sanity.items_per_second
        )
        inference_calls += math.ceil(
            sanity.inference_calls
            * inputs.pending_clean_samples
            / sanity.terminal_samples
        )
    if inputs.pending_chunks:
        profile = inputs.profile
        if profile is None or profile.terminal_items == 0:
            raise EstimationError("perturbed throughput is unavailable")
        estimated_seconds += (
            inputs.pending_perturbed_items / profile.items_per_second
        )
        inference_calls += math.ceil(
            profile.inference_calls
            * inputs.pending_perturbed_items
            / profile.terminal_items
        )

    total_dump = (
        None
        if class_count is None
        else _estimate_dump_bytes(
            inputs.total_clean_samples,
            inputs.total_perturbed_items,
            class_count,
            inputs.options.dump_size,
            include_manifest=True,
        )
    )
    remaining_dump = (
        0
        if no_pending_work
        else _estimate_dump_bytes(
            inputs.pending_clean_samples,
            inputs.pending_perturbed_items,
            class_count,
            inputs.options.dump_size,
            include_manifest=False,
        )
    )
    exceedances = _find_exceedances(
        inputs.pending_perturbed_items,
        estimated_seconds,
        remaining_dump,
        inputs.options.limits,
    )
    advisories = list(inputs.sanity.advisories if inputs.sanity else ())
    if inputs.profile is not None:
        advisories.extend(inputs.profile.advisories)
    advisories.extend(
        Advisory(
            AdvisoryCode.LIMIT_EXCEEDED,
            f"Estimated {item.kind.value} exceeds its configured limit; "
            "reduce the sample fraction, region/control count, or seed_salts.",
        )
        for item in exceedances
    )

    recommended_variants = _recommended_variants_per_chunk(
        inputs.variants_per_chunk,
        inputs.profile,
        inputs.options.prepared_chunk_budget_bytes,
    )
    if inputs.profile is not None and (
        inputs.profile.max_prepared_chunk_bytes
        > inputs.options.prepared_chunk_budget_bytes
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
        else max(
            0.0,
            min(1.0, *(item.allowed_fraction for item in exceedances)),
        )
    )

    sanity_requires_confirmation = (
        inputs.sanity is not None
        and inputs.options.minimum_accuracy is not None
        and inputs.sanity.passed is False
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
        total_clean_samples=inputs.total_clean_samples,
        pending_clean_samples=inputs.pending_clean_samples,
        total_chunks=inputs.total_chunks,
        pending_chunks=inputs.pending_chunks,
        total_perturbed_items=inputs.total_perturbed_items,
        pending_perturbed_items=inputs.pending_perturbed_items,
        class_count=class_count,
        estimated_inference_calls=inference_calls,
        estimated_remaining_seconds=estimated_seconds,
        estimated_total_dump_bytes=total_dump,
        estimated_remaining_dump_bytes=remaining_dump,
        profile=inputs.profile,
        sanity=inputs.sanity,
        options=inputs.options,
        limits=inputs.options.limits,
        dump_size_assumptions=inputs.options.dump_size,
        exceedances=exceedances,
        advisories=tuple(advisories),
        confirmation_required=(
            bool(exceedances) or sanity_requires_confirmation
        ),
        recommended_sample_fraction=sample_fraction,
        recommended_variants_per_chunk=recommended_variants,
        recommendations=tuple(recommendations),
    )


def _resolve_class_count(
    spec: AdapterSpec,
    sanity: SanityCheckResult | None,
    profile: ProfileResult | None,
) -> int | None:
    """Resolve one consistent class count from metadata and observations.

    Args:
        spec: Adapter metadata with optional class names.
        sanity: Optional clean measurement result.
        profile: Optional perturbed measurement result.

    Returns:
        The consistent class count, or ``None`` when no source provides one.

    Raises:
        EstimationError: If available sources disagree.
    """

    values: list[int] = []
    if spec.class_names is not None:
        values.append(len(spec.class_names))
    if sanity is not None and sanity.class_count is not None:
        values.append(sanity.class_count)
    if profile is not None:
        values.append(profile.class_count)
    if not values:
        return None
    if len(set(values)) != 1:
        raise EstimationError(
            "AdapterSpec and observed logits class counts disagree"
        )
    return values[0]


def _estimate_dump_bytes(
    clean_rows: int,
    perturbed_rows: int,
    class_count: int | None,
    assumptions: DumpSizeAssumptions,
    *,
    include_manifest: bool,
) -> int:
    """Estimate compressed dump bytes from row-level assumptions.

    Args:
        clean_rows: Number of clean result rows.
        perturbed_rows: Number of perturbed result rows.
        class_count: Number of logits stored per successful row.
        assumptions: Row overhead and compression assumptions.
        include_manifest: Whether to include the fixed manifest estimate.

    Returns:
        Estimated compressed bytes.

    Raises:
        EstimationError: If class count is unavailable.
    """

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
    return compressed + (
        assumptions.manifest_bytes if include_manifest else 0
    )


def _find_exceedances(
    pending_items: int,
    estimated_seconds: float,
    remaining_dump_bytes: int,
    limits: EstimationLimits,
) -> tuple[LimitExceedance, ...]:
    """Return configured limits exceeded by current estimates.

    Args:
        pending_items: Number of pending perturbed work items.
        estimated_seconds: Estimated remaining execution duration.
        remaining_dump_bytes: Estimated remaining dump size.
        limits: Optional confirmation limits.

    Returns:
        Strictly exceeded limits in stable category order.
    """

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
    """Recommend a smaller chunk size from observed prepared memory.

    Args:
        current: Current variants-per-chunk setting.
        profile: Optional perturbed memory profile.
        budget_bytes: Maximum desired prepared chunk allocation.

    Returns:
        A smaller positive chunk size, or ``None`` when no change is needed.
    """

    if (
        profile is None
        or profile.max_prepared_chunk_bytes <= budget_bytes
        or profile.max_prepared_item_bytes <= 0
    ):
        return None
    capacity = max(1, budget_bytes // profile.max_prepared_item_bytes)
    recommended = min(current, capacity)
    return recommended if recommended < current else None
