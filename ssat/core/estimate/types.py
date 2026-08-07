"""Define immutable public contracts for estimation and sanity checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

from ssat.core.types import ItemStatus

GIB = 1024**3
MIB = 1024**2


class AdvisoryCode(str, Enum):
    """Identify stable machine-readable preflight advisory categories.

    Attributes:
        PROFILE_PARTIAL_FAILURES: Perturbed profiling contained failures.
        SANITY_PARTIAL_FAILURES: Clean sanity checking contained failures.
        SANITY_NO_LABELED_OUTPUTS: No valid labeled output was measured.
        SANITY_ACCURACY_BELOW_MINIMUM: Accuracy missed the configured threshold.
        PREPARED_CHUNK_MEMORY_HIGH: Prepared memory exceeded its budget.
        LIMIT_EXCEEDED: An analytical confirmation limit was exceeded.
    """

    PROFILE_PARTIAL_FAILURES = "profile_partial_failures"
    SANITY_PARTIAL_FAILURES = "sanity_partial_failures"
    SANITY_NO_LABELED_OUTPUTS = "sanity_no_labeled_outputs"
    SANITY_ACCURACY_BELOW_MINIMUM = "sanity_accuracy_below_minimum"
    PREPARED_CHUNK_MEMORY_HIGH = "prepared_chunk_memory_high"
    LIMIT_EXCEEDED = "limit_exceeded"


class LimitKind(str, Enum):
    """Identify quantities that can require caller confirmation.

    Attributes:
        PENDING_ITEMS: Number of remaining perturbed items.
        ESTIMATED_SECONDS: Estimated remaining execution duration.
        REMAINING_DUMP_BYTES: Estimated bytes not yet persisted.
    """

    PENDING_ITEMS = "pending_items"
    ESTIMATED_SECONDS = "estimated_seconds"
    REMAINING_DUMP_BYTES = "remaining_dump_bytes"


@dataclass(frozen=True, slots=True)
class EstimationLimits:
    """Configure conservative thresholds for confirmation decisions.

    Attributes:
        max_pending_items: Optional maximum pending perturbation count.
        max_estimated_seconds: Optional maximum remaining duration.
        max_remaining_dump_bytes: Optional maximum remaining dump size.
    """

    max_pending_items: int | None = 1_000_000
    max_estimated_seconds: float | None = 24.0 * 60.0 * 60.0
    max_remaining_dump_bytes: int | None = 100 * GIB

    def __post_init__(self) -> None:
        """Validate positive optional confirmation limits.

        Raises:
            ValueError: If a configured limit is nonpositive or nonfinite.
        """

        for name, value in (
            ("max_pending_items", self.max_pending_items),
            ("max_remaining_dump_bytes", self.max_remaining_dump_bytes),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        seconds = self.max_estimated_seconds
        if seconds is not None and (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError(
                "max_estimated_seconds must be positive finite or None"
            )


@dataclass(frozen=True, slots=True)
class DumpSizeAssumptions:
    """Configure transparent analytical assumptions for dump sizing.

    Attributes:
        float_bytes: Bytes stored for each logit value.
        compression_ratio: Estimated compressed-to-raw byte ratio.
        clean_row_overhead_bytes: Non-logit bytes per clean row.
        perturbed_row_overhead_bytes: Non-logit bytes per perturbed row.
        index_row_overhead_bytes: Bytes per perturbed index row.
        manifest_bytes: Fixed manifest size included in total estimates.
    """

    float_bytes: int = 4
    compression_ratio: float = 0.6
    clean_row_overhead_bytes: int = 128
    perturbed_row_overhead_bytes: int = 384
    index_row_overhead_bytes: int = 96
    manifest_bytes: int = 16 * 1024

    def __post_init__(self) -> None:
        """Validate byte counts and compression ratio.

        Raises:
            ValueError: If a byte count or compression ratio is invalid.
        """

        for name, value in (
            ("float_bytes", self.float_bytes),
            ("clean_row_overhead_bytes", self.clean_row_overhead_bytes),
            ("perturbed_row_overhead_bytes", self.perturbed_row_overhead_bytes),
            ("index_row_overhead_bytes", self.index_row_overhead_bytes),
            ("manifest_bytes", self.manifest_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.float_bytes == 0:
            raise ValueError("float_bytes must be positive")
        if (
            isinstance(self.compression_ratio, bool)
            or not isinstance(self.compression_ratio, (int, float))
            or not math.isfinite(self.compression_ratio)
            or not 0.0 < self.compression_ratio <= 1.0
        ):
            raise ValueError("compression_ratio must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class EstimateOptions:
    """Control bounded profiling without changing the audit configuration.

    Attributes:
        max_profile_chunks: Maximum perturbed chunks measured directly.
        max_sanity_samples: Maximum clean samples measured directly.
        minimum_accuracy: Optional inclusive clean top-1 threshold.
        prepared_chunk_budget_bytes: Desired maximum prepared chunk allocation.
        limits: Confirmation thresholds applied to final estimates.
        dump_size: Assumptions used to estimate compressed dump size.
    """

    max_profile_chunks: int = 20
    max_sanity_samples: int = 20
    minimum_accuracy: float | None = None
    prepared_chunk_budget_bytes: int = 64 * MIB
    limits: EstimationLimits = field(default_factory=EstimationLimits)
    dump_size: DumpSizeAssumptions = field(default_factory=DumpSizeAssumptions)

    def __post_init__(self) -> None:
        """Validate profiling bounds and nested option types.

        Raises:
            TypeError: If a nested option has an invalid type.
            ValueError: If a profiling bound or threshold is invalid.
        """

        for name, value in (
            ("max_profile_chunks", self.max_profile_chunks),
            ("max_sanity_samples", self.max_sanity_samples),
            ("prepared_chunk_budget_bytes", self.prepared_chunk_budget_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.limits, EstimationLimits):
            raise TypeError("limits must be an EstimationLimits")
        if not isinstance(self.dump_size, DumpSizeAssumptions):
            raise TypeError("dump_size must be a DumpSizeAssumptions")
        if self.minimum_accuracy is not None and (
            isinstance(self.minimum_accuracy, bool)
            or not isinstance(self.minimum_accuracy, (int, float))
            or not math.isfinite(self.minimum_accuracy)
            or not 0.0 <= self.minimum_accuracy <= 1.0
        ):
            raise ValueError("minimum_accuracy must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class Advisory:
    """Describe one stable warning or recommendation in a report.

    Attributes:
        code: Machine-readable advisory category.
        message: Human-readable explanation or recommendation.
    """

    code: AdvisoryCode
    message: str

    def __post_init__(self) -> None:
        """Validate advisory category and message.

        Raises:
            TypeError: If ``code`` is not an ``AdvisoryCode``.
            ValueError: If ``message`` is empty.
        """

        if not isinstance(self.code, AdvisoryCode):
            raise TypeError("advisory code must be an AdvisoryCode")
        if not self.message:
            raise ValueError("advisory message must not be empty")


@dataclass(frozen=True, slots=True)
class LimitExceedance:
    """Describe one estimate that is strictly greater than its limit.

    Attributes:
        kind: Quantity that exceeded its configured limit.
        estimated: Positive finite estimated value.
        limit: Positive finite configured threshold.
    """

    kind: LimitKind
    estimated: float
    limit: float

    def __post_init__(self) -> None:
        """Validate the exceedance category and positive values.

        Raises:
            TypeError: If ``kind`` is not a ``LimitKind``.
            ValueError: If an estimated or limit value is invalid.
        """

        if not isinstance(self.kind, LimitKind):
            raise TypeError("kind must be a LimitKind")
        if not math.isfinite(self.estimated) or self.estimated <= 0:
            raise ValueError("estimated must be positive and finite")
        if not math.isfinite(self.limit) or self.limit <= 0:
            raise ValueError("limit must be positive and finite")

    @property
    def allowed_fraction(self) -> float:
        """Return the fraction of estimated work permitted by the limit.

        Returns:
            The configured limit divided by the estimated value.
        """

        return self.limit / self.estimated


@dataclass(frozen=True, slots=True)
class SanityCheckResult:
    """Report clean throughput, validity, and optional top-1 accuracy.

    Attributes:
        selected_samples: Number of clean samples selected for measurement.
        terminal_samples: Number of samples with terminal outcomes.
        successful_predictions: Number of valid successful outputs.
        labeled_predictions: Number of valid labeled outputs.
        correct_predictions: Number of correct labeled top-1 outputs.
        unlabeled_predictions: Number of successful unlabeled outputs.
        invalid_label_predictions: Number of out-of-range labels.
        invalid_logit_predictions: Number of empty or nonfinite logits.
        status_counts: Complete terminal-status counts.
        elapsed_seconds: Measured end-to-end duration.
        items_per_second: Measured terminal throughput.
        inference_calls: Adapter calls including OOM retries.
        oom_events: OOM events observed during the sanity pass.
        initial_batch_size: Batch cap at the start of measurement.
        final_batch_size: Batch cap after adaptive OOM handling.
        class_count: Consistent measured logit dimension, when available.
        accuracy: Labeled top-1 accuracy, when available.
        minimum_accuracy: Configured accuracy threshold, when present.
        passed: Threshold outcome, or ``None`` when not evaluated.
        advisories: Stable warnings produced by the sanity pass.
    """

    selected_samples: int
    terminal_samples: int
    successful_predictions: int
    labeled_predictions: int
    correct_predictions: int
    unlabeled_predictions: int
    invalid_label_predictions: int
    invalid_logit_predictions: int
    status_counts: dict[ItemStatus, int]
    elapsed_seconds: float
    items_per_second: float
    inference_calls: int
    oom_events: int
    initial_batch_size: int
    final_batch_size: int
    class_count: int | None
    accuracy: float | None
    minimum_accuracy: float | None
    passed: bool | None
    advisories: tuple[Advisory, ...] = ()


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """Report perturbed end-to-end throughput without dump I/O.

    Attributes:
        selected_chunks: Number of representative chunks measured.
        selected_items: Number of planned items in selected chunks.
        terminal_items: Number of items with terminal outcomes.
        successful_predictions: Number of successful model outputs.
        status_counts: Complete terminal-status counts.
        elapsed_seconds: Measured end-to-end duration.
        items_per_second: Measured terminal throughput.
        inference_calls: Adapter calls including OOM retries.
        class_count: Consistent measured logit dimension.
        oom_events: OOM events observed during profiling.
        initial_batch_size: Batch cap at the start of profiling.
        final_batch_size: Batch cap after adaptive OOM handling.
        max_prepared_item_bytes: Largest prepared item allocation.
        max_prepared_chunk_bytes: Largest prepared chunk allocation.
        advisories: Stable warnings produced by profiling.
    """

    selected_chunks: int
    selected_items: int
    terminal_items: int
    successful_predictions: int
    status_counts: dict[ItemStatus, int]
    elapsed_seconds: float
    items_per_second: float
    inference_calls: int
    class_count: int
    oom_events: int
    initial_batch_size: int
    final_batch_size: int
    max_prepared_item_bytes: int
    max_prepared_chunk_bytes: int
    advisories: tuple[Advisory, ...] = ()


@dataclass(frozen=True, slots=True)
class EstimateReport:
    """Provide the complete preflight report consumed by callers.

    Attributes:
        total_clean_samples: Number of clean samples in the complete plan.
        pending_clean_samples: Number of resume-filtered clean samples.
        total_chunks: Number of chunks in the complete plan.
        pending_chunks: Number of resume-filtered chunks.
        total_perturbed_items: Number of perturbed items in the complete plan.
        pending_perturbed_items: Number of resume-filtered perturbed items.
        class_count: Resolved logit dimension, when available.
        estimated_inference_calls: Estimated remaining adapter calls.
        estimated_remaining_seconds: Estimated remaining execution duration.
        estimated_total_dump_bytes: Estimated final dump size, when available.
        estimated_remaining_dump_bytes: Estimated bytes not yet persisted.
        profile: Optional perturbed measurement details.
        sanity: Optional clean measurement details.
        options: Options used to build the report.
        limits: Confirmation limits copied from the options.
        dump_size_assumptions: Dump assumptions copied from the options.
        exceedances: Limits strictly exceeded by current estimates.
        advisories: Combined sanity, profile, memory, and limit warnings.
        confirmation_required: Whether a caller should confirm execution.
        recommended_sample_fraction: Suggested pending sample fraction.
        recommended_variants_per_chunk: Suggested smaller chunk size.
        recommendations: Human-readable configuration recommendations.
    """

    total_clean_samples: int
    pending_clean_samples: int
    total_chunks: int
    pending_chunks: int
    total_perturbed_items: int
    pending_perturbed_items: int
    class_count: int | None
    estimated_inference_calls: int
    estimated_remaining_seconds: float
    estimated_total_dump_bytes: int | None
    estimated_remaining_dump_bytes: int
    profile: ProfileResult | None
    sanity: SanityCheckResult | None
    options: EstimateOptions
    limits: EstimationLimits
    dump_size_assumptions: DumpSizeAssumptions
    exceedances: tuple[LimitExceedance, ...]
    advisories: tuple[Advisory, ...]
    confirmation_required: bool
    recommended_sample_fraction: float | None
    recommended_variants_per_chunk: int | None
    recommendations: tuple[str, ...]
