"""Immutable public contracts for cost estimation and sanity checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

from ssat.core.types import ItemStatus

GIB = 1024**3
MIB = 1024**2


class AdvisoryCode(str, Enum):
    """Stable machine-readable preflight advisory categories."""

    PROFILE_PARTIAL_FAILURES = "profile_partial_failures"
    SANITY_PARTIAL_FAILURES = "sanity_partial_failures"
    SANITY_NO_LABELED_OUTPUTS = "sanity_no_labeled_outputs"
    SANITY_ACCURACY_BELOW_MINIMUM = "sanity_accuracy_below_minimum"
    PREPARED_CHUNK_MEMORY_HIGH = "prepared_chunk_memory_high"
    LIMIT_EXCEEDED = "limit_exceeded"


class LimitKind(str, Enum):
    """Quantities that can require caller confirmation."""

    PENDING_ITEMS = "pending_items"
    ESTIMATED_SECONDS = "estimated_seconds"
    REMAINING_DUMP_BYTES = "remaining_dump_bytes"


@dataclass(frozen=True, slots=True)
class EstimationLimits:
    """Conservative default limits used only for confirmation decisions."""

    max_pending_items: int | None = 1_000_000
    max_estimated_seconds: float | None = 24.0 * 60.0 * 60.0
    max_remaining_dump_bytes: int | None = 100 * GIB

    def __post_init__(self) -> None:
        for name, value in (
            ("max_pending_items", self.max_pending_items),
            ("max_remaining_dump_bytes", self.max_remaining_dump_bytes),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        seconds = self.max_estimated_seconds
        if seconds is not None and (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("max_estimated_seconds must be positive finite or None")


@dataclass(frozen=True, slots=True)
class DumpSizeAssumptions:
    """Transparent analytical assumptions for zstd Parquet sizing."""

    float_bytes: int = 4
    compression_ratio: float = 0.6
    clean_row_overhead_bytes: int = 128
    perturbed_row_overhead_bytes: int = 384
    index_row_overhead_bytes: int = 96
    manifest_bytes: int = 16 * 1024

    def __post_init__(self) -> None:
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
    """Control bounded profiling without changing the audit config."""

    max_profile_chunks: int = 20
    max_sanity_samples: int = 20
    minimum_accuracy: float | None = None
    prepared_chunk_budget_bytes: int = 64 * MIB
    limits: EstimationLimits = field(default_factory=EstimationLimits)
    dump_size: DumpSizeAssumptions = field(default_factory=DumpSizeAssumptions)

    def __post_init__(self) -> None:
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
    """One stable warning or recommendation attached to a report."""

    code: AdvisoryCode
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, AdvisoryCode):
            raise TypeError("advisory code must be an AdvisoryCode")
        if not self.message:
            raise ValueError("advisory message must not be empty")


@dataclass(frozen=True, slots=True)
class LimitExceedance:
    """Describe one estimate that is strictly greater than its limit."""

    kind: LimitKind
    estimated: float
    limit: float

    def __post_init__(self) -> None:
        if not isinstance(self.kind, LimitKind):
            raise TypeError("kind must be a LimitKind")
        if not math.isfinite(self.estimated) or self.estimated <= 0:
            raise ValueError("estimated must be positive and finite")
        if not math.isfinite(self.limit) or self.limit <= 0:
            raise ValueError("limit must be positive and finite")

    @property
    def allowed_fraction(self) -> float:
        return self.limit / self.estimated


@dataclass(frozen=True, slots=True)
class SanityCheckResult:
    """Clean throughput, output validity, and optional top-1 accuracy."""

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
    """Observed perturbed end-to-end throughput without dump I/O."""

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
    """Complete preflight report consumed by the future CLI."""

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
