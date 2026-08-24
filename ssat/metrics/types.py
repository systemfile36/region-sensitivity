"""Long-form contract types shared across the metrics engine.

Every aggregate type in this module is long-form: one row per entity plus
one row per registered metric name (``(entity, metric_name)``). Adding a new
Metric therefore never changes any of these schemas, only the number of rows
produced — the same principle ItemMetrics itself is built on.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ssat.core.types import RegionKind


def region_key(region_id: str, region_instance_id: str) -> str:
    """Format the stable concrete-region identity used across ``ssat.metrics``.

    Shared by ``ssat.metrics.aggregate`` and ``ssat.metrics.viz.mask_check``,
    which previously each hand-wrote this same ``"{region_id}::{region_instance_id}"``
    format. The analogous ``ssat.analysis`` side keeps its own independent
    copy (``ssat.analysis.types.region_key``) rather than importing this one,
    per that package's documented dependency-direction boundary.

    Args:
        region_id: Region family identity.
        region_instance_id: Concrete instance identity within the family.

    Returns:
        The joined region key.
    """

    return f"{region_id}::{region_instance_id}"


class ExclusionReason(str, Enum):
    """Identify why a perturbed item's metric value could not be computed.

    Attributes:
        PERTURBED_STATUS_NOT_OK: The item's perturbed status was not "ok"
            while its clean counterpart succeeded. Clean-side failures
            remove the entire sample before it reaches
            ItemMetrics and are reported via DumpHandle.summary() instead —
            they never appear here.
        GT_LABEL_UNKNOWN: The item's sample has no ground-truth label
            (``gt_label is None`` — a scenario the core/source layer already
            supports explicitly for label-free, inference-only auditing,
            ``ssat/core/source/types.py``'s ``gt_label: int | None``). Every
            currently registered metric is defined relative to a reference
            class (accuracy/flip/margin all need to know which class is
            "correct"), so none can be computed for such an item — this is
            distinct from ``PERTURBED_STATUS_NOT_OK`` because the perturbed
            side may have succeeded perfectly; the gap is entirely on the
            clean/label side. Follows the same "never silently drop, always
            record a reason" principle used throughout this module.
    """

    PERTURBED_STATUS_NOT_OK = "perturbed_status_not_ok"
    GT_LABEL_UNKNOWN = "gt_label_unknown"


@dataclass(frozen=True, slots=True)
class RegionGeometryRef:
    """Carry the minimum recipe needed to re-resolve a region's mask.

    Attributes:
        region_kind: Mask materialization strategy.
        region_params_json: Canonical JSON parameters for procedural regions.
        ref: Resolved explicit-mask path, when applicable.
        ref_hash: Content hash for the explicit mask, when applicable.
    """

    region_kind: RegionKind
    region_params_json: str
    ref: str | None = None
    ref_hash: str | None = None

    def __post_init__(self) -> None:
        """Validate the region kind and its reference fields.

        Raises:
            TypeError: If ``region_kind`` is not a ``RegionKind``.
            ValueError: If ``region_params_json`` is empty, or ``ref``/
                ``ref_hash`` are present or absent inconsistently with kind.
        """

        if not isinstance(self.region_kind, RegionKind):
            raise TypeError("region_kind must be a RegionKind")
        if not self.region_params_json:
            raise ValueError("region_params_json must not be empty")
        if self.region_kind is RegionKind.EXPLICIT:
            if self.ref is None or self.ref_hash is None:
                raise ValueError("explicit regions require ref and ref_hash")
        elif self.ref is not None or self.ref_hash is not None:
            raise ValueError("procedural regions cannot define ref or ref_hash")


@dataclass(frozen=True, slots=True)
class ItemMetrics:
    """Carry one metric's clean/perturbed comparison for one work item.

    Attributes:
        item_id: Identity of the perturbed WorkItem this row measures.
        sample_id: Owning sample, denormalized for convenient grouping.
        metric_name: Registered Metric name this row was computed with.
        clean_correct: Whether the clean prediction matched ground truth;
            ``None`` only when ``excluded_reason`` is ``GT_LABEL_UNKNOWN`` —
            correctness cannot be judged without a reference class. Present
            (never ``None``) for every other row, including
            ``PERTURBED_STATUS_NOT_OK`` exclusions, since those still know
            the clean side's own correctness.
        value_clean: Raw clean-side metric value, when available.
        value_perturbed: Raw perturbed-side metric value, when available.
        degradation: Sign-normalized clean-minus-perturbed (or reverse)
            value; positive always means worse performance.
        available: Whether this metric could be computed for this item.
        excluded_reason: Why the value is missing, when unavailable.
    """

    item_id: str
    sample_id: str
    metric_name: str
    clean_correct: bool | None
    value_clean: float | None
    value_perturbed: float | None
    degradation: float | None
    available: bool
    excluded_reason: ExclusionReason | None = None

    def __post_init__(self) -> None:
        """Validate identity fields, the availability/value toggle, and clean_correct.

        Raises:
            ValueError: If an identity field is empty, metric values are
                present while unavailable/excluded, absent while healthy, or
                ``clean_correct`` is ``None`` without (or non-``None`` with)
                ``excluded_reason`` being ``GT_LABEL_UNKNOWN``.
        """

        if not self.item_id or not self.sample_id or not self.metric_name:
            raise ValueError("item_id, sample_id, and metric_name must not be empty")
        missing = not self.available or self.excluded_reason is not None
        values = (self.value_clean, self.value_perturbed, self.degradation)
        if missing and any(value is not None for value in values):
            raise ValueError(
                "unavailable or excluded rows must not carry metric values"
            )
        if not missing and any(value is None for value in values):
            raise ValueError(
                "available rows without an excluded_reason require all metric values"
            )
        is_gt_unknown = self.excluded_reason is ExclusionReason.GT_LABEL_UNKNOWN
        if is_gt_unknown and self.clean_correct is not None:
            raise ValueError("gt_label_unknown rows must have clean_correct=None")
        if not is_gt_unknown and self.clean_correct is None:
            raise ValueError(
                "clean_correct may only be None when excluded_reason is gt_label_unknown"
            )


@dataclass(frozen=True, slots=True)
class SampleMetrics:
    """Aggregate one metric's item-level values for one sample.

    Attributes:
        sample_id: Sample this row summarizes.
        metric_name: Registered Metric name this row was computed with.
        gt_label: Ground-truth class, when known.
        clean_correct: Whether the clean prediction matched ground truth,
            when gt_label is known.
        n_items: Number of perturbed items enumerated for this sample.
        n_valid: Number of items with an available, unexcluded value.
        flip_rate: Fraction of valid items that flipped, when this metric
            is binary; ``None`` for continuous metrics.
        vulnerability_score: Sample-level score from the configured primary
            metric, denormalized across every metric_name row.
        metric_mean: Mean degradation across valid items.
        metric_max: Maximum degradation across valid items.
        metric_std: Standard deviation of degradation across valid items.
    """

    sample_id: str
    metric_name: str
    gt_label: int | None
    clean_correct: bool | None
    n_items: int
    n_valid: int
    flip_rate: float | None
    vulnerability_score: float | None
    metric_mean: float | None
    metric_max: float | None
    metric_std: float | None

    def __post_init__(self) -> None:
        """Validate identity fields, item counts, and flip_rate range.

        Raises:
            ValueError: If an identity field is empty, counts are invalid,
                or flip_rate is outside the unit interval.
        """

        if not self.sample_id or not self.metric_name:
            raise ValueError("sample_id and metric_name must not be empty")
        _validate_counts(self.n_items, self.n_valid)
        _validate_unit_interval_or_none(self.flip_rate, field_name="flip_rate")


@dataclass(frozen=True, slots=True)
class RegionMetrics:
    """Aggregate one metric's item-level values for one region.

    Attributes:
        region_key: Stable identity of the concrete region (region_id +
            region_instance_id, joined as documented by the Aggregator).
        metric_name: Registered Metric name this row was computed with.
        region_kind: Mask materialization strategy.
        intended_area_px: Selected pixels before model preprocessing.
        effective_area_px: Selected pixels in model input space, when
            the adapter can report it.
        n_samples: Number of distinct samples contributing to this region.
        n_valid: Number of items with an available, unexcluded value.
        flip_rate: Fraction of valid items that flipped, when binary.
        metric_mean: Mean degradation across valid items.
    """

    region_key: str
    metric_name: str
    region_kind: RegionKind
    intended_area_px: int | None
    effective_area_px: int | None
    n_samples: int
    n_valid: int
    flip_rate: float | None
    metric_mean: float | None

    def __post_init__(self) -> None:
        """Validate identity fields, area fields, counts, and flip_rate.

        Raises:
            TypeError: If ``region_kind`` is not a ``RegionKind``.
            ValueError: If an identity field is empty, an area is negative,
                counts are invalid, or flip_rate is outside the unit interval.
        """

        if not self.region_key or not self.metric_name:
            raise ValueError("region_key and metric_name must not be empty")
        if not isinstance(self.region_kind, RegionKind):
            raise TypeError("region_kind must be a RegionKind")
        for name, value in (
            ("intended_area_px", self.intended_area_px),
            ("effective_area_px", self.effective_area_px),
        ):
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ValueError(f"{name} must be non-negative when provided")
        _validate_counts(self.n_samples, self.n_valid, total_field="n_samples")
        _validate_unit_interval_or_none(self.flip_rate, field_name="flip_rate")


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """Aggregate one metric's item-level values for one ground-truth class.

    Attributes:
        gt_label: Ground-truth class this row summarizes.
        metric_name: Registered Metric name this row was computed with.
        n_samples: Number of distinct samples with this ground-truth class.
        flip_rate: Fraction of valid items that flipped, when binary.
        metric_mean: Mean degradation across valid items.
    """

    gt_label: int
    metric_name: str
    n_samples: int
    flip_rate: float | None
    metric_mean: float | None

    def __post_init__(self) -> None:
        """Validate the metric name, sample count, and flip_rate range.

        Raises:
            ValueError: If metric_name is empty, n_samples is negative, or
                flip_rate is outside the unit interval.
        """

        if not self.metric_name:
            raise ValueError("metric_name must not be empty")
        if isinstance(self.n_samples, bool) or self.n_samples < 0:
            raise ValueError("n_samples must be non-negative")
        _validate_unit_interval_or_none(self.flip_rate, field_name="flip_rate")


@dataclass(frozen=True, slots=True)
class SpatialProfile:
    """Carry one metric's degradation for one sample-region pair.

    This is DebugViz's direct input: every row already carries enough
    region geometry to re-resolve a mask without a second dump join.

    Attributes:
        sample_id: Sample this row belongs to.
        region_key: Stable identity of the concrete region.
        metric_name: Registered Metric name this row was computed with.
        region_geometry_ref: Minimum recipe to re-resolve this region's mask.
        degradation: Sign-normalized degradation value, when available.
    """

    sample_id: str
    region_key: str
    metric_name: str
    region_geometry_ref: RegionGeometryRef
    degradation: float | None

    def __post_init__(self) -> None:
        """Validate identity fields and the geometry reference type.

        Raises:
            TypeError: If region_geometry_ref is not a RegionGeometryRef.
            ValueError: If an identity field is empty.
        """

        if not self.sample_id or not self.region_key or not self.metric_name:
            raise ValueError(
                "sample_id, region_key, and metric_name must not be empty"
            )
        if not isinstance(self.region_geometry_ref, RegionGeometryRef):
            raise TypeError("region_geometry_ref must be a RegionGeometryRef")


def _validate_counts(total: int, valid: int, *, total_field: str = "n_items") -> None:
    """Validate that a non-negative valid count does not exceed its total.

    Raises:
        ValueError: If either count is negative or valid exceeds total.
    """

    if isinstance(total, bool) or total < 0:
        raise ValueError(f"{total_field} must be non-negative")
    if isinstance(valid, bool) or valid < 0:
        raise ValueError("n_valid must be non-negative")
    if valid > total:
        raise ValueError(f"n_valid must not exceed {total_field}")


def _validate_unit_interval_or_none(value: float | None, *, field_name: str) -> None:
    """Validate that an optional fraction lies within [0, 1].

    Raises:
        ValueError: If value is provided and outside the unit interval.
    """

    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
