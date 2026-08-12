"""Contract types for the control/stability analysis module.

``AnchorKey`` identifies what was measured (sample x region x invert_mask);
``ConditionKey`` identifies how it was measured (perturb op x params, with
the seed axis deliberately excluded — see CONTROL_STABILITY_DESIGN_v1.md §2
and IMPLE_PLAN_CONTROL_STABILITY_v1.md §1 항목4: the dump only stores
``seed_used``, a per-item hash that differs across regions even for the
same seed_salt, so seed cannot be part of a key that must compare equal
across different AnchorKeys — it becomes a repeat-count axis instead).

Every row type below maps to a parquet table produced by A7 AnalysisStore
(design §A7). This module has zero dependencies, including on
``ssat.core.types`` — string-typed fields such as ``perturb_op`` are kept as
plain ``str`` rather than ``PerturbationOp`` so that later modules can
depend on this one without pulling in the core package too
(IMPLE_PLAN_CONTROL_STABILITY_v1.md §3.3 "analysis.types → (없음)").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# --- Keys --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnchorKey:
    """Identify the measurement target: a sample's region under a mask polarity.

    Attributes:
        sample_id: Owning sample identity.
        region_key: Stable concrete-region identity, formatted as
            ``f"{region_id}::{region_instance_id}"`` — the same convention
            used by ``ssat.metrics.aggregate`` and
            ``ssat.metrics.viz.mask_check`` (intentionally duplicated here
            rather than extracted into a shared helper; see
            IMPLE_PLAN_CONTROL_STABILITY_v1.md §9).
        invert_mask: Whether the region was preserved (True) or removed
            (False) by the perturbation.
    """

    sample_id: str
    region_key: str
    invert_mask: bool

    def __post_init__(self) -> None:
        """Validate identity fields.

        Raises:
            ValueError: If sample_id or region_key is empty.
        """

        if not self.sample_id or not self.region_key:
            raise ValueError("sample_id and region_key must not be empty")


@dataclass(frozen=True, slots=True)
class ConditionKey:
    """Identify the measurement method: perturbation operator and parameters.

    Deliberately excludes seed (IMPLE_PLAN_CONTROL_STABILITY_v1.md §1
    항목4): the dump only stores ``seed_used``, a per-item hash that differs
    across regions even for the same seed_salt, so seed cannot be part of a
    key that must compare equal across different AnchorKeys. Seed instead
    becomes a repeat-count axis (``n_seeds``/``n_conditions``), not a key
    field.

    Attributes:
        perturb_op: Registered perturbation operator name, as stored
            verbatim in the dump's ``perturb_op`` column (``str``, not
            ``ssat.core.types.PerturbationOp`` — see module docstring).
        perturb_params_hash: ``hashlib.sha256(perturb_params_json).hexdigest()``
            of the dump's already-canonical ``perturb_params_json`` string
            (no re-serialization — computed by ``analysis.indexer``, not
            here; this type only checks non-emptiness, not hash format).
    """

    perturb_op: str
    perturb_params_hash: str

    def __post_init__(self) -> None:
        """Validate identity fields.

        Raises:
            ValueError: If perturb_op or perturb_params_hash is empty.
        """

        if not self.perturb_op or not self.perturb_params_hash:
            raise ValueError("perturb_op and perturb_params_hash must not be empty")


# --- Shared enums --------------------------------------------------------


class FlagValue(Enum):
    """Three-state flag distinguishing "no" from "not computed".

    A plain ``Enum`` (no str/int mixin) so every member is equally truthy —
    ``if flag:`` is unusable for any member, forcing explicit
    ``flag is FlagValue.TRUE`` comparisons everywhere. This enforces the
    design principle "unavailable을 false로 취급하지 않는다"
    (CONTROL_STABILITY_DESIGN_v1.md §A6): UNAVAILABLE must never be mistaken
    for FALSE.

    Attributes:
        TRUE: The condition was evaluated and holds.
        FALSE: The condition was evaluated and does not hold.
        UNAVAILABLE: The condition could not be evaluated (e.g. no control
            items were requested, or too few repeats existed) — distinct
            from FALSE.
    """

    TRUE = "true"
    FALSE = "false"
    UNAVAILABLE = "unavailable"


class ReliabilityGrade(str, Enum):
    """Overall reliability label assigned by A6 ReliabilityScorer.

    Attributes:
        HIGH: ``sign_consistent`` + ``exceeds_control`` + ``multi_strategy``
            + ``ci_excludes_zero`` all hold.
        MODERATE: Some of the above hold, no negative flags.
        LOW: Sign is consistent but the effect is marginal or conditions
            are insufficient.
        UNRELIABLE: ``sign_consistent`` is false — direction flips across
            conditions (design §A6: this is the grade that matters most).
    """

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNRELIABLE = "unreliable"


class MatchMethod(str, Enum):
    """How a control was paired with its target region (design §A1).

    Only used for successfully matched pairs — a control that could not be
    paired at all is not represented as a row (see ``ControlPairRow``); it
    is only reflected in ``CoverageReport.n_controls_unmatched``.

    Attributes:
        EXACT_REFERENCE: Paired via the ``target_region`` recipe embedded
            in the control's ``region_params_json`` (the normal-path
            outcome for every control the core actually produces today,
            per IMPLE_PLAN_CONTROL_STABILITY_v1.md §1 항목1).
        AREA_TOLERANCE: Paired via the ``area_match_tolerance`` fallback
            because no reference recipe was present or resolvable.
    """

    EXACT_REFERENCE = "exact_reference"
    AREA_TOLERANCE = "area_tolerance"


class Alignment(str, Enum):
    """Per-operator agreement between declared and empirical grouping.

    Attributes:
        ALIGNED: This op's declared attribute group and empirical cluster
            membership agree.
        PARTIAL: Partial overlap between declared and empirical grouping.
        DIVERGENT: The op's empirical cluster contradicts its declared
            grouping.
    """

    ALIGNED = "aligned"
    PARTIAL = "partial"
    DIVERGENT = "divergent"


# --- Row types -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnchorRow:
    """One AnchorKey with its enumerated condition axis (A1 AnchorTable).

    Attributes:
        anchor_key: This row's identity.
        sample_id: Denormalized from anchor_key, for flat parquet access.
        region_key: Denormalized from anchor_key.
        invert_mask: Denormalized from anchor_key.
        intended_area_px: Selected pixels before model preprocessing.
        effective_area_px: Selected pixels in model input space, when the
            adapter can report it.
        is_control: Whether this anchor is a random-area-match control.
        n_conditions: Number of distinct ConditionKeys observed for this
            anchor (the seed/strategy repeat count, per the redefined
            ConditionKey — §1 항목4).
        condition_keys: Every ConditionKey observed for this anchor.
    """

    anchor_key: AnchorKey
    sample_id: str
    region_key: str
    invert_mask: bool
    intended_area_px: int | None
    effective_area_px: int | None
    is_control: bool
    n_conditions: int
    condition_keys: tuple[ConditionKey, ...]

    def __post_init__(self) -> None:
        """Validate key/type consistency, areas, and condition counts.

        Raises:
            TypeError: If anchor_key or any condition_keys element has the
                wrong type.
            ValueError: If the denormalized fields disagree with
                anchor_key, an area is negative, n_conditions is negative,
                or n_conditions does not match len(condition_keys).
        """

        if not isinstance(self.anchor_key, AnchorKey):
            raise TypeError("anchor_key must be an AnchorKey")
        if (
            self.anchor_key.sample_id != self.sample_id
            or self.anchor_key.region_key != self.region_key
            or self.anchor_key.invert_mask != self.invert_mask
        ):
            raise ValueError(
                "sample_id, region_key, and invert_mask must match anchor_key"
            )
        for name, value in (
            ("intended_area_px", self.intended_area_px),
            ("effective_area_px", self.effective_area_px),
        ):
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ValueError(f"{name} must be non-negative when provided")
        if isinstance(self.n_conditions, bool) or self.n_conditions < 0:
            raise ValueError("n_conditions must be non-negative")
        if not all(isinstance(key, ConditionKey) for key in self.condition_keys):
            raise TypeError("condition_keys elements must be ConditionKey")
        if len(self.condition_keys) != self.n_conditions:
            raise ValueError("n_conditions must equal len(condition_keys)")


@dataclass(frozen=True, slots=True)
class ControlPairRow:
    """One target-control pairing under a shared ConditionKey (A1 ControlPairs).

    Attributes:
        target_anchor_key: The anchor being explained.
        control_anchor_key: The random-area-match control paired to it.
        condition_key: Shared perturbation condition (guaranteed equal on
            the normal path — target and control are enumerated from the
            same perturbation loop, per ``plan/builder.py:276-288``).
        match_method: How this pairing was established.
        area_match_ratio: control ``intended_area_px`` divided by target
            ``intended_area_px``.
    """

    target_anchor_key: AnchorKey
    control_anchor_key: AnchorKey
    condition_key: ConditionKey
    match_method: MatchMethod
    area_match_ratio: float | None

    def __post_init__(self) -> None:
        """Validate key types, distinctness, method, and ratio range.

        Raises:
            TypeError: If a key field or match_method has the wrong type.
            ValueError: If target_anchor_key equals control_anchor_key, or
                area_match_ratio is present and negative.
        """

        if not isinstance(self.target_anchor_key, AnchorKey) or not isinstance(
            self.control_anchor_key, AnchorKey
        ):
            raise TypeError("target_anchor_key and control_anchor_key must be AnchorKey")
        if not isinstance(self.condition_key, ConditionKey):
            raise TypeError("condition_key must be a ConditionKey")
        if not isinstance(self.match_method, MatchMethod):
            raise TypeError("match_method must be a MatchMethod")
        if self.target_anchor_key == self.control_anchor_key:
            raise ValueError("target_anchor_key and control_anchor_key must differ")
        if self.area_match_ratio is not None and self.area_match_ratio < 0.0:
            raise ValueError("area_match_ratio must be non-negative when provided")


@dataclass(frozen=True, slots=True)
class ControlComparisonRow:
    """Target-vs-control comparison for one (AnchorKey, ConditionKey, metric).

    A2 ControlComparator output.

    Attributes:
        target_anchor_key: Anchor being compared against its controls.
        condition_key: Condition this comparison was computed under.
        metric_name: Registered Metric name.
        control_available: Whether any control was matched for this
            target/condition — UNAVAILABLE (not FALSE) when zero controls
            matched (design §A2).
        area_matched: Whether area_match_ratio fell within tolerance;
            UNAVAILABLE whenever control_available is UNAVAILABLE, since
            area matching is meaningless without a matched control.
        control_mean: Mean degradation across matched controls.
        control_std: Standard deviation of degradation across matched
            controls.
        n_controls: Number of matched controls contributing.
        excess: target_degradation - control_mean.
        ratio: target_degradation / control_mean; None when
            abs(control_mean) is near zero.
        z_vs_control: (target - control_mean) / control_std; None when
            control_std is 0 or n_controls < 2.
    """

    target_anchor_key: AnchorKey
    condition_key: ConditionKey
    metric_name: str
    control_available: FlagValue
    area_matched: FlagValue
    control_mean: float | None
    control_std: float | None
    n_controls: int
    excess: float | None
    ratio: float | None
    z_vs_control: float | None

    def __post_init__(self) -> None:
        """Validate identity, flags, counts, and the unavailable-row cluster.

        Raises:
            TypeError: If a key/flag field has the wrong type.
            ValueError: If metric_name is empty, n_controls or control_std
                is negative, or control_available is not TRUE while any
                numeric field is non-None, n_controls is nonzero, or
                area_matched is not UNAVAILABLE.
        """

        if not self.metric_name:
            raise ValueError("metric_name must not be empty")
        if not isinstance(self.target_anchor_key, AnchorKey):
            raise TypeError("target_anchor_key must be an AnchorKey")
        if not isinstance(self.condition_key, ConditionKey):
            raise TypeError("condition_key must be a ConditionKey")
        if not isinstance(self.control_available, FlagValue) or not isinstance(
            self.area_matched, FlagValue
        ):
            raise TypeError("control_available and area_matched must be FlagValue")
        if isinstance(self.n_controls, bool) or self.n_controls < 0:
            raise ValueError("n_controls must be non-negative")
        if self.control_std is not None and self.control_std < 0.0:
            raise ValueError("control_std must be non-negative when provided")
        if self.control_available is not FlagValue.TRUE:
            unavailable_numeric = (
                self.control_mean,
                self.control_std,
                self.excess,
                self.ratio,
                self.z_vs_control,
            )
            if (
                any(value is not None for value in unavailable_numeric)
                or self.n_controls != 0
                or self.area_matched is not FlagValue.UNAVAILABLE
            ):
                raise ValueError(
                    "rows with control_available != TRUE must have null control "
                    "statistics, n_controls == 0, and area_matched == UNAVAILABLE"
                )


@dataclass(frozen=True, slots=True)
class SeedStabilityRow:
    """Repeat-trial stability for one (AnchorKey, ConditionKey, metric).

    A3(a) output — pure stochastic variation with op/params held fixed.

    Attributes:
        anchor_key: Anchor these repeats belong to.
        condition_key: Condition (op + params) held fixed across repeats.
        metric_name: Registered Metric name.
        seed_mean: Mean degradation across repeat trials.
        seed_std: Standard deviation of degradation across repeat trials.
        seed_cv: seed_std / abs(seed_mean); None when seed_mean is near
            zero.
        n_seeds: Number of repeat trials contributing.
    """

    anchor_key: AnchorKey
    condition_key: ConditionKey
    metric_name: str
    seed_mean: float | None
    seed_std: float | None
    seed_cv: float | None
    n_seeds: int

    def __post_init__(self) -> None:
        """Validate identity fields and non-negative counts/spreads.

        Raises:
            TypeError: If anchor_key or condition_key has the wrong type.
            ValueError: If metric_name is empty, or n_seeds/seed_std/
                seed_cv is negative.
        """

        if not self.metric_name:
            raise ValueError("metric_name must not be empty")
        if not isinstance(self.anchor_key, AnchorKey):
            raise TypeError("anchor_key must be an AnchorKey")
        if not isinstance(self.condition_key, ConditionKey):
            raise TypeError("condition_key must be a ConditionKey")
        if isinstance(self.n_seeds, bool) or self.n_seeds < 0:
            raise ValueError("n_seeds must be non-negative")
        for name, value in (("seed_std", self.seed_std), ("seed_cv", self.seed_cv)):
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative when provided")


@dataclass(frozen=True, slots=True)
class StrategyStabilityRow:
    """Per-anchor cross-operator stability for one (AnchorKey, metric).

    A3(c) per-anchor output. Deliberately preserves per-op values instead
    of averaging them away (design §A3(c): averaging hides sign flips).

    Attributes:
        anchor_key: Anchor these operators were compared on.
        metric_name: Registered Metric name.
        strategy_signs: Per-op degradation sign, one of {-1, 0, 1}, keyed
            by perturb_op.
        strategy_values: Per-op degradation value, keyed by perturb_op
            (same key set as strategy_signs).
        sign_agreement_ratio: Fraction of ops sharing the most common sign.
        n_strategies: Number of distinct ops contributing (size of the
            dicts above).
    """

    anchor_key: AnchorKey
    metric_name: str
    strategy_signs: dict[str, int]
    strategy_values: dict[str, float]
    sign_agreement_ratio: float
    n_strategies: int

    def __post_init__(self) -> None:
        """Validate identity, key-set consistency, signs, and ratio range.

        Raises:
            TypeError: If anchor_key has the wrong type.
            ValueError: If metric_name is empty, strategy_signs and
                strategy_values have different key sets, n_strategies
                disagrees with their size, a sign is outside {-1, 0, 1},
                or sign_agreement_ratio is outside [0, 1].
        """

        if not self.metric_name:
            raise ValueError("metric_name must not be empty")
        if not isinstance(self.anchor_key, AnchorKey):
            raise TypeError("anchor_key must be an AnchorKey")
        if set(self.strategy_signs) != set(self.strategy_values):
            raise ValueError("strategy_signs and strategy_values must share keys")
        if len(self.strategy_signs) != self.n_strategies:
            raise ValueError("n_strategies must equal len(strategy_signs)")
        if any(sign not in (-1, 0, 1) for sign in self.strategy_signs.values()):
            raise ValueError("strategy_signs values must be one of -1, 0, 1")
        _validate_unit_interval_or_none(
            self.sign_agreement_ratio, field_name="sign_agreement_ratio"
        )


@dataclass(frozen=True, slots=True)
class RankCorrelationRow:
    """Dataset-level rank correlation between two operators (A3(c) tail).

    Attributes:
        op_a: First perturb_op in the pair.
        op_b: Second perturb_op in the pair (op_a != op_b).
        spearman: Spearman correlation of per-region mean degradation ranks
            between op_a and op_b, over this row's scope.
        n_regions: Number of regions contributing to spearman.
        spearman_excl_top1: Same correlation excluding the top-k regions
            (design §A3(c): guards against one dominant signal inflating
            the correlation).
        scope: Free-form description of what population this correlation
            was computed over, e.g. ``"full_dataset"`` (kept as ``str``
            rather than a closed enum: v1 only implements the dataset-wide
            case — IMPLE_PLAN_CONTROL_STABILITY_v1.md §5 단계4 — so the
            full vocabulary is not yet fixed).
    """

    op_a: str
    op_b: str
    spearman: float | None
    n_regions: int
    spearman_excl_top1: float | None
    scope: str

    def __post_init__(self) -> None:
        """Validate identity fields, op distinctness, and value ranges.

        Raises:
            ValueError: If op_a/op_b/scope is empty, op_a equals op_b,
                n_regions is negative, or a correlation is outside [-1, 1].
        """

        if not self.op_a or not self.op_b or not self.scope:
            raise ValueError("op_a, op_b, and scope must not be empty")
        if self.op_a == self.op_b:
            raise ValueError("op_a and op_b must differ")
        if isinstance(self.n_regions, bool) or self.n_regions < 0:
            raise ValueError("n_regions must be non-negative")
        _validate_correlation_or_none(self.spearman, field_name="spearman")
        _validate_correlation_or_none(
            self.spearman_excl_top1, field_name="spearman_excl_top1"
        )


@dataclass(frozen=True, slots=True)
class StrategyProfileRow:
    """One operator's declared attributes, empirical grouping, and summary.

    A4 StrategyProfiler output, per perturb_op.

    Attributes:
        perturb_op: Operator name (``str``, not ``PerturbationOp`` — see
            module docstring; ``analysis.strategy_profile`` is the one
            module allowed to cross-reference
            ``ssat.core.types.PerturbationOp``, and only as a declarative-
            table lookup key, not by changing this field's type).
        preserves_statistics: Declared attribute — preserves color and
            brightness statistics.
        preserves_local_texture: Declared attribute — preserves local
            texture.
        is_global_operation: Declared attribute — operates on the full
            frame before compositing.
        cluster_id: Empirical cluster id from correlation-based grouping;
            None if not yet clustered (e.g. too few ops).
        mean_corr_within: Mean correlation to same-cluster ops.
        mean_corr_across: Mean correlation to other-cluster ops.
        alignment: Whether this op's declared group and empirical cluster
            agree — a per-op projection of the dataset-level alignment
            judgment defined in design §A4 (the dataset-level
            declared_groups/empirical_groups/agreement triple itself is
            out of scope for this module; see design note near
            ``Alignment``).
        mean_degradation_excl_top: Mean degradation excluding the top-k
            regions by magnitude.
        sign_ratio_positive: Fraction of anchors with positive degradation.
        n_anchors: Number of anchors contributing to this op's summary.
    """

    perturb_op: str
    preserves_statistics: bool
    preserves_local_texture: bool
    is_global_operation: bool
    cluster_id: int | None
    mean_corr_within: float | None
    mean_corr_across: float | None
    alignment: Alignment
    mean_degradation_excl_top: float | None
    sign_ratio_positive: float | None
    n_anchors: int

    def __post_init__(self) -> None:
        """Validate identity, alignment type, correlation ranges, and counts.

        Raises:
            TypeError: If alignment is not an Alignment.
            ValueError: If perturb_op is empty, cluster_id/n_anchors is
                negative, a correlation is outside [-1, 1], or
                sign_ratio_positive is outside [0, 1].
        """

        if not self.perturb_op:
            raise ValueError("perturb_op must not be empty")
        if not isinstance(self.alignment, Alignment):
            raise TypeError("alignment must be an Alignment")
        if self.cluster_id is not None and (
            isinstance(self.cluster_id, bool) or self.cluster_id < 0
        ):
            raise ValueError("cluster_id must be non-negative when provided")
        _validate_correlation_or_none(
            self.mean_corr_within, field_name="mean_corr_within"
        )
        _validate_correlation_or_none(
            self.mean_corr_across, field_name="mean_corr_across"
        )
        _validate_unit_interval_or_none(
            self.sign_ratio_positive, field_name="sign_ratio_positive"
        )
        if isinstance(self.n_anchors, bool) or self.n_anchors < 0:
            raise ValueError("n_anchors must be non-negative")


@dataclass(frozen=True, slots=True)
class IntervalRow:
    """Bootstrap confidence interval for one (region_key, metric) estimate.

    A5 IntervalEstimator output.

    Attributes:
        region_key: Region this interval covers.
        metric: Registered Metric name.
        point_estimate: Region-level point estimate (e.g. mean
            degradation).
        ci_low: Lower bound of the bootstrap interval.
        ci_high: Upper bound of the bootstrap interval.
        ci_method: Interval construction method, e.g. ``"percentile"``.
        n_bootstrap: Number of bootstrap resamples used.
        excludes_zero: Whether the interval excludes zero — must equal
            ``ci_low > 0 or ci_high < 0``.
    """

    region_key: str
    metric: str
    point_estimate: float
    ci_low: float
    ci_high: float
    ci_method: str
    n_bootstrap: int
    excludes_zero: bool

    def __post_init__(self) -> None:
        """Validate identity fields, bound ordering, and the zero-exclusion flag.

        Raises:
            ValueError: If region_key/metric/ci_method is empty, ci_low
                exceeds ci_high, n_bootstrap is negative, or excludes_zero
                disagrees with the actual bounds.
        """

        if not self.region_key or not self.metric or not self.ci_method:
            raise ValueError("region_key, metric, and ci_method must not be empty")
        if self.ci_low > self.ci_high:
            raise ValueError("ci_low must not exceed ci_high")
        if isinstance(self.n_bootstrap, bool) or self.n_bootstrap < 0:
            raise ValueError("n_bootstrap must be non-negative")
        expected_excludes_zero = self.ci_low > 0.0 or self.ci_high < 0.0
        if self.excludes_zero != expected_excludes_zero:
            raise ValueError("excludes_zero must equal (ci_low > 0 or ci_high < 0)")


@dataclass(frozen=True, slots=True)
class ReliabilityRow:
    """Reliability verdict for one (AnchorKey, metric): flags + grade + why.

    A6 ReliabilityScorer output.

    Attributes:
        anchor_key: Anchor this verdict covers.
        metric_name: Registered Metric name.
        sign_consistent: Same degradation sign across all conditions.
        exceeds_control: z_vs_control exceeds the configured threshold.
        seed_stable: Seed coefficient of variation below threshold.
        jitter_stable: Always UNAVAILABLE in v1 (core has no jitter
            support — see ``analysis.stability``'s jitter stub).
        multi_strategy: Reproduced in 2 or more operators.
        ci_excludes_zero: Bootstrap CI excludes zero.
        area_matched: Control area match within tolerance.
        reliability_grade: Overall verdict.
        reliability_reasons: Human-readable justifications for the grade.
    """

    anchor_key: AnchorKey
    metric_name: str
    sign_consistent: FlagValue
    exceeds_control: FlagValue
    seed_stable: FlagValue
    jitter_stable: FlagValue
    multi_strategy: FlagValue
    ci_excludes_zero: FlagValue
    area_matched: FlagValue
    reliability_grade: ReliabilityGrade
    reliability_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate identity, flag types, and grade type.

        Raises:
            TypeError: If anchor_key, a flag field, or reliability_grade
                has the wrong type.
            ValueError: If metric_name is empty.
        """

        if not self.metric_name:
            raise ValueError("metric_name must not be empty")
        if not isinstance(self.anchor_key, AnchorKey):
            raise TypeError("anchor_key must be an AnchorKey")
        for name, value in (
            ("sign_consistent", self.sign_consistent),
            ("exceeds_control", self.exceeds_control),
            ("seed_stable", self.seed_stable),
            ("jitter_stable", self.jitter_stable),
            ("multi_strategy", self.multi_strategy),
            ("ci_excludes_zero", self.ci_excludes_zero),
            ("area_matched", self.area_matched),
        ):
            if not isinstance(value, FlagValue):
                raise TypeError(f"{name} must be a FlagValue")
        if not isinstance(self.reliability_grade, ReliabilityGrade):
            raise TypeError("reliability_grade must be a ReliabilityGrade")


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Dataset-wide accounting of skipped/incomplete combinations (A1 output).

    Attributes:
        n_anchors: Total AnchorKeys enumerated.
        n_conditions_insufficient: AnchorKeys with n_conditions < 2 (design
            §2 "불완전 조합의 처리").
        n_controls_unmatched: Control items that could not be paired to a
            target (exact-reference and area-tolerance both failed).
        n_area_mismatch_warnings: Matched control pairs whose
            area_match_ratio fell outside tolerance.
    """

    n_anchors: int
    n_conditions_insufficient: int
    n_controls_unmatched: int
    n_area_mismatch_warnings: int

    def __post_init__(self) -> None:
        """Validate non-negative counts and the insufficient/anchors bound.

        Raises:
            ValueError: If any count is negative, or
                n_conditions_insufficient exceeds n_anchors.
        """

        _validate_counts(
            self.n_anchors, self.n_conditions_insufficient, total_field="n_anchors"
        )
        for name, value in (
            ("n_controls_unmatched", self.n_controls_unmatched),
            ("n_area_mismatch_warnings", self.n_area_mismatch_warnings),
        ):
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be non-negative")


# --- Reader availability report (A0 output) --------------------------------


@dataclass(frozen=True, slots=True)
class AvailableAnalyses:
    """Report which stability/control analyses one dump+metrics pair supports.

    Produced by ``analysis.reader.AnalysisReader.available_analyses()``
    (design §A0, IMPLE_PLAN_CONTROL_STABILITY_v1.md §5 단계1). Every field
    is a plain ``bool`` — validated explicitly here because pandas/numpy
    reductions (``Series.any()``, comparisons on numpy scalars) produce
    ``numpy.bool_``, which is not a ``bool`` subclass; a caller that forgot
    to wrap one in ``bool(...)`` should fail loudly here rather than
    silently leak a numpy scalar into what is documented as a pure Python
    contract type.

    Attributes:
        control_comparison: True if the item context contains at least one
            ``is_control=True`` row.
        fill_strategy_stability: True if non-control items exercise two or
            more distinct ``perturb_op`` values.
        seed_stability: True if any group of items sharing
            ``(AnchorKey, ConditionKey)`` has size >= 2 — i.e. at least one
            anchor/condition pair was run under more than one seed.
        jitter_stability: Always False. The core has no jitter axis at all
            (IMPLE_PLAN_CONTROL_STABILITY_v1.md §1 항목2) — kept as an
            explicit fourth field, rather than omitted, so every consumer
            gets one uniform four-key availability report, and so a future
            core jitter feature has a single field to flip.
    """

    control_comparison: bool
    fill_strategy_stability: bool
    seed_stability: bool
    jitter_stability: bool

    def __post_init__(self) -> None:
        """Validate every field is a genuine bool, not a numpy/int lookalike.

        Raises:
            TypeError: If any field is not exactly a bool.
        """

        for name in (
            "control_comparison",
            "fill_strategy_stability",
            "seed_stability",
            "jitter_stability",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")


# --- Shared private validators -------------------------------------------


def _validate_counts(total: int, valid: int, *, total_field: str = "n_items") -> None:
    """Validate that a non-negative valid count does not exceed its total.

    Raises:
        ValueError: If either count is negative or valid exceeds total.
    """

    if isinstance(total, bool) or total < 0:
        raise ValueError(f"{total_field} must be non-negative")
    if isinstance(valid, bool) or valid < 0:
        raise ValueError("n_conditions_insufficient must be non-negative")
    if valid > total:
        raise ValueError(f"n_conditions_insufficient must not exceed {total_field}")


def _validate_unit_interval_or_none(value: float | None, *, field_name: str) -> None:
    """Validate that an optional fraction lies within [0, 1].

    Raises:
        ValueError: If value is provided and outside the unit interval.
    """

    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")


def _validate_correlation_or_none(value: float | None, *, field_name: str) -> None:
    """Validate that an optional correlation lies within [-1, 1].

    Raises:
        ValueError: If value is provided and outside [-1, 1].
    """

    if value is not None and not -1.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between -1 and 1")
