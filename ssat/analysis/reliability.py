"""A6 ReliabilityScorer: combine A2-A5 outputs into per-anchor verdicts.

Produces one ``ReliabilityRow`` per (AnchorKey, metric_name) — seven
independently-interpretable ``FlagValue`` flags, an overall
``ReliabilityGrade``, and a human-readable reason per flag (design
CONTROL_STABILITY_DESIGN_v1.md §A6 "등급만 주고 끝내지 않는다").

Like ``analysis.strategy_profile`` consuming A3's typed output,
``compute_reliability`` takes A2/A3(a)/A3(c)/A5's row lists directly as
parameters rather than importing their code or raw ``item_values``
(IMPLE_PLAN_CONTROL_STABILITY_v1.md §5 단계7 "A2~A5의 산출물을 모아"; §3.3
declares only ``analysis.reliability → analysis.types``). ``jitter_stable``
needs no input at all: it is always ``FlagValue.UNAVAILABLE``, mirroring
``analysis.stability.compute_jitter_stability``'s own hardcoded stub (the
core has no jitter axis to report on).

Two of A2/A3(a)'s row types are keyed per (AnchorKey, ConditionKey, metric)
— finer-grained than ``ReliabilityRow``'s (AnchorKey, metric) — so this
module reduces across the condition axis per anchor before deriving a flag,
using the same "reveal, don't average away" philosophy as the rest of the
package: a flag becomes ``TRUE`` only when the *best* available evidence
clears the threshold (``exceeds_control``) or *every* evaluated condition
agrees (``seed_stable``, ``area_matched``) — never a silent average.
``IntervalRow`` (A5) is keyed only by (region_key, metric), coarser than
AnchorKey (design decision recorded when A5 was implemented — that grain
matches ``ssat.metrics.aggregate.RegionMetrics`` exactly); every anchor
sharing a region_key shares that region's CI verdict.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from ssat.analysis.types import (
    AnchorKey,
    ControlComparisonRow,
    FlagValue,
    IntervalRow,
    ReliabilityGrade,
    ReliabilityRow,
    SeedStabilityRow,
    StrategyStabilityRow,
)

DEFAULT_Z_VS_CONTROL_THRESHOLD = 2.0
DEFAULT_SEED_CV_THRESHOLD = 0.2

_JITTER_REASON = "jitter stability unavailable (core has no jitter support)"


def compute_reliability(
    control_rows: Sequence[ControlComparisonRow],
    seed_rows: Sequence[SeedStabilityRow],
    strategy_rows: Sequence[StrategyStabilityRow],
    interval_rows: Sequence[IntervalRow],
    *,
    z_vs_control_threshold: float = DEFAULT_Z_VS_CONTROL_THRESHOLD,
    seed_cv_threshold: float = DEFAULT_SEED_CV_THRESHOLD,
) -> list[ReliabilityRow]:
    """Score every (AnchorKey, metric_name) observed in the strategy/control inputs.

    Args:
        control_rows: A2 output, per (target AnchorKey, ConditionKey, metric).
        seed_rows: A3(a) output, per (AnchorKey, ConditionKey, metric) —
            includes control anchors, but only target keys (from
            strategy_rows/control_rows) are ever looked up here.
        strategy_rows: A3(c) per-anchor output, per (AnchorKey, metric);
            target-only by construction.
        interval_rows: A5 output, per (region_key, metric).
        z_vs_control_threshold: ``exceeds_control`` is TRUE when the best
            evaluable ``z_vs_control`` across this anchor's conditions
            exceeds this value.
        seed_cv_threshold: ``seed_stable`` is TRUE when every evaluable
            ``seed_cv`` across this anchor's conditions is below this value.

    Returns:
        One row per (AnchorKey, metric_name), sorted by
        (sample_id, region_key, invert_mask, metric_name). The key set is
        the union of keys observed in ``strategy_rows`` and
        ``control_rows`` — the two inputs guaranteed to enumerate every
        target anchor actually measured.
    """

    strategy_by_key: dict[tuple[AnchorKey, str], StrategyStabilityRow] = {
        (row.anchor_key, row.metric_name): row for row in strategy_rows
    }
    control_by_key: dict[tuple[AnchorKey, str], list[ControlComparisonRow]] = defaultdict(list)
    for row in control_rows:
        control_by_key[(row.target_anchor_key, row.metric_name)].append(row)
    seed_by_key: dict[tuple[AnchorKey, str], list[SeedStabilityRow]] = defaultdict(list)
    for row in seed_rows:
        seed_by_key[(row.anchor_key, row.metric_name)].append(row)
    interval_by_region_metric: dict[tuple[str, str], IntervalRow] = {
        (row.region_key, row.metric): row for row in interval_rows
    }

    keys = set(strategy_by_key) | set(control_by_key)

    rows: list[ReliabilityRow] = []
    for anchor_key, metric_name in sorted(
        keys, key=lambda key: (key[0].sample_id, key[0].region_key, key[0].invert_mask, key[1])
    ):
        strategy_row = strategy_by_key.get((anchor_key, metric_name))
        control_group = control_by_key.get((anchor_key, metric_name), [])
        seed_group = seed_by_key.get((anchor_key, metric_name), [])
        interval_row = interval_by_region_metric.get((anchor_key.region_key, metric_name))

        sign_consistent, sign_reason = _sign_consistent_flag(strategy_row)
        exceeds_control, exceeds_reason = _exceeds_control_flag(
            control_group, z_vs_control_threshold
        )
        seed_stable, seed_reason = _seed_stable_flag(seed_group, seed_cv_threshold)
        multi_strategy, multi_reason = _multi_strategy_flag(strategy_row)
        ci_excludes_zero, ci_reason = _ci_excludes_zero_flag(interval_row)
        area_matched, area_reason = _area_matched_flag(control_group)

        grade = _grade(sign_consistent, exceeds_control, multi_strategy, ci_excludes_zero)

        rows.append(
            ReliabilityRow(
                anchor_key=anchor_key,
                metric_name=metric_name,
                sign_consistent=sign_consistent,
                exceeds_control=exceeds_control,
                seed_stable=seed_stable,
                jitter_stable=FlagValue.UNAVAILABLE,
                multi_strategy=multi_strategy,
                ci_excludes_zero=ci_excludes_zero,
                area_matched=area_matched,
                reliability_grade=grade,
                reliability_reasons=(
                    sign_reason,
                    exceeds_reason,
                    seed_reason,
                    _JITTER_REASON,
                    multi_reason,
                    ci_reason,
                    area_reason,
                ),
            )
        )
    return rows


def _grade(
    sign_consistent: FlagValue,
    exceeds_control: FlagValue,
    multi_strategy: FlagValue,
    ci_excludes_zero: FlagValue,
) -> ReliabilityGrade:
    """Assign the overall grade (design §A6 table).

    ``sign_consistent is FALSE`` alone forces UNRELIABLE (design "값이 작은
    것이 아니라 방향이 갈리는 것") — not merely "not TRUE": an anchor whose
    sign consistency is UNAVAILABLE (no fill-strategy data at all) is
    withheld from HIGH/MODERATE below but is not accused of an actual sign
    flip, so it falls through to LOW rather than UNRELIABLE.
    """

    if sign_consistent is FlagValue.FALSE:
        return ReliabilityGrade.UNRELIABLE

    core = (exceeds_control, multi_strategy, ci_excludes_zero)
    if sign_consistent is FlagValue.TRUE and all(flag is FlagValue.TRUE for flag in core):
        return ReliabilityGrade.HIGH
    if (
        sign_consistent is FlagValue.TRUE
        and any(flag is FlagValue.TRUE for flag in core)
        and not any(flag is FlagValue.FALSE for flag in core)
    ):
        return ReliabilityGrade.MODERATE
    return ReliabilityGrade.LOW


def _sign_consistent_flag(row: StrategyStabilityRow | None) -> tuple[FlagValue, str]:
    """All conditions (operators) share one degradation sign for this anchor."""

    if row is None or not row.strategy_signs:
        return FlagValue.UNAVAILABLE, "sign consistency unavailable (no fill-strategy data)"
    if row.sign_agreement_ratio == 1.0:
        return FlagValue.TRUE, f"sign consistent across {row.n_strategies} operator(s)"
    counts = Counter(row.strategy_signs.values())
    pos, neg, zero = counts.get(1, 0), counts.get(-1, 0), counts.get(0, 0)
    detail = f"{pos} positive, {neg} negative" + (f", {zero} zero" if zero else "")
    return FlagValue.FALSE, f"sign differs across fill strategies ({detail})"


def _multi_strategy_flag(row: StrategyStabilityRow | None) -> tuple[FlagValue, str]:
    """The dominant sign is reproduced by 2 or more operators."""

    if row is None or not row.strategy_signs:
        return (
            FlagValue.UNAVAILABLE,
            "multi-strategy reproduction unavailable (no fill-strategy data)",
        )
    _, dominant_count = Counter(row.strategy_signs.values()).most_common(1)[0]
    if dominant_count >= 2:
        return FlagValue.TRUE, f"reproduced in {dominant_count} of {row.n_strategies} operators"
    return (
        FlagValue.FALSE,
        f"reproduced in only {dominant_count} of {row.n_strategies} operator(s)",
    )


def _exceeds_control_flag(
    rows: list[ControlComparisonRow], threshold: float
) -> tuple[FlagValue, str]:
    """The best evaluable z_vs_control across this anchor's conditions clears the threshold."""

    evaluable = [
        row.z_vs_control
        for row in rows
        if row.control_available is FlagValue.TRUE and row.z_vs_control is not None
    ]
    if not evaluable:
        return (
            FlagValue.UNAVAILABLE,
            "control comparison unavailable (no matched control with a usable value)",
        )
    best = max(evaluable)
    if best > threshold:
        return FlagValue.TRUE, f"exceeds control (z={best:.2f} > threshold {threshold:g})"
    return (
        FlagValue.FALSE,
        f"does not exceed control (max z={best:.2f}, threshold {threshold:g})",
    )


def _seed_stable_flag(rows: list[SeedStabilityRow], threshold: float) -> tuple[FlagValue, str]:
    """Every evaluable seed_cv across this anchor's conditions is below the threshold."""

    evaluable = [row.seed_cv for row in rows if row.seed_cv is not None]
    if not evaluable:
        return FlagValue.UNAVAILABLE, "seed stability unavailable (fewer than 2 seeds observed)"
    worst = max(evaluable)
    if worst < threshold:
        return FlagValue.TRUE, f"seed-stable (max cv={worst:.3f} < {threshold:g})"
    return (
        FlagValue.FALSE,
        f"seed variation exceeds threshold (max cv={worst:.3f} >= {threshold:g})",
    )


def _area_matched_flag(rows: list[ControlComparisonRow]) -> tuple[FlagValue, str]:
    """Every matched control's area falls within tolerance, for every evaluable condition."""

    evaluable = [row.area_matched for row in rows if row.control_available is FlagValue.TRUE]
    if not evaluable:
        return FlagValue.UNAVAILABLE, "control area match unavailable (no matched control)"
    if any(value is FlagValue.FALSE for value in evaluable):
        return FlagValue.FALSE, "control area match outside tolerance for at least one condition"
    return FlagValue.TRUE, "control area match within tolerance"


def _ci_excludes_zero_flag(row: IntervalRow | None) -> tuple[FlagValue, str]:
    """This anchor's region-level bootstrap CI excludes zero."""

    if row is None:
        return (
            FlagValue.UNAVAILABLE,
            "bootstrap CI unavailable (no interval computed for this region/metric)",
        )
    if row.excludes_zero:
        return FlagValue.TRUE, f"bootstrap CI excludes zero ([{row.ci_low:.3f}, {row.ci_high:.3f}])"
    return FlagValue.FALSE, f"bootstrap CI includes zero ([{row.ci_low:.3f}, {row.ci_high:.3f}])"
