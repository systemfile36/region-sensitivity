"""A3 StabilityAnalyzer.

Answers "does the result hold up across conditions?" by comparing the same
AnchorKey across three axes that are deliberately kept separate (design
CONTROL_STABILITY_DESIGN_v1.md §A3: "섞으면 원인을 알 수 없다" -- mixing them
makes the cause unrecoverable):

- **seed** (:func:`compute_seed_stability`): pure stochastic variation, same
  op/params, only the seed differs.
- **jitter** (:func:`compute_jitter_stability`): mask-boundary sensitivity.
  The core has no jitter axis at all (IMPLE_PLAN_CONTROL_STABILITY_v1.md §1
  항목2), so this is an interface stub that always returns
  ``FlagValue.UNAVAILABLE`` -- there is no ``JitterStabilityRow`` type in
  ``analysis.types`` to populate, only ``ReliabilityRow.jitter_stable``,
  which is documented there as always UNAVAILABLE in v1.
- **fill strategy** (:func:`compute_strategy_stability`): cross-operator
  agreement, both per-anchor (sign/value preserved, never averaged away --
  design §A3(c): averaging hides sign flips) and as a dataset-level
  region-rank correlation between operator pairs.

Like ``analysis.control``, this module re-derives ``AnchorKey``/
``ConditionKey`` from raw item rows rather than importing
``analysis.indexer``/``analysis.control`` (IMPLE_PLAN §3.3
"analysis.stability → analysis.types" only) and consumes the same
``item_values`` frame shape ``analysis.control.compare_to_controls`` does:
``AnalysisReader.item_context()``'s identity columns joined with one row per
(item, metric_name) (``metric_name``, ``degradation``, ``available``).

``RankCorrelationRow`` (design §A7 parquet) has no ``metric_name`` field --
unlike every other row type here, it is scoped to a single primary metric,
matching ``ssat.metrics.aggregate.DEFAULT_PRIMARY_METRIC`` and the fact that
L3's Check2 only ever analyzed ``margin_drop``. ``DEFAULT_PRIMARY_METRIC`` is
redefined locally rather than imported, extending the same duplication this
module already accepts for ``region_key``/``ConditionKey`` formulas
(IMPLE_PLAN §9 리스크 표).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ssat.analysis.types import (
    AnchorKey,
    ConditionKey,
    FlagValue,
    RankCorrelationRow,
    SeedStabilityRow,
    StrategyStabilityRow,
)
from ssat.utils.io import sha256_bytes

DEFAULT_CV_ZERO_THRESHOLD = 1e-6
DEFAULT_TOP_K_EXCLUDE = 1
DEFAULT_SCOPE = "full_dataset"
DEFAULT_PRIMARY_METRIC = "margin_drop"


def compute_seed_stability(
    item_values: pd.DataFrame,
    *,
    metric_names: Sequence[str] | None = None,
    cv_zero_threshold: float = DEFAULT_CV_ZERO_THRESHOLD,
) -> list[SeedStabilityRow]:
    """Measure repeat-trial (seed) variation for every (Anchor, Condition, metric).

    Applies to every AnchorKey regardless of ``is_control`` -- a control
    anchor's own repeat-trial noise is as meaningful a QC signal as a
    target's, and A1's AnchorTable itself does not distinguish them either.

    Args:
        item_values: Item-grain frame combining
            ``AnalysisReader.item_context()``'s identity columns with one
            row per (item, metric_name): ``metric_name``, ``degradation``,
            ``available``.
        metric_names: Metric names to compute rows for; defaults to every
            distinct ``metric_name`` present in ``item_values``.
        cv_zero_threshold: ``seed_cv`` is ``None`` when ``abs(seed_mean)``
            falls below this (same 0-denominator guard as A2's ``ratio``).

    Returns:
        One row per (AnchorKey, ConditionKey, metric_name) that has at
        least one available value. ``seed_std``/``seed_cv`` are ``None``
        when ``n_seeds < 2`` -- a single trial cannot support a claim of
        "no variation" (same principle as A2's ``z_vs_control`` requiring
        ``n_controls >= 2``). Rows with ``n_seeds < 2`` are still emitted;
        filtering them out as "insufficient" is left to the consumer (A6),
        mirroring A1's "flag, don't filter" rule for
        ``n_conditions_insufficient``.
    """

    candidate_metric_names = (
        set(metric_names) if metric_names is not None else set(item_values["metric_name"])
    )
    raw_values = _raw_values_by_anchor_condition_metric(item_values)

    rows: list[SeedStabilityRow] = []
    for (anchor_key, condition_key, metric_name), values in raw_values.items():
        if metric_name not in candidate_metric_names:
            continue
        n_seeds = len(values)
        seed_mean = float(np.mean(values))
        if n_seeds >= 2:
            seed_std = float(np.std(values))
            seed_cv = (
                seed_std / abs(seed_mean) if abs(seed_mean) >= cv_zero_threshold else None
            )
        else:
            seed_std = None
            seed_cv = None
        rows.append(
            SeedStabilityRow(
                anchor_key=anchor_key,
                condition_key=condition_key,
                metric_name=metric_name,
                seed_mean=seed_mean,
                seed_std=seed_std,
                seed_cv=seed_cv,
                n_seeds=n_seeds,
            )
        )

    rows.sort(
        key=lambda row: (
            row.anchor_key.sample_id,
            row.anchor_key.region_key,
            row.anchor_key.invert_mask,
            row.condition_key.perturb_op,
            row.condition_key.perturb_params_hash,
            row.metric_name,
        )
    )
    return rows


def compute_jitter_stability(item_values: pd.DataFrame) -> FlagValue:
    """Jitter stability interface stub -- unreachable until the core supports jitter.

    Always returns ``FlagValue.UNAVAILABLE`` (IMPLE_PLAN_CONTROL_STABILITY_v1.md
    §1 항목2, §5 단계4(b)): no ``RegionKind`` or config concept for mask-boundary
    jitter exists anywhere in ``ssat.core`` today, so there is nothing this
    function could compute. It accepts ``item_values`` purely so its call
    signature matches the other two axes -- if the core ever grows jitter
    support, this stub is the only place that needs to change.
    """

    del item_values
    return FlagValue.UNAVAILABLE


def compute_strategy_stability(
    item_values: pd.DataFrame,
    *,
    metric_names: Sequence[str] | None = None,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
    top_k_exclude: int = DEFAULT_TOP_K_EXCLUDE,
    scope: str = DEFAULT_SCOPE,
) -> tuple[list[StrategyStabilityRow], list[RankCorrelationRow]]:
    """Compare degradation across fill-strategy operators (design §A3(c)).

    Two outputs over two different populations, per design decision: the
    per-anchor view preserves each operator's sign/value instead of
    averaging them away, while the dataset-level view ranks target regions
    per operator and correlates those rankings between every operator pair.

    Args:
        item_values: Same shape as :func:`compute_seed_stability`'s.
        metric_names: Metric names the per-anchor rows are computed for;
            defaults to every distinct ``metric_name`` present.
        primary_metric: The single metric the dataset-level rank
            correlation is computed over -- ``RankCorrelationRow`` has no
            ``metric_name`` field (module docstring), so only one metric can
            be represented per call.
        top_k_exclude: Number of top-ranked (by ``op_a``'s value) shared
            regions excluded when computing ``spearman_excl_top1`` (design
            §A3(c) "모든 op에서 1위인 강한 신호가 상관을 인위적으로 끌어올린다").
        scope: Free-form population label stored on every
            ``RankCorrelationRow`` (design §A3(c) "scope는 상관 계산 범위를
            기록한다").

    Returns:
        ``(strategy_rows, rank_correlation_rows)``. ``strategy_rows`` cover
        only non-control (target) anchors with at least one operator value
        -- ``ReliabilityRow.multi_strategy`` (A6) is per target AnchorKey,
        so this output is scoped to match. ``rank_correlation_rows`` cover
        every pair of operators (``op_a < op_b`` alphabetically, no
        duplicates) that each have at least one target region value for
        ``primary_metric``; a pair sharing fewer than 2 regions still gets a
        row with ``spearman=None`` (undefined, not omitted).
    """

    strategy_rows = _per_anchor_strategy_rows(item_values, metric_names=metric_names)
    rank_rows = _rank_correlation_rows(
        item_values, primary_metric=primary_metric, top_k_exclude=top_k_exclude, scope=scope
    )
    return strategy_rows, rank_rows


def _per_anchor_strategy_rows(
    item_values: pd.DataFrame, *, metric_names: Sequence[str] | None
) -> list[StrategyStabilityRow]:
    """Build one StrategyStabilityRow per (target AnchorKey, metric_name)."""

    candidate_metric_names = (
        set(metric_names) if metric_names is not None else set(item_values["metric_name"])
    )
    op_means, is_control_by_anchor = _op_level_anchor_means(item_values)

    grouped: dict[tuple[AnchorKey, str], dict[str, float]] = defaultdict(dict)
    for (anchor_key, perturb_op, metric_name), value in op_means.items():
        if metric_name not in candidate_metric_names:
            continue
        if is_control_by_anchor.get(anchor_key, False):
            continue
        grouped[(anchor_key, metric_name)][perturb_op] = value

    rows: list[StrategyStabilityRow] = []
    for (anchor_key, metric_name), values_by_op in grouped.items():
        strategy_signs = {op: _sign(value) for op, value in values_by_op.items()}
        n_strategies = len(strategy_signs)
        _, top_count = Counter(strategy_signs.values()).most_common(1)[0]
        rows.append(
            StrategyStabilityRow(
                anchor_key=anchor_key,
                metric_name=metric_name,
                strategy_signs=strategy_signs,
                strategy_values=dict(values_by_op),
                sign_agreement_ratio=top_count / n_strategies,
                n_strategies=n_strategies,
            )
        )

    rows.sort(
        key=lambda row: (
            row.anchor_key.sample_id,
            row.anchor_key.region_key,
            row.anchor_key.invert_mask,
            row.metric_name,
        )
    )
    return rows


def _rank_correlation_rows(
    item_values: pd.DataFrame, *, primary_metric: str, top_k_exclude: int, scope: str
) -> list[RankCorrelationRow]:
    """Build one RankCorrelationRow per (op_a, op_b) pair for ``primary_metric``.

    Stage 2 of the macro-average: pools op-level anchor values sharing a
    ``region_key`` (across ``sample_id``/``invert_mask``) into one value per
    (region_key, perturb_op) -- mirrors ``ssat.metrics.aggregate``'s
    spatial_profile -> region_metrics reduction, stratified by
    ``perturb_op`` (``region_metrics.parquet`` itself cannot be reused here
    since it has already collapsed that axis -- IMPLE_PLAN §1 항목3).
    """

    op_means, is_control_by_anchor = _op_level_anchor_means(item_values)

    region_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (anchor_key, perturb_op, metric_name), value in op_means.items():
        if metric_name != primary_metric:
            continue
        if is_control_by_anchor.get(anchor_key, False):
            continue
        region_values[perturb_op][anchor_key.region_key].append(value)

    region_value_by_op: dict[str, dict[str, float]] = {
        op: {region_key: float(np.mean(values)) for region_key, values in per_region.items()}
        for op, per_region in region_values.items()
    }

    ops = sorted(region_value_by_op)
    rows: list[RankCorrelationRow] = []
    for i, op_a in enumerate(ops):
        for op_b in ops[i + 1 :]:
            rows.append(
                _one_rank_correlation(
                    op_a,
                    op_b,
                    region_value_by_op[op_a],
                    region_value_by_op[op_b],
                    top_k_exclude=top_k_exclude,
                    scope=scope,
                )
            )
    return rows


def _one_rank_correlation(
    op_a: str,
    op_b: str,
    values_a: dict[str, float],
    values_b: dict[str, float],
    *,
    top_k_exclude: int,
    scope: str,
) -> RankCorrelationRow:
    """Compute one operator pair's full and top-k-excluded Spearman correlation."""

    shared_keys = sorted(set(values_a) & set(values_b))
    spearman = _spearman_correlation(
        {key: values_a[key] for key in shared_keys},
        {key: values_b[key] for key in shared_keys},
    )

    # Exclude the top_k_exclude regions ranked highest by op_a's value
    # (ties broken alphabetically by region_key for determinism) -- design
    # §A3(c): a single dominant signal shared by every operator mechanically
    # pulls the full correlation toward +1.
    dominant = sorted(shared_keys, key=lambda key: (-values_a[key], key))[:top_k_exclude]
    remaining_keys = [key for key in shared_keys if key not in set(dominant)]
    spearman_excl_top1 = _spearman_correlation(
        {key: values_a[key] for key in remaining_keys},
        {key: values_b[key] for key in remaining_keys},
    )

    return RankCorrelationRow(
        op_a=op_a,
        op_b=op_b,
        spearman=spearman,
        n_regions=len(shared_keys),
        spearman_excl_top1=spearman_excl_top1,
        scope=scope,
    )


def _spearman_correlation(reference: dict[str, float], other: dict[str, float]) -> float | None:
    """Spearman rank correlation between two region_key -> value mappings.

    Reimplemented rather than imported -- this module depends only on
    ``analysis.types`` (IMPLE_PLAN §3.3) -- mirroring
    ``experiments/synthetic_shortcut/analyze_section35_sensitivity.py``'s
    private helper of the same name: Pearson correlation of the two series'
    ranks, avoiding a scipy dependency (IMPLE_PLAN §2.1).
    """

    shared_keys = sorted(set(reference) & set(other))
    if len(shared_keys) < 2:
        return None
    reference_ranks = pd.Series([reference[key] for key in shared_keys]).rank()
    other_ranks = pd.Series([other[key] for key in shared_keys]).rank()
    correlation = reference_ranks.corr(other_ranks)
    return None if pd.isna(correlation) else float(correlation)


def _sign(value: float) -> int:
    """Map a degradation value to {-1, 0, 1}."""

    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _raw_values_by_anchor_condition_metric(
    item_values: pd.DataFrame,
) -> dict[tuple[AnchorKey, ConditionKey, str], list[float]]:
    """Group available degradation values by (AnchorKey, ConditionKey, metric).

    Includes both target and control anchors -- seed stability is a
    property of one anchor's own repeat trials, independent of whether that
    anchor is a target or a control.
    """

    raw_values: dict[tuple[AnchorKey, ConditionKey, str], list[float]] = defaultdict(list)
    for row in item_values.itertuples(index=False):
        if not row.available:
            continue
        anchor_key = _anchor_key(row)
        condition_key = _condition_key(row)
        raw_values[(anchor_key, condition_key, row.metric_name)].append(float(row.degradation))
    return raw_values


def _op_level_anchor_means(
    item_values: pd.DataFrame,
) -> tuple[dict[tuple[AnchorKey, str, str], float], dict[AnchorKey, bool]]:
    """Collapse per-(Anchor, Condition, metric) means to one value per (Anchor, op, metric).

    First reduces raw item values to one mean per (AnchorKey, ConditionKey,
    metric) -- absorbing seed repeats, the same first macro-average stage A2
    uses -- then averages across any ConditionKeys that share a
    ``perturb_op`` (distinct params under the same op), so each anchor
    contributes exactly one value per operator.
    """

    raw_values = _raw_values_by_anchor_condition_metric(item_values)
    condition_means = {key: float(np.mean(values)) for key, values in raw_values.items()}

    is_control_by_anchor: dict[AnchorKey, bool] = {}
    for row in item_values.itertuples(index=False):
        anchor_key = _anchor_key(row)
        is_control_by_anchor.setdefault(anchor_key, bool(row.is_control))

    grouped: dict[tuple[AnchorKey, str, str], list[float]] = defaultdict(list)
    for (anchor_key, condition_key, metric_name), value in condition_means.items():
        grouped[(anchor_key, condition_key.perturb_op, metric_name)].append(value)

    op_means = {key: float(np.mean(values)) for key, values in grouped.items()}
    return op_means, is_control_by_anchor


def _anchor_key(row: Any) -> AnchorKey:
    """Reconstruct one item row's AnchorKey (same formula as A1's indexer)."""

    region_key = f"{row.region_id}::{row.region_instance_id}"
    return AnchorKey(
        sample_id=row.sample_id, region_key=region_key, invert_mask=bool(row.invert_mask)
    )


def _condition_key(row: Any) -> ConditionKey:
    """Reconstruct one item row's ConditionKey (same formula as A1's indexer)."""

    return ConditionKey(
        perturb_op=row.perturb_op,
        perturb_params_hash=sha256_bytes(row.perturb_params_json.encode("utf-8")),
    )
