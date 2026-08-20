"""A4 StrategyProfiler.

Answers "what do these fill-strategy operators actually measure
differently?" by contrasting a **declarative** attribute table (a
hypothesis about each operator's own nature) against an **empirical**
grouping derived from the fill-strategy correlation data A3 already
produced (an observation about how the operators actually behaved on this
dataset). Presenting declared attributes as a hypothesis and empirical
groups as an observation, then reporting whether they agree, is the honest
way to design this comparison.

This module consumes A3's *output types*, not its code: ``strategy_rows``
(``analysis.stability``'s per-anchor ``StrategyStabilityRow`` list) and
``rank_correlation_rows`` (its dataset-level ``RankCorrelationRow`` list) are
passed in by the caller, the same composition pattern A2 already uses for
A1's ``control_pairs`` (this step reuses, as-is, the op×op correlation
matrix the previous step built). Because
``StrategyStabilityRow.strategy_values`` already carries one degradation
value per (anchor, op), this module never needs to re-join raw item data --
it depends on nothing but ``analysis.types``, ``analysis.errors``, and
``ssat.core.types.PerturbationOp`` (used only as a declarative-table lookup
key, as an explicit exception to this package's usual dependency rules).

Like ``RankCorrelationRow``, ``StrategyProfileRow`` has no ``metric_name``
field, so the dataset-level pieces of this module (clustering, the sign-group
summary) are scoped to a single ``primary_metric`` -- ``DEFAULT_PRIMARY_METRIC``
is redefined locally rather than imported, the same duplication policy
already applied to ``region_key``/``ConditionKey`` formulas and
``ssat.metrics.aggregate.DEFAULT_PRIMARY_METRIC``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from ssat.analysis.errors import AnalysisCorruptionError
from ssat.analysis.types import (
    Alignment,
    RankCorrelationRow,
    StrategyProfileRow,
    StrategyStabilityRow,
)
from ssat.core.types import PerturbationOp

DEFAULT_PRIMARY_METRIC = "margin_drop"
DEFAULT_CLUSTER_THRESHOLD = 0.3
DEFAULT_TOP_K_EXCLUDE = 1

# (preserves_statistics, preserves_local_texture, is_global_operation) per op.
# preserves_statistics values are the design's own worked examples; the
# remaining ten cells are this module's proposed values, read off
# ssat/core/perturb/operators.py's actual compositing behavior:
# preserves_local_texture is True only for patch_shuffle (every other op
# overwrites the signal outright); is_global_operation is True only for
# blur and patch_shuffle, since both build their replacement candidate from
# the *entire* frame before compositing back through the mask -- so pixels
# from outside the selected region can leak in -- while constant_fill,
# mean_fill (a pre-resolved dataset-level constant), and gaussian_noise (an
# elementwise op with no cross-pixel dependency, despite being computed over
# the full array shape for vectorization) never look outside the mask.
_DECLARED_ATTRIBUTES: dict[PerturbationOp, tuple[bool, bool, bool]] = {
    PerturbationOp.CONSTANT_FILL: (False, False, False),
    PerturbationOp.MEAN_FILL: (True, False, False),
    PerturbationOp.BLUR: (True, False, True),
    PerturbationOp.GAUSSIAN_NOISE: (False, False, False),
    PerturbationOp.PATCH_SHUFFLE: (True, True, True),
}


def compute_strategy_profile(
    strategy_rows: Sequence[StrategyStabilityRow],
    rank_correlation_rows: Sequence[RankCorrelationRow],
    *,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    top_k_exclude: int = DEFAULT_TOP_K_EXCLUDE,
) -> list[StrategyProfileRow]:
    """Profile every observed fill-strategy operator.

    Args:
        strategy_rows: A3's per-anchor fill-strategy output
            (``analysis.stability.compute_strategy_stability``'s first
            return value) -- supplies each op's per-anchor degradation
            values for the sign-group summary.
        rank_correlation_rows: A3's dataset-level rank-correlation output
            (that function's second return value) -- supplies the op×op
            correlation matrix clustering is based on.
        primary_metric: The single metric this profile is computed over
            (module docstring: neither ``StrategyProfileRow`` nor
            ``RankCorrelationRow`` carries a metric axis).
        cluster_threshold: Minimum signed correlation for two ops to be
            connected in the empirical grouping graph (simple
            threshold-based connected components).
        top_k_exclude: Number of an op's own highest-value contributing
            anchors excluded from ``mean_degradation_excl_top`` (same
            top-k-exclusion idea A3(c) uses for ``spearman_excl_top1``).

    Returns:
        One row per op observed in ``strategy_rows`` for ``primary_metric``
        (ops never tested are simply absent, not zero-filled), sorted by
        ``perturb_op``.

    Raises:
        AnalysisCorruptionError: If an observed ``perturb_op`` string is not
            one of the five ``PerturbationOp`` values the declarative table
            covers -- the core cannot produce any other operator today.
    """

    ops = _observed_ops(strategy_rows, primary_metric)
    correlation_lookup = _correlation_lookup(rank_correlation_rows)
    cluster_id_by_op = _cluster_ops(ops, correlation_lookup, cluster_threshold)
    declared_by_op = {op: _declared_attributes(op) for op in ops}
    declared_group_by_op = {op: declared_by_op[op][0] for op in ops}

    rows: list[StrategyProfileRow] = []
    for op in ops:
        preserves_statistics, preserves_local_texture, is_global_operation = declared_by_op[op]
        mean_degradation_excl_top, sign_ratio_positive, n_anchors = _sign_group_summary(
            op, strategy_rows, primary_metric=primary_metric, top_k_exclude=top_k_exclude
        )
        rows.append(
            StrategyProfileRow(
                perturb_op=op,
                preserves_statistics=preserves_statistics,
                preserves_local_texture=preserves_local_texture,
                is_global_operation=is_global_operation,
                cluster_id=cluster_id_by_op.get(op),
                mean_corr_within=_mean_corr(
                    op, ops, cluster_id_by_op, correlation_lookup, within=True
                ),
                mean_corr_across=_mean_corr(
                    op, ops, cluster_id_by_op, correlation_lookup, within=False
                ),
                alignment=_alignment(op, ops, declared_group_by_op, cluster_id_by_op),
                mean_degradation_excl_top=mean_degradation_excl_top,
                sign_ratio_positive=sign_ratio_positive,
                n_anchors=n_anchors,
            )
        )

    rows.sort(key=lambda row: row.perturb_op)
    return rows


def _observed_ops(
    strategy_rows: Sequence[StrategyStabilityRow], primary_metric: str
) -> list[str]:
    """Every distinct perturb_op contributing to primary_metric, sorted."""

    ops: set[str] = set()
    for row in strategy_rows:
        if row.metric_name != primary_metric:
            continue
        ops.update(row.strategy_values)
    return sorted(ops)


def _declared_attributes(op: str) -> tuple[bool, bool, bool]:
    """Look up an op's declarative table entry.

    Raises:
        AnalysisCorruptionError: If op is not one of the five declared
            PerturbationOp values.
    """

    try:
        canonical = PerturbationOp(op)
    except ValueError as exc:
        raise AnalysisCorruptionError(f"unrecognized perturb_op: {op!r}") from exc
    return _DECLARED_ATTRIBUTES[canonical]


def _correlation_lookup(
    rank_correlation_rows: Sequence[RankCorrelationRow],
) -> dict[tuple[str, str], float | None]:
    """Map each (op_a, op_b) pair to its preferred correlation value.

    Prefers ``spearman_excl_top1`` over ``spearman`` -- A3(c) built that
    field specifically because a single dominant signal shared by every
    operator mechanically pulls the plain correlation toward +1, so the
    excl_top1 value is the more honest measure of true behavioral
    similarity between two operators.
    """

    lookup: dict[tuple[str, str], float | None] = {}
    for row in rank_correlation_rows:
        weight = row.spearman_excl_top1 if row.spearman_excl_top1 is not None else row.spearman
        lookup[(row.op_a, row.op_b)] = weight
    return lookup


def _lookup_weight(
    op_a: str, op_b: str, correlation_lookup: dict[tuple[str, str], float | None]
) -> float | None:
    """Fetch a pair's correlation weight regardless of argument order."""

    if (op_a, op_b) in correlation_lookup:
        return correlation_lookup[(op_a, op_b)]
    return correlation_lookup.get((op_b, op_a))


def _cluster_ops(
    ops: list[str],
    correlation_lookup: dict[tuple[str, str], float | None],
    threshold: float,
) -> dict[str, int | None]:
    """Connected-components clustering on signed correlation >= threshold edges.

    Uses the *signed* value, not its magnitude: a strong negative
    correlation means two operators respond in opposite directions, not
    that they behave similarly (L3's blur vs constant_fill, -0.843, is
    exactly this case). Every op gets an integer cluster_id, even
    an isolated one with no qualifying edges (its own singleton cluster) --
    ``None`` is reserved for "too few ops to cluster at all"
    (``StrategyProfileRow`` docstring), not "clustered alone".
    """

    if len(ops) < 2:
        return {op: None for op in ops}

    parent = {op: op for op in ops}

    def find(op: str) -> str:
        while parent[op] != op:
            parent[op] = parent[parent[op]]
            op = parent[op]
        return op

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for i, op_a in enumerate(ops):
        for op_b in ops[i + 1 :]:
            weight = _lookup_weight(op_a, op_b, correlation_lookup)
            if weight is not None and weight >= threshold:
                union(op_a, op_b)

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for op in ops:
        members_by_root[find(op)].append(op)

    sorted_clusters = sorted(members_by_root.values(), key=min)
    return {
        op: cluster_id
        for cluster_id, members in enumerate(sorted_clusters)
        for op in members
    }


def _mean_corr(
    op: str,
    ops: list[str],
    cluster_id_by_op: dict[str, int | None],
    correlation_lookup: dict[tuple[str, str], float | None],
    *,
    within: bool,
) -> float | None:
    """Mean signed correlation to same-cluster (within) or other-cluster (across) ops."""

    this_cluster = cluster_id_by_op.get(op)
    if this_cluster is None:
        return None

    values: list[float] = []
    for other in ops:
        if other == op:
            continue
        same_cluster = cluster_id_by_op.get(other) == this_cluster
        if same_cluster != within:
            continue
        weight = _lookup_weight(op, other, correlation_lookup)
        if weight is not None:
            values.append(weight)
    return float(np.mean(values)) if values else None


def _alignment(
    op: str,
    ops: list[str],
    declared_group_by_op: dict[str, bool],
    cluster_id_by_op: dict[str, int | None],
) -> Alignment:
    """Jaccard-compare op's declared group against its empirical cluster."""

    declared = {other for other in ops if declared_group_by_op[other] == declared_group_by_op[op]}
    this_cluster = cluster_id_by_op.get(op)
    empirical = (
        {other for other in ops if cluster_id_by_op.get(other) == this_cluster}
        if this_cluster is not None
        else {op}
    )

    union_size = len(declared | empirical)
    jaccard = len(declared & empirical) / union_size if union_size else 1.0
    if jaccard >= 1.0:
        return Alignment.ALIGNED
    if jaccard >= 0.5:
        return Alignment.PARTIAL
    return Alignment.DIVERGENT


def _sign_group_summary(
    op: str,
    strategy_rows: Sequence[StrategyStabilityRow],
    *,
    primary_metric: str,
    top_k_exclude: int,
) -> tuple[float | None, float | None, int]:
    """Compute (mean_degradation_excl_top, sign_ratio_positive, n_anchors) for one op."""

    values = [
        (row.anchor_key.region_key, row.strategy_values[op])
        for row in strategy_rows
        if row.metric_name == primary_metric and op in row.strategy_values
    ]
    n_anchors = len(values)
    if n_anchors == 0:
        return None, None, 0

    sign_ratio_positive = sum(1 for _, value in values if value > 0.0) / n_anchors

    # Exclude the top_k_exclude anchors ranked highest by this op's own
    # value (ties broken alphabetically by region_key), same convention as
    # A3(c)'s spearman_excl_top1 exclusion.
    ranked = sorted(values, key=lambda item: (-item[1], item[0]))
    remaining = ranked[top_k_exclude:]
    mean_degradation_excl_top = float(np.mean([v for _, v in remaining])) if remaining else None

    return mean_degradation_excl_top, sign_ratio_positive, n_anchors
