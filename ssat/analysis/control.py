"""A2 ControlComparator.

Compares each target AnchorKey/ConditionKey's degradation against its
matched random-area-match controls, answering "is this sensitivity
region-specific, or just because something was occluded?" (design
CONTROL_STABILITY_DESIGN_v1.md §A2, IMPLE_PLAN_CONTROL_STABILITY_v1.md §5
단계3).

This module deliberately re-derives ``AnchorKey``/``ConditionKey`` from raw
item rows rather than importing ``analysis.indexer``/``analysis.reader`` --
the same pattern IMPLE_PLAN §5 단계4 (A3(c)) already prescribes ("A1의
AnchorTable + item 컨텍스트에서 ... 직접 재집계한다"), extending the
already-accepted ``region_key`` formula duplication (§9 리스크 표) to
``ConditionKey`` as well. ``compare_to_controls`` therefore depends only on
``analysis.types`` plus ``pandas``/``numpy``/stdlib, consuming a plain
``item_values`` frame the caller assembles from
``AnalysisReader.item_context()`` + ``AnalysisReader.item_metrics`` -- never
``ssat.metrics.types.ItemMetrics`` objects directly.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ssat.analysis.types import (
    AnchorKey,
    ConditionKey,
    ControlComparisonRow,
    ControlPairRow,
    FlagValue,
)
from ssat.utils.io import sha256_bytes

DEFAULT_RATIO_ZERO_THRESHOLD = 1e-6
DEFAULT_AREA_MATCH_TOLERANCE = 0.05


def compare_to_controls(
    item_values: pd.DataFrame,
    control_pairs: Sequence[ControlPairRow],
    *,
    metric_names: Sequence[str] | None = None,
    ratio_zero_threshold: float = DEFAULT_RATIO_ZERO_THRESHOLD,
    area_match_tolerance: float = DEFAULT_AREA_MATCH_TOLERANCE,
) -> list[ControlComparisonRow]:
    """Compare each target's degradation to its matched controls (design §A2).

    Args:
        item_values: Item-grain frame combining
            ``AnalysisReader.item_context()``'s identity columns
            (``sample_id``, ``region_id``, ``region_instance_id``,
            ``invert_mask``, ``perturb_op``, ``perturb_params_json``,
            ``is_control``) with one row per (item, metric_name): also
            ``metric_name``, ``degradation`` (``float``, ignored when
            ``available`` is falsy), and ``available`` (``bool``).
        control_pairs: A1's matched (control anchor, target anchor,
            ConditionKey) pairs --
            ``ssat.analysis.indexer.ComparisonIndexer.control_pairs``.
        metric_names: Metric names to compute rows for; defaults to every
            distinct ``metric_name`` present in ``item_values``.
        ratio_zero_threshold: ``ratio`` is ``None`` when
            ``abs(control_mean)`` falls below this (design §A2 "분모가 0에
            가까울 때 폭발").
        area_match_tolerance: Fractional tolerance used to judge
            ``area_matched`` from each matched pair's ``area_match_ratio``
            (same default as A1's ``area_match_tolerance``).

    Returns:
        One ``ControlComparisonRow`` per (target AnchorKey, ConditionKey,
        metric_name) that the target itself has at least one available
        value for -- a metric the target was never computed for produces no
        row (mirrors ``ssat.metrics.aggregate`` only emitting rows for
        combinations actually present in the item data). A target/condition
        with no matched controls -- or matched controls that all lack a
        usable value for that metric -- gets
        ``control_available=FlagValue.UNAVAILABLE`` and every numeric field
        ``None`` (design §A2 "control 부재 시").
    """

    anchor_means, is_control_by_anchor = _anchor_level_means(item_values)
    candidate_metric_names = (
        set(metric_names) if metric_names is not None else set(item_values["metric_name"])
    )

    pairs_by_target: dict[tuple[AnchorKey, ConditionKey], list[ControlPairRow]] = defaultdict(list)
    for pair in control_pairs:
        pairs_by_target[(pair.target_anchor_key, pair.condition_key)].append(pair)

    rows: list[ControlComparisonRow] = []
    for (anchor_key, condition_key, metric_name), target_degradation in anchor_means.items():
        if metric_name not in candidate_metric_names:
            continue
        if is_control_by_anchor.get(anchor_key, False):
            continue
        matched_pairs = pairs_by_target.get((anchor_key, condition_key), [])
        rows.append(
            _compare_one(
                anchor_key,
                condition_key,
                metric_name,
                target_degradation,
                matched_pairs,
                anchor_means,
                ratio_zero_threshold=ratio_zero_threshold,
                area_match_tolerance=area_match_tolerance,
            )
        )

    rows.sort(
        key=lambda row: (
            row.target_anchor_key.sample_id,
            row.target_anchor_key.region_key,
            row.condition_key.perturb_op,
            row.condition_key.perturb_params_hash,
            row.metric_name,
        )
    )
    return rows


def _compare_one(
    anchor_key: AnchorKey,
    condition_key: ConditionKey,
    metric_name: str,
    target_degradation: float,
    matched_pairs: list[ControlPairRow],
    anchor_means: dict[tuple[AnchorKey, ConditionKey, str], float],
    *,
    ratio_zero_threshold: float,
    area_match_tolerance: float,
) -> ControlComparisonRow:
    """Build one ControlComparisonRow from a target's anchor-level value.

    Control availability is judged independently of the target: a
    structurally matched control anchor (present in ``matched_pairs``) that
    happens to lack a usable value for this ``metric_name`` simply does not
    contribute -- if none of the matched controls have a value,
    ``control_available`` is ``UNAVAILABLE`` even though A1 did find
    matches, because "unavailable" tracks usable values, not structural
    matching (design principle "unavailable을 false로 취급하지 않는다"
    applied to this axis too).
    """

    control_values = [
        value
        for pair in matched_pairs
        if (value := anchor_means.get((pair.control_anchor_key, condition_key, metric_name)))
        is not None
    ]

    if not control_values:
        return ControlComparisonRow(
            target_anchor_key=anchor_key,
            condition_key=condition_key,
            metric_name=metric_name,
            control_available=FlagValue.UNAVAILABLE,
            area_matched=FlagValue.UNAVAILABLE,
            control_mean=None,
            control_std=None,
            n_controls=0,
            excess=None,
            ratio=None,
            z_vs_control=None,
        )

    control_mean = float(np.mean(control_values))
    control_std = float(np.std(control_values))
    n_controls = len(control_values)

    excess = target_degradation - control_mean
    ratio = (
        target_degradation / control_mean
        if abs(control_mean) >= ratio_zero_threshold
        else None
    )
    z_vs_control = (
        (target_degradation - control_mean) / control_std
        if control_std != 0.0 and n_controls >= 2
        else None
    )

    # "Reveal inconsistency, don't smooth it over" (design §0): area_matched
    # is TRUE only if every matched pair's area_match_ratio is known and
    # within tolerance; any single mismatch or undefined ratio flags the
    # whole row FALSE rather than silently passing.
    area_matched = FlagValue.TRUE
    for pair in matched_pairs:
        ratio_value = pair.area_match_ratio
        if ratio_value is None or not (
            1.0 - area_match_tolerance <= ratio_value <= 1.0 + area_match_tolerance
        ):
            area_matched = FlagValue.FALSE
            break

    return ControlComparisonRow(
        target_anchor_key=anchor_key,
        condition_key=condition_key,
        metric_name=metric_name,
        control_available=FlagValue.TRUE,
        area_matched=area_matched,
        control_mean=control_mean,
        control_std=control_std,
        n_controls=n_controls,
        excess=excess,
        ratio=ratio,
        z_vs_control=z_vs_control,
    )


def _anchor_level_means(
    item_values: pd.DataFrame,
) -> tuple[dict[tuple[AnchorKey, ConditionKey, str], float], dict[AnchorKey, bool]]:
    """Reduce raw available item values to one mean per (anchor, condition, metric).

    Mirrors ``ssat.metrics.aggregate``'s macro-average policy (module
    docstring): raw values are pooled per key and averaged once, so seed
    repeats sharing one ``ConditionKey`` (design §1 항목4 -- seed is
    deliberately excluded from ``ConditionKey``) collapse to a single
    anchor-level value before any control/target comparison happens.
    """

    raw_values: dict[tuple[AnchorKey, ConditionKey, str], list[float]] = defaultdict(list)
    is_control_by_anchor: dict[AnchorKey, bool] = {}

    for row in item_values.itertuples(index=False):
        anchor_key = _anchor_key(row)
        is_control_by_anchor.setdefault(anchor_key, bool(row.is_control))
        if not row.available:
            continue
        condition_key = _condition_key(row)
        raw_values[(anchor_key, condition_key, row.metric_name)].append(float(row.degradation))

    means = {key: float(np.mean(values)) for key, values in raw_values.items()}
    return means, is_control_by_anchor


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
