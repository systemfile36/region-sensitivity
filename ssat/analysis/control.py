"""A2 ControlComparator.

Compares each target AnchorKey/ConditionKey's degradation against its
matched random-area-match controls, answering "is this sensitivity
region-specific, or just because something was occluded?"

This module deliberately re-derives ``AnchorKey``/``ConditionKey`` from raw
item rows rather than importing ``analysis.indexer``/``analysis.reader`` --
the same pattern A3(c) already follows (recomputing directly from A1's
AnchorTable plus item context rather than importing it), accepting the same
``ConditionKey`` formula duplication A3(c) already does (``region_key`` no
longer needs to be duplicated for this: both modules share
``analysis.types.region_key_column``). ``compare_to_controls`` therefore
depends only on
``analysis.types`` plus ``pandas``/``numpy``/stdlib, consuming a plain
``item_values`` frame the caller assembles from
``AnalysisReader.item_context()`` + ``AnalysisReader.item_metrics`` -- never
``ssat.metrics.types.ItemMetrics`` objects directly.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd

from ssat.analysis.types import (
    AnchorKey,
    ConditionKey,
    ControlComparisonRow,
    ControlPairRow,
    FlagValue,
    region_key_column,
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
    """Compare each target's degradation to its matched controls.

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
            ``abs(control_mean)`` falls below this (avoids the ratio
            exploding as the denominator approaches zero).
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
        ``None`` (i.e., when no usable control is available).
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
    matching (the same "never treat unavailable as false" principle applied
    to this axis too).
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

    # "Reveal inconsistency, don't smooth it over": area_matched
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
    repeats sharing one ``ConditionKey`` (seed is deliberately excluded
    from ``ConditionKey``) collapse to a single anchor-level value before
    any control/target comparison happens.

    Grouped with pandas rather than a per-row Python loop: ``item_values``
    can run into the tens of millions of rows for a real dataset, so
    ``AnchorKey``/``ConditionKey`` objects are only constructed once per
    grouped result (anchors x conditions x metrics), not once per raw row.
    """

    df = item_values.assign(
        region_key=region_key_column(item_values),
        perturb_params_hash=_perturb_params_hash_column(item_values),
    )

    # .first() takes the first row's value per group in frame order, same
    # as the previous itertuples + dict.setdefault behavior.
    is_control_by_group = df.groupby(
        ["sample_id", "region_key", "invert_mask"], sort=False
    )["is_control"].first()
    is_control_by_anchor = {
        AnchorKey(sample_id=sample_id, region_key=region_key, invert_mask=bool(invert_mask)): bool(
            is_control
        )
        for (sample_id, region_key, invert_mask), is_control in is_control_by_group.items()
    }

    available = df[df["available"]]
    grouped_means = available.groupby(
        [
            "sample_id",
            "region_key",
            "invert_mask",
            "perturb_op",
            "perturb_params_hash",
            "metric_name",
        ],
        sort=False,
    )["degradation"].mean()

    means: dict[tuple[AnchorKey, ConditionKey, str], float] = {}
    for (
        sample_id,
        region_key,
        invert_mask,
        perturb_op,
        perturb_params_hash,
        metric_name,
    ), value in grouped_means.items():
        anchor_key = AnchorKey(
            sample_id=sample_id, region_key=region_key, invert_mask=bool(invert_mask)
        )
        condition_key = ConditionKey(
            perturb_op=perturb_op, perturb_params_hash=perturb_params_hash
        )
        means[(anchor_key, condition_key, metric_name)] = float(value)

    return means, is_control_by_anchor


def _perturb_params_hash_column(df: pd.DataFrame) -> pd.Series:
    """Vectorized form of ``ConditionKey.perturb_params_hash``.

    ``perturb_params_json`` takes only a handful of distinct values across
    an entire frame (one per configured (op, params) combination actually
    run), so this hashes each distinct value once via ``.map()`` instead of
    calling ``sha256_bytes`` once per row.
    """

    distinct = df["perturb_params_json"].unique()
    hash_by_json = {value: sha256_bytes(value.encode("utf-8")) for value in distinct}
    return df["perturb_params_json"].map(hash_by_json)
