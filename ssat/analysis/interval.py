"""IntervalEstimator: bootstrap confidence intervals for region-level estimates.

``IntervalRow`` (``analysis.types``) has grain ``(region_key, metric)`` only —
no ``sample_id``, ``invert_mask``, or ``ConditionKey``. This is deliberately
the same grain as ``ssat.metrics.aggregate.RegionMetrics`` ("one row per
(region_key, metric_name)"), and this module computes its ``point_estimate``
the same way that module does: a two-stage macro-average over non-control
items — (1) reduce each sample's items for a given (region_key, metric_name)
to one per-sample mean (pooling across whatever perturb_op/invert_mask/seed
combinations exist for that sample and region — the same population
``SpatialProfile`` pools over), then (2) average those per-sample values
across samples. This module's contribution is bootstrapping stage (2) by
resampling *samples* with replacement to attach a percentile confidence
interval to that same point estimate.

No other analysis submodule is imported (``analysis.interval → analysis.types``
only) — this module re-derives the per-sample/per-region reduction directly
from a raw ``item_values`` frame (the same shape ``analysis.stability``/
``analysis.control`` consume), rather than depending on the anchor-table
module's ``AnchorTable`` (which carries no degradation values) or any other
analysis submodule's output.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ssat.analysis.types import IntervalRow, region_key_column

DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_CI_LOW_PCT = 2.5
DEFAULT_CI_HIGH_PCT = 97.5
DEFAULT_RANDOM_SEED = 0
CI_METHOD = "percentile"


def compute_intervals(
    item_values: pd.DataFrame,
    *,
    metric_names: Sequence[str] | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_low_pct: float = DEFAULT_CI_LOW_PCT,
    ci_high_pct: float = DEFAULT_CI_HIGH_PCT,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> list[IntervalRow]:
    """Bootstrap a percentile CI for each observed (region_key, metric) pair.

    Args:
        item_values: Identity columns (``sample_id``, ``region_id``,
            ``region_instance_id``, ``is_control``, ``metric_name``) joined
            with one row per (item, metric_name): ``degradation``,
            ``available``.
        metric_names: Metrics to compute intervals for; ``None`` uses every
            distinct ``metric_name`` present.
        n_bootstrap: Number of bootstrap resamples per (region_key, metric).
        ci_low_pct: Lower percentile of the bootstrap distribution.
        ci_high_pct: Upper percentile of the bootstrap distribution.
        random_seed: Seed for the resampling RNG — same input always yields
            the same CI unless a caller opts into a different seed.

    Returns:
        One ``IntervalRow`` per (region_key, metric) that has at least one
        contributing sample; combinations with zero available non-control
        data are omitted rather than emitted with null statistics.
    """

    per_sample_means = _per_sample_region_metric_means(item_values, metric_names=metric_names)

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (_, region_key, metric_name), value in per_sample_means.items():
        grouped[(region_key, metric_name)].append(value)

    rng = np.random.default_rng(random_seed)
    rows: list[IntervalRow] = []
    for region_key, metric_name in sorted(grouped):
        values = np.asarray(grouped[(region_key, metric_name)], dtype=float)
        point_estimate = float(values.mean())
        resample_means = rng.choice(
            values, size=(n_bootstrap, values.size), replace=True
        ).mean(axis=1)
        ci_low = float(np.percentile(resample_means, ci_low_pct))
        ci_high = float(np.percentile(resample_means, ci_high_pct))
        excludes_zero = ci_low > 0.0 or ci_high < 0.0
        rows.append(
            IntervalRow(
                region_key=region_key,
                metric=metric_name,
                point_estimate=point_estimate,
                ci_low=ci_low,
                ci_high=ci_high,
                ci_method=CI_METHOD,
                n_bootstrap=n_bootstrap,
                excludes_zero=excludes_zero,
            )
        )
    return rows


def _per_sample_region_metric_means(
    item_values: pd.DataFrame, *, metric_names: Sequence[str] | None
) -> dict[tuple[str, str, str], float]:
    """Reduce non-control available items to one mean per (sample, region, metric).

    Stage 1 of the two-stage macro-average (module docstring): pools across
    whatever perturb_op/invert_mask/seed combinations a sample contributes
    for a given region and metric, mirroring
    ``ssat.metrics.aggregate``'s ``SpatialProfile`` grain and its exclusion
    of control items before any aggregation.

    Returns:
        Mapping from ``(sample_id, region_key, metric_name)`` to the mean of
        that group's ``degradation`` values.
    """

    available = item_values[item_values["available"] & ~item_values["is_control"]]
    if metric_names is not None:
        available = available[available["metric_name"].isin(metric_names)]

    if available.empty:
        return {}

    region_key = region_key_column(available)
    grouped = available.assign(region_key=region_key).groupby(
        ["sample_id", "region_key", "metric_name"], sort=False
    )["degradation"].mean()

    return {key: float(value) for key, value in grouped.items()}
