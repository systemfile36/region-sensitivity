"""A1 step 1: build the per-anchor feature table the grade engine sweeps over.

For each run and metric, reads the stored analysis tables (strategy
stability, control comparison, seed stability, reliability) and recomputes
the region bootstrap CIs at every registered level. The CIs reuse the stored
analysis's random stream: ``compute_intervals`` draws from one generator
seeded once and consumed over every (region_key, metric) pair in sorted
order, so the draws for skipped metrics are replayed (their shapes come from
``spatial_profile.parquet``). The 95 % result must reproduce the stored
``intervals.parquet`` (``excludes_zero`` and ``point_estimate`` exactly,
bounds within 1e-12), otherwise the build fails.

Output: ``<output-dir>/<run>__<metric>.parquet`` plus a ``.json`` sidecar
with the condition list, run scales, and the interval parity result.

Example:
    python experiments/revision_1/a1_threshold/build_features.py --runs all
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from experiments.revision_1.common.grade_engine import CI_PERCENTILES, compute_scales
from experiments.revision_1.common.loading import (
    iter_latest_perturbed,
    read_metric_table,
    verify_metrics_source,
)
from experiments.revision_1.common.provenance import write_json_shared
from experiments.revision_1.common.runs import BASELINE_RUNS, REVISION_RESULTS_DIR, RunRef
from ssat.analysis.interval import _per_sample_region_metric_means
from ssat.utils.io import load_json, sha256_file

KEY = ["sample_id", "region_key", "invert_mask", "metric_name"]
DEFAULT_OUTPUT_DIR = REVISION_RESULTS_DIR / "a1" / "features"
PARITY_BOUND_TOLERANCE = 1e-12
_CELL = re.compile(r"/r(\d+)/c(\d+)$")


# --- stored analysis tables --------------------------------------------------


def strategy_table(analysis_dir: Path, metric: str) -> pd.DataFrame:
    """Return one row per target anchor with an ``op__<op>`` column per operator."""

    table = read_metric_table(analysis_dir / "strategy_stability.parquet", [metric])
    values = pd.DataFrame([json.loads(text) for text in table["strategy_values_json"]], index=table.index)
    values = values.reindex(columns=sorted(values.columns)).add_prefix("op__")
    return pd.concat([table[KEY], values], axis=1)


def control_table(analysis_dir: Path, metric: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Return one row per target anchor with ``cond<i>__*`` columns, and the condition list."""

    table = read_metric_table(analysis_dir / "control_comparison.parquet", [metric])
    conditions = sorted(set(zip(table["perturb_op"], table["perturb_params_hash"])))
    merged: pd.DataFrame | None = None
    for index, (op, params_hash) in enumerate(conditions):
        part = table[(table["perturb_op"] == op) & (table["perturb_params_hash"] == params_hash)]
        part = pd.DataFrame(
            {
                **{column: part[column].to_numpy() for column in KEY},
                f"cond{index}__excess": part["excess"].to_numpy(dtype=float),
                f"cond{index}__control_mean": part["control_mean"].to_numpy(dtype=float),
                f"cond{index}__std0": part["control_std"].to_numpy(dtype=float),
                f"cond{index}__n": part["n_controls"].to_numpy(dtype=np.int64),
                f"cond{index}__available": (part["control_available"] == "true").to_numpy(),
            }
        )
        merged = part if merged is None else merged.merge(part, on=KEY, how="outer")
    if merged is None:
        merged = pd.DataFrame(columns=KEY)
    for index in range(len(conditions)):
        merged[f"cond{index}__n"] = merged[f"cond{index}__n"].fillna(0).astype(np.int64)
        merged[f"cond{index}__available"] = merged[f"cond{index}__available"].fillna(False).astype(bool)
    listed = [
        {"index": index, "perturb_op": op, "perturb_params_hash": params_hash}
        for index, (op, params_hash) in enumerate(conditions)
    ]
    return merged, listed


def seed_table(analysis_dir: Path, metric: str) -> pd.DataFrame:
    """Return the max evaluable ``seed_cv`` per anchor (anchors with none are absent)."""

    table = read_metric_table(
        analysis_dir / "seed_stability.parquet",
        [metric],
        columns=("sample_id", "region_key", "invert_mask", "metric_name", "seed_cv"),
    )
    table = table[table["seed_cv"].notna()]
    return table.groupby(KEY, sort=False)["seed_cv"].max().rename("seed_cv_max").reset_index()


def stored_reliability(analysis_dir: Path, metric: str) -> pd.DataFrame:
    """Return the stored flags and grade, prefixed ``stored__``."""

    table = read_metric_table(
        analysis_dir / "reliability.parquet",
        [metric],
        columns=(
            *KEY,
            "sign_consistent",
            "exceeds_control",
            "multi_strategy",
            "ci_excludes_zero",
            "seed_stable",
            "reliability_grade",
        ),
    )
    return table.rename(columns={column: f"stored__{column}" for column in table.columns if column not in KEY})


# --- bootstrap intervals -------------------------------------------------------


def bootstrap_draw_counts(metrics_dir: Path) -> dict[tuple[str, str], int]:
    """Return the number of contributing samples per (region_key, metric) in the store."""

    table = pq.read_table(
        metrics_dir / "spatial_profile.parquet", columns=["region_key", "metric_name", "degradation"]
    ).to_pandas()
    table = table[table["degradation"].notna()]
    counts = table.groupby(["region_key", "metric_name"], sort=False).size()
    return {key: int(value) for key, value in counts.items()}


def target_item_values(run: RunRef, metric: str) -> pd.DataFrame:
    """Return target-item values for one metric in ``AnalysisReader.item_values`` order."""

    columns = ["item_id", "sample_id", "region_id", "region_instance_id", "is_control"]
    targets = pc.field("is_control") == False  # noqa: E712
    context = pa.concat_tables(iter_latest_perturbed(run.dump, columns, filter_expression=targets))
    context = context.select(columns).to_pandas()
    values = read_metric_table(
        run.metrics / "item_metrics.parquet",
        [metric],
        columns=("item_id", "metric_name", "degradation", "available"),
    )
    return context.merge(values, on="item_id", how="inner")


def replicate_intervals(
    item_values: pd.DataFrame,
    draw_counts: Mapping[tuple[str, str], int],
    metric: str,
    *,
    levels: Sequence[str] = tuple(CI_PERCENTILES),
    n_bootstrap: int = 1000,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Recompute ``compute_intervals`` for ``metric`` at several CI levels.

    Args:
        item_values: Target item values for ``metric`` in the analysis input order.
        draw_counts: Samples per (region_key, metric) over every metric the
            stored analysis bootstrapped; draws for other metrics are replayed.
        metric: Metric to return intervals for.
        levels: Keys of ``CI_PERCENTILES``.
        n_bootstrap: Bootstrap resamples per region.
        random_seed: Seed of the shared generator.

    Returns:
        One row per region_key with ``point_estimate``, ``n_samples``, and
        ``ci<level>__low`` / ``ci<level>__high`` columns.

    Raises:
        ValueError: If ``draw_counts`` disagrees with ``item_values``.
    """

    per_sample = _per_sample_region_metric_means(item_values, metric_names=[metric])
    grouped: dict[str, list[float]] = {}
    for (_, region_key, _), value in per_sample.items():
        grouped.setdefault(region_key, []).append(value)
    expected = {region_key for region_key, name in draw_counts if name == metric}
    if expected != set(grouped):
        raise ValueError(f"{metric}: spatial_profile regions differ from item_values regions")

    rng = np.random.default_rng(random_seed)
    rows = []
    for region_key, name in sorted(draw_counts):
        n = draw_counts[(region_key, name)]
        if name != metric:
            rng.choice(n, size=(n_bootstrap, n), replace=True)
            continue
        values = np.asarray(grouped[region_key], dtype=float)
        if values.size != n:
            raise ValueError(f"{region_key}/{metric}: {values.size} samples, spatial_profile has {n}")
        means = rng.choice(values, size=(n_bootstrap, values.size), replace=True).mean(axis=1)
        row: dict[str, object] = {"region_key": region_key, "point_estimate": float(values.mean()), "n_samples": n}
        for level in levels:
            low_pct, high_pct = CI_PERCENTILES[level]
            row[f"ci{level}__low"] = float(np.percentile(means, low_pct))
            row[f"ci{level}__high"] = float(np.percentile(means, high_pct))
        rows.append(row)
    return pd.DataFrame(rows)


def interval_parity(intervals: pd.DataFrame, analysis_dir: Path, metric: str) -> dict[str, object]:
    """Compare the recomputed 95 % intervals with ``intervals.parquet``."""

    stored = read_metric_table(analysis_dir / "intervals.parquet", [metric], metric_column="metric")
    joined = stored.merge(intervals, on="region_key", how="outer", indicator=True)
    both = joined[joined["_merge"] == "both"]
    recomputed_excludes = (both["ci95__low"] > 0) | (both["ci95__high"] < 0)
    bound_diff = np.maximum(
        (both["ci_low"] - both["ci95__low"]).abs(), (both["ci_high"] - both["ci95__high"]).abs()
    )
    result = {
        "n_regions": int(len(stored)),
        "unmatched_regions": int((joined["_merge"] != "both").sum()),
        "excludes_zero_mismatches": int((recomputed_excludes != both["excludes_zero"]).sum()),
        "point_estimate_mismatches": int((both["point_estimate_x"] != both["point_estimate_y"]).sum()),
        "max_bound_abs_diff": float(bound_diff.max()) if len(both) else 0.0,
        "bounds_bitwise_equal": bool((bound_diff == 0).all()),
    }
    result["passed"] = (
        result["unmatched_regions"] == 0
        and result["excludes_zero_mismatches"] == 0
        and result["point_estimate_mismatches"] == 0
        and result["max_bound_abs_diff"] <= PARITY_BOUND_TOLERANCE
    )
    return result


# --- assembly ------------------------------------------------------------------


def cell_columns(region_key: pd.Series) -> pd.DataFrame:
    """Return ``region_group``, ``region_cell``, and ``cell_type`` for each region key."""

    group = region_key.str.split("::", n=1).str[0]
    match = region_key.str.extract(_CELL)
    rows = pd.to_numeric(match[0])
    cols = pd.to_numeric(match[1])
    cell = pd.Series(pd.NA, index=region_key.index, dtype=object)
    cell_type = pd.Series(pd.NA, index=region_key.index, dtype=object)
    has_cell = rows.notna() & cols.notna()
    if has_cell.any():
        n_rows, n_cols = int(rows.max()) + 1, int(cols.max()) + 1
        cell[has_cell] = "r" + rows[has_cell].astype(int).astype(str) + "/c" + cols[has_cell].astype(int).astype(str)
        row_edge = rows.isin([0, n_rows - 1])
        col_edge = cols.isin([0, n_cols - 1])
        cell_type[has_cell] = np.where(
            row_edge & col_edge, "corner", np.where(~row_edge & ~col_edge, "center", "edge")
        )[has_cell.to_numpy()]
    return pd.DataFrame({"region_group": group, "region_cell": cell, "cell_type": cell_type})


def build_features(
    run: RunRef, metric: str = "margin_drop", *, n_bootstrap: int = 1000, random_seed: int = 0
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the feature table and sidecar metadata for one run and metric.

    Raises:
        ValueError: If the stores do not chain to the dump, or the recomputed
            95 % intervals fail the parity check.
    """

    verify_metrics_source(run)
    manifest = load_json(run.analysis / "analysis_manifest.json")
    if manifest["source_metrics_manifest_hash"] != sha256_file(run.metrics / "metrics_manifest.json"):
        raise ValueError(f"{run.name}: analysis store was not computed from this metrics store")
    if (manifest["n_bootstrap"], manifest["random_seed"]) != (n_bootstrap, random_seed):
        raise ValueError(f"{run.name}: stored analysis used n_bootstrap/seed {manifest['n_bootstrap']}/{manifest['random_seed']}")

    strategy = strategy_table(run.analysis, metric)
    control, conditions = control_table(run.analysis, metric)
    features = strategy.merge(control, on=KEY, how="outer")
    features = features.merge(seed_table(run.analysis, metric), on=KEY, how="left")

    intervals = replicate_intervals(
        target_item_values(run, metric),
        bootstrap_draw_counts(run.metrics),
        metric,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    parity = interval_parity(intervals, run.analysis, metric)
    if not parity["passed"]:
        raise ValueError(f"{run.name}: recomputed 95% intervals do not match intervals.parquet: {parity}")
    ci_columns = [column for column in intervals.columns if column.startswith("ci")]
    features = features.merge(intervals[["region_key", *ci_columns]], on="region_key", how="left")

    stored = stored_reliability(run.analysis, metric)
    features = features.merge(stored, on=KEY, how="outer", indicator=True)
    if (features["_merge"] != "both").any():
        raise ValueError(f"{run.name}: feature anchors differ from reliability.parquet anchors")
    features = features.drop(columns="_merge").sort_values(KEY, kind="stable").reset_index(drop=True)
    features = pd.concat([features, cell_columns(features["region_key"])], axis=1)

    scales = compute_scales(features)
    meta = {
        "run": run.name,
        "metric": metric,
        "n_anchors": int(len(features)),
        "operators": [column[4:] for column in features.columns if column.startswith("op__")],
        "conditions": conditions,
        "scales": {"sign_scale": scales.sign_scale, "std_scale": scales.std_scale},
        "n_bootstrap": n_bootstrap,
        "random_seed": random_seed,
        "interval_parity_95": parity,
        "n_interval_regions": int(len(intervals)),
        "inputs": {
            "analysis_manifest_sha256": sha256_file(run.analysis / "analysis_manifest.json"),
            "metrics_manifest_sha256": sha256_file(run.metrics / "metrics_manifest.json"),
            "run_manifest_sha256": sha256_file(run.dump / "run_manifest.json"),
        },
    }
    return features, meta


def feature_paths(output_dir: Path, run_name: str, metric: str) -> tuple[Path, Path]:
    """Return the ``(parquet, json)`` paths of one feature table."""

    stem = output_dir / f"{run_name}__{metric}"
    return stem.with_suffix(".parquet"), stem.with_suffix(".json")


def load_features(output_dir: Path, run_name: str, metric: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a feature table and its sidecar."""

    table_path, meta_path = feature_paths(output_dir, run_name, metric)
    return pd.read_parquet(table_path), load_json(meta_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", nargs="+", default=["all"], help="run names from BASELINE_RUNS, or 'all'")
    parser.add_argument("--metric", default="margin_drop")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    names = list(BASELINE_RUNS) if args.runs == ["all"] else args.runs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        started = time.monotonic()
        features, meta = build_features(BASELINE_RUNS[name], args.metric)
        table_path, meta_path = feature_paths(args.output_dir, name, args.metric)
        features.to_parquet(table_path, index=False)
        table_path.chmod(0o644)
        meta["feature_table_sha256"] = sha256_file(table_path)
        write_json_shared(meta_path, meta)
        parity = meta["interval_parity_95"]
        print(
            f"{name}: {meta['n_anchors']} anchors, {len(meta['conditions'])} conditions, "
            f"interval parity ok (max bound diff {parity['max_bound_abs_diff']:.3g}), "
            f"{time.monotonic() - started:.0f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
