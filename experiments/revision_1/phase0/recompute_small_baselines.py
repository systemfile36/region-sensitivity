#!/usr/bin/env python3
"""P0-7: recompute metrics/analysis of the small baselines with the v1.0.0 code.

The stored NTU and synthetic stores cannot all be read as-is by the
revision-1 tooling:

* the synthetic metrics stores use metrics schema 1.0.0, which the current
  ``ssat.metrics.store`` rejects;
* ``ntu60_tsm_crop_free``'s ``run_manifest.json`` was rewritten after its
  metrics were computed, so the metrics-to-dump hash chain no longer holds;
* all four analyses predate the last change to ``ssat/analysis``.

This script reruns ``ssat metrics`` and ``ssat analyze`` (default settings)
over the unchanged dumps into ``results/phase0/recomputed/<run>/`` and
compares every stored parquet table with its recomputed counterpart, so the
revision can use stores with a valid provenance chain while showing they
reproduce the stored (paper) numbers. The original stores are not modified.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from experiments.revision_1.common.provenance import write_json_shared, write_provenance
from experiments.revision_1.common.runs import ORIGINAL_SMALL_RUNS, RECOMPUTED_DIR, RunRef


SUMMARY_DIR = Path(__file__).resolve().parent / "summary"
DEFAULT_RUNS = tuple(ORIGINAL_SMALL_RUNS)

TABLE_KEYS: dict[str, tuple[str, ...]] = {
    "metrics/item_metrics": ("item_id", "metric_name"),
    "metrics/sample_metrics": ("sample_id", "metric_name"),
    "metrics/region_metrics": ("region_key", "metric_name"),
    "metrics/class_metrics": ("gt_label", "metric_name"),
    "metrics/spatial_profile": ("sample_id", "region_key", "metric_name"),
    "analysis/control_comparison": (
        "sample_id", "region_key", "invert_mask", "perturb_op", "perturb_params_hash", "metric_name",
    ),
    "analysis/seed_stability": (
        "sample_id", "region_key", "invert_mask", "perturb_op", "perturb_params_hash", "metric_name",
    ),
    "analysis/strategy_stability": ("sample_id", "region_key", "invert_mask", "metric_name"),
    "analysis/rank_correlation": ("op_a", "op_b", "scope"),
    "analysis/strategy_profile": ("perturb_op",),
    "analysis/intervals": ("region_key", "metric"),
    "analysis/reliability": ("sample_id", "region_key", "invert_mask", "metric_name"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--output-root", type=Path, default=RECOMPUTED_DIR)
    parser.add_argument("--skip-compute", action="store_true", help="Only compare existing recomputed stores.")
    parser.add_argument("--output", type=Path, default=SUMMARY_DIR / "recomputed_baselines.json")
    return parser.parse_args(argv)


def recompute(run: RunRef, target: Path) -> None:
    """Run ``ssat metrics`` and ``ssat analyze`` for ``run`` into ``target``."""

    metrics_dir, analysis_dir = target / "metrics", target / "analysis"
    for command in (
        [sys.executable, "-m", "ssat", "metrics", str(run.dump), "--metrics-dir", str(metrics_dir)],
        [sys.executable, "-m", "ssat", "analyze", str(run.dump), "--metrics-dir", str(metrics_dir),
         "--analysis-dir", str(analysis_dir)],
    ):
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object and frame[column].map(lambda v: isinstance(v, (list, np.ndarray))).any():
            frame[column] = frame[column].map(lambda v: None if v is None else tuple(v))
    return frame


def compare_tables(stored_path: Path, recomputed_path: Path, keys: tuple[str, ...]) -> dict[str, Any]:
    """Compare two parquet tables row-by-row on ``keys``.

    The recomputed table is first restricted to the metrics present in the
    stored one (the synthetic stores were computed for ``margin_drop``
    only). Float columns report the maximum absolute difference
    (NaN == NaN); every other column must be exactly equal.
    """

    stored = _normalize(pq.read_table(stored_path).to_pandas())
    recomputed = _normalize(pq.read_table(recomputed_path).to_pandas())
    result: dict[str, Any] = {"n_stored": len(stored), "n_recomputed_total": len(recomputed)}
    metric_column = next((column for column in ("metric_name", "metric") if column in keys), None)
    if metric_column is not None:
        metrics = sorted(stored[metric_column].unique())
        recomputed = recomputed[recomputed[metric_column].isin(metrics)]
        result["metrics_compared"] = metrics
    result["n_recomputed"] = len(recomputed)
    if set(stored.columns) != set(recomputed.columns):
        result["column_mismatch"] = sorted(set(stored.columns) ^ set(recomputed.columns))
    stored = stored.sort_values(list(keys)).reset_index(drop=True)
    recomputed = recomputed.sort_values(list(keys)).reset_index(drop=True)
    keys_equal = len(stored) == len(recomputed) and stored[list(keys)].equals(recomputed[list(keys)])
    result["keys_equal"] = bool(keys_equal)
    if not keys_equal:
        result["identical"] = False
        return result
    differing: dict[str, Any] = {}
    for column in sorted(set(stored.columns) & set(recomputed.columns) - set(keys)):
        a, b = stored[column], recomputed[column]
        if pd.api.types.is_float_dtype(a) or pd.api.types.is_float_dtype(b):
            a_values = a.to_numpy(dtype=np.float64, na_value=np.nan)
            b_values = b.to_numpy(dtype=np.float64, na_value=np.nan)
            both_nan = np.isnan(a_values) & np.isnan(b_values)
            nan_mismatch = int((np.isnan(a_values) != np.isnan(b_values)).sum())
            diff = np.where(both_nan, 0.0, np.abs(a_values - b_values))
            max_diff = float(np.nanmax(diff)) if len(diff) else 0.0
            if nan_mismatch or max_diff > 0.0:
                differing[column] = {"max_abs_diff": max_diff, "nan_mismatch": nan_mismatch}
        else:
            unequal = int((~((a == b) | (a.isna() & b.isna()))).sum())
            if unequal:
                differing[column] = {"n_unequal": unequal}
    result["differing_columns"] = differing
    result["identical"] = not differing and "column_mismatch" not in result
    return result


def compare_run(run: RunRef, target: Path) -> dict[str, Any]:
    """Compare every stored table of ``run`` with the recomputed one."""

    tables = {}
    for name, keys in TABLE_KEYS.items():
        kind, table = name.split("/")
        stored = (run.metrics if kind == "metrics" else run.analysis) / f"{table}.parquet"
        tables[name] = compare_tables(stored, target / kind / f"{table}.parquet", keys)
    reliability = tables["analysis/reliability"]
    return {
        "stored": {"metrics": str(run.metrics), "analysis": str(run.analysis)},
        "recomputed": str(target),
        "tables": tables,
        "all_identical": all(table["identical"] for table in tables.values()),
        "reliability_identical": reliability["identical"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = {}
    for name in args.runs:
        run = ORIGINAL_SMALL_RUNS[name]
        target = args.output_root / name
        if not args.skip_compute:
            if target.exists() and any(target.iterdir()):
                raise SystemExit(f"{target} is not empty; remove it or pass --skip-compute")
            recompute(run, target)
        results[name] = compare_run(run, target)
        summary = {table: result["identical"] for table, result in results[name]["tables"].items()}
        print(f"{name}: all_identical={results[name]['all_identical']} {summary}", flush=True)
    payload = {"runs": results, "all_identical": all(r["all_identical"] for r in results.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_shared(args.output, payload)
    inputs = {f"{name}_dump": ORIGINAL_SMALL_RUNS[name].dump for name in args.runs}
    inputs.update({f"{name}_recomputed_metrics": args.output_root / name / "metrics" for name in args.runs})
    inputs.update({f"{name}_recomputed_analysis": args.output_root / name / "analysis" for name in args.runs})
    write_provenance(args.output.parent, inputs=inputs, filename=f"{args.output.stem}.provenance.json")
    print(f"wrote {args.output} (all_identical={payload['all_identical']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
