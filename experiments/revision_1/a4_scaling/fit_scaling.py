#!/usr/bin/env python3
"""A4 step 2: flatten the measurements, fit cost models, and derive the workload tables (implementation plan section 7.5).

Reads ``results/a4/{measurements,estimates}.jsonl`` and
``summary/components.json`` and writes to ``summary/``:

* ``measurements.csv``: one row per measurement (phase times and peak RSS,
  audit-loop time, GPU peak and mean utilization, bytes per subdirectory);
* ``settings.csv``: per setting, mean / std / CV over repeats and throughput;
* ``fits.csv``: per axis and pooled, ``y = a + b * items`` (OLS) for phase
  times and peak RSS, and ``bytes = c * items`` (through the origin);
* ``reference_points.csv``: the 10,000-sample K=3 baseline (manifest
  timestamps) and the A2 K=20 exact run (``run_matrix_log.jsonl``), measured
  differently from the sweep;
* ``workload_table.csv``: predicted cost of the small / medium / large
  workloads and the item count at which each phase's peak RSS would reach
  the host's memory (linear extrapolation);
* ``coarse_to_fine.csv``: analytic item counts of exhaustive 8x8 vs 2x2
  followed by a 4x4 refinement of the top 2x2 cell;
* ``estimate_accuracy.csv``: ``ssat estimate`` prediction vs measured ``run``;
* ``components.csv``: the stage breakdown from ``profile_components.py``.

Example:
    python experiments/revision_1/a4_scaling/fit_scaling.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.revision_1.a4_scaling.sweep import A4_DIR, A4_RESULTS_DIR, AXES, Setting
from experiments.revision_1.common.provenance import write_provenance
from experiments.revision_1.common.runs import A2_RESULTS_DIR, BASELINE_RUNS

SUMMARY_DIR = A4_DIR / "summary"
PHASES = ("run", "metrics", "analyze", "report")
SUBDIRS = ("clean", "perturbed", "index", "metrics", "analysis", "report", "other", "total")
TIME_COLUMNS = ("run_s", "run_loop_s", "preflight_s", "metrics_s", "analyze_s", "report_s", "pipeline_s")
RSS_COLUMNS = tuple(f"{phase}_rss_gib" for phase in PHASES)
BYTES_COLUMNS = ("raw_bytes", "total_bytes")
WORKLOADS = {
    "small": Setting(n=1000, grid=4, v=5, k=3),
    "medium (paper setting)": Setting(n=10000, grid=4, v=5, k=3),
    "large": Setting(n=10000, grid=8, v=5, k=10),
}
HOST_MEMORY_GIB = 125.0


def flatten(records: list[dict]) -> pd.DataFrame:
    """One row per measurement with phase times (s), peak RSS (GiB), and bytes."""

    rows = []
    for record in records:
        row = {key: record[key] for key in ("axis", "level", "repeat", "setting", "n", "grid", "regions", "v", "k",
                                            "planned_items", "manifest_items", "started_at", "run_loop_s",
                                            "gpu_peak_mib", "gpu_util_mean")}
        for phase in PHASES:
            row[f"{phase}_s"] = record["phases"][phase]["elapsed_s"]
            row[f"{phase}_rss_gib"] = record["phases"][phase]["peak_rss_kb"] / 2**20
        row["preflight_s"] = row["run_s"] - row["run_loop_s"] if row["run_loop_s"] is not None else np.nan
        row["pipeline_s"] = sum(row[f"{phase}_s"] for phase in PHASES)
        after_run, final = record["bytes_after_run"], record["bytes_final"]
        row["raw_bytes"] = after_run["total"]
        row["total_bytes"] = final["total"]
        row.update({f"bytes_{name}": final[name] for name in SUBDIRS})
        row["items"] = record["manifest_items"]
        row["loop_items_per_s"] = row["items"] / row["run_loop_s"] if row["run_loop_s"] else np.nan
        row["run_items_per_s"] = row["items"] / row["run_s"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    if (frame["planned_items"] != frame["manifest_items"]).any():
        raise ValueError("manifest item counts differ from the planned counts")
    return frame


def setting_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, and CV over repeats per setting (shared settings keep their owning axis)."""

    columns = [*TIME_COLUMNS, *RSS_COLUMNS, "loop_items_per_s", "run_items_per_s", "gpu_peak_mib", "gpu_util_mean", *BYTES_COLUMNS]
    grouped = frame.groupby(["setting", "n", "grid", "regions", "v", "k", "items"], sort=False)
    out = grouped[columns].mean().add_suffix("_mean")
    out = out.join(grouped[list(TIME_COLUMNS)].std(ddof=1).add_suffix("_std"))
    for column in TIME_COLUMNS:
        out[f"{column}_cv"] = out[f"{column}_std"] / out[f"{column}_mean"]
    out["n_repeats"] = grouped.size()
    return out.reset_index()


def ols(x: np.ndarray, y: np.ndarray, *, intercept: bool = True) -> dict[str, float]:
    """Least squares ``y = a + b x`` (or ``y = b x``) with R^2 (uncentered when through the origin)."""

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    design = np.column_stack([np.ones_like(x), x]) if intercept else x[:, None]
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    total = ((y - y.mean()) ** 2).sum() if intercept else (y**2).sum()
    return {"a": float(coef[0]) if intercept else 0.0, "b": float(coef[-1]), "r2": float(1 - (residual**2).sum() / total),
            "n_points": int(len(x)), "x_min": float(x.min()), "x_max": float(x.max())}


def axis_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Measurements of each axis, including the shared settings measured by another axis."""

    parts = []
    for axis, settings in AXES.items():
        keys = {setting.key for setting in settings}
        parts.append(frame[frame["setting"].isin(keys)].assign(fit_axis=axis))
    return pd.concat([*parts, frame.assign(fit_axis="pooled")], ignore_index=True)


def fit_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Per axis and pooled fits of time, RSS (affine) and bytes (through the origin) against items."""

    rows = []
    for axis, group in axis_rows(frame).groupby("fit_axis", sort=False):
        for column in (*TIME_COLUMNS, *RSS_COLUMNS):
            rows.append({"axis": axis, "y": column, "model": "a + b*items", **ols(group["items"], group[column])})
        for column in BYTES_COLUMNS:
            rows.append({"axis": axis, "y": column, "model": "c*items", **ols(group["items"], group[column], intercept=False)})
    return pd.DataFrame(rows)


def reference_rows() -> pd.DataFrame:
    """The 10k K=3 baseline run (manifest timestamps) and the A2 K=20 exact run (run_matrix log)."""

    baseline = BASELINE_RUNS["imagenet_mnv2_050_exact_k3"]
    manifest = json.loads((baseline.dump / "run_manifest.json").read_text(encoding="utf-8"))
    seconds = (datetime.fromisoformat(manifest["finished_at"].replace("Z", "+00:00"))
               - datetime.fromisoformat(manifest["started_at"].replace("Z", "+00:00"))).total_seconds()
    items = int(sum(manifest["counts_by_status"].values()))
    raw = sum(p.stat().st_size for name in ("clean", "perturbed", "index") for p in (baseline.dump / name).rglob("*") if p.is_file())
    rows = [{"reference": baseline.name, "n": 10000, "regions": 16, "v": 5, "k": 3, "items": items, "seconds": seconds,
             "raw_bytes": raw, "method": "run_manifest started_at -> finished_at (audit loop plus dump finalization; no preflight)"}]
    log = A2_RESULTS_DIR / "run_matrix_log.jsonl"
    for line in log.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["run"] == "a2_imagenet_mnv2_050_exact_k20" and record["step"] == "run":
            rows.append({"reference": record["run"], "n": 2000, "regions": 16, "v": 5, "k": 20,
                         "items": Setting(2000, 4, 5, 20).planned_items, "seconds": record["elapsed_s"], "raw_bytes": np.nan,
                         "method": "run_matrix_log elapsed of `ssat run` (includes preflight; not an isolated wrapper)"})
    frame = pd.DataFrame(rows)
    frame["items_per_s"] = frame["items"] / frame["seconds"]
    return frame


def workload_rows(fits: pd.DataFrame) -> pd.DataFrame:
    """Predicted time, storage, and peak RSS of the reference workloads from the pooled fits."""

    pooled = fits[fits["axis"] == "pooled"].set_index("y")
    max_measured = pooled.loc["run_s", "x_max"]
    rows = []
    for name, setting in WORKLOADS.items():
        items = setting.planned_items
        row = {"workload": name, "n": setting.n, "grid": setting.grid, "v": setting.v, "k": setting.k, "items": items,
               "extrapolated": items > max_measured}
        for column in (*TIME_COLUMNS, *RSS_COLUMNS):
            row[f"pred_{column}"] = pooled.loc[column, "a"] + pooled.loc[column, "b"] * items
        for column in BYTES_COLUMNS:
            row[f"pred_{column}"] = pooled.loc[column, "b"] * items
        rows.append(row)
    for column in RSS_COLUMNS:
        a, b = pooled.loc[column, "a"], pooled.loc[column, "b"]
        rows.append({"workload": f"items at which {column} reaches {HOST_MEMORY_GIB:.0f} GiB", "items": (HOST_MEMORY_GIB - a) / b if b > 0 else np.nan,
                     "extrapolated": True})
    return pd.DataFrame(rows)


def coarse_to_fine_rows(v: int = 5, k: int = 3) -> pd.DataFrame:
    """Items per sample: exhaustive grids vs a 2x2 pass followed by a 4x4 refinement of the top 2x2 cell."""

    per_region = v * (1 + k)
    strategies = {"exhaustive 4x4": 16, "exhaustive 8x8": 64, "2x2 then 4x4 inside the top cell (8x8 resolution there)": 4 + 16}
    exhaustive = 1 + 64 * per_region
    return pd.DataFrame([{"strategy": name, "regions_evaluated": regions, "items_per_sample": 1 + regions * per_region,
                          "share_of_exhaustive_8x8": (1 + regions * per_region) / exhaustive, "v": v, "k": k}
                         for name, regions in strategies.items()])


def estimate_rows(estimates: list[dict], settings: pd.DataFrame) -> pd.DataFrame:
    """``ssat estimate`` remaining-seconds prediction vs mean measured run and loop time."""

    if not estimates:
        return pd.DataFrame()
    frame = pd.DataFrame(estimates).drop_duplicates("setting", keep="last")
    merged = frame.merge(settings[["setting", "items", "run_s_mean", "run_loop_s_mean"]], on="setting", how="inner")
    merged["predicted_over_run"] = merged["estimated_remaining_seconds"] / merged["run_s_mean"]
    merged["predicted_over_loop"] = merged["estimated_remaining_seconds"] / merged["run_loop_s_mean"]
    return merged


def component_rows(components: dict) -> pd.DataFrame:
    """Per-item cost of each stage and the throughput ceiling it implies."""

    inference = components["inference"]
    stages = [
        ("(a) source decode, one process", components["decode"], "per sample"),
        ("(b) chunk preparation, worker pool", components["preparation_pool"], "worker pool plus main-process hydration"),
        ("(b) chunk preparation, one process", components["preparation_serial"], "per-item CPU cost"),
        ("(c) adapter preprocessing (main process)", inference["preprocess"], "main process"),
        ("(c) model forward incl. transfer (GPU)", inference["forward"], "main process waits"),
        ("effective-area mask transform (main process)", inference["effective_area_mask_transform"], "main process"),
    ]
    rows = [{"stage": name, "ms_per_item": stats["ms_per_item"], "items_per_s": stats["items_per_s"], "items": stats["items"], "note": note}
            for name, stats, note in stages]
    main_ms = sum(inference[key]["ms_per_item"] for key in ("preprocess", "forward", "effective_area_mask_transform"))
    rows.append({"stage": "main process total (c + mask transform), excluding dump write", "ms_per_item": main_ms,
                 "items_per_s": 1000 / main_ms, "items": inference["items"], "note": "ceiling of the audit loop"})
    return pd.DataFrame(rows)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-root", type=Path, default=A4_RESULTS_DIR)
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    args = parser.parse_args(argv)
    out = args.summary_dir

    frame = flatten(_read_jsonl(args.results_root / "measurements.jsonl"))
    frame.to_csv(out / "measurements.csv", index=False)
    settings = setting_summary(frame)
    settings.to_csv(out / "settings.csv", index=False)
    fits = fit_rows(frame)
    fits.to_csv(out / "fits.csv", index=False)
    reference_rows().to_csv(out / "reference_points.csv", index=False)
    workload_rows(fits).to_csv(out / "workload_table.csv", index=False)
    coarse_to_fine_rows().to_csv(out / "coarse_to_fine.csv", index=False)
    estimate_rows(_read_jsonl(args.results_root / "estimates.jsonl"), settings).to_csv(out / "estimate_accuracy.csv", index=False)
    components = out / "components.json"
    if components.is_file():
        component_rows(json.loads(components.read_text(encoding="utf-8"))).to_csv(out / "components.csv", index=False)
    write_provenance(out, inputs={"measurements": args.results_root / "measurements.jsonl",
                                  "estimates": args.results_root / "estimates.jsonl", "components": components},
                     filename="fit_scaling.provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
