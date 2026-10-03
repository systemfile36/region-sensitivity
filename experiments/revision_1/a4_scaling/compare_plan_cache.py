#!/usr/bin/env python3
"""A4 post hoc: compare the sweep before and after the planner fix (``deviations.md`` D-013).

Reads the pre-registered sweep (``results/a4/``) and the re-measured
settings (``results/a4_plan_cache/``), flattens both with
``fit_scaling.flatten``, and writes to ``summary_plan_cache/``:

* ``measurements.csv``: the re-measured rows, as in ``summary/measurements.csv``;
* ``comparison.csv``: per re-measured setting, chunks per sample, mean
  audit-loop and ``ssat run`` time per item before and after, their ratio
  (after / before), the same ratio divided by that of the drift control
  ``n500_g4_v5_k0`` (one chunk per sample, so the fix cannot change it),
  CVs, peak RSS, GPU utilization, and stored bytes;
* ``fits.csv``: ``time = a + b * items`` over the re-measured settings only,
  before and after, for the audit loop and ``ssat run``, with the predicted
  time of the 3.21 M-item Sec. 3.2 setting;
* ``estimate_accuracy.csv``: ``ssat estimate`` against the re-measured run.

Example:
    python experiments/revision_1/a4_scaling/compare_plan_cache.py
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.revision_1.a4_scaling.fit_scaling import _read_jsonl, estimate_rows, flatten, ols, setting_summary
from experiments.revision_1.a4_scaling.sweep import A4_DIR, A4_RESULTS_DIR, Setting
from experiments.revision_1.common.provenance import write_provenance
from experiments.revision_1.common.runs import REVISION_RESULTS_DIR

POST_RESULTS_DIR = REVISION_RESULTS_DIR / "a4_plan_cache"
SUMMARY_DIR = A4_DIR / "summary_plan_cache"
VARIANTS_PER_CHUNK = 128
DRIFT_CONTROL = "n500_g4_v5_k0"
PAPER_SETTING = Setting(n=10000, grid=4, v=5, k=3)
TIMES = {"loop": "run_loop_s", "run": "run_s"}


def chunks_per_sample(frame: pd.DataFrame) -> pd.Series:
    """Perturbed chunks per sample: ``ceil(R * V * (1 + K) / variants_per_chunk)``."""

    perturbed = frame["regions"] * frame["v"] * (1 + frame["k"])
    return perturbed.map(lambda value: math.ceil(value / VARIANTS_PER_CHUNK))


def comparison_rows(pre: pd.DataFrame, post: pd.DataFrame) -> pd.DataFrame:
    """Per re-measured setting, before and after values and their ratios.

    Raises:
        ValueError: If a re-measured setting is missing from the pre-registered sweep.
    """

    before = setting_summary(pre).set_index("setting")
    after = setting_summary(post).set_index("setting")
    missing = set(after.index) - set(before.index)
    if missing:
        raise ValueError(f"settings not in the pre-registered sweep: {sorted(missing)}")
    before = before.loc[after.index]
    rows = after[["n", "grid", "regions", "v", "k", "items"]].copy()
    rows["chunks_per_sample"] = chunks_per_sample(rows)
    rows["n_repeats_before"] = before["n_repeats"]
    rows["n_repeats_after"] = after["n_repeats"]
    for name, column in TIMES.items():
        for label, table in (("before", before), ("after", after)):
            rows[f"{name}_ms_per_item_{label}"] = 1e3 * table[f"{column}_mean"] / table["items"]
            rows[f"{name}_cv_{label}"] = table[f"{column}_cv"]
        rows[f"{name}_ratio"] = rows[f"{name}_ms_per_item_after"] / rows[f"{name}_ms_per_item_before"]
        control = rows[f"{name}_ratio"].get(DRIFT_CONTROL, np.nan)
        rows[f"{name}_ratio_vs_drift_control"] = rows[f"{name}_ratio"] / control
    for column in ("metrics_s", "analyze_s", "report_s", "run_rss_gib", "analyze_rss_gib", "gpu_util_mean", "gpu_peak_mib", "total_bytes"):
        rows[f"{column}_before"] = before[f"{column}_mean"]
        rows[f"{column}_after"] = after[f"{column}_mean"]
    return rows.reset_index()


def subset_fits(pre: pd.DataFrame, post: pd.DataFrame) -> pd.DataFrame:
    """``time = a + b * items`` over the re-measured settings, before and after the fix."""

    settings = set(post["setting"])
    rows = []
    for label, frame in (("before", pre[pre["setting"].isin(settings)]), ("after", post)):
        for name, column in TIMES.items():
            fit = ols(frame["items"].to_numpy(float), frame[column].to_numpy(float))
            rows.append({"code": label, "time": name, **fit, "ms_per_item": 1e3 * fit["b"], "n_points": len(frame),
                         "predicted_paper_setting_s": fit["a"] + fit["b"] * PAPER_SETTING.planned_items})
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pre-root", type=Path, default=A4_RESULTS_DIR)
    parser.add_argument("--post-root", type=Path, default=POST_RESULTS_DIR)
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    args = parser.parse_args(argv)
    out = args.summary_dir
    out.mkdir(parents=True, exist_ok=True)

    pre = flatten(_read_jsonl(args.pre_root / "measurements.jsonl"))
    post = flatten(_read_jsonl(args.post_root / "measurements.jsonl"))
    outputs = {
        "measurements.csv": post,
        "comparison.csv": comparison_rows(pre, post),
        "fits.csv": subset_fits(pre, post),
        "estimate_accuracy.csv": estimate_rows(_read_jsonl(args.post_root / "estimates.jsonl"), setting_summary(post)),
    }
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)
        (out / name).chmod(0o644)
    write_provenance(out, inputs={name: root / f"{kind}.jsonl" for name, root, kind in (
        ("pre_measurements", args.pre_root, "measurements"), ("post_measurements", args.post_root, "measurements"),
        ("post_estimates", args.post_root, "estimates"))}, filename="compare_plan_cache.provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
