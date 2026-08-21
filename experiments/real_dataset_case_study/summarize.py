#!/usr/bin/env python3
"""Create tracked CSV/JSON summaries from completed Phase-3 output stores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ssat.utils.io import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/real_dataset_case_study"))
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("experiments/real_dataset_case_study/summary"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    matrix = json.loads((root / "matrix.json").read_text(encoding="utf-8"))
    region_frames = []
    class_frames = []
    strategy_frames = []
    run_summaries = []
    for run in matrix["runs"]:
        name = run["name"]
        output = args.output_root / name
        metrics = output / "metrics"
        analysis = output / "analysis"
        required = [metrics / "region_metrics.parquet", metrics / "class_metrics.parquet"]
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(f"run {name!r} is incomplete under {output}")
        region = pd.read_parquet(required[0]).assign(run=name)
        classes = pd.read_parquet(required[1]).assign(run=name)
        region_frames.append(region)
        class_frames.append(classes)
        strategy_path = analysis / "strategy_profile.parquet"
        if strategy_path.is_file():
            strategy_frames.append(pd.read_parquet(strategy_path).assign(run=name))
        primary = region[region["metric_name"] == "margin_drop"]
        run_summaries.append(
            {
                "run": name,
                "n_regions": int(len(primary)),
                "mean_region_margin_drop": float(primary["metric_mean"].mean()),
                "mean_region_flip_rate": float(primary["flip_rate"].mean()),
                "minimum_accuracy": run["minimum_accuracy"],
            }
        )
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(region_frames, ignore_index=True).to_csv(
        args.summary_dir / "region_metrics.csv", index=False
    )
    pd.concat(class_frames, ignore_index=True).to_csv(
        args.summary_dir / "class_metrics.csv", index=False
    )
    if strategy_frames:
        pd.concat(strategy_frames, ignore_index=True).to_csv(
            args.summary_dir / "strategy_profiles.csv", index=False
        )
    write_json_atomic(args.summary_dir / "run_summary.json", {"runs": run_summaries})


if __name__ == "__main__":
    main()
