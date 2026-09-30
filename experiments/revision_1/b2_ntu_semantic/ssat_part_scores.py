#!/usr/bin/env python3
"""B2 step 2: class x body-part-group SSAT scores from the stored NTU runs (implementation plan section 8.4).

For each run and sample population, the per-(sample, region) margin-drop
degradation in ``metrics/spatial_profile.parquet`` is reduced to one value
per (sample, semantic group) by averaging the group's regions (left and
right), then to ``S[c, g]``, the mean over the samples of class ``c``. This
is the report's class x semantic-group matrix (``ssat.report.assembler``).
Scores:
- ``raw``: ``S``;
- ``lift`` (primary): ``S`` z-scored per group across classes (ddof 0);
- ``area_adjusted_lift``: the same after replacing each (sample, part)
  value with the residual of ``log1p(max(d, 0))`` regressed on
  ``log(effective_area_px)`` over all (sample, atomic part) pairs.

Refuses to run unless every annotator sheet is complete and committed
unchanged (blind condition, protocol.md). Writes
``summary/class_group_scores.csv``.

Example:
    python experiments/revision_1/b2_ntu_semantic/ssat_part_scores.py --annotators A B
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.revision_1.b2_ntu_semantic.annotations import require_blind_annotations
from experiments.revision_1.common.provenance import write_provenance
from experiments.revision_1.common.runs import CASE_STUDY_DIR

B2_DIR = Path(__file__).resolve().parent
SUMMARY_DIR = B2_DIR / "summary"
METRIC = "margin_drop"
RUNS = {"exact": "ntu60_tsm_exact", "crop_free": "ntu60_tsm_crop_free"}
POPULATIONS = ("all", "clean_correct")
ALL_GROUPS = ("head", "torso", "arms", "hands", "legs", "upper_body", "lower_body")
ATOMIC_GROUPS = ALL_GROUPS[:5]


def run_dir(run: str) -> Path:
    return CASE_STUDY_DIR / "results" / RUNS[run]


def config_path(run: str) -> Path:
    return CASE_STUDY_DIR / "configs" / f"{RUNS[run]}.yaml"


def region_groups(path: Path) -> dict[str, str]:
    """``region_id -> semantic_group`` from an audit config."""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {region["region_id"]: region.get("semantic_group", region["region_id"]) for region in config["regions"]}


def region_id(region_key: str) -> str:
    """Region family id from a region key (``head::head/<sample>`` -> ``head``)."""

    return region_key.split("::", 1)[0]


def load_region_values(directory: Path, metric: str = METRIC) -> pd.DataFrame:
    """Per (sample, region) degradation with effective area, gt label, and clean correctness."""

    metrics = directory / "metrics"
    spatial = pd.read_parquet(metrics / "spatial_profile.parquet", columns=["sample_id", "region_key", "metric_name", "degradation"])
    spatial = spatial[spatial["metric_name"] == metric].drop(columns="metric_name")
    areas = pd.read_parquet(metrics / "region_metrics.parquet", columns=["region_key", "metric_name", "effective_area_px"])
    areas = areas[areas["metric_name"] == metric].drop(columns="metric_name")
    samples = pd.read_parquet(metrics / "sample_metrics.parquet", columns=["sample_id", "metric_name", "gt_label", "clean_correct"])
    samples = samples[samples["metric_name"] == metric].drop(columns="metric_name")
    frame = spatial.merge(areas, on="region_key", how="left", validate="one_to_one").merge(samples, on="sample_id", how="left",
                                                                                            validate="many_to_one")
    frame["region_id"] = frame["region_key"].map(region_id)
    return frame


def area_residuals(frame: pd.DataFrame, groups: Mapping[str, str]) -> pd.Series:
    """Residual of ``log1p(max(d, 0))`` on ``log(effective_area_px)`` for atomic-group regions (NaN elsewhere)."""

    mask = frame["region_id"].map(groups).isin(ATOMIC_GROUPS) & frame["degradation"].notna() & (frame["effective_area_px"] > 0)
    y = np.log1p(frame.loc[mask, "degradation"].clip(lower=0).to_numpy(dtype=float))
    x = np.log(frame.loc[mask, "effective_area_px"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    residuals = pd.Series(np.nan, index=frame.index)
    residuals[mask] = y - (intercept + slope * x)
    return residuals


def class_group_matrix(frame: pd.DataFrame, groups: Mapping[str, str], value: str = "degradation") -> pd.DataFrame:
    """``S[c, g]``: mean over class samples of each sample's mean over the group's regions.

    Returns a long frame with ``label_id``, ``group``, ``value``, ``n_samples``.
    """

    valid = frame[frame[value].notna()].assign(group=lambda f: f["region_id"].map(groups).fillna(f["region_id"]))
    per_sample = valid.groupby(["sample_id", "gt_label", "group"], as_index=False)[value].mean()
    per_class = per_sample.groupby(["gt_label", "group"], as_index=False).agg(value=(value, "mean"), n_samples=("sample_id", "nunique"))
    return per_class.rename(columns={"gt_label": "label_id"}).astype({"label_id": int})


def lift(matrix: pd.DataFrame) -> pd.Series:
    """Per-group z-score of ``value`` across classes (population sd)."""

    grouped = matrix.groupby("group")["value"]
    return (matrix["value"] - grouped.transform("mean")) / grouped.transform(lambda values: values.std(ddof=0))


def score_rows(frame: pd.DataFrame, groups: Mapping[str, str]) -> pd.DataFrame:
    """Raw, lift, and area-adjusted-lift scores for one run's region values, for each population."""

    frame = frame.assign(area_residual=area_residuals(frame, groups))
    rows = []
    for population in POPULATIONS:
        subset = frame if population == "all" else frame[frame["clean_correct"].astype(bool)]
        raw = class_group_matrix(subset, groups)
        rows.append(raw.assign(score="raw", population=population))
        rows.append(raw.assign(value=lift(raw), score="lift", population=population))
        adjusted = class_group_matrix(subset, groups, value="area_residual")
        rows.append(adjusted.assign(value=lift(adjusted), score="area_adjusted_lift", population=population))
    return pd.concat(rows, ignore_index=True)


def all_scores() -> pd.DataFrame:
    """Scores for every run in ``RUNS``."""

    return pd.concat([score_rows(load_region_values(run_dir(run)), region_groups(config_path(run))).assign(run=run) for run in RUNS],
                     ignore_index=True)[["run", "population", "score", "label_id", "group", "value", "n_samples"]]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--annotators", nargs="+", default=["A", "B"])
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    args = parser.parse_args(argv)
    require_blind_annotations(args.annotators)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    path = args.summary_dir / "class_group_scores.csv"
    all_scores().sort_values(["run", "population", "score", "label_id", "group"]).to_csv(path, index=False)
    path.chmod(0o644)
    inputs = {f"{run}_{name}": run_dir(run) / "metrics" / f"{name}.parquet" for run in RUNS
              for name in ("spatial_profile", "region_metrics", "sample_metrics")}
    write_provenance(args.summary_dir, inputs=inputs, extra={"annotators": args.annotators, "metric": METRIC},
                     filename="ssat_part_scores.provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
