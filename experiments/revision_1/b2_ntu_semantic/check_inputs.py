#!/usr/bin/env python3
"""B2 step 0: check the stored NTU runs' structure without reading any score value.

Safe to run before annotation (blind condition): it checks sample, label,
region, and group coverage and prints SHA-256 of the inputs, but never
reads or prints degradation values.

Checks:
- the video manifest has 20 samples per label for all 60 labels, and each
  sample id's action number equals label + 1;
- each run's config maps regions to the expected semantic groups;
- each run's spatial profile has every (sample, region) pair for the
  metric, its sample metrics have the manifest's labels, and every region
  has an effective area (zero-area pairs are counted, not rejected).

Example:
    python experiments/revision_1/b2_ntu_semantic/check_inputs.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from experiments.revision_1.b2_ntu_semantic.annotations import NTU60_ACTIONS
from experiments.revision_1.b2_ntu_semantic.ssat_part_scores import METRIC, RUNS, config_path, region_groups, region_id, run_dir
from experiments.revision_1.common.provenance import sha256_file
from experiments.revision_1.common.runs import REPO_ROOT

MANIFEST = REPO_ROOT / "data" / "phase3" / "ntu60_xsub_test_20" / "video_manifest.json"
SAMPLES_PER_CLASS = 20
EXPECTED_GROUPS = {"head": "head", "torso": "torso", "left_arm": "arms", "right_arm": "arms", "left_hand": "hands",
                   "right_hand": "hands", "left_leg": "legs", "right_leg": "legs", "upper_body": "upper_body",
                   "lower_body": "lower_body"}


def manifest_labels(path: Path = MANIFEST) -> dict[str, int]:
    """``sample_id -> gt_label``, checking class balance and action numbers."""

    samples = json.loads(path.read_text(encoding="utf-8"))["samples"]
    labels = {sample["sample_id"]: int(sample["gt_label"]) for sample in samples}
    counts = Counter(labels.values())
    if sorted(counts) != list(range(len(NTU60_ACTIONS))) or set(counts.values()) != {SAMPLES_PER_CLASS}:
        raise ValueError(f"expected {SAMPLES_PER_CLASS} samples for each of {len(NTU60_ACTIONS)} labels, got {dict(counts)}")
    wrong = [sample for sample, label in labels.items() if int(sample.split("A")[-1]) != label + 1]
    if wrong:
        raise ValueError(f"action number != label + 1 for {wrong[:5]}")
    return labels


def check_run(run: str, labels: dict[str, int]) -> dict[str, object]:
    """Structural checks for one run; returns counts only."""

    groups = region_groups(config_path(run))
    if groups != EXPECTED_GROUPS:
        raise ValueError(f"{run}: region -> group mapping differs: {groups}")
    metrics = run_dir(run) / "metrics"
    spatial = pd.read_parquet(metrics / "spatial_profile.parquet", columns=["sample_id", "region_key", "metric_name"])
    spatial = spatial[spatial["metric_name"] == METRIC]
    pairs = set(zip(spatial["sample_id"], spatial["region_key"].map(region_id)))
    expected = {(sample, region) for sample in labels for region in groups}
    if pairs != expected or len(spatial) != len(expected):
        raise ValueError(f"{run}: spatial profile covers {len(pairs)} of {len(expected)} (sample, region) pairs")
    samples = pd.read_parquet(metrics / "sample_metrics.parquet", columns=["sample_id", "metric_name", "gt_label", "clean_correct"])
    samples = samples[samples["metric_name"] == METRIC]
    if dict(zip(samples["sample_id"], samples["gt_label"].astype(int))) != labels:
        raise ValueError(f"{run}: sample_metrics labels differ from the manifest")
    areas = pd.read_parquet(metrics / "region_metrics.parquet", columns=["region_key", "metric_name", "effective_area_px"])
    areas = areas[(areas["metric_name"] == METRIC) & areas["region_key"].isin(spatial["region_key"])]
    if len(areas) != len(spatial) or areas["effective_area_px"].isna().any():
        raise ValueError(f"{run}: effective_area_px missing for some regions")
    zero = areas[areas["effective_area_px"] <= 0]
    return {"run": run, "samples": len(labels), "regions": len(groups), "pairs": len(pairs),
            "clean_correct_samples": int(samples["clean_correct"].astype(bool).sum()),
            "zero_area_pairs": len(zero), "zero_area_samples": int(zero["region_key"].str.split("/").str[-1].nunique()),
            "inputs_sha256": {path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
                              for path in [config_path(run), *(metrics / f"{name}.parquet" for name in
                                                               ("spatial_profile", "region_metrics", "sample_metrics"))]}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args(argv)
    labels = manifest_labels()
    result = {"manifest": {MANIFEST.relative_to(REPO_ROOT).as_posix(): sha256_file(MANIFEST)},
              "runs": [check_run(run, labels) for run in RUNS]}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
