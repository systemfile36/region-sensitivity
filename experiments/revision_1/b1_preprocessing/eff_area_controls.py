#!/usr/bin/env python3
"""B1-4: P_exact grades with effective-area-matched controls (P_exact_EA).

Under the exact protocol a control's visible area rarely matches its
target's (B1-1). On the A2 K=20 exact run, this keeps only control items
whose effective area is within +-5 % of the target's, averages the eligible
items of each control anchor, and computes z from at least three such
anchors (``protocol.json`` ``b1_4_effective_area_controls``). Grades are
compared with (a) the K=3 prefix, (b) all 20 controls under P_exact, and
(d) all 20 controls of the A2 crop-free run, on the same 2,000 samples.

Output: ``summary/eff_area_controls.csv`` (long format: section, comparison,
cell_type, stat, value).

Example:
    python experiments/revision_1/b1_preprocessing/eff_area_controls.py
"""

from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.revision_1.a2_control_count.build_control_tensor import (
    DEFAULT_OUTPUT_DIR as TENSOR_DIR,
    control_targets,
    subset_stats,
)
from experiments.revision_1.a2_control_count.run_ablation import FEATURE_DIR, aligned_inputs, cell_masks
from experiments.revision_1.common.agreement import GRADE_ORDER, grade_agreement
from experiments.revision_1.common.grade_engine import (
    UNAVAILABLE,
    GradeParams,
    compute_scales,
    grade_anchors,
    grade_labels,
    with_control_stats,
)
from experiments.revision_1.common.loading import load_item_values
from experiments.revision_1.common.provenance import write_provenance
from experiments.revision_1.common.runs import A2_RUNS
from ssat.analysis.control import _perturb_params_hash_column
from ssat.analysis.types import region_key_column

SUMMARY_DIR = Path(__file__).resolve().parent / "summary"
EXACT_RUN = "a2_imagenet_mnv2_050_exact_k20"
CF_RUN = "a2_imagenet_mnv2_050_cf_k20"
TOLERANCE = 0.05
MIN_CONTROLS = 3
ANCHOR = ["sample_id", "region_key", "invert_mask"]


def eligible_items(item_values: pd.DataFrame, tolerance: float = TOLERANCE) -> pd.DataFrame:
    """Return control items with their target anchor, condition, control index, and eligibility.

    Raises:
        ValueError: If a target anchor's items disagree on effective area.
    """

    frame = item_values.assign(
        region_key=region_key_column(item_values), perturb_params_hash=_perturb_params_hash_column(item_values)
    )
    targets = frame[~frame["is_control"]]
    areas = targets.groupby(ANCHOR)["effective_area_px"].agg(["min", "max"])
    if (areas["min"] != areas["max"]).any():
        raise ValueError("target effective area varies across a target anchor's items")
    target_area = areas["min"].rename("target_area").reset_index()

    controls = frame[frame["is_control"]].rename(columns={"region_key": "control_key"})
    params = controls.groupby(["sample_id", "control_key", "invert_mask"], sort=False)["region_params_json"].first()
    controls = controls.merge(control_targets(params), on=["sample_id", "control_key", "invert_mask"], how="left")
    controls = controls.merge(target_area, on=ANCHOR, how="left")
    if controls["target_area"].isna().any():
        raise ValueError("controls reference a target anchor without items")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = controls["effective_area_px"].to_numpy(float) / controls["target_area"].to_numpy(float)
    controls["eligible"] = (controls["target_area"].to_numpy() > 0) & (np.abs(ratio - 1.0) <= tolerance)
    return controls[[*ANCHOR, "perturb_op", "perturb_params_hash", "control_index", "degradation", "available", "eligible"]]


def ea_control_stats(
    anchor_pos: np.ndarray, condition_pos: np.ndarray, control_index: np.ndarray, values: np.ndarray,
    eligible: np.ndarray, shape: tuple[int, int, int], min_controls: int = MIN_CONTROLS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Effective-area-matched control mean, ddof-0 std, and count per (anchor, condition).

    Each control anchor contributes the mean of its eligible items; conditions
    with fewer than ``min_controls`` contributing anchors get NaN statistics.
    """

    keep = eligible & np.isfinite(values)
    sums = np.zeros(shape)
    counts = np.zeros(shape)
    np.add.at(sums, (anchor_pos[keep], condition_pos[keep], control_index[keep]), values[keep])
    np.add.at(counts, (anchor_pos[keep], condition_pos[keep], control_index[keep]), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_control = np.where(counts > 0, sums / counts, np.nan)
    n = np.isfinite(per_control).sum(axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
        mean = np.nanmean(per_control, axis=-1)
        std0 = np.nanstd(per_control, axis=-1)
    usable = n >= min_controls
    return np.where(usable, mean, np.nan), np.where(usable, std0, np.nan), n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metric", default="margin_drop")
    parser.add_argument("--output", type=Path, default=SUMMARY_DIR / "eff_area_controls.csv")
    args = parser.parse_args(argv)

    exact_run = A2_RUNS[EXACT_RUN]
    features, tensor, _ = aligned_inputs(exact_run, args.metric)
    cf_features, cf_tensor, _ = aligned_inputs(A2_RUNS[CF_RUN], args.metric)
    if not (np.array_equal(features["sample_id"].to_numpy(object), cf_features["sample_id"].to_numpy(object))
            and np.array_equal(features["region_key"].to_numpy(object), cf_features["region_key"].to_numpy(object))):
        raise ValueError("exact and crop-free anchors differ")

    def grade(feat: pd.DataFrame, target: np.ndarray, mean: np.ndarray, std0: np.ndarray, n) -> pd.DataFrame:
        return grade_anchors(with_control_stats(feat, target, mean, std0, n), GradeParams(), scales=compute_scales(feat))

    items = eligible_items(load_item_values(exact_run, (args.metric,)))
    anchor_pos = pd.Series(np.arange(len(features)), index=pd.MultiIndex.from_frame(features[ANCHOR]))
    condition_pos = {condition: index for index, condition in enumerate(tensor.conditions)}
    a_pos = anchor_pos.reindex(pd.MultiIndex.from_frame(items[ANCHOR])).to_numpy()
    c_pos = np.array([condition_pos[key] for key in zip(items["perturb_op"], items["perturb_params_hash"])])
    values = np.where(items["available"].to_numpy(), items["degradation"].to_numpy(float), np.nan)
    shape = (len(features), len(tensor.conditions), tensor.n_controls)
    ea_mean, ea_std0, ea_n = ea_control_stats(a_pos, c_pos, items["control_index"].to_numpy(), values,
                                              items["eligible"].to_numpy(), shape)

    graded = {
        "a_exact_k3": grade(features, tensor.target, *subset_stats(tensor, range(3))),
        "b_exact_k20": grade(features, tensor.target, *subset_stats(tensor, range(tensor.n_controls))),
        "c_exact_ea": grade(features, tensor.target, ea_mean, ea_std0, ea_n),
        "d_cf_k20": grade(cf_features, cf_tensor.target, *subset_stats(cf_tensor, range(cf_tensor.n_controls))),
    }
    masks = cell_masks(features["cell_type"])
    rows = []
    for name, table in graded.items():
        for cell_type, mask in masks.items():
            codes = table["grade"].to_numpy()[mask]
            for code, grade_name in enumerate(GRADE_ORDER):
                rows.append(("grade_distribution", name, cell_type, f"{grade_name}_share", float((codes == code).mean())))
            rows.append(("grade_distribution", name, cell_type, "exceeds_control_unavailable_share",
                         float((table["exceeds_control"].to_numpy()[mask] == UNAVAILABLE).mean())))
    for (name_a, a), (name_b, b) in itertools.combinations(graded.items(), 2):
        for cell_type, mask in masks.items():
            stats = grade_agreement(pd.Series(grade_labels(a["grade"].to_numpy()[mask])),
                                    pd.Series(grade_labels(b["grade"].to_numpy()[mask])))
            for stat in ("agreement", "kappa_linear", "delta_ge2"):
                rows.append(("agreement", f"{name_a}_vs_{name_b}", cell_type, stat, stats[stat]))
    item_cell = features["cell_type"].to_numpy()[a_pos]
    for cell_type, mask in masks.items():
        per_condition = ea_n[mask].ravel()
        item_mask = np.ones(len(items), bool) if cell_type == "all" else item_cell == cell_type
        rows.extend([
            ("eligibility", "c_exact_ea", cell_type, "eligible_item_share", float(items["eligible"].to_numpy()[item_mask].mean())),
            ("eligibility", "c_exact_ea", cell_type, "eligible_controls_mean", float(per_condition.mean())),
            *[("eligibility", "c_exact_ea", cell_type, f"eligible_controls_p{q}", float(np.percentile(per_condition, q)))
              for q in (10, 50, 90)],
            ("eligibility", "c_exact_ea", cell_type, "share_conditions_ge3", float((per_condition >= MIN_CONTROLS).mean())),
            ("eligibility", "c_exact_ea", cell_type, "share_conditions_0", float((per_condition == 0).mean())),
        ])
    frame = pd.DataFrame(rows, columns=["section", "comparison", "cell_type", "stat", "value"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    args.output.chmod(0o644)
    write_provenance(
        args.output.parent,
        inputs={EXACT_RUN: exact_run.dump, CF_RUN: A2_RUNS[CF_RUN].dump,
                "exact_tensor": TENSOR_DIR / f"{EXACT_RUN}__{args.metric}.npz",
                "cf_tensor": TENSOR_DIR / f"{CF_RUN}__{args.metric}.npz",
                "exact_features": FEATURE_DIR / f"{EXACT_RUN}__{args.metric}.parquet"},
        extra={"tolerance": TOLERANCE, "min_controls": MIN_CONTROLS, "metric": args.metric},
        filename="eff_area_controls.provenance.json",
    )
    print(frame[frame["cell_type"] == "all"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
