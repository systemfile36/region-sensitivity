#!/usr/bin/env python3
"""P0-2: recompute the paper's reported numbers from the frozen baseline stores.

Every expected value below is quoted from ``docs/internal/paper/manuscript.tex``
(the submitted manuscript) at the rounding the paper uses; the observed
value is recomputed from the stored parquet outputs, not copied from the
report. Checks:

* ImageNet reliability grade shares (K=3 runs; paper Sec. 3.2, 2 d.p.);
* ImageNet clean top-1 accuracy and mean region ``margin_drop`` (K=1 runs,
  which generated the paper table; the K=3 runs are reported alongside);
* ImageNet top-region share: the four central cells are the most frequent
  most-vulnerable cells at roughly 14-17 % (report figure run: mnv2_050 exact);
* NTU body-part ranking (upper_body largest at 2.657, then arms and torso,
  head smallest at 0.060) and the class-conditioned lower-body values;
* synthetic Q1-Q5 values against ``results_crop_free/verdicts.json``.

A mismatch fails the script; the plan requires recording the cause and
stopping before further experiments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.revision_1.common.loading import (
    load_region_metrics,
    load_reliability,
    load_sample_meta,
    load_spatial_profile,
)
from experiments.revision_1.common.provenance import write_json_shared, write_provenance
from experiments.revision_1.common.runs import BASELINE_RUNS, K1_PARITY_RUNS, SYNTHETIC_DIR, RunRef
from ssat.utils.io import load_json

SUMMARY_DIR = Path(__file__).resolve().parent / "summary"
PAPER = "docs/internal/paper/manuscript.tex"

IMAGENET_GRADES = {  # manuscript.tex l.490-492: HIGH/LOW/UNRELIABLE %
    "imagenet_mnv2_050_exact_k3": (14.28, 26.69, 59.03),
    "imagenet_mnv2_050_crop_free_k3": (14.54, 23.85, 61.61),
    "imagenet_mnv2_100_exact_k3": (13.93, 26.87, 59.20),
    "imagenet_mnv2_100_crop_free_k3": (14.29, 24.57, 61.14),
}
IMAGENET_TABLE = {  # manuscript.tex l.430-436: clean top-1 %, mean region margin_drop
    "imagenet_mnv2_050_exact": (66.06, 0.0438),
    "imagenet_mnv2_050_crop_free": (62.20, 0.0398),
    "imagenet_mnv2_100_exact": (73.03, 0.0404),
    "imagenet_mnv2_100_crop_free": (70.94, 0.0376),
}
CENTRAL_CELLS = ("grid_4x4/r1/c1", "grid_4x4/r1/c2", "grid_4x4/r2/c1", "grid_4x4/r2/c2")
TOP_SHARE_RANGE_PCT = (14, 17)  # manuscript.tex l.466-468: "roughly 14--17%"
TOP_SHARE_FIGURE_RUNS = ("imagenet_mnv2_050_exact_k3", "imagenet_mnv2_050_exact_k1")
NTU_GROUPS = {"upper_body": 2.657, "head": 0.060}  # manuscript.tex l.515-517 (3 d.p.)
NTU_CLAPPING_LABEL, NTU_CLAPPING_LOWER_BODY = 9, -0.09  # "A10: clapping", l.526-527 (2 d.p.)
NTU_SHOE_LABEL = 16  # "A17: take off a shoe": lower_body is the largest group, l.528-529
SYNTHETIC_Q = {  # manuscript.tex Table tab:q1q5 and l.396-399
    "Q1_patch_region_rank": 1,
    "Q2_multiplier": 175.65,
    "Q3_patch_rank_in_m_normal": 16,
    "Q4_reproduced": 5,
    "Q5_margin_pp": 95.75,
    "Q5_shortcut_drop_pp": 89.55,
    "Q5_normal_drop_pp": -6.20,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=SUMMARY_DIR / "paper_numbers.json")
    return parser.parse_args(argv)


def _check(name: str, expected: Any, observed: Any, passed: bool, **details: Any) -> dict[str, Any]:
    return {"check": name, "expected": expected, "observed": observed, "pass": bool(passed), **details}


def imagenet_grade_checks() -> list[dict[str, Any]]:
    """Compare HIGH/LOW/UNRELIABLE shares (and MODERATE = 0) with the paper."""

    checks = []
    for name, expected in IMAGENET_GRADES.items():
        reliability = load_reliability(BASELINE_RUNS[name])
        shares = reliability["reliability_grade"].value_counts(normalize=True) * 100
        observed = tuple(round(float(shares.get(grade, 0.0)), 2) for grade in ("high", "low", "unreliable"))
        moderate = float(shares.get("moderate", 0.0))
        checks.append(
            _check(
                f"grades_pct[{name}]",
                list(expected),
                list(observed),
                observed == expected and moderate == 0.0 and len(reliability) == 160_000,
                moderate_pct=moderate,
                n_anchors=len(reliability),
            )
        )
    return checks


def _clean_accuracy_pct(run: RunRef) -> tuple[float, int]:
    meta = load_sample_meta(run)
    return 100.0 * float(meta["clean_correct"].mean()), len(meta)


def imagenet_table_checks() -> list[dict[str, Any]]:
    """Compare clean accuracy and mean region margin_drop (K=1 source, K=3 shown)."""

    checks = []
    for stem, (accuracy, mean_drop) in IMAGENET_TABLE.items():
        k1, k3 = K1_PARITY_RUNS[f"{stem}_k1"], BASELINE_RUNS[f"{stem}_k3"]
        acc_k1, n_k1 = _clean_accuracy_pct(k1)
        acc_k3, _ = _clean_accuracy_pct(k3)
        drop_k1 = float(load_region_metrics(k1)["metric_mean"].mean())
        drop_k3 = float(load_region_metrics(k3)["metric_mean"].mean())
        checks.append(
            _check(
                f"clean_accuracy_pct[{stem}]",
                accuracy,
                round(acc_k1, 2),
                round(acc_k1, 2) == accuracy and round(acc_k3, 2) == accuracy and n_k1 == 10_000,
                k3_value=round(acc_k3, 2),
                n_samples=n_k1,
            )
        )
        checks.append(
            _check(
                f"mean_region_margin_drop[{stem}]",
                mean_drop,
                round(drop_k1, 4),
                round(drop_k1, 4) == mean_drop and round(drop_k3, 4) == mean_drop,
                k1_unrounded=drop_k1,
                k3_unrounded=drop_k3,
            )
        )
    return checks


def top_region_shares(run: RunRef) -> pd.Series:
    """Share of samples whose largest-margin_drop cell is each cell (ties: region_key ascending)."""

    profile = load_spatial_profile(run).dropna(subset=["degradation"])
    ordered = profile.assign(neg=-profile["degradation"]).sort_values(["sample_id", "neg", "region_key"])
    top = ordered.drop_duplicates("sample_id", keep="first")
    cell = top["region_key"].str.split("::", n=1).str[1]
    return cell.value_counts(normalize=True)


def imagenet_top_region_checks() -> list[dict[str, Any]]:
    """Check the central four cells lead the top-region share at ~14-17 %."""

    checks = []
    runs = [*(BASELINE_RUNS[name] for name in IMAGENET_GRADES), *K1_PARITY_RUNS.values()]
    low, high = TOP_SHARE_RANGE_PCT
    for run in runs:
        shares = top_region_shares(run)
        top4 = tuple(sorted(shares.nlargest(4).index))
        central = {cell: round(100.0 * float(shares.get(cell, 0.0)), 2) for cell in CENTRAL_CELLS}
        in_range = all(low <= round(value) <= high for value in central.values())
        is_figure_run = run.name in TOP_SHARE_FIGURE_RUNS
        checks.append(
            _check(
                f"top_region_share[{run.name}]",
                {"top4": sorted(CENTRAL_CELLS), "central_pct_rounded_in": [low, high]},
                {"top4": list(top4), "central_pct": central},
                (top4 == tuple(sorted(CENTRAL_CELLS)) and in_range) if is_figure_run else True,
                gating=is_figure_run,
                central_in_range=in_range,
                central_are_top4=top4 == tuple(sorted(CENTRAL_CELLS)),
            )
        )
    return checks


def ntu_group_degradation(run: RunRef) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-group means and the class x group matrix, as the report computes them."""

    regions = load_json(run.dump / "run_manifest.json")["resolved_config"]["regions"]
    group_of = {region["region_id"]: region.get("semantic_group") or region["region_id"] for region in regions}
    profile = load_spatial_profile(run).dropna(subset=["degradation"])
    profile["group"] = profile["region_key"].str.split("::", n=1).str[0].map(group_of)
    per_sample = profile.groupby(["sample_id", "group"], as_index=False)["degradation"].mean()
    group_means = per_sample.groupby("group")["degradation"].mean().sort_values(ascending=False)
    meta = load_sample_meta(run)
    by_class = per_sample.merge(meta[["sample_id", "gt_label"]], on="sample_id")
    matrix = by_class.groupby(["gt_label", "group"])["degradation"].mean().unstack("group")
    return group_means.to_frame("mean_degradation"), matrix


def ntu_checks() -> list[dict[str, Any]]:
    """Check the NTU body-part ranking and class-conditioned examples (exact run)."""

    group_means, matrix = ntu_group_degradation(BASELINE_RUNS["ntu60_tsm_exact"])
    means = group_means["mean_degradation"]
    order = list(means.index)
    checks = [
        _check(
            "ntu_group_order",
            {"first": "upper_body", "next_two": ["arms", "torso"], "last": "head"},
            order,
            order[0] == "upper_body" and set(order[1:3]) == {"arms", "torso"} and order[-1] == "head",
            group_means={group: float(value) for group, value in means.items()},
        )
    ]
    for group, expected in NTU_GROUPS.items():
        observed = round(float(means[group]), 3)
        checks.append(_check(f"ntu_group_mean[{group}]", expected, observed, observed == expected))
    clapping = round(float(matrix.loc[NTU_CLAPPING_LABEL, "lower_body"]), 2)
    checks.append(
        _check("ntu_class_lower_body[A10 clapping]", NTU_CLAPPING_LOWER_BODY, clapping, clapping == NTU_CLAPPING_LOWER_BODY)
    )
    shoe_row = matrix.loc[NTU_SHOE_LABEL]
    checks.append(
        _check(
            "ntu_class_top_group[A17 take off a shoe]",
            "lower_body",
            str(shoe_row.idxmax()),
            shoe_row.idxmax() == "lower_body",
            row={group: float(value) for group, value in shoe_row.items()},
        )
    )
    return checks


def synthetic_checks() -> list[dict[str, Any]]:
    """Compare the stored Q1-Q5 verdicts with the paper's table."""

    verdicts = load_json(SYNTHETIC_DIR / "results_crop_free" / "verdicts.json")
    observed = {
        "Q1_patch_region_rank": verdicts["Q1_identifies_patch_region"]["patch_region_rank"],
        "Q2_multiplier": round(verdicts["Q2_separated_from_baseline"]["multiplier"], 2),
        "Q3_patch_rank_in_m_normal": verdicts["Q3_distinguishes_normal_model"]["patch_region_rank_in_m_normal"],
        "Q4_reproduced": len(verdicts["Q4_robust_to_fill_strategy"]["reproduced_in"]),
        "Q5_margin_pp": round(verdicts["Q5_predicts_generalization_gap"]["margin_points"], 2),
        "Q5_shortcut_drop_pp": round(verdicts["Q5_predicts_generalization_gap"]["shortcut_accuracy_drop_points"], 2),
        "Q5_normal_drop_pp": round(verdicts["Q5_predicts_generalization_gap"]["normal_accuracy_drop_points"], 2),
    }
    all_pass = all(verdict.get("pass", True) for key, verdict in verdicts.items() if key.startswith("Q"))
    return [
        _check(f"synthetic[{key}]", expected, observed[key], np.isclose(observed[key], expected) and all_pass)
        for key, expected in SYNTHETIC_Q.items()
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checks = [
        *imagenet_grade_checks(),
        *imagenet_table_checks(),
        *imagenet_top_region_checks(),
        *ntu_checks(),
        *synthetic_checks(),
    ]
    failed = [check["check"] for check in checks if not check["pass"]]
    for check in checks:
        print(f"{'PASS' if check['pass'] else 'FAIL'}  {check['check']}: expected {check['expected']}, observed {check['observed']}")
    payload = {"paper_source": PAPER, "all_pass": not failed, "failed": failed, "checks": checks}
    write_json_shared(args.output, payload)
    inputs: dict[str, Path] = {"paper": Path(__file__).resolve().parents[3] / PAPER}
    for run in [*BASELINE_RUNS.values(), *K1_PARITY_RUNS.values()]:
        inputs[f"{run.name}_metrics"] = run.metrics
        inputs[f"{run.name}_analysis"] = run.analysis
    inputs["synthetic_verdicts"] = SYNTHETIC_DIR / "results_crop_free" / "verdicts.json"
    write_provenance(args.output.parent, inputs=inputs, filename=f"{args.output.stem}.provenance.json")
    print(f"wrote {args.output}: {len(checks) - len(failed)}/{len(checks)} pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
