#!/usr/bin/env python3
"""Judge Q1-Q5 and the section 3.5 fill-strategy sensitivity check, and report.

Implements docs/STAGE9_SYNTHETIC_SHORTCUT_DESIGN_v1.md sections 5-9. This
script only *reads* the dumps/metrics run_audit.py already produced and the
accuracy numbers evaluate_accuracy.py already produced; it makes no model or
audit calls of its own. Thresholds are loaded from thresholds.json rather
than hardcoded here, because that file is the pre-registered record of what
counts as a pass (design section 5/9: "결과를 본 뒤 변경하지 않는다") --
keeping it as data, not code, makes it obvious nothing here was tuned after
seeing the numbers.

Per design section 9, a failed threshold is reported as failed, not adjusted
-- this script never raises or exits non-zero on a Q1-Q5 failure; it always
finishes and writes report.md with whatever the pre-registered criteria
actually say.

Run as: python3 experiments/synthetic_shortcut/evaluate.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from common import PATCH_REGION_KEY
from run_audit import FILL_PARAMS, PRIMARY_METRIC, RUN_SPECS
from ssat.metrics.store import load_metrics
from ssat.metrics.viz.heatmap import save_heatmap_views

FILL_STRATEGIES = tuple(FILL_PARAMS)  # ("constant_fill", "mean_fill", "blur", "gaussian_noise", "patch_shuffle")

# Representative heatmaps only -- design section 8 asks for "M_shortcut/
# M_normal 각각 대표 샘플", not one heatmap set per fill strategy.
HEATMAP_RUN_IDS = ("shortcut_A_constant_fill", "normal_A_constant_fill", "shortcut_B_constant_fill")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one evaluation pass."""

    parser = argparse.ArgumentParser(description="Judge Q1-Q5 and write the L3 experiment report.")
    parser.add_argument(
        "--results-dir", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path(__file__).resolve().parent / "thresholds.json",
    )
    return parser.parse_args()


def _load_region_means(metrics_dir: Path) -> dict[str, float]:
    """Load one run's region_key -> mean margin_drop degradation.

    Rows with metric_mean=None (no valid items for that region) are dropped;
    a region absent here means it could not be scored, not that it scored
    zero.
    """

    _, result, _ = load_metrics(metrics_dir)
    return {
        row.region_key: row.metric_mean
        for row in result.region_metrics
        if row.metric_name == PRIMARY_METRIC and row.metric_mean is not None
    }


def _write_region_csv(path: Path, region_means: dict[str, float]) -> None:
    """Save one run's region ranking (most to least degraded) as a CSV."""

    ranking = sorted(region_means.items(), key=lambda item: item[1], reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "region_key", "metric_mean"])
        for rank, (region_key, mean) in enumerate(ranking, start=1):
            writer.writerow([rank, region_key, mean])


def _patch_rank(region_means: dict[str, float]) -> int:
    """Return the patch region's 1-indexed rank by descending mean degradation.

    Higher margin_drop means the classification margin fell further after
    perturbing that region, i.e. the model relied on it more (see
    ssat/metrics/builtin_metrics/continuous.py::MarginDrop.compute) -- so
    rank 1 is the single most-relied-upon region.
    """

    ranking = sorted(region_means.items(), key=lambda item: item[1], reverse=True)
    for rank, (region_key, _mean) in enumerate(ranking, start=1):
        if region_key == PATCH_REGION_KEY:
            return rank
    raise ValueError(f"{PATCH_REGION_KEY!r} not found among this run's audited regions")


def _patch_multiplier(region_means: dict[str, float]) -> float:
    """Return patch-region mean degradation divided by the other regions' mean (Q2)."""

    patch_mean = region_means[PATCH_REGION_KEY]
    others = [value for key, value in region_means.items() if key != PATCH_REGION_KEY]
    other_mean = sum(others) / len(others)
    if other_mean == 0:
        return float("inf") if patch_mean > 0 else 0.0
    return patch_mean / other_mean


def _spearman_correlation(reference: dict[str, float], other: dict[str, float]) -> float:
    """Spearman rank correlation between two runs' per-region degradation.

    Spearman correlation is exactly the Pearson correlation of the two
    series' ranks, so this rank-transforms first and then calls pandas'
    default (Pearson) Series.corr -- pandas' own method="spearman" option
    delegates to scipy internally, which is not a project dependency; this
    avoids adding one just for this single call.
    """

    shared_keys = sorted(set(reference) & set(other))
    reference_ranks = pd.Series([reference[key] for key in shared_keys]).rank()
    other_ranks = pd.Series([other[key] for key in shared_keys]).rank()
    return float(reference_ranks.corr(other_ranks))


def _judge(
    thresholds: dict, region_means_by_run: dict[str, dict[str, float]], accuracy: dict
) -> dict:
    """Evaluate every pre-registered Q1-Q5 criterion from thresholds.json."""

    baseline = region_means_by_run["shortcut_A_constant_fill"]
    q1_rank = _patch_rank(baseline)
    q2_multiplier = _patch_multiplier(baseline)
    normal_rank = _patch_rank(region_means_by_run["normal_A_constant_fill"])
    reproduced_in = [
        fill
        for fill in FILL_STRATEGIES
        if _patch_rank(region_means_by_run[f"shortcut_A_{fill}"]) == 1
    ]

    shortcut_drop = accuracy["shortcut"]["A"] - accuracy["shortcut"]["C"]
    normal_drop = accuracy["normal"]["A"] - accuracy["normal"]["C"]
    q5_margin_points = (shortcut_drop - normal_drop) * 100.0

    return {
        "Q1_identifies_patch_region": {
            "pass": q1_rank == 1,
            "patch_region_rank": q1_rank,
        },
        "Q2_separated_from_baseline": {
            "pass": q2_multiplier >= thresholds["q2_multiplier"],
            "multiplier": q2_multiplier,
            "threshold": thresholds["q2_multiplier"],
        },
        "Q3_distinguishes_normal_model": {
            "pass": normal_rank != 1,
            "patch_region_rank_in_m_normal": normal_rank,
        },
        "Q4_robust_to_fill_strategy": {
            "pass": len(reproduced_in) >= thresholds["q4_min_fill_strategies"],
            "reproduced_in": reproduced_in,
            "min_required": thresholds["q4_min_fill_strategies"],
        },
        "Q5_predicts_generalization_gap": {
            "pass": q5_margin_points >= thresholds["q5_min_margin_points"],
            "shortcut_accuracy_drop_points": shortcut_drop * 100.0,
            "normal_accuracy_drop_points": normal_drop * 100.0,
            "margin_points": q5_margin_points,
            "threshold_points": thresholds["q5_min_margin_points"],
        },
        "B_auxiliary_control": {
            "patch_region_rank_on_random_placement": _patch_rank(
                region_means_by_run["shortcut_B_constant_fill"]
            ),
            "note": (
                "not part of the pre-registered Q1-Q5 verdict; expected to be "
                "far from rank 1 if the tool tracks the patch's *position* "
                "rather than some unrelated bias toward the top-left cell"
            ),
        },
    }


def _render_report(verdicts: dict, correlations: dict[str, float], thresholds: dict) -> str:
    """Render the Q1-Q5 verdicts and section 3.5 correlations as Markdown."""

    lines = ["# L3 Synthetic-Shortcut Experiment Report", ""]
    lines.append("## Q1-Q5 (pre-registered, docs/STAGE9_SYNTHETIC_SHORTCUT_DESIGN_v1.md section 5)")
    lines.append("")
    lines.append("| Question | Result | Detail |")
    lines.append("|---|---|---|")
    question_labels = {
        "Q1_identifies_patch_region": "Q1. identifies patch region",
        "Q2_separated_from_baseline": "Q2. separated from baseline",
        "Q3_distinguishes_normal_model": "Q3. distinguishes M_normal",
        "Q4_robust_to_fill_strategy": "Q4. robust to fill strategy",
        "Q5_predicts_generalization_gap": "Q5. predicts generalization gap",
    }
    for key, label in question_labels.items():
        verdict = verdicts[key]
        status = "PASS" if verdict["pass"] else "FAIL"
        detail = ", ".join(f"{k}={v}" for k, v in verdict.items() if k != "pass")
        lines.append(f"| {label} | **{status}** | {detail} |")
    lines.append("")
    control = verdicts["B_auxiliary_control"]
    lines.append(
        f"**B auxiliary control** (not a pass/fail criterion): patch region rank under "
        f"random placement = {control['patch_region_rank_on_random_placement']}. "
        f"{control['note']}."
    )
    lines.append("")
    lines.append(
        "## Section 3.5 -- fill-strategy sensitivity "
        "(Spearman correlation of region ranking against constant_fill)"
    )
    lines.append("")
    lines.append("| fill strategy | Spearman correlation |")
    lines.append("|---|---|")
    for fill, correlation in correlations.items():
        lines.append(f"| {fill} | {correlation:.3f} |")
    lines.append("")
    lines.append(f"_Thresholds used (from thresholds.json): {json.dumps(thresholds)}_")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Load every run's metrics/accuracy, judge Q1-Q5, and write the report."""

    args = parse_args()
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    metrics_dir = args.results_dir / "metrics"

    region_means_by_run = {
        spec.run_id: _load_region_means(metrics_dir / spec.run_id) for spec in RUN_SPECS
    }
    for run_id, region_means in region_means_by_run.items():
        _write_region_csv(args.results_dir / f"region_metrics_{run_id}.csv", region_means)

    accuracy = json.loads((args.results_dir / "accuracy.json").read_text(encoding="utf-8"))
    verdicts = _judge(thresholds, region_means_by_run, accuracy)

    baseline = region_means_by_run["shortcut_A_constant_fill"]
    correlations = {
        fill: _spearman_correlation(baseline, region_means_by_run[f"shortcut_A_{fill}"])
        for fill in FILL_STRATEGIES
        if fill != "constant_fill"
    }

    report = _render_report(verdicts, correlations, thresholds)
    report_path = args.results_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"saved report to {report_path}")

    # Representative heatmaps (design section 8), reusing the already-shipped
    # DebugViz V2 view rather than building a second rendering path.
    for run_id in HEATMAP_RUN_IDS:
        save_heatmap_views(
            args.results_dir / "dumps" / run_id,
            args.results_dir / "metrics" / run_id,
            args.results_dir / "heatmaps" / run_id,
            metric_name=PRIMARY_METRIC,
            n_samples=5,
        )
    print(f"saved heatmaps under {args.results_dir / 'heatmaps'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
