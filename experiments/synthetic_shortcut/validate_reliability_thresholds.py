#!/usr/bin/env python3
"""Check whether ssat.analysis's z_vs_control_threshold / seed_cv_threshold
defaults (2.0 / 0.2, ssat/analysis/reliability.py) behave sensibly against
real control-region and multi-seed data -- something the five original
shortcut_A_* runs could not exercise at all (IMPLE_PLAN_CONTROL_STABILITY_v1.md
§5 단계9 follow-up). Reads the run run_threshold_validation.py produces
(shortcut_A_constant_fill_thresholds: 2 controls/target-region, 3 seeds).

Important caveat this script's own output calls out explicitly: this run
uses only one perturb_op (constant_fill), so every anchor's
StrategyStabilityRow.n_strategies == 1, and A6's ``multi_strategy`` flag is
*always* FALSE here (ssat/analysis/reliability.py::_multi_strategy_flag
requires the dominant sign to be reproduced by >=2 operators) -- not
UNAVAILABLE. Since ``_grade`` treats a real FALSE among
{exceeds_control, multi_strategy, ci_excludes_zero} as blocking both HIGH
*and* MODERATE, every reliability_grade in this run is capped at LOW
regardless of how strong the control/seed evidence is. That is expected and
is not what this script is checking -- the actual target is the raw
exceeds_control/seed_stable *flags* and the z_vs_control/seed_cv values
feeding them, not the final grade.

Depends on experiments/synthetic_shortcut/results/ (gitignored) -- not
included in pytest collection.

Run as: python3 experiments/synthetic_shortcut/validate_reliability_thresholds.py
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path

from common import PATCH_REGION_KEY, build_item_values
from run_audit import PRIMARY_METRIC
from run_threshold_validation import RUN_ID

from ssat.analysis import (
    ComparisonIndexer,
    ControlComparisonRow,
    FlagValue,
    ReliabilityRow,
    SeedStabilityRow,
    compare_to_controls,
    compute_intervals,
    compute_reliability,
    compute_seed_stability,
    compute_strategy_stability,
)
from ssat.analysis.reader import AnalysisReader
from ssat.analysis.reliability import DEFAULT_SEED_CV_THRESHOLD, DEFAULT_Z_VS_CONTROL_THRESHOLD


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one threshold-validation pass."""

    parser = argparse.ArgumentParser(
        description="Validate z_vs_control_threshold/seed_cv_threshold against real control/seed data."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    return parser.parse_args()


def _summary(values: list[float]) -> str:
    """Render a compact min/median/max summary, or a placeholder if empty."""

    if not values:
        return "n=0"
    return (
        f"n={len(values)}, min={min(values):.3f}, median={statistics.median(values):.3f}, "
        f"max={max(values):.3f}"
    )


def _z_vs_control_report(control_rows: list[ControlComparisonRow]) -> str:
    """Compare the patch region's z_vs_control against every other region's."""

    by_region: dict[str, list[ControlComparisonRow]] = {}
    for row in control_rows:
        by_region.setdefault(row.target_anchor_key.region_key, []).append(row)

    patch_rows = by_region.get(PATCH_REGION_KEY, [])
    patch_z = [row.z_vs_control for row in patch_rows if row.z_vs_control is not None]
    non_patch_z = [
        row.z_vs_control
        for region_key, rows in by_region.items()
        if region_key != PATCH_REGION_KEY
        for row in rows
        if row.z_vs_control is not None
    ]
    unavailable = sum(1 for row in control_rows if row.control_available is not FlagValue.TRUE)

    exceeding = sum(1 for z in non_patch_z if z > DEFAULT_Z_VS_CONTROL_THRESHOLD)
    lines = [
        f"- patch region ({PATCH_REGION_KEY}) z_vs_control: {_summary(patch_z)}",
        f"- non-patch regions (15) z_vs_control: {_summary(non_patch_z)}",
        f"- non-patch anchors exceeding threshold {DEFAULT_Z_VS_CONTROL_THRESHOLD:g}: "
        f"{exceeding}/{len(non_patch_z)} ({exceeding / len(non_patch_z):.1%})"
        if non_patch_z
        else "- non-patch anchors exceeding threshold: n=0",
        f"- control_available != TRUE (no usable control): {unavailable}/{len(control_rows)}",
    ]
    return "\n".join(lines)


def _seed_cv_report(seed_rows: list[SeedStabilityRow]) -> str:
    """Compare seed_cv across patch, non-patch target, and control anchors.

    compute_seed_stability covers control anchors too, and they are the only
    ones whose mask is actually re-drawn per seed salt -- a target region's
    mask is fixed, so under a deterministic operator its seed_cv is 0 by
    construction. Lumping the two together would make target regions look
    far noisier than they are, so they are reported separately.
    """

    patch_cv: list[float] = []
    target_cv: list[float] = []
    control_cv: list[float] = []
    for row in seed_rows:
        if row.seed_cv is None:
            continue
        region_key = row.anchor_key.region_key
        if region_key.startswith("control:"):
            control_cv.append(row.seed_cv)
        elif region_key == PATCH_REGION_KEY:
            patch_cv.append(row.seed_cv)
        else:
            target_cv.append(row.seed_cv)

    lines = [
        f"- patch region ({PATCH_REGION_KEY}) seed_cv: {_summary(patch_cv)}",
        f"- non-patch target regions (15) seed_cv: {_summary(target_cv)}",
        f"- control anchors seed_cv: {_summary(control_cv)}",
    ]
    if control_cv:
        below = sum(1 for cv in control_cv if cv < DEFAULT_SEED_CV_THRESHOLD)
        lines.append(
            f"- control anchors below threshold {DEFAULT_SEED_CV_THRESHOLD:g} "
            f"(seed_stable=TRUE): {below}/{len(control_cv)} "
            f"({below / len(control_cv):.1%})"
        )
    return "\n".join(lines)


def _flag_distribution_report(reliability_rows: list[ReliabilityRow]) -> str:
    """Summarize every flag's TRUE/FALSE/UNAVAILABLE counts across all anchors."""

    lines = []
    for flag_name in (
        "sign_consistent",
        "exceeds_control",
        "seed_stable",
        "multi_strategy",
        "ci_excludes_zero",
        "area_matched",
    ):
        counts = Counter(getattr(row, flag_name).value for row in reliability_rows)
        lines.append(f"- {flag_name}: {dict(counts)}")
    grade_counts = Counter(row.reliability_grade.value for row in reliability_rows)
    lines.append(f"- reliability_grade: {dict(grade_counts)}")
    return "\n".join(lines)


def main() -> int:
    """Load the control+multi-seed run and report threshold behavior."""

    args = parse_args()
    reader = AnalysisReader(
        args.results_dir / "dumps" / RUN_ID, args.results_dir / "metrics" / RUN_ID
    )
    item_values = build_item_values(reader)
    indexer = ComparisonIndexer(reader.item_context())
    available = reader.available_analyses()

    control_rows = compare_to_controls(item_values, indexer.control_pairs)
    seed_rows = compute_seed_stability(item_values)
    strategy_rows, _rank_rows = compute_strategy_stability(item_values, primary_metric=PRIMARY_METRIC)
    interval_rows = compute_intervals(item_values)
    reliability_rows = compute_reliability(control_rows, seed_rows, strategy_rows, interval_rows)

    report_lines = [
        "# Reliability Threshold Validation -- control regions + multiple seeds",
        "",
        f"Run: `{RUN_ID}` (constant_fill only; 2 controls/target-region, 3 seeds/item).",
        "",
        f"- available_analyses: {available}",
        f"- coverage_report: {indexer.coverage_report}",
        f"- n_control_rows={len(control_rows)}, n_seed_rows={len(seed_rows)}, "
        f"n_reliability_rows={len(reliability_rows)}",
        "",
        "**Caveat**: this run has only one perturb_op, so `multi_strategy` is "
        "always FALSE here (see module docstring) -- every `reliability_grade` "
        "below is capped at LOW regardless of the control/seed evidence. Judge "
        "the thresholds from the flag-level and raw-value sections, not the grade.",
        "",
        f"## z_vs_control (threshold={DEFAULT_Z_VS_CONTROL_THRESHOLD:g})",
        "",
        _z_vs_control_report(control_rows),
        "",
        f"## seed_cv (threshold={DEFAULT_SEED_CV_THRESHOLD:g})",
        "",
        _seed_cv_report(seed_rows),
        "",
        "## Flag distributions across all anchors",
        "",
        _flag_distribution_report(reliability_rows),
        "",
    ]
    report = "\n".join(report_lines)
    report_path = args.results_dir / "threshold_validation_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"saved report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
