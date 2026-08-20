#!/usr/bin/env python3
"""Final z_vs_control_threshold/seed_cv_threshold recalibration pass, after
re-analyzing B3 (proposed defaults z=2.0, cv=0.2 --
ssat/analysis/reliability.py's DEFAULT_Z_VS_CONTROL_THRESHOLD/
DEFAULT_SEED_CV_THRESHOLD). Reads the run run_threshold_validation_full.py
produces: crop-free preprocessing, all 5 fill strategies, 2 controls/
target-region, 3 seeds/item.

Unlike validate_reliability_thresholds.py (which read a constant_fill-only
run and therefore had `multi_strategy` structurally FALSE for every anchor,
capping every `reliability_grade` at LOW regardless of evidence), this run
gives `multi_strategy` real TRUE/FALSE data -- see the new `_strategy_report`
section below.

Depends on experiments/synthetic_shortcut/results_crop_free/ (gitignored) --
not included in pytest collection.

Run as: python3 experiments/synthetic_shortcut/validate_reliability_thresholds_full.py
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path

from common import PATCH_REGION_KEY, build_item_values
from run_audit import PRIMARY_METRIC
from run_threshold_validation_full import RUN_ID

from ssat.analysis import (
    ComparisonIndexer,
    ControlComparisonRow,
    FlagValue,
    ReliabilityRow,
    SeedStabilityRow,
    StrategyStabilityRow,
    compare_to_controls,
    compute_intervals,
    compute_reliability,
    compute_seed_stability,
    compute_strategy_stability,
)
from ssat.analysis.reader import AnalysisReader
from ssat.analysis.reliability import DEFAULT_SEED_CV_THRESHOLD, DEFAULT_Z_VS_CONTROL_THRESHOLD


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one full threshold-validation pass."""

    parser = argparse.ArgumentParser(
        description="Final-pass validation of z_vs_control_threshold/seed_cv_threshold."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results_crop_free",
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


def _strategy_report(strategy_rows: list[StrategyStabilityRow]) -> str:
    """Summarize n_strategies and dominant-sign reproduction across anchors.

    This is the section validate_reliability_thresholds.py's single-op run
    could never populate meaningfully: with 5 fill strategies in this run,
    ``n_strategies`` can actually be >1 per anchor, and the dominant sign's
    reproduction count (what A6's multi_strategy flag is computed from,
    ssat/analysis/reliability.py::_multi_strategy_flag) has real spread.
    """

    n_strategies_counts = Counter(row.n_strategies for row in strategy_rows)
    dominant_counts: list[int] = []
    for row in strategy_rows:
        if not row.strategy_signs:
            continue
        _, count = Counter(row.strategy_signs.values()).most_common(1)[0]
        dominant_counts.append(count)
    reproduced_by_2plus = sum(1 for count in dominant_counts if count >= 2)

    lines = [
        f"- n_strategies distribution across anchors: {dict(sorted(n_strategies_counts.items()))}",
        f"- anchors with dominant sign reproduced by >=2 operators "
        f"(multi_strategy=TRUE candidates): {reproduced_by_2plus}/{len(dominant_counts)}",
    ]
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
    """Load the crop-free, all-ops, control+multi-seed run and report threshold behavior."""

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
        "# Reliability Threshold Validation (final pass) -- crop-free, all fill strategies",
        "",
        f"Run: `{RUN_ID}` (5 fill strategies; 2 controls/target-region, 3 seeds/item; crop-free preprocessing).",
        "",
        f"- available_analyses: {available}",
        f"- coverage_report: {indexer.coverage_report}",
        f"- n_control_rows={len(control_rows)}, n_seed_rows={len(seed_rows)}, "
        f"n_strategy_rows={len(strategy_rows)}, n_reliability_rows={len(reliability_rows)}",
        "",
        f"## z_vs_control (threshold={DEFAULT_Z_VS_CONTROL_THRESHOLD:g})",
        "",
        _z_vs_control_report(control_rows),
        "",
        f"## seed_cv (threshold={DEFAULT_SEED_CV_THRESHOLD:g})",
        "",
        _seed_cv_report(seed_rows),
        "",
        "## multi_strategy (fill-strategy sign reproduction)",
        "",
        _strategy_report(strategy_rows),
        "",
        "## Flag distributions across all anchors",
        "",
        _flag_distribution_report(reliability_rows),
        "",
    ]
    report = "\n".join(report_lines)
    report_path = args.results_dir / "threshold_validation_report_full.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"saved report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
