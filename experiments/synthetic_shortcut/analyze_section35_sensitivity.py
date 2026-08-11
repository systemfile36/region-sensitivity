#!/usr/bin/env python3
"""Follow-up analysis for the section 3.5 fill-strategy sensitivity result.

evaluate.py's already-completed run reported blur's Spearman correlation
against constant_fill as -0.518 (docs/STAGE9_SYNTHETIC_SHORTCUT_DESIGN_v1.md
section 6). A negative correlation -- not just a low one -- raised the
question of whether blur is measuring something systematically different
from the other fill strategies, rather than just being noisier. This script
answers that with three checks, all against the already-stored metrics under
results/metrics/shortcut_A_*/ (no new training or auditing required):

1. Distribution of the 15 non-patch regions' degradation. If they are all
   tiny relative to the patch region and indistinguishable from 0 given
   their own sampling noise, then an unstable rank order among them (and
   therefore an unstable correlation) is the *expected* behavior of ranking
   noise, not a sign of anything being measured incorrectly.
2. Spearman correlation recomputed with the patch region excluded. The
   reported -0.518 includes the patch region, which is rank 1 in every fill
   strategy (Q1/Q4) and therefore mechanically pulls every correlation
   toward +1 regardless of what happens among the other 15 -- excluding it
   isolates how (un)stable the *weak-signal* ranking really is.
3. Whether blur's behavior is actually blur-specific, or shared by any fill
   strategy that happens to produce a similarly neutral replacement color.
   ssat's BlurOperator computes a full-frame Gaussian blur before compositing
   only the target region's pixels back in (ssat/core/perturb/operators.py),
   so at sigma=4 on a 32x32 image the blurred candidate for a region next to
   the patch can carry some of the patch's color -- a plausible source of a
   *distance-dependent* effect. This is tested directly by correlating each
   region's |degradation| with its grid (Manhattan) distance from the patch,
   per fill strategy, and by placing the two regions orthogonally adjacent to
   the patch (r0/c1, r1/c0) side by side across all five strategies.

This script is intentionally read-only with respect to the experiment
pipeline: it does not call AuditApplication, does not retrain or re-audit
anything, and does not modify evaluate.py/run_audit.py or their outputs. It
only adds two new files under results/ (section35_sensitivity_analysis.md,
section35_region_distance_degradation.csv).

Run as: python3 experiments/synthetic_shortcut/analyze_section35_sensitivity.py
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import pandas as pd
from common import PATCH_COL, PATCH_REGION_KEY, PATCH_ROW
from run_audit import FILL_PARAMS, PRIMARY_METRIC
from ssat.metrics.store import load_metrics

# Only the five M_shortcut/A runs carry a region ranking that section 3.5
# compares across fill strategies -- normal_A_* and shortcut_B_* exist for
# Q3/the B control instead, and are out of scope for this follow-up.
FILL_STRATEGIES = tuple(FILL_PARAMS)
BASELINE_FILL = "constant_fill"

# A region_key looks like "grid::grid/r{row}/c{col}" (common.py). Parsing it
# back into (row, col) here -- rather than adding a helper to common.py --
# keeps this analysis-only need out of the shared module the stable pipeline
# scripts depend on.
_REGION_KEY_PATTERN = re.compile(r"^grid::grid/r(\d+)/c(\d+)$")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one analysis pass."""

    parser = argparse.ArgumentParser(
        description="Investigate the section 3.5 blur-correlation anomaly using already-computed metrics."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    return parser.parse_args()


def _region_row_col(region_key: str) -> tuple[int, int]:
    """Parse a grid region_key back into its (row, col) grid coordinates."""

    match = _REGION_KEY_PATTERN.match(region_key)
    if match is None:
        raise ValueError(f"unrecognized grid region_key: {region_key!r}")
    return int(match.group(1)), int(match.group(2))


def _manhattan_distance_from_patch(region_key: str) -> int:
    """Grid distance from a region to the patch cell (0,0 by design section 3)."""

    row, col = _region_row_col(region_key)
    return abs(row - PATCH_ROW) + abs(col - PATCH_COL)


def _spearman_correlation(reference: dict[str, float], other: dict[str, float]) -> float | None:
    """Spearman rank correlation between two region_key -> value mappings.

    Reimplemented here (mirroring, not importing, evaluate.py's private
    helper of the same name) per this codebase's existing convention of
    keeping each experiment script's dependencies to public names only
    (see run_audit.py's docstring on tests/fixtures/synthetic_dump_builder.py
    for the precedent). Spearman is the Pearson correlation of the two
    series' ranks, which avoids needing scipy (not a project dependency).
    """

    shared_keys = sorted(set(reference) & set(other))
    if len(shared_keys) < 2:
        return None
    reference_ranks = pd.Series([reference[key] for key in shared_keys]).rank()
    other_ranks = pd.Series([other[key] for key in shared_keys]).rank()
    correlation = reference_ranks.corr(other_ranks)
    return None if pd.isna(correlation) else float(correlation)


class _RunData:
    """One fill strategy's per-region mean/std/SE, loaded from its metrics store."""

    def __init__(self, results_dir: Path, fill: str) -> None:
        metrics_dir = results_dir / "metrics" / f"shortcut_A_{fill}"
        _, aggregation, _ = load_metrics(metrics_dir)

        # region_metrics gives the mean each fill strategy's own report.md /
        # region_metrics_*.csv already used for ranking; kept here so this
        # script's numbers can be cross-checked against those files directly.
        self.region_mean: dict[str, float] = {
            row.region_key: row.metric_mean
            for row in aggregation.region_metrics
            if row.metric_name == PRIMARY_METRIC and row.metric_mean is not None
        }

        # spatial_profile is per-(sample, region) -- the finer grain needed
        # to compute a standard error per region, which RegionMetrics itself
        # does not carry (ssat/metrics/types.py). This is the piece that
        # lets this script tell "small value" apart from "value plausibly
        # indistinguishable from zero given how few samples back it".
        per_region_values: dict[str, list[float]] = {}
        for row in aggregation.spatial_profile:
            if row.metric_name != PRIMARY_METRIC or row.degradation is None:
                continue
            per_region_values.setdefault(row.region_key, []).append(row.degradation)

        self.region_std: dict[str, float] = {}
        self.region_se: dict[str, float] = {}
        self.region_n: dict[str, int] = {}
        for region_key, values in per_region_values.items():
            n = len(values)
            self.region_n[region_key] = n
            if n < 2:
                continue
            series = pd.Series(values)
            std = float(series.std(ddof=1))
            self.region_std[region_key] = std
            self.region_se[region_key] = std / math.sqrt(n)


def _check_distribution(runs: dict[str, _RunData]) -> list[str]:
    """Check (1): describe the 15 non-patch regions' degradation per fill strategy."""

    lines = [
        "## Check 1 -- non-patch region degradation distribution",
        "",
        "If the 15 non-patch regions' values are all tiny and statistically "
        "indistinguishable from 0 (|mean| within ~2 standard errors of 0), an "
        "unstable rank order among them is expected sampling noise, not a "
        "sign that the tool is measuring something wrong.",
        "",
        "| fill | patch mean | non-patch mean | non-patch std | non-patch min | non-patch max | # non-patch regions distinguishable from 0 (|mean|>2*SE) |",
        "|---|---|---|---|---|---|---|",
    ]
    for fill in FILL_STRATEGIES:
        run = runs[fill]
        patch_mean = run.region_mean.get(PATCH_REGION_KEY)
        non_patch = {k: v for k, v in run.region_mean.items() if k != PATCH_REGION_KEY}
        values = pd.Series(list(non_patch.values()))
        distinguishable = 0
        for region_key, mean in non_patch.items():
            se = run.region_se.get(region_key)
            if se is not None and se > 0 and abs(mean) > 2 * se:
                distinguishable += 1
        lines.append(
            f"| {fill} | {patch_mean:.4f} | {values.mean():.4f} | {values.std(ddof=1):.4f} "
            f"| {values.min():.4f} | {values.max():.4f} | {distinguishable}/{len(non_patch)} |"
        )
    lines.append("")
    return lines


def _check_correlation_with_and_without_patch(runs: dict[str, _RunData]) -> list[str]:
    """Check (2): recompute section 3.5's correlation excluding the patch region."""

    baseline_all = runs[BASELINE_FILL].region_mean
    baseline_non_patch = {k: v for k, v in baseline_all.items() if k != PATCH_REGION_KEY}

    lines = [
        "## Check 2 -- Spearman correlation with vs. without the patch region",
        "",
        f"The patch region is rank 1 in every fill strategy (Q1/Q4), so including it "
        f"mechanically pulls every pairwise correlation toward +1. This isolates how "
        f"stable the ranking of the *other 15* (weak-signal) regions really is.",
        "",
        "| fill (vs. constant_fill) | all 16 regions | 15 non-patch regions only |",
        "|---|---|---|",
    ]
    for fill in FILL_STRATEGIES:
        if fill == BASELINE_FILL:
            continue
        other_all = runs[fill].region_mean
        other_non_patch = {k: v for k, v in other_all.items() if k != PATCH_REGION_KEY}
        all_corr = _spearman_correlation(baseline_all, other_all)
        non_patch_corr = _spearman_correlation(baseline_non_patch, other_non_patch)
        all_str = "n/a" if all_corr is None else f"{all_corr:.3f}"
        non_patch_str = "n/a" if non_patch_corr is None else f"{non_patch_corr:.3f}"
        lines.append(f"| {fill} | {all_str} | {non_patch_str} |")
    lines.append("")
    return lines


def _check_blur_specificity(
    runs: dict[str, _RunData], results_dir: Path
) -> tuple[list[str], list[dict[str, object]]]:
    """Check (3): is the pattern blur-specific, or shared by any neutral-color fill?

    Also writes the full (fill, region, distance, degradation) table to CSV
    so the raw numbers behind this section's claims can be inspected
    directly, not just the correlation summaries.
    """

    rows: list[dict[str, object]] = []
    for fill in FILL_STRATEGIES:
        for region_key, mean in runs[fill].region_mean.items():
            if region_key == PATCH_REGION_KEY:
                continue
            rows.append(
                {
                    "fill": fill,
                    "region_key": region_key,
                    "manhattan_distance_from_patch": _manhattan_distance_from_patch(region_key),
                    "degradation_mean": mean,
                }
            )

    csv_path = results_dir / "section35_region_distance_degradation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["fill", "region_key", "manhattan_distance_from_patch", "degradation_mean"]
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "## Check 3 -- is this blur-specific, or shared by any neutral-color fill?",
        "",
        "BlurOperator computes a full-frame Gaussian blur before compositing only the "
        "target region back in (ssat/core/perturb/operators.py), so at sigma=4 on a "
        "32x32 image, a region near the patch can plausibly pick up some of the "
        "patch's color when it is blurred. If that mechanism is what's driving the "
        "-0.518 correlation, |degradation| among the 15 non-patch regions should "
        "correlate with distance from the patch *specifically for blur*.",
        "",
        "| fill | Spearman(|degradation|, distance from patch) |",
        "|---|---|",
    ]
    for fill in FILL_STRATEGIES:
        distances: dict[str, float] = {}
        magnitudes: dict[str, float] = {}
        for region_key, mean in runs[fill].region_mean.items():
            if region_key == PATCH_REGION_KEY:
                continue
            # Use region_key itself as the shared "index" for the correlation
            # helper -- it only needs matching keys between the two series.
            distances[region_key] = float(_manhattan_distance_from_patch(region_key))
            magnitudes[region_key] = abs(mean)
        corr = _spearman_correlation(distances, magnitudes)
        corr_str = "n/a" if corr is None else f"{corr:.3f}"
        lines.append(f"| {fill} | {corr_str} |")
    lines.append("")

    # The two regions orthogonally adjacent to the patch cell (design section
    # 3 places the patch at (0,0), so its only in-bounds orthogonal
    # neighbors are (0,1) and (1,0)) side by side across every fill --
    # directly re-checking, with the real numbers, the near-coincidence
    # between mean_fill and blur noticed by inspecting region_metrics_*.csv
    # by hand before writing this script.
    adjacent_keys = ("grid::grid/r0/c1", "grid::grid/r1/c0")
    lines.append(
        "The two regions orthogonally adjacent to the patch cell, across all five fills "
        "(a spatial, non-blur-specific mechanism would predict these look similar across "
        "fills that produce a similarly neutral replacement color, not just for blur):"
    )
    lines.append("")
    lines.append("| region | " + " | ".join(FILL_STRATEGIES) + " |")
    lines.append("|---|" + "---|" * len(FILL_STRATEGIES))
    for region_key in adjacent_keys:
        cells = [f"{runs[fill].region_mean.get(region_key, float('nan')):.4f}" for fill in FILL_STRATEGIES]
        lines.append(f"| {region_key} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines, rows


def main() -> int:
    """Load every shortcut_A_* run's metrics and write the section 3.5 follow-up report."""

    args = parse_args()
    runs = {fill: _RunData(args.results_dir, fill) for fill in FILL_STRATEGIES}

    report_lines = [
        "# Section 3.5 Sensitivity Follow-Up: the blur=-0.518 Anomaly",
        "",
        "Investigates why evaluate.py's section 3.5 output reported blur's "
        "Spearman correlation against constant_fill as -0.518, using only the "
        "already-completed run's stored metrics (no retraining or re-auditing). "
        "See the module docstring of analyze_section35_sensitivity.py for the "
        "three checks below.",
        "",
    ]
    report_lines += _check_distribution(runs)
    report_lines += _check_correlation_with_and_without_patch(runs)
    blur_lines, _rows = _check_blur_specificity(runs, args.results_dir)
    report_lines += blur_lines

    report_path = args.results_dir / "section35_sensitivity_analysis.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    print(f"saved report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
