#!/usr/bin/env python3
"""Phase 4 (crop-free re-run follow-up): check whether the fill-strategy sign
group split -- docs/CONTROL_STABILITY_DESIGN_v1.md section 0's driving
observation ({constant_fill, gaussian_noise} positive vs. {mean_fill, blur,
patch_shuffle} negative rank correlation against constant_fill) -- survives
removing the CenterCrop-induced model-space area confound, or was
substantially an artifact of it.

Loads BOTH the original (cropped) results/ and the new crop-free
results_crop_free/ shortcut_A_* runs and, for each:

1. reports the distribution of effective_area_px across the 15 non-patch
   regions (expected: {2304, 3072, 4096} for the cropped run, a single
   constant {3136} for the crop-free run -- confirming Fix 1 actually
   removed the confound, not just relocated it);
2. recomputes Spearman(degradation, effective_area_px) across those 15
   regions per fill strategy -- under crop-free this is a correlation with
   a near-zero-variance variable, so it is expected to become undefined/near
   zero regardless of what happens to the sign split itself;
3. recomputes each fill strategy's region-ranking Spearman against
   constant_fill (both runs, 15-non-patch-region grain -- the same quantity
   analyze_control_stability.py's Q3 checks against the cropped run's
   pre-registered values) and reports whether the same two sign groups hold
   under crop-free.

This script only reads already-produced dumps/metrics and does not touch
run_audit.py/evaluate.py's outputs. The full A0-A6 reproduction check
(Q1-Q5-style) against the crop-free run is left to re-running
analyze_control_stability.py --results-dir experiments/synthetic_shortcut/results_crop_free,
which already implements that machinery -- this script's job is specifically
the area-confound diagnostic comparison between the two runs.

Depends on experiments/synthetic_shortcut/results/ and results_crop_free/
(both gitignored) -- not included in pytest collection.

Run as: python3 experiments/synthetic_shortcut/analyze_sign_group_premise.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from common import PATCH_REGION_KEY
from run_audit import FILL_PARAMS, PRIMARY_METRIC
from ssat.metrics.store import load_metrics

FILL_STRATEGIES = tuple(FILL_PARAMS)
BASELINE_FILL = "constant_fill"
_EXPECTED_POSITIVE_GROUP = frozenset({"constant_fill", "gaussian_noise"})
_EXPECTED_NEGATIVE_GROUP = frozenset({"mean_fill", "blur", "patch_shuffle"})


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one sign-group comparison pass."""

    parser = argparse.ArgumentParser(
        description="Compare the fill-strategy sign group split between cropped and crop-free runs."
    )
    parser.add_argument(
        "--cropped-results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--crop-free-results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results_crop_free",
    )
    return parser.parse_args()


def _load_region_data(results_dir: Path, fill: str) -> tuple[dict[str, float], dict[str, int]]:
    """Load one fill strategy's per-region mean degradation and effective_area_px.

    Loader is local (not imported from analyze_section35_sensitivity.py's
    private _RunData) per this codebase's established convention of
    duplicating small store-reading helpers across experiment scripts
    rather than importing another script's internals (see run_audit.py's
    docstring on _compute_and_save_metrics for the precedent).
    """

    _, aggregation, _ = load_metrics(results_dir / "metrics" / f"shortcut_A_{fill}")
    region_mean: dict[str, float] = {}
    region_area: dict[str, int] = {}
    for row in aggregation.region_metrics:
        if row.metric_name != PRIMARY_METRIC:
            continue
        if row.metric_mean is not None:
            region_mean[row.region_key] = row.metric_mean
        if row.effective_area_px is not None:
            region_area[row.region_key] = row.effective_area_px
    return region_mean, region_area


def _spearman_correlation(reference: dict[str, float], other: dict[str, float]) -> float | None:
    """Spearman rank correlation between two region_key -> value mappings.

    Reimplemented (not imported) per this codebase's convention -- see
    analyze_section35_sensitivity.py's helper of the same name.
    """

    shared_keys = sorted(set(reference) & set(other))
    if len(shared_keys) < 2:
        return None
    reference_ranks = pd.Series([reference[key] for key in shared_keys]).rank()
    other_ranks = pd.Series([other[key] for key in shared_keys]).rank()
    correlation = reference_ranks.corr(other_ranks)
    return None if pd.isna(correlation) else float(correlation)


def _area_report(label: str, results_dir: Path) -> list[str]:
    """Check (1): describe the 15 non-patch regions' effective_area_px distribution."""

    _, region_area = _load_region_data(results_dir, BASELINE_FILL)
    non_patch_areas = {k: v for k, v in region_area.items() if k != PATCH_REGION_KEY}
    distinct = sorted(set(non_patch_areas.values()))
    return [
        f"- **{label}** ({results_dir}): distinct effective_area_px values across "
        f"15 non-patch regions: {distinct}"
    ]


def _area_degradation_correlation_report(label: str, results_dir: Path) -> list[str]:
    """Check (2): Spearman(degradation, effective_area_px) per fill strategy."""

    lines = [f"### {label}", "", "| fill | Spearman(degradation, effective_area_px) |", "|---|---|"]
    for fill in FILL_STRATEGIES:
        region_mean, region_area = _load_region_data(results_dir, fill)
        non_patch_mean = {k: v for k, v in region_mean.items() if k != PATCH_REGION_KEY}
        non_patch_area = {
            k: float(v) for k, v in region_area.items() if k != PATCH_REGION_KEY
        }
        corr = _spearman_correlation(non_patch_mean, non_patch_area)
        corr_str = "n/a (undefined -- likely zero-variance area)" if corr is None else f"{corr:.3f}"
        lines.append(f"| {fill} | {corr_str} |")
    lines.append("")
    return lines


def _sign_group_report(label: str, results_dir: Path) -> list[str]:
    """Check (3): each fill strategy's 15-non-patch-region rank correlation vs. constant_fill."""

    baseline_mean, _ = _load_region_data(results_dir, BASELINE_FILL)
    baseline_non_patch = {k: v for k, v in baseline_mean.items() if k != PATCH_REGION_KEY}

    lines = [
        f"### {label}",
        "",
        "| fill (vs. constant_fill) | spearman_excl_top1 | sign |",
        "|---|---|---|",
    ]
    positive_group = {BASELINE_FILL}
    negative_group: set[str] = set()
    for fill in FILL_STRATEGIES:
        if fill == BASELINE_FILL:
            continue
        other_mean, _ = _load_region_data(results_dir, fill)
        other_non_patch = {k: v for k, v in other_mean.items() if k != PATCH_REGION_KEY}
        corr = _spearman_correlation(baseline_non_patch, other_non_patch)
        if corr is None:
            lines.append(f"| {fill} | n/a | n/a |")
            continue
        sign = "+" if corr > 0 else "-"
        (positive_group if corr > 0 else negative_group).add(fill)
        lines.append(f"| {fill} | {corr:.3f} | {sign} |")

    matches_original_split = (
        positive_group == _EXPECTED_POSITIVE_GROUP and negative_group == _EXPECTED_NEGATIVE_GROUP
    )
    lines.append("")
    lines.append(
        f"- positive group: {sorted(positive_group)}, negative group: {sorted(negative_group)}"
    )
    lines.append(
        f"- matches original cropped-run split "
        f"({sorted(_EXPECTED_POSITIVE_GROUP)} vs {sorted(_EXPECTED_NEGATIVE_GROUP)}): "
        f"{'YES' if matches_original_split else 'NO'}"
    )
    lines.append("")
    return lines


def main() -> int:
    """Compare the sign-group split between the cropped and crop-free runs."""

    args = parse_args()

    report_lines = [
        "# Sign-Group Premise Re-examination -- cropped vs. crop-free",
        "",
        "Compares docs/CONTROL_STABILITY_DESIGN_v1.md section 0's fill-strategy "
        "sign-group split between the original (CenterCrop-confounded) run and "
        "the crop-free re-run, to determine whether the split was substantially "
        "a model-space area artifact.",
        "",
        "## Check 1 -- effective_area_px distribution (15 non-patch regions)",
        "",
        *_area_report("cropped", args.cropped_results_dir),
        *_area_report("crop-free", args.crop_free_results_dir),
        "",
        "## Check 2 -- Spearman(degradation, effective_area_px) per fill strategy",
        "",
        *_area_degradation_correlation_report("cropped", args.cropped_results_dir),
        *_area_degradation_correlation_report("crop-free", args.crop_free_results_dir),
        "## Check 3 -- sign-group split, 15-non-patch-region spearman_excl_top1",
        "",
        *_sign_group_report("cropped", args.cropped_results_dir),
        *_sign_group_report("crop-free", args.crop_free_results_dir),
    ]
    report = "\n".join(report_lines)
    report_path = args.crop_free_results_dir / "sign_group_premise_reexamination.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"saved report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
