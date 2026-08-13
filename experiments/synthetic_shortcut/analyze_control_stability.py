#!/usr/bin/env python3
"""B3 -- L3 re-analysis: apply ssat.analysis (A0-A6) to the already-completed
shortcut_A_* runs and check whether it reproduces, automatically, the
structure docs/L3_Synthetic-Shortcut Experiment Report.md found by hand
(patch-region reliability, fill-strategy sign disagreement,
spearman_excl_top1). Implements
docs/IMPLE_PLAN_CONTROL_STABILITY_v1.md §5 단계9 / §4.4.

This script only *reads* the dumps/metrics run_audit.py already produced; it
makes no audit or training calls of its own, and does not persist an
analysis/ directory (A7) -- the five shortcut_A_* runs are separate
dump+metrics pairs, and comparing fill strategies against each other means
combining all five runs' item_values into one frame before calling A2-A6.
Since that combined frame no longer traces back to a single metrics store,
recording it via A7's save_analysis would misrepresent
AnalysisManifest.source_metrics_manifest_hash's single-source provenance
chain -- A7 itself is already covered by tests/unit/test_analysis_store.py.

Per the same convention evaluate.py already established, a failed check is
reported as failed, not adjusted -- this script never raises or exits
non-zero on a Q1-Q5-style mismatch; it always finishes and writes the report
with whatever the checks actually found.

Depends on experiments/synthetic_shortcut/results/ (gitignored, produced by
run_audit.py) -- not included in pytest collection.

Run as: python3 experiments/synthetic_shortcut/analyze_control_stability.py
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from common import PATCH_REGION_KEY, build_item_values
from run_audit import PRIMARY_METRIC, RUN_SPECS

from ssat.analysis import (
    Alignment,
    AnalysisReader,
    AvailableAnalyses,
    ComparisonIndexer,
    ControlComparisonRow,
    CoverageReport,
    IntervalRow,
    RankCorrelationRow,
    ReliabilityGrade,
    ReliabilityRow,
    SeedStabilityRow,
    StrategyProfileRow,
    StrategyStabilityRow,
    compare_to_controls,
    compute_intervals,
    compute_reliability,
    compute_seed_stability,
    compute_strategy_profile,
    compute_strategy_stability,
)

# Only the five M_shortcut/A runs carry a per-op region ranking that fill
# strategies can be compared across -- normal_A_* and shortcut_B_* exist for
# evaluate.py's Q3/B control instead and are out of scope here.
_SHORTCUT_A_RUN_IDS = tuple(
    spec.run_id for spec in RUN_SPECS if spec.model == "shortcut" and spec.dataset == "A"
)

# Pre-registered expected values, design CONTROL_STABILITY_DESIGN_v1.md §4.4 /
# IMPLE_PLAN_CONTROL_STABILITY_v1.md §5 단계9, transcribed from L3's Check 2
# table (docs/L3_Synthetic-Shortcut Experiment Report.md) -- the 15-non-patch-
# region column, not the "all 16 regions" column that section 3.5 reported.
_EXPECTED_SPEARMAN_EXCL_TOP1 = {
    "mean_fill": -0.311,
    "blur": -0.843,
    "gaussian_noise": 0.729,
    "patch_shuffle": -0.218,
}
_SPEARMAN_TOLERANCE = 0.001
_EXPECTED_POSITIVE_GROUP = frozenset({"constant_fill", "gaussian_noise"})
_EXPECTED_NEGATIVE_GROUP = frozenset({"mean_fill", "blur", "patch_shuffle"})


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one re-analysis pass."""

    parser = argparse.ArgumentParser(
        description="Re-analyze the L3 synthetic-shortcut shortcut_A_* runs with ssat.analysis."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    return parser.parse_args()


def _combined_available_analyses(item_values: pd.DataFrame) -> AvailableAnalyses:
    """Recompute AvailableAnalyses over all five runs' item_values combined.

    ``AnalysisReader.available_analyses()`` is scoped to one dump (design
    §A0) -- called per run here it would always report
    ``fill_strategy_stability=False``, since each individual
    ``shortcut_A_<op>`` dump only ever contains one ``perturb_op``. This
    applies the same criteria (IMPLE_PLAN_CONTROL_STABILITY_v1.md §5 단계1
    table) to the combined frame instead, where >=2 distinct ops are
    actually present.
    """

    non_control = item_values.loc[~item_values["is_control"]]
    return AvailableAnalyses(
        control_comparison=bool(item_values["is_control"].any()),
        fill_strategy_stability=bool(non_control["perturb_op"].nunique() >= 2),
        seed_stability=_has_repeated_condition_group(item_values),
        # Always False: the core has no jitter axis (same reasoning as
        # AnalysisReader.available_analyses()).
        jitter_stability=False,
    )


def _has_repeated_condition_group(item_values: pd.DataFrame) -> bool:
    """Return True if any (sample, region, invert_mask, op, params) group has >=2 rows.

    Groups directly on the raw identity columns rather than reconstructing
    AnchorKey/ConditionKey objects -- since this dataset registers exactly
    one metric (``margin_drop``), ``item_values`` is already one row per
    item and grouping on these columns is equivalent to counting seed
    repeats per (AnchorKey, ConditionKey).
    """

    group_sizes = item_values.groupby(
        [
            "sample_id",
            "region_id",
            "region_instance_id",
            "invert_mask",
            "perturb_op",
            "perturb_params_json",
        ],
        dropna=False,
    ).size()
    return bool((group_sizes >= 2).any())


def _combine_coverage_reports(reports: list[CoverageReport]) -> CoverageReport:
    """Sum every run's CoverageReport fields into one dataset-wide total."""

    return CoverageReport(
        n_anchors=sum(report.n_anchors for report in reports),
        n_conditions_insufficient=sum(report.n_conditions_insufficient for report in reports),
        n_controls_unmatched=sum(report.n_controls_unmatched for report in reports),
        n_area_mismatch_warnings=sum(report.n_area_mismatch_warnings for report in reports),
    )


def _grade_distribution_by_region(
    reliability_rows: list[ReliabilityRow],
) -> dict[str, Counter[str]]:
    """Group ReliabilityRows by region_key and count each grade within it.

    ReliabilityRow is per (sample_id, region_key, invert_mask, metric) --
    finer than the region-level questions B3 asks (design §4.4). This
    reduces the sample axis the same way A7's
    ``AnalysisManifest.grade_distribution`` reduces the whole dataset
    (IMPLE_PLAN §5 단계8): a ``Counter`` of grade values, just scoped per
    region instead of dataset-wide.
    """

    distribution: dict[str, Counter[str]] = defaultdict(Counter)
    for row in reliability_rows:
        distribution[row.anchor_key.region_key][row.reliability_grade.value] += 1
    return dict(distribution)


def _check_patch_region_grade(distribution_by_region: dict[str, Counter[str]]) -> str:
    """Q1: the patch region's grade distribution should be dominated by `high`."""

    counts = distribution_by_region.get(PATCH_REGION_KEY, Counter())
    total = sum(counts.values())
    dominant, dominant_count = counts.most_common(1)[0] if counts else (None, 0)
    status = "PASS" if dominant == ReliabilityGrade.HIGH.value else "FAIL"
    return (
        f"**{status}** -- {PATCH_REGION_KEY}: {dict(counts)} "
        f"(dominant={dominant}, {dominant_count}/{total})"
    )


def _check_non_patch_grades(distribution_by_region: dict[str, Counter[str]]) -> str:
    """Q2: the 15 non-patch regions' combined grades should be mostly `unreliable`."""

    combined: Counter[str] = Counter()
    per_region_lines = []
    for region_key, counts in sorted(distribution_by_region.items()):
        if region_key == PATCH_REGION_KEY:
            continue
        combined.update(counts)
        dominant, _dominant_count = counts.most_common(1)[0] if counts else (None, 0)
        per_region_lines.append(f"  - {region_key}: {dict(counts)} (dominant={dominant})")

    total = sum(combined.values())
    unreliable_ratio = (
        combined.get(ReliabilityGrade.UNRELIABLE.value, 0) / total if total else 0.0
    )
    status = "PASS" if unreliable_ratio >= 0.5 else "FAIL"
    lines = [
        f"**{status}** -- combined non-patch grade distribution: {dict(combined)} "
        f"(unreliable ratio={unreliable_ratio:.2%})"
    ]
    lines.extend(per_region_lines)
    return "\n".join(lines)


def _check_spearman_excl_top1(rank_rows: list[RankCorrelationRow]) -> str:
    """Q3: spearman_excl_top1 (vs constant_fill) should match the L3 report's Check 2."""

    lines = []
    all_match = True
    for op, expected in _EXPECTED_SPEARMAN_EXCL_TOP1.items():
        actual = next(
            (
                row.spearman_excl_top1
                for row in rank_rows
                if {row.op_a, row.op_b} == {"constant_fill", op}
            ),
            None,
        )
        matched = actual is not None and abs(actual - expected) < _SPEARMAN_TOLERANCE
        all_match = all_match and matched
        actual_text = f"{actual:.3f}" if actual is not None else "None"
        lines.append(
            f"  - {op} vs constant_fill: expected={expected:.3f}, actual={actual_text} "
            f"({'match' if matched else 'MISMATCH'})"
        )
    status = "PASS" if all_match else "FAIL"
    return "\n".join([f"**{status}**"] + lines)


def _check_sign_groups(profile_rows: list[StrategyProfileRow]) -> str:
    """Q4: the empirical clusters should split {constant_fill, gaussian_noise} from the rest."""

    cluster_by_op = {row.perturb_op: row.cluster_id for row in profile_rows}
    positive_clusters = {cluster_by_op.get(op) for op in _EXPECTED_POSITIVE_GROUP}
    negative_clusters = {cluster_by_op.get(op) for op in _EXPECTED_NEGATIVE_GROUP}
    matched = (
        len(positive_clusters) == 1
        and None not in positive_clusters
        and len(negative_clusters) == 1
        and None not in negative_clusters
        and positive_clusters != negative_clusters
    )
    status = "PASS" if matched else "FAIL"
    detail = ", ".join(f"{op}=cluster{cid}" for op, cid in sorted(cluster_by_op.items()))
    return f"**{status}** -- {detail}"


def _check_alignment(profile_rows: list[StrategyProfileRow]) -> str:
    """Q5: no op's declarative/empirical alignment should come back divergent."""

    alignments = {row.perturb_op: row.alignment for row in profile_rows}
    matched = all(alignment is not Alignment.DIVERGENT for alignment in alignments.values())
    status = "PASS" if matched else "FAIL"
    detail = ", ".join(f"{op}={alignment.value}" for op, alignment in sorted(alignments.items()))
    return f"**{status}** -- {detail}"


def main() -> int:
    """Load all five shortcut_A_* runs, run A1-A6 combined, and check reproduction."""

    args = parse_args()

    per_run_item_values: list[pd.DataFrame] = []
    per_run_coverage: list[CoverageReport] = []
    control_pairs_all: list = []
    for run_id in _SHORTCUT_A_RUN_IDS:
        reader = AnalysisReader(
            args.results_dir / "dumps" / run_id, args.results_dir / "metrics" / run_id
        )
        per_run_item_values.append(build_item_values(reader))
        indexer = ComparisonIndexer(reader.item_context())
        per_run_coverage.append(indexer.coverage_report)
        control_pairs_all.extend(indexer.control_pairs)

    item_values = pd.concat(per_run_item_values, ignore_index=True)
    available = _combined_available_analyses(item_values)
    coverage_report = _combine_coverage_reports(per_run_coverage)

    control_rows: list[ControlComparisonRow] = compare_to_controls(item_values, control_pairs_all)
    seed_rows: list[SeedStabilityRow] = compute_seed_stability(item_values)
    strategy_rows, rank_rows = compute_strategy_stability(
        item_values, primary_metric=PRIMARY_METRIC
    )
    profile_rows: list[StrategyProfileRow] = compute_strategy_profile(
        strategy_rows, rank_rows, primary_metric=PRIMARY_METRIC
    )
    interval_rows: list[IntervalRow] = compute_intervals(item_values)
    reliability_rows: list[ReliabilityRow] = compute_reliability(
        control_rows, seed_rows, strategy_rows, interval_rows
    )

    distribution_by_region = _grade_distribution_by_region(reliability_rows)

    report_lines = [
        "# Control/Stability Analysis (B3) -- L3 Synthetic-Shortcut Re-analysis",
        "",
        "Applies ssat.analysis (A0-A6) to the already-completed "
        f"{', '.join(_SHORTCUT_A_RUN_IDS)} runs, combined into one item_values frame, "
        "and checks whether it reproduces the structure "
        "docs/L3_Synthetic-Shortcut Experiment Report.md found by hand "
        "(IMPLE_PLAN_CONTROL_STABILITY_v1.md §5 단계9). A7 (persistence) is "
        "intentionally not exercised here -- see this script's module docstring.",
        "",
        f"- available_analyses: {available}",
        f"- coverage_report: {coverage_report}",
        f"- n_reliability_rows={len(reliability_rows)}, n_regions={len(distribution_by_region)}",
        "",
        "## Q1. Patch region (grid::grid/r0/c0) grade -- expect dominated by `high`",
        "",
        _check_patch_region_grade(distribution_by_region),
        "",
        "## Q2. Non-patch region grades -- expect mostly `unreliable`",
        "",
        _check_non_patch_grades(distribution_by_region),
        "",
        "## Q3. spearman_excl_top1 vs constant_fill (L3 report Check 2)",
        "",
        _check_spearman_excl_top1(rank_rows),
        "",
        "## Q4. Empirical sign groups -- expect "
        "{constant_fill, gaussian_noise} vs {mean_fill, blur, patch_shuffle}",
        "",
        _check_sign_groups(profile_rows),
        "",
        "## Q5. Declarative/empirical alignment -- expect aligned or partial, never divergent",
        "",
        _check_alignment(profile_rows),
        "",
    ]
    report = "\n".join(report_lines)
    report_path = args.results_dir / "control_stability_analysis.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"saved report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
