#!/usr/bin/env python3
"""C3 -- real-data validation of the reporting layer (``ssat.report`` R0-R4)
against the crop-free, all-fill-strategy, control+multi-seed dump
``run_threshold_validation_full.py`` produced. Implements
docs/IMPLE_PLAN_REPORTING_v1.md §5 단계8.

Unlike analyze_control_stability.py's B3 (which had to combine five
separate shortcut_A_* dumps at script level, since each held only one fill
strategy), this dump already holds all 5 fill strategies plus controls and
multiple seeds by itself (run_threshold_validation_full.py §51-121) --
exactly the single dump+metrics+analysis triple ``AuditApplication.
generate_report`` expects (IMPLE_PLAN_REPORTING_v1.md §5 단계7,
``generate_report`` takes exactly one ``analysis_dir``). No script-level
combining is needed here.

metrics/ for this run already exists (run_threshold_validation_full.py
computes and saves it); analysis/ does not yet exist, so this script runs
``AuditApplication.analyze`` once to produce it before running
``AuditApplication.generate_report``.

Depends on experiments/synthetic_shortcut/results_crop_free/ (gitignored,
produced by run_threshold_validation_full.py) -- not included in pytest
collection. After running, open ``<results-dir>/report/<run_id>/report.html``
directly in a browser (no server needed -- see docs/REPORT_LAYER_DESIGN_v1.md
§0's offline principle) and check it against the table in
IMPLE_PLAN_REPORTING_v1.md §5 단계8.

``--model {shortcut,normal}`` mirrors run_threshold_validation_full.py's
flag of the same name and must match whichever run that script produced --
this script imports its ``run_id_for`` helper directly rather than
duplicating the naming convention, so the two scripts can never disagree on
what a given ``--model`` value's run_id is. ``--model normal`` is the
comparison-report step documented in README.md.

Run as: python3 experiments/synthetic_shortcut/generate_report.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from run_threshold_validation_full import run_id_for

from ssat.application import AnalyzeRequest, AuditApplication, ReportRequest
from ssat.metrics.aggregate import DEFAULT_PRIMARY_METRIC


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one analyze+report pass."""

    parser = argparse.ArgumentParser(
        description=(
            "Run ssat analyze then ssat report against the crop-free, "
            "all-ops threshold-validation dump, for C3 real-data validation "
            "of the reporting layer."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results_crop_free",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Most-vulnerable samples to feature in the report gallery.",
    )
    parser.add_argument(
        "--bottom-k",
        type=int,
        default=20,
        help="Most-robust samples to feature in the report gallery.",
    )
    parser.add_argument(
        "--model",
        choices=("shortcut", "normal"),
        default="shortcut",
        help=(
            "Which run_threshold_validation_full.py run to report on -- must "
            "match that script's --model."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run analyze then generate_report for the given --model's run_id, and print a checklist summary."""

    args = parse_args()
    run_id = run_id_for(args.model)
    dump = args.results_dir / "dumps" / run_id
    metrics_dir = args.results_dir / "metrics" / run_id
    analysis_dir = args.results_dir / "analysis" / run_id
    report_dir = args.results_dir / "report" / run_id

    app = AuditApplication()

    print(f"== 1. ssat analyze ({dump}) ==")
    analyze_result = app.analyze(
        AnalyzeRequest(dump, metrics_dir, analysis_dir, DEFAULT_PRIMARY_METRIC)
    )
    print(f"  analysis dir: {analyze_result.analysis_dir}")
    print(f"  reliability rows: {analyze_result.n_reliability_rows:,}")
    print(f"  reliability grades: {dict(analyze_result.grade_distribution)}")

    print(f"== 2. ssat report ({dump}) ==")
    report_result = app.generate_report(
        ReportRequest(
            dump,
            metrics_dir,
            analysis_dir,
            report_dir,
            DEFAULT_PRIMARY_METRIC,
            args.top_k,
            args.bottom_k,
        )
    )
    print(f"  report dir: {report_result.report_dir}")
    print(f"  samples: {report_result.n_samples:,}, regions: {report_result.n_regions:,}")
    print(f"  reliability grades: {dict(report_result.grade_distribution)}")

    # A minimal automated signal for §5 단계8's headline success condition:
    # the region_summary's reliability_distribution field (§1 격차#3's
    # worst-case-grade mitigation) must actually carry more than one grade
    # for at least one non-patch region, not just decorate a single value.
    # This is a necessary-but-not-sufficient check -- the plan's remaining
    # five checklist items still require opening report.html by eye.
    region_summary_path = report_dir / "data" / "region_summary.csv"
    _check_region_summary_has_mixed_distributions(region_summary_path)

    print()
    print(f"report.html: {report_dir / 'report.html'}")
    print(
        "Open the file above directly in a browser and check it against the "
        "table in docs/IMPLE_PLAN_REPORTING_v1.md §5 단계8."
    )
    return 0


def _check_region_summary_has_mixed_distributions(region_summary_path: Path) -> None:
    """Warn (not fail) if every region's reliability_distribution is a single grade.

    Per the plan's success condition: if every region ends up with exactly
    one grade among its ``{high,moderate,low,unreliable}_count`` columns
    (exporter.py's ``_region_summary_row`` flattening of ``RegionRow.
    reliability_distribution``), the worst-case-grade mitigation field
    (IMPLE_PLAN_REPORTING_v1.md §1 격차#3) is not actually carrying
    information for this dump, and R0's join/aggregation policy would need
    re-examination. This script only surfaces the signal; it does not raise,
    matching evaluate.py's convention of always finishing and reporting
    whatever the checks actually found.
    """

    import csv

    if not region_summary_path.is_file():
        print(f"  [warn] {region_summary_path} not found; skipping distribution check")
        return

    grade_columns = ("high_count", "moderate_count", "low_count", "unreliable_count")
    with region_summary_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    mixed = 0
    for row in rows:
        n_grades_present = sum(
            1 for column in grade_columns if int(row.get(column, "") or 0) > 0
        )
        if n_grades_present > 1:
            mixed += 1

    print(
        f"  region_summary rows with a mixed reliability_distribution "
        f"(>1 grade present): {mixed}/{len(rows)}"
    )
    if rows and mixed == 0:
        print(
            "  [warn] every region's reliability_distribution is a single grade -- "
            "re-examine R0's join/aggregation policy per §5 단계8's success condition."
        )


if __name__ == "__main__":
    raise SystemExit(main())
