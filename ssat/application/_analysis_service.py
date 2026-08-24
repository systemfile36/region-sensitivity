"""``analyze`` body for ``AuditApplication`` (facade delegation)."""

from __future__ import annotations

from ssat.analysis.control import compare_to_controls
from ssat.analysis.indexer import ComparisonIndexer
from ssat.analysis.interval import compute_intervals
from ssat.analysis.reader import AnalysisReader
from ssat.analysis.reliability import compute_reliability
from ssat.analysis.stability import compute_seed_stability, compute_strategy_stability
from ssat.analysis.store import save_analysis
from ssat.analysis.strategy_profile import compute_strategy_profile
from ssat.application.types import (
    AnalyzeRequest,
    AnalyzeResult,
    ApplicationError,
    ApplicationErrorCode,
)


def analyze(request: AnalyzeRequest) -> AnalyzeResult:
    """Compute and persist the control/stability analysis for an existing dump+metrics pair.

    This is the Application-layer counterpart of what experiment scripts
    (experiments/synthetic_shortcut/analyze_control_stability.py,
    validate_reliability_thresholds_full.py) previously had to hand-roll:
    opening an AnalysisReader, indexing anchors/controls, and running
    A2-A6 in sequence. Scoped to one dump+metrics pair, matching
    AnalysisReader/compute_metrics's own scope; combining several runs
    into one item_values frame stays a script-level concern.
    """

    dump = request.dump.expanduser().resolve(strict=True)
    metrics_dir = (request.metrics_dir or dump / "metrics").expanduser().resolve()
    analysis_dir = (request.analysis_dir or dump / "analysis").expanduser().resolve()
    try:
        reader = AnalysisReader(dump, metrics_dir)
        item_values = reader.item_values()
        indexer = ComparisonIndexer(
            reader.item_context(), area_match_tolerance=request.area_match_tolerance
        )

        control_rows = compare_to_controls(
            item_values,
            indexer.control_pairs,
            area_match_tolerance=request.area_match_tolerance,
        )
        seed_rows = compute_seed_stability(item_values)
        strategy_rows, rank_rows = compute_strategy_stability(
            item_values, primary_metric=request.primary_metric
        )
        profile_rows = compute_strategy_profile(
            strategy_rows, rank_rows, primary_metric=request.primary_metric
        )
        interval_rows = compute_intervals(
            item_values, n_bootstrap=request.n_bootstrap, random_seed=request.random_seed
        )
        reliability_rows = compute_reliability(
            control_rows,
            seed_rows,
            strategy_rows,
            interval_rows,
            z_vs_control_threshold=request.z_vs_control_threshold,
            seed_cv_threshold=request.seed_cv_threshold,
        )
        available = reader.available_analyses()

        manifest = save_analysis(
            analysis_dir,
            control_rows=control_rows,
            seed_rows=seed_rows,
            strategy_rows=strategy_rows,
            rank_correlation_rows=rank_rows,
            strategy_profile_rows=profile_rows,
            interval_rows=interval_rows,
            reliability_rows=reliability_rows,
            coverage_report=indexer.coverage_report,
            available_analyses=available,
            thresholds={
                "z_vs_control_threshold": request.z_vs_control_threshold,
                "seed_cv_threshold": request.seed_cv_threshold,
                "area_match_tolerance": request.area_match_tolerance,
            },
            n_bootstrap=request.n_bootstrap,
            random_seed=request.random_seed,
            source_metrics_manifest_path=metrics_dir / "metrics_manifest.json",
        )
    except Exception as error:
        # Catches AnalysisReader's unwrapped MetricsCorruptionError/
        # MetricsSchemaError (module docstring on ssat.analysis.reader),
        # ssat.analysis.errors.* raised by A1-A6/A7 themselves, and
        # anything DumpHandle surfaces -- all map to the same ANALYSIS
        # error code, mirroring compute_metrics's identical rationale.
        raise ApplicationError(
            ApplicationErrorCode.ANALYSIS, f"cannot compute analysis: {error}"
        ) from error

    return AnalyzeResult(
        dump=dump,
        metrics_dir=metrics_dir,
        analysis_dir=analysis_dir,
        available_analyses=available,
        coverage_report=indexer.coverage_report,
        grade_distribution=dict(manifest.grade_distribution),
        n_reliability_rows=len(reliability_rows),
        computed_at=manifest.computed_at.isoformat(),
    )
