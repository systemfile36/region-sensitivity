"""A7 AnalysisStore: persist and reload every control/stability analysis output.

Writes ``<analysis_dir>/control_comparison.parquet``, ``seed_stability.parquet``,
``strategy_stability.parquet``, ``rank_correlation.parquet``,
``strategy_profile.parquet``, ``intervals.parquet``, ``reliability.parquet``,
``coverage_report.json``, and ``analysis_manifest.json`` (file-per-row-type
layout confirmed with the user over an earlier higher-level 6-file sketch —
see ``ssat/analysis/schema.py``'s module docstring). Follows the same
provenance principle that ``MetricsManifest`` established: this analysis is
only as trustworthy as the metrics it was computed from, tracked here via
``source_metrics_manifest_hash`` — a SHA-256 of the source metrics store's
actual ``metrics_manifest.json`` bytes, not a re-serialization of its parsed
contents.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa

from ssat.analysis._storage import atomic_write_parquet, read_validated_table
from ssat.analysis.errors import AnalysisCorruptionError, AnalysisSchemaError
from ssat.analysis.schema import (
    ANALYSIS_SCHEMA_VERSION,
    CONTROL_COMPARISON_SCHEMA,
    INTERVALS_SCHEMA,
    RANK_CORRELATION_SCHEMA,
    RELIABILITY_SCHEMA,
    SEED_STABILITY_SCHEMA,
    STRATEGY_PROFILE_SCHEMA,
    STRATEGY_STABILITY_SCHEMA,
)
from ssat.analysis.types import (
    Alignment,
    AnchorKey,
    AvailableAnalyses,
    ConditionKey,
    ControlComparisonRow,
    CoverageReport,
    FlagValue,
    IntervalRow,
    RankCorrelationRow,
    ReliabilityGrade,
    ReliabilityRow,
    SeedStabilityRow,
    StrategyProfileRow,
    StrategyStabilityRow,
)
from ssat.utils.io import load_json, sha256_file, write_json_atomic


@dataclass(frozen=True, slots=True)
class AnalysisManifest:
    """Describe one stored analysis run and the metrics store it was computed from.

    Attributes:
        analysis_schema_version: Version of the on-disk analysis format itself.
        source_metrics_manifest_hash: SHA-256 of the source metrics store's
            ``metrics_manifest.json`` bytes at computation time.
        available_analyses: A0's availability report, recorded verbatim.
        thresholds: Every threshold actually used to derive flags/grades
            (A2-A6), keyed by name — free-form so callers are not forced
            into a fixed vocabulary before an orchestrator exists (v1 has
            none yet).
        n_bootstrap: A5's bootstrap resample count for this run.
        random_seed: A5's resampling seed for this run.
        computed_at: When this analysis run was persisted.
        grade_distribution: Count of ``reliability_rows`` per
            ``ReliabilityGrade`` value, derived by ``save_analysis`` itself
            rather than supplied by the caller (single source of truth).
    """

    analysis_schema_version: str
    source_metrics_manifest_hash: str
    available_analyses: AvailableAnalyses
    thresholds: Mapping[str, float]
    n_bootstrap: int
    random_seed: int
    computed_at: datetime
    grade_distribution: Mapping[str, int]

    def __post_init__(self) -> None:
        """Validate identity, hash format, availability type, and timestamp.

        Raises:
            TypeError: If available_analyses has the wrong type.
            ValueError: If analysis_schema_version is empty,
                source_metrics_manifest_hash is not a lowercase SHA-256 hex
                digest, computed_at is not timezone-aware, or n_bootstrap is
                negative.
        """

        if not self.analysis_schema_version:
            raise ValueError("analysis_schema_version must not be empty")
        _validate_sha256_hex(self.source_metrics_manifest_hash)
        if not isinstance(self.available_analyses, AvailableAnalyses):
            raise TypeError("available_analyses must be an AvailableAnalyses")
        if self.computed_at.tzinfo is None or self.computed_at.utcoffset() is None:
            raise ValueError("computed_at must be timezone-aware")
        if isinstance(self.n_bootstrap, bool) or self.n_bootstrap < 0:
            raise ValueError("n_bootstrap must be non-negative")


def save_analysis(
    analysis_dir: Path,
    *,
    control_rows: Sequence[ControlComparisonRow],
    seed_rows: Sequence[SeedStabilityRow],
    strategy_rows: Sequence[StrategyStabilityRow],
    rank_correlation_rows: Sequence[RankCorrelationRow],
    strategy_profile_rows: Sequence[StrategyProfileRow],
    interval_rows: Sequence[IntervalRow],
    reliability_rows: Sequence[ReliabilityRow],
    coverage_report: CoverageReport,
    available_analyses: AvailableAnalyses,
    thresholds: Mapping[str, float],
    n_bootstrap: int,
    random_seed: int,
    source_metrics_manifest_path: Path,
    computed_at: datetime | None = None,
) -> AnalysisManifest:
    """Persist every A2-A6 output plus coverage_report.json and analysis_manifest.json.

    Args:
        analysis_dir: Destination directory (created if missing), typically
            ``<run_dir>/analysis``.
        control_rows: A2 output.
        seed_rows: A3(a) output.
        strategy_rows: A3(c) per-anchor output.
        rank_correlation_rows: A3(c) dataset-level output.
        strategy_profile_rows: A4 output.
        interval_rows: A5 output.
        reliability_rows: A6 output.
        coverage_report: A1's dataset-wide accounting.
        available_analyses: A0's availability report.
        thresholds: Every threshold used to derive this run's flags/grades.
        n_bootstrap: A5's bootstrap resample count.
        random_seed: A5's resampling seed.
        source_metrics_manifest_path: Path to the source metrics store's
            ``metrics_manifest.json``, hashed to produce
            ``source_metrics_manifest_hash``.
        computed_at: Timestamp to record; defaults to now (UTC).

    Returns:
        The ``AnalysisManifest`` that was written.
    """

    analysis_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(
        analysis_dir / "control_comparison.parquet", _control_comparison_to_table(control_rows)
    )
    atomic_write_parquet(
        analysis_dir / "seed_stability.parquet", _seed_stability_to_table(seed_rows)
    )
    atomic_write_parquet(
        analysis_dir / "strategy_stability.parquet", _strategy_stability_to_table(strategy_rows)
    )
    atomic_write_parquet(
        analysis_dir / "rank_correlation.parquet",
        _rank_correlation_to_table(rank_correlation_rows),
    )
    atomic_write_parquet(
        analysis_dir / "strategy_profile.parquet",
        _strategy_profile_to_table(strategy_profile_rows),
    )
    atomic_write_parquet(analysis_dir / "intervals.parquet", _intervals_to_table(interval_rows))
    atomic_write_parquet(
        analysis_dir / "reliability.parquet", _reliability_to_table(reliability_rows)
    )
    write_json_atomic(analysis_dir / "coverage_report.json", _coverage_report_to_json(coverage_report))

    grade_distribution = dict(Counter(row.reliability_grade.value for row in reliability_rows))
    manifest = AnalysisManifest(
        analysis_schema_version=ANALYSIS_SCHEMA_VERSION,
        source_metrics_manifest_hash=sha256_file(source_metrics_manifest_path),
        available_analyses=available_analyses,
        thresholds=dict(thresholds),
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
        computed_at=computed_at or datetime.now(timezone.utc),
        grade_distribution=grade_distribution,
    )
    write_json_atomic(analysis_dir / "analysis_manifest.json", _manifest_to_json(manifest))
    return manifest


def load_analysis(analysis_dir: Path) -> tuple[
    list[ControlComparisonRow],
    list[SeedStabilityRow],
    list[StrategyStabilityRow],
    list[RankCorrelationRow],
    list[StrategyProfileRow],
    list[IntervalRow],
    list[ReliabilityRow],
    CoverageReport,
    AnalysisManifest,
]:
    """Reload every analysis output and its manifest.

    Raises:
        AnalysisCorruptionError: If a parquet/json fragment or the manifest
            is missing or unreadable.
        AnalysisSchemaError: If ``analysis_schema_version`` or any parquet
            fragment's version/schema does not match this module's version.
    """

    manifest = _load_manifest(analysis_dir / "analysis_manifest.json")

    control_rows = _table_to_control_comparison(
        read_validated_table(
            analysis_dir / "control_comparison.parquet",
            CONTROL_COMPARISON_SCHEMA,
            expected_version=ANALYSIS_SCHEMA_VERSION,
        )
    )
    seed_rows = _table_to_seed_stability(
        read_validated_table(
            analysis_dir / "seed_stability.parquet",
            SEED_STABILITY_SCHEMA,
            expected_version=ANALYSIS_SCHEMA_VERSION,
        )
    )
    strategy_rows = _table_to_strategy_stability(
        read_validated_table(
            analysis_dir / "strategy_stability.parquet",
            STRATEGY_STABILITY_SCHEMA,
            expected_version=ANALYSIS_SCHEMA_VERSION,
        )
    )
    rank_correlation_rows = _table_to_rank_correlation(
        read_validated_table(
            analysis_dir / "rank_correlation.parquet",
            RANK_CORRELATION_SCHEMA,
            expected_version=ANALYSIS_SCHEMA_VERSION,
        )
    )
    strategy_profile_rows = _table_to_strategy_profile(
        read_validated_table(
            analysis_dir / "strategy_profile.parquet",
            STRATEGY_PROFILE_SCHEMA,
            expected_version=ANALYSIS_SCHEMA_VERSION,
        )
    )
    interval_rows = _table_to_intervals(
        read_validated_table(
            analysis_dir / "intervals.parquet",
            INTERVALS_SCHEMA,
            expected_version=ANALYSIS_SCHEMA_VERSION,
        )
    )
    reliability_rows = _table_to_reliability(
        read_validated_table(
            analysis_dir / "reliability.parquet",
            RELIABILITY_SCHEMA,
            expected_version=ANALYSIS_SCHEMA_VERSION,
        )
    )
    coverage_report = _load_coverage_report(analysis_dir / "coverage_report.json")

    return (
        control_rows,
        seed_rows,
        strategy_rows,
        rank_correlation_rows,
        strategy_profile_rows,
        interval_rows,
        reliability_rows,
        coverage_report,
        manifest,
    )


def verify_source_metrics(manifest: AnalysisManifest, metrics_manifest_path: Path) -> None:
    """Raise if the source metrics store's metrics_manifest.json no longer matches.

    Raises:
        AnalysisCorruptionError: If the source metrics were recomputed or
            otherwise changed since this analysis was computed.
    """

    actual_hash = sha256_file(metrics_manifest_path)
    if actual_hash != manifest.source_metrics_manifest_hash:
        raise AnalysisCorruptionError(
            "source metrics have changed since this analysis was computed: "
            f"expected metrics_manifest hash {manifest.source_metrics_manifest_hash}, "
            f"found {actual_hash}"
        )


# --- manifest / coverage report JSON -----------------------------------


def _manifest_to_json(manifest: AnalysisManifest) -> dict[str, object]:
    return {
        "analysis_schema_version": manifest.analysis_schema_version,
        "source_metrics_manifest_hash": manifest.source_metrics_manifest_hash,
        "available_analyses": {
            "control_comparison": manifest.available_analyses.control_comparison,
            "fill_strategy_stability": manifest.available_analyses.fill_strategy_stability,
            "seed_stability": manifest.available_analyses.seed_stability,
            "jitter_stability": manifest.available_analyses.jitter_stability,
        },
        "thresholds": dict(manifest.thresholds),
        "n_bootstrap": manifest.n_bootstrap,
        "random_seed": manifest.random_seed,
        "computed_at": manifest.computed_at.isoformat(),
        "grade_distribution": dict(manifest.grade_distribution),
    }


def _load_manifest(path: Path) -> AnalysisManifest:
    if not path.is_file():
        raise AnalysisCorruptionError(f"analysis manifest does not exist: {path}")
    try:
        payload = load_json(path)
    except Exception as error:
        raise AnalysisCorruptionError(f"cannot read analysis manifest: {path}") from error
    if not isinstance(payload, dict):
        raise AnalysisCorruptionError("analysis manifest must contain a JSON object")

    version = payload.get("analysis_schema_version")
    if version != ANALYSIS_SCHEMA_VERSION:
        raise AnalysisSchemaError(
            f"analysis schema version mismatch: expected {ANALYSIS_SCHEMA_VERSION}, found {version}"
        )

    try:
        available = payload["available_analyses"]
        return AnalysisManifest(
            analysis_schema_version=payload["analysis_schema_version"],
            source_metrics_manifest_hash=payload["source_metrics_manifest_hash"],
            available_analyses=AvailableAnalyses(
                control_comparison=available["control_comparison"],
                fill_strategy_stability=available["fill_strategy_stability"],
                seed_stability=available["seed_stability"],
                jitter_stability=available["jitter_stability"],
            ),
            thresholds=payload["thresholds"],
            n_bootstrap=payload["n_bootstrap"],
            random_seed=payload["random_seed"],
            computed_at=datetime.fromisoformat(payload["computed_at"]),
            grade_distribution=payload["grade_distribution"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AnalysisCorruptionError("analysis manifest does not match its schema") from error


def _coverage_report_to_json(report: CoverageReport) -> dict[str, object]:
    return {
        "n_anchors": report.n_anchors,
        "n_conditions_insufficient": report.n_conditions_insufficient,
        "n_controls_unmatched": report.n_controls_unmatched,
        "n_area_mismatch_warnings": report.n_area_mismatch_warnings,
    }


def _load_coverage_report(path: Path) -> CoverageReport:
    if not path.is_file():
        raise AnalysisCorruptionError(f"coverage report does not exist: {path}")
    try:
        payload = load_json(path)
        return CoverageReport(
            n_anchors=payload["n_anchors"],
            n_conditions_insufficient=payload["n_conditions_insufficient"],
            n_controls_unmatched=payload["n_controls_unmatched"],
            n_area_mismatch_warnings=payload["n_area_mismatch_warnings"],
        )
    except Exception as error:
        raise AnalysisCorruptionError(f"cannot read coverage report: {path}") from error


# --- ControlComparisonRow -----------------------------------------------


def _control_comparison_to_table(rows: Sequence[ControlComparisonRow]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "sample_id": row.target_anchor_key.sample_id,
                "region_key": row.target_anchor_key.region_key,
                "invert_mask": row.target_anchor_key.invert_mask,
                "perturb_op": row.condition_key.perturb_op,
                "perturb_params_hash": row.condition_key.perturb_params_hash,
                "metric_name": row.metric_name,
                "control_available": row.control_available.value,
                "area_matched": row.area_matched.value,
                "control_mean": row.control_mean,
                "control_std": row.control_std,
                "n_controls": row.n_controls,
                "excess": row.excess,
                "ratio": row.ratio,
                "z_vs_control": row.z_vs_control,
            }
            for row in rows
        ],
        schema=CONTROL_COMPARISON_SCHEMA,
    )


def _table_to_control_comparison(table: pa.Table) -> list[ControlComparisonRow]:
    return [
        ControlComparisonRow(
            target_anchor_key=AnchorKey(
                sample_id=row["sample_id"],
                region_key=row["region_key"],
                invert_mask=row["invert_mask"],
            ),
            condition_key=ConditionKey(
                perturb_op=row["perturb_op"], perturb_params_hash=row["perturb_params_hash"]
            ),
            metric_name=row["metric_name"],
            control_available=FlagValue(row["control_available"]),
            area_matched=FlagValue(row["area_matched"]),
            control_mean=row["control_mean"],
            control_std=row["control_std"],
            n_controls=row["n_controls"],
            excess=row["excess"],
            ratio=row["ratio"],
            z_vs_control=row["z_vs_control"],
        )
        for row in table.to_pylist()
    ]


# --- SeedStabilityRow ----------------------------------------------------


def _seed_stability_to_table(rows: Sequence[SeedStabilityRow]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "sample_id": row.anchor_key.sample_id,
                "region_key": row.anchor_key.region_key,
                "invert_mask": row.anchor_key.invert_mask,
                "perturb_op": row.condition_key.perturb_op,
                "perturb_params_hash": row.condition_key.perturb_params_hash,
                "metric_name": row.metric_name,
                "seed_mean": row.seed_mean,
                "seed_std": row.seed_std,
                "seed_cv": row.seed_cv,
                "n_seeds": row.n_seeds,
            }
            for row in rows
        ],
        schema=SEED_STABILITY_SCHEMA,
    )


def _table_to_seed_stability(table: pa.Table) -> list[SeedStabilityRow]:
    return [
        SeedStabilityRow(
            anchor_key=AnchorKey(
                sample_id=row["sample_id"],
                region_key=row["region_key"],
                invert_mask=row["invert_mask"],
            ),
            condition_key=ConditionKey(
                perturb_op=row["perturb_op"], perturb_params_hash=row["perturb_params_hash"]
            ),
            metric_name=row["metric_name"],
            seed_mean=row["seed_mean"],
            seed_std=row["seed_std"],
            seed_cv=row["seed_cv"],
            n_seeds=row["n_seeds"],
        )
        for row in table.to_pylist()
    ]


# --- StrategyStabilityRow ------------------------------------------------


def _strategy_stability_to_table(rows: Sequence[StrategyStabilityRow]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "sample_id": row.anchor_key.sample_id,
                "region_key": row.anchor_key.region_key,
                "invert_mask": row.anchor_key.invert_mask,
                "metric_name": row.metric_name,
                "strategy_signs_json": json.dumps(row.strategy_signs, sort_keys=True),
                "strategy_values_json": json.dumps(row.strategy_values, sort_keys=True),
                "sign_agreement_ratio": row.sign_agreement_ratio,
                "n_strategies": row.n_strategies,
            }
            for row in rows
        ],
        schema=STRATEGY_STABILITY_SCHEMA,
    )


def _table_to_strategy_stability(table: pa.Table) -> list[StrategyStabilityRow]:
    return [
        StrategyStabilityRow(
            anchor_key=AnchorKey(
                sample_id=row["sample_id"],
                region_key=row["region_key"],
                invert_mask=row["invert_mask"],
            ),
            metric_name=row["metric_name"],
            strategy_signs=json.loads(row["strategy_signs_json"]),
            strategy_values=json.loads(row["strategy_values_json"]),
            sign_agreement_ratio=row["sign_agreement_ratio"],
            n_strategies=row["n_strategies"],
        )
        for row in table.to_pylist()
    ]


# --- RankCorrelationRow ----------------------------------------------------


def _rank_correlation_to_table(rows: Sequence[RankCorrelationRow]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "op_a": row.op_a,
                "op_b": row.op_b,
                "spearman": row.spearman,
                "n_regions": row.n_regions,
                "spearman_excl_top1": row.spearman_excl_top1,
                "scope": row.scope,
            }
            for row in rows
        ],
        schema=RANK_CORRELATION_SCHEMA,
    )


def _table_to_rank_correlation(table: pa.Table) -> list[RankCorrelationRow]:
    return [
        RankCorrelationRow(
            op_a=row["op_a"],
            op_b=row["op_b"],
            spearman=row["spearman"],
            n_regions=row["n_regions"],
            spearman_excl_top1=row["spearman_excl_top1"],
            scope=row["scope"],
        )
        for row in table.to_pylist()
    ]


# --- StrategyProfileRow ----------------------------------------------------


def _strategy_profile_to_table(rows: Sequence[StrategyProfileRow]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "perturb_op": row.perturb_op,
                "preserves_statistics": row.preserves_statistics,
                "preserves_local_texture": row.preserves_local_texture,
                "is_global_operation": row.is_global_operation,
                "cluster_id": row.cluster_id,
                "mean_corr_within": row.mean_corr_within,
                "mean_corr_across": row.mean_corr_across,
                "alignment": row.alignment.value,
                "mean_degradation_excl_top": row.mean_degradation_excl_top,
                "sign_ratio_positive": row.sign_ratio_positive,
                "n_anchors": row.n_anchors,
            }
            for row in rows
        ],
        schema=STRATEGY_PROFILE_SCHEMA,
    )


def _table_to_strategy_profile(table: pa.Table) -> list[StrategyProfileRow]:
    return [
        StrategyProfileRow(
            perturb_op=row["perturb_op"],
            preserves_statistics=row["preserves_statistics"],
            preserves_local_texture=row["preserves_local_texture"],
            is_global_operation=row["is_global_operation"],
            cluster_id=row["cluster_id"],
            mean_corr_within=row["mean_corr_within"],
            mean_corr_across=row["mean_corr_across"],
            alignment=Alignment(row["alignment"]),
            mean_degradation_excl_top=row["mean_degradation_excl_top"],
            sign_ratio_positive=row["sign_ratio_positive"],
            n_anchors=row["n_anchors"],
        )
        for row in table.to_pylist()
    ]


# --- IntervalRow -----------------------------------------------------------


def _intervals_to_table(rows: Sequence[IntervalRow]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "region_key": row.region_key,
                "metric": row.metric,
                "point_estimate": row.point_estimate,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "ci_method": row.ci_method,
                "n_bootstrap": row.n_bootstrap,
                "excludes_zero": row.excludes_zero,
            }
            for row in rows
        ],
        schema=INTERVALS_SCHEMA,
    )


def _table_to_intervals(table: pa.Table) -> list[IntervalRow]:
    return [
        IntervalRow(
            region_key=row["region_key"],
            metric=row["metric"],
            point_estimate=row["point_estimate"],
            ci_low=row["ci_low"],
            ci_high=row["ci_high"],
            ci_method=row["ci_method"],
            n_bootstrap=row["n_bootstrap"],
            excludes_zero=row["excludes_zero"],
        )
        for row in table.to_pylist()
    ]


# --- ReliabilityRow ----------------------------------------------------


def _reliability_to_table(rows: Sequence[ReliabilityRow]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "sample_id": row.anchor_key.sample_id,
                "region_key": row.anchor_key.region_key,
                "invert_mask": row.anchor_key.invert_mask,
                "metric_name": row.metric_name,
                "sign_consistent": row.sign_consistent.value,
                "exceeds_control": row.exceeds_control.value,
                "seed_stable": row.seed_stable.value,
                "jitter_stable": row.jitter_stable.value,
                "multi_strategy": row.multi_strategy.value,
                "ci_excludes_zero": row.ci_excludes_zero.value,
                "area_matched": row.area_matched.value,
                "reliability_grade": row.reliability_grade.value,
                "reliability_reasons": list(row.reliability_reasons),
            }
            for row in rows
        ],
        schema=RELIABILITY_SCHEMA,
    )


def _table_to_reliability(table: pa.Table) -> list[ReliabilityRow]:
    return [
        ReliabilityRow(
            anchor_key=AnchorKey(
                sample_id=row["sample_id"],
                region_key=row["region_key"],
                invert_mask=row["invert_mask"],
            ),
            metric_name=row["metric_name"],
            sign_consistent=FlagValue(row["sign_consistent"]),
            exceeds_control=FlagValue(row["exceeds_control"]),
            seed_stable=FlagValue(row["seed_stable"]),
            jitter_stable=FlagValue(row["jitter_stable"]),
            multi_strategy=FlagValue(row["multi_strategy"]),
            ci_excludes_zero=FlagValue(row["ci_excludes_zero"]),
            area_matched=FlagValue(row["area_matched"]),
            reliability_grade=ReliabilityGrade(row["reliability_grade"]),
            reliability_reasons=tuple(row["reliability_reasons"]),
        )
        for row in table.to_pylist()
    ]


def _validate_sha256_hex(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("source_metrics_manifest_hash must be a lowercase SHA-256 hex digest")
