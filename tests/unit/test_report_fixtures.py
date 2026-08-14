"""Regression tests for the report-layer test fixture extension.

Covers ``tests/fixtures/synthetic_dump_builder.compute_and_save_analysis``
(IMPLE_PLAN_REPORTING_v1.md §5 단계0) — the thin wrapper future report-layer
tests will use to get a ready-to-load dump+metrics+analysis triple without
hand-rolling the A0-A6 pipeline themselves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from synthetic_dump_builder import (
    build_resolved_config,
    clean_record,
    compute_and_save_analysis,
    compute_and_save_metrics,
    perturbed_record,
    write_dump,
)

from ssat.analysis.store import load_analysis
from ssat.application import AnalyzeRequest, AuditApplication
from ssat.core.config.schema import ResolvedRegionConfig
from ssat.core.types import PerturbationOp, RegionKind
from ssat.metrics.builtin_metrics.continuous import GtLogitDrop
from ssat.metrics.registry import MetricRegistry

_METRIC_NAME = "gt_logit_drop"
_TARGET_REGIONS = (
    ResolvedRegionConfig(region_id="grid", kind=RegionKind.GRID, params={"rows": 1, "cols": 1}),
)
_LOGITS = np.array([1.0, 0.0])


def _registry() -> MetricRegistry:
    registry = MetricRegistry()
    registry.register(GtLogitDrop())
    return registry


def _build_dump_with_metrics(tmp_path: Path) -> tuple[Path, Path]:
    """Build a synthetic dump with a control and two fill strategies.

    Mirrors ``test_analysis_reader.py``'s ``_build_reader_fixture`` shape —
    enough condition variety (two ops, one control) that A2 (control
    comparison) and A3(c) (fill-strategy stability) both actually run,
    rather than trivially reporting "unavailable".
    """

    config = build_resolved_config(tmp_path, regions=_TARGET_REGIONS)
    sample_id = "s0"
    perturbed_records = [
        perturbed_record(
            0,
            sample_id=sample_id,
            region_id="grid",
            region_instance_id="grid/r0/c0",
            logits=_LOGITS,
            perturb_op=PerturbationOp.CONSTANT_FILL,
        ),
        perturbed_record(
            1,
            sample_id=sample_id,
            region_id="grid",
            region_instance_id="grid/r0/c0",
            logits=_LOGITS,
            perturb_op=PerturbationOp.MEAN_FILL,
        ),
        perturbed_record(
            2,
            sample_id=sample_id,
            region_id="control",
            region_instance_id="control/0",
            logits=_LOGITS,
            region_kind=RegionKind.RANDOM_AREA_MATCH,
            is_control=True,
        ),
    ]
    dump_root = tmp_path / "dump"
    write_dump(
        dump_root,
        config,
        clean_records=(clean_record(sample_id, logits=_LOGITS),),
        perturbed_records=tuple(perturbed_records),
    )
    metrics_dir = tmp_path / "metrics"
    compute_and_save_metrics(
        dump_root, config, metrics_dir, registry=_registry(), primary_metric=_METRIC_NAME
    )
    return dump_root, metrics_dir


def test_compute_and_save_analysis_persists_a_loadable_analysis(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_dump_with_metrics(tmp_path)
    analysis_dir = tmp_path / "analysis"

    manifest = compute_and_save_analysis(
        dump_root, metrics_dir, analysis_dir, primary_metric=_METRIC_NAME
    )

    assert manifest.available_analyses.control_comparison
    assert manifest.available_analyses.fill_strategy_stability
    (*_rest, reliability_rows, coverage_report, reloaded_manifest) = load_analysis(analysis_dir)
    assert reloaded_manifest == manifest
    assert sum(manifest.grade_distribution.values()) == len(reliability_rows)
    assert coverage_report.n_anchors > 0


def test_compute_and_save_analysis_matches_direct_application_analyze(tmp_path: Path) -> None:
    """The fixture helper must not silently diverge from calling analyze() directly.

    Both paths start from the identical dump+metrics pair and default
    thresholds/random_seed, so every persisted row and every manifest field
    except the wall-clock ``computed_at`` timestamp must match exactly.
    """

    dump_root, metrics_dir = _build_dump_with_metrics(tmp_path)

    via_fixture_dir = tmp_path / "analysis_via_fixture"
    via_fixture_manifest = compute_and_save_analysis(
        dump_root, metrics_dir, via_fixture_dir, primary_metric=_METRIC_NAME
    )

    via_direct_dir = tmp_path / "analysis_via_direct"
    AuditApplication().analyze(
        AnalyzeRequest(
            dump=dump_root,
            metrics_dir=metrics_dir,
            analysis_dir=via_direct_dir,
            primary_metric=_METRIC_NAME,
        )
    )
    *_rest, via_direct_manifest = load_analysis(via_direct_dir)

    fixture_rows = load_analysis(via_fixture_dir)[:-1]
    direct_rows = load_analysis(via_direct_dir)[:-1]
    assert fixture_rows == direct_rows

    assert via_fixture_manifest.analysis_schema_version == via_direct_manifest.analysis_schema_version
    assert (
        via_fixture_manifest.source_metrics_manifest_hash
        == via_direct_manifest.source_metrics_manifest_hash
    )
    assert via_fixture_manifest.available_analyses == via_direct_manifest.available_analyses
    assert via_fixture_manifest.thresholds == via_direct_manifest.thresholds
    assert via_fixture_manifest.n_bootstrap == via_direct_manifest.n_bootstrap
    assert via_fixture_manifest.random_seed == via_direct_manifest.random_seed
    assert via_fixture_manifest.grade_distribution == via_direct_manifest.grade_distribution
