"""Unit tests for ssat.report.assembler's pure helpers and input validation.

Complements tests/integration/test_report_synthetic_dump.py, which exercises
:class:`ReportDataAssembler` end-to-end against real store data — these
tests isolate the small, deterministic pieces (grade reduction, dataset-name
derivation, failure-rate arithmetic, constructor guards) that do not need a
dump/metrics/analysis triple at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from synthetic_dump_builder import (
    build_resolved_config,
    clean_record,
    compute_and_save_metrics,
    perturbed_record,
    write_dump,
)

from ssat.analysis.types import AnchorKey, FlagValue, ReliabilityGrade, ReliabilityRow
from ssat.core.config.schema import ResolvedRegionConfig
from ssat.core.types import RegionKind
from ssat.metrics.builtin_metrics.continuous import GtLogitDrop
from ssat.metrics.registry import MetricRegistry
from ssat.report.adapters import ClassificationAdapter
from ssat.report.assembler import (
    ReportDataAssembler,
    _dataset_name,
    _failure_rate,
    _group_report_grades,
    _worst_grade,
)
from ssat.report.errors import ReportDataError
from ssat.report.types import ReportGrade

_METRIC_NAME = "gt_logit_drop"


def _reliability_row(
    *,
    sample_id: str,
    region_key: str,
    metric_name: str = _METRIC_NAME,
    grade: ReliabilityGrade,
) -> ReliabilityRow:
    return ReliabilityRow(
        anchor_key=AnchorKey(sample_id=sample_id, region_key=region_key, invert_mask=False),
        metric_name=metric_name,
        sign_consistent=FlagValue.UNAVAILABLE,
        exceeds_control=FlagValue.UNAVAILABLE,
        seed_stable=FlagValue.UNAVAILABLE,
        jitter_stable=FlagValue.UNAVAILABLE,
        multi_strategy=FlagValue.UNAVAILABLE,
        ci_excludes_zero=FlagValue.UNAVAILABLE,
        area_matched=FlagValue.UNAVAILABLE,
        reliability_grade=grade,
        reliability_reasons=(),
    )


# --- _worst_grade -----------------------------------------------------------


def test_worst_grade_returns_none_for_empty_sequence() -> None:
    assert _worst_grade(()) is None


def test_worst_grade_picks_unreliable_over_high() -> None:
    assert _worst_grade([ReportGrade.HIGH, ReportGrade.UNRELIABLE]) is ReportGrade.UNRELIABLE


def test_worst_grade_full_severity_order() -> None:
    assert _worst_grade([ReportGrade.HIGH, ReportGrade.MODERATE]) is ReportGrade.MODERATE
    assert _worst_grade([ReportGrade.MODERATE, ReportGrade.LOW]) is ReportGrade.LOW
    assert _worst_grade([ReportGrade.LOW, ReportGrade.UNRELIABLE]) is ReportGrade.UNRELIABLE


def test_worst_grade_single_element_is_itself() -> None:
    assert _worst_grade([ReportGrade.MODERATE]) is ReportGrade.MODERATE


# --- _group_report_grades ----------------------------------------------------


def test_group_report_grades_filters_by_metric_name_and_groups_by_key() -> None:
    rows = [
        _reliability_row(sample_id="s0", region_key="r0", grade=ReliabilityGrade.HIGH),
        _reliability_row(sample_id="s1", region_key="r0", grade=ReliabilityGrade.UNRELIABLE),
        _reliability_row(sample_id="s2", region_key="r1", grade=ReliabilityGrade.LOW),
        _reliability_row(
            sample_id="s3", region_key="r0", metric_name="other_metric", grade=ReliabilityGrade.HIGH
        ),
    ]

    grouped = _group_report_grades(rows, _METRIC_NAME, key=lambda row: row.anchor_key.region_key)

    assert grouped == {
        "r0": [ReportGrade.HIGH, ReportGrade.UNRELIABLE],
        "r1": [ReportGrade.LOW],
    }


def test_group_report_grades_empty_rows_yields_empty_dict() -> None:
    assert _group_report_grades([], _METRIC_NAME, key=lambda row: row.anchor_key.sample_id) == {}


# --- _dataset_name -----------------------------------------------------------


class _FakeSourceProvenance:
    def __init__(self, manifest: Path) -> None:
        self.manifest = manifest


class _FakeResolvedConfig:
    def __init__(self, *, source_provenance: object = None, config_source: Path | None = None) -> None:
        self.source_provenance = source_provenance
        self.config_source = config_source


def test_dataset_name_from_source_provenance_manifest_parent(tmp_path: Path) -> None:
    manifest_path = tmp_path / "shortcut_A" / "manifest.json"
    config = _FakeResolvedConfig(source_provenance=_FakeSourceProvenance(manifest_path))

    assert _dataset_name(config) == "shortcut_A"


def test_dataset_name_falls_back_to_config_source_stem(tmp_path: Path) -> None:
    config = _FakeResolvedConfig(config_source=tmp_path / "my_audit.yaml")

    assert _dataset_name(config) == "my_audit"


def test_dataset_name_falls_back_to_unknown() -> None:
    assert _dataset_name(_FakeResolvedConfig()) == "unknown"


# --- _failure_rate -----------------------------------------------------------


def test_failure_rate_computes_fraction() -> None:
    summary = {"total_perturbed_items": 20, "items_excluded_perturbed_failed": 5}
    assert _failure_rate(summary) == 0.25


def test_failure_rate_none_when_no_perturbed_items() -> None:
    summary = {"total_perturbed_items": 0, "items_excluded_perturbed_failed": 0}
    assert _failure_rate(summary) is None


# --- constructor / assemble() validation -------------------------------------


def _minimal_dump_and_metrics(tmp_path: Path) -> tuple[Path, Path]:
    config = build_resolved_config(
        tmp_path,
        regions=(ResolvedRegionConfig(region_id="grid", kind=RegionKind.GRID, params={"rows": 1, "cols": 1}),),
    )
    dump_root = tmp_path / "dump"
    write_dump(
        dump_root,
        config,
        clean_records=(clean_record("s0", logits=np.array([1.0, 0.0])),),
        perturbed_records=(
            perturbed_record(
                0,
                sample_id="s0",
                region_id="grid",
                region_instance_id="grid/r0/c0",
                logits=np.array([0.5, 0.0]),
            ),
        ),
    )
    metrics_dir = tmp_path / "metrics"
    registry = MetricRegistry()
    registry.register(GtLogitDrop())
    compute_and_save_metrics(dump_root, config, metrics_dir, registry=registry, primary_metric=_METRIC_NAME)
    return dump_root, metrics_dir


@pytest.mark.parametrize("kwargs", [{"top_k": -1}, {"bottom_k": -1}, {"region_top_k": -1}])
def test_constructor_rejects_negative_selection_sizes(tmp_path: Path, kwargs: dict) -> None:
    adapter = ClassificationAdapter(primary_metric=_METRIC_NAME)
    with pytest.raises(ValueError, match="non-negative"):
        ReportDataAssembler(tmp_path / "dump", tmp_path / "metrics", adapter=adapter, **kwargs)


def test_assemble_rejects_empty_primary_metric(tmp_path: Path) -> None:
    dump_root, metrics_dir = _minimal_dump_and_metrics(tmp_path)
    adapter = ClassificationAdapter(primary_metric=_METRIC_NAME)
    assembler = ReportDataAssembler(dump_root, metrics_dir, adapter=adapter)

    with pytest.raises(ValueError, match="primary_metric"):
        assembler.assemble("")


def test_assemble_rejects_unregistered_primary_metric(tmp_path: Path) -> None:
    dump_root, metrics_dir = _minimal_dump_and_metrics(tmp_path)
    adapter = ClassificationAdapter(primary_metric=_METRIC_NAME)
    assembler = ReportDataAssembler(dump_root, metrics_dir, adapter=adapter)

    with pytest.raises(ReportDataError, match="not_a_real_metric"):
        assembler.assemble("not_a_real_metric")
