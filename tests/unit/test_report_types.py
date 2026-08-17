"""Validation and serialization tests for the reporting-layer contract types."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import json
from pathlib import Path

import pytest

import ssat.report
from ssat.report.types import (
    FlaggedItem,
    MetricCard,
    ProvenanceInfo,
    RegionRow,
    RegionSummary,
    ReliabilitySpotlight,
    ReportGrade,
    ReportMeta,
    ReportModel,
    ReportSchemaVersions,
    RunSummary,
    SampleCard,
    SampleRankings,
    SpatialConcentration,
    TaskKind,
    TopRegionEntry,
    VulnerabilityDistribution,
    VulnerabilitySummaryStats,
)


# --- builders ----------------------------------------------------------------


def _metric_card(**overrides: object) -> MetricCard:
    defaults: dict[str, object] = {
        "key": "accuracy",
        "label": "Clean Accuracy",
        "value": 0.9,
        "unit": "%",
        "higher_is_better": True,
        "note": None,
    }
    defaults.update(overrides)
    return MetricCard(**defaults)  # type: ignore[arg-type]


def _top_region_entry(**overrides: object) -> TopRegionEntry:
    defaults: dict[str, object] = {
        "region_key": "grid::0",
        "degradation": 0.4,
        "reliability_grade": ReportGrade.HIGH,
    }
    defaults.update(overrides)
    return TopRegionEntry(**defaults)  # type: ignore[arg-type]


def _sample_card(**overrides: object) -> SampleCard:
    defaults: dict[str, object] = {
        "sample_id": "sample-1",
        "gt_label": 0,
        "clean_correct": True,
        "vulnerability_score": 0.8,
        "reliability_grade": ReportGrade.HIGH,
        "heatmap_asset_ref": "assets/img/heatmaps/sample_1.png",
        "thumbnail_asset_ref": "assets/img/thumbnails/sample_1.png",
        "top_regions": (_top_region_entry(),),
        "task_extra": {},
    }
    defaults.update(overrides)
    return SampleCard(**defaults)  # type: ignore[arg-type]


def _region_row(**overrides: object) -> RegionRow:
    defaults: dict[str, object] = {
        "region_key": "grid::0",
        "region_kind": "grid",
        "intended_area_px": 64,
        "effective_area_px": 60,
        "mean_degradation": 0.3,
        "flip_rate": 0.2,
        "n_valid": 10,
        "reliability_grade": ReportGrade.UNRELIABLE,
        "reliability_distribution": {"high": 1, "unreliable": 1},
        "top_region_share": 0.5,
        "high_rate": 0.5,
    }
    defaults.update(overrides)
    return RegionRow(**defaults)  # type: ignore[arg-type]


def _spatial_concentration(**overrides: object) -> SpatialConcentration:
    defaults: dict[str, object] = {
        "dominant_region_key": "grid::0",
        "dominant_region_share": 0.5,
        "spatial_entropy": 0.8,
        "n_scored_samples": 100,
    }
    defaults.update(overrides)
    return SpatialConcentration(**defaults)  # type: ignore[arg-type]


def _flagged_item(**overrides: object) -> FlaggedItem:
    defaults: dict[str, object] = {
        "anchor_key_repr": "sample-1::grid::0::False",
        "reason_summary": "sign flips across fill strategies",
        "reliability_reasons": ("blur:+0.1", "mean_fill:-0.2"),
    }
    defaults.update(overrides)
    return FlaggedItem(**defaults)  # type: ignore[arg-type]


def _run_summary(**overrides: object) -> RunSummary:
    defaults: dict[str, object] = {
        "dataset_name": "shortcut_A",
        "n_samples": 100,
        "n_regions_per_sample": 4,
        "n_conditions": 5,
        "duration_seconds": 120.5,
        "failure_rate": 0.01,
        "model_id": "resnet18",
        "preprocessing_desc": "224x224 center crop",
    }
    defaults.update(overrides)
    return RunSummary(**defaults)  # type: ignore[arg-type]


def _report_meta(**overrides: object) -> ReportMeta:
    defaults: dict[str, object] = {
        "run_id": "shortcut_A_all_ops",
        "generated_at": "2026-08-14T00:00:00+00:00",
        "tool_version": "1.0.0",
        "schema_versions": ReportSchemaVersions(
            dump="1.0.0", metrics="1.0.0", analysis="1.0.0", report="1.0.0"
        ),
        "task_kind": TaskKind.CLASSIFICATION,
    }
    defaults.update(overrides)
    return ReportMeta(**defaults)  # type: ignore[arg-type]


def _provenance_info(**overrides: object) -> ProvenanceInfo:
    defaults: dict[str, object] = {
        "dump_path": "/data/dump",
        "metrics_dir": "/data/dump/metrics",
        "analysis_dir": "/data/dump/analysis",
        "run_manifest_hash": "c" * 64,
        "metrics_manifest_hash": "a" * 64,
        "analysis_manifest_hash": "b" * 64,
        "thresholds": {"z_vs_control_threshold": 2.0},
    }
    defaults.update(overrides)
    return ProvenanceInfo(**defaults)  # type: ignore[arg-type]


def _report_model(**overrides: object) -> ReportModel:
    defaults: dict[str, object] = {
        "meta": _report_meta(),
        "run_summary": _run_summary(),
        "scorecard": (_metric_card(),),
        "vulnerability_distribution": VulnerabilityDistribution(
            histogram_asset_ref="assets/img/charts/histogram.svg",
            summary_stats=VulnerabilitySummaryStats(mean=0.3, median=0.25, p90=0.6, p99=0.9),
        ),
        "sample_rankings": SampleRankings(
            most_vulnerable=(_sample_card(),), most_robust=(_sample_card(sample_id="sample-2"),)
        ),
        "region_summary": RegionSummary(
            rows=(_region_row(),),
            reliability_distribution={"high": 1, "unreliable": 1},
            chart_asset_ref="assets/img/charts/region_bar.svg",
        ),
        "spatial_concentration": _spatial_concentration(),
        "fill_strategy_correlation_asset_ref": "assets/img/charts/fill_strategy_correlation.svg",
        "reliability_spotlight": ReliabilitySpotlight(flagged_examples=(_flagged_item(),)),
        "provenance": _provenance_info(),
    }
    defaults.update(overrides)
    return ReportModel(**defaults)  # type: ignore[arg-type]


# --- package scaffolding ------------------------------------------------------


def test_report_package_imports() -> None:
    importlib.import_module("ssat.report")


def test_report_package_exports_every_public_symbol() -> None:
    for name in ssat.report.__all__:
        assert hasattr(ssat.report, name), f"missing export: {name}"


def test_report_types_module_has_no_analysis_or_metrics_or_core_imports() -> None:
    """Statically enforce the §3.3 dependency rule: report.types → (없음).

    Parses the module source with ``ast`` instead of importing it, so this
    check does not depend on whether the forbidden modules happen to already
    be importable in the test environment (design intent, not import
    machinery, is what is being verified).
    """

    source_path = (
        Path(__file__).resolve().parents[2] / "ssat" / "report" / "types.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_prefixes = ("ssat.analysis", "ssat.metrics", "ssat.core", "ssat.application")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(forbidden_prefixes), alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith(forbidden_prefixes), module


# --- serialization -------------------------------------------------------------


def test_report_model_serializes_to_json_via_dataclasses_asdict() -> None:
    model = _report_model()
    payload = dataclasses.asdict(model)
    encoded = json.dumps(payload, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded["meta"]["run_id"] == "shortcut_A_all_ops"
    assert decoded["meta"]["task_kind"] == "classification"
    assert decoded["run_summary"]["dataset_name"] == "shortcut_A"
    assert decoded["scorecard"][0]["key"] == "accuracy"
    assert decoded["sample_rankings"]["most_vulnerable"][0]["sample_id"] == "sample-1"
    assert (
        decoded["sample_rankings"]["most_vulnerable"][0]["top_regions"][0]["reliability_grade"]
        == "high"
    )
    assert decoded["region_summary"]["rows"][0]["reliability_grade"] == "unreliable"
    assert decoded["reliability_spotlight"]["flagged_examples"][0]["reason_summary"]
    assert decoded["provenance"]["metrics_manifest_hash"] == "a" * 64


def test_report_model_with_no_analysis_serializes_with_none_markers() -> None:
    """"해당 없음" must round-trip as JSON null, not vanish or become false."""

    model = _report_model(
        meta=_report_meta(
            schema_versions=ReportSchemaVersions(
                dump="1.0.0", metrics="1.0.0", analysis=None, report="1.0.0"
            )
        ),
        sample_rankings=SampleRankings(
            most_vulnerable=(
                _sample_card(reliability_grade=None, top_regions=()),
            ),
            most_robust=(),
        ),
        region_summary=RegionSummary(
            rows=(_region_row(reliability_grade=None, reliability_distribution={}),),
            reliability_distribution={},
            chart_asset_ref=None,
        ),
        fill_strategy_correlation_asset_ref=None,
        reliability_spotlight=ReliabilitySpotlight(flagged_examples=()),
        provenance=_provenance_info(analysis_dir=None, analysis_manifest_hash=None),
    )
    decoded = json.loads(json.dumps(dataclasses.asdict(model), sort_keys=True))

    assert decoded["meta"]["schema_versions"]["analysis"] is None
    assert decoded["sample_rankings"]["most_vulnerable"][0]["reliability_grade"] is None
    assert decoded["region_summary"]["rows"][0]["reliability_grade"] is None
    assert decoded["provenance"]["analysis_dir"] is None


# --- ReportModel.from_dict (round trip) --------------------------------------


def test_report_model_from_dict_round_trips_full_model() -> None:
    model = _report_model()
    payload = json.loads(json.dumps(dataclasses.asdict(model), sort_keys=True))

    rebuilt = ReportModel.from_dict(payload)

    assert rebuilt == model


def test_report_model_from_dict_round_trips_none_markers() -> None:
    """The "해당 없음" path must reconstruct with the same None markers, not defaults."""

    model = _report_model(
        meta=_report_meta(
            schema_versions=ReportSchemaVersions(
                dump="1.0.0", metrics="1.0.0", analysis=None, report="1.0.0"
            )
        ),
        sample_rankings=SampleRankings(
            most_vulnerable=(_sample_card(reliability_grade=None, top_regions=()),),
            most_robust=(),
        ),
        region_summary=RegionSummary(
            rows=(_region_row(reliability_grade=None, reliability_distribution={}),),
            reliability_distribution={},
            chart_asset_ref=None,
        ),
        fill_strategy_correlation_asset_ref=None,
        reliability_spotlight=ReliabilitySpotlight(flagged_examples=()),
        provenance=_provenance_info(analysis_dir=None, analysis_manifest_hash=None),
    )
    payload = json.loads(json.dumps(dataclasses.asdict(model), sort_keys=True))

    rebuilt = ReportModel.from_dict(payload)

    assert rebuilt == model
    assert rebuilt.meta.schema_versions.analysis is None
    assert rebuilt.sample_rankings.most_vulnerable[0].reliability_grade is None


def test_report_model_from_dict_rejects_missing_field() -> None:
    payload = dataclasses.asdict(_report_model())
    del payload["provenance"]

    with pytest.raises(KeyError):
        ReportModel.from_dict(payload)


# --- MetricCard ----------------------------------------------------------------


def test_metric_card_accepts_valid_fields() -> None:
    card = _metric_card()
    assert card.value == 0.9


def test_metric_card_allows_none_value_with_note() -> None:
    card = _metric_card(value=None, note="flip_rate not applicable for continuous metrics")
    assert card.value is None
    assert card.note


def test_metric_card_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _metric_card(key="")


def test_metric_card_rejects_empty_label() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _metric_card(label="")


# --- TopRegionEntry --------------------------------------------------------------


def test_top_region_entry_accepts_none_grade() -> None:
    entry = _top_region_entry(reliability_grade=None)
    assert entry.reliability_grade is None


def test_top_region_entry_rejects_non_grade_type() -> None:
    with pytest.raises(TypeError, match="ReportGrade"):
        _top_region_entry(reliability_grade="high")


def test_top_region_entry_rejects_empty_region_key() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _top_region_entry(region_key="")


# --- SampleCard ----------------------------------------------------------------


def test_sample_card_rejects_empty_sample_id() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _sample_card(sample_id="")


def test_sample_card_rejects_non_top_region_entry_elements() -> None:
    with pytest.raises(TypeError, match="TopRegionEntry"):
        _sample_card(top_regions=("not-an-entry",))


def test_sample_card_rejects_non_string_task_extra_value() -> None:
    with pytest.raises(TypeError, match="unsupported value type"):
        _sample_card(task_extra={"n_missed": object()})


def test_sample_card_accepts_nested_json_task_extra() -> None:
    card = _sample_card(task_extra={"missed_objects": [{"object_id": "o1", "iou_drop": 0.4}]})
    assert card.task_extra["missed_objects"][0]["object_id"] == "o1"


# --- RegionRow -----------------------------------------------------------------


def test_region_row_rejects_negative_area() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _region_row(intended_area_px=-1)


def test_region_row_rejects_flip_rate_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        _region_row(flip_rate=1.5)


def test_region_row_rejects_unknown_distribution_key() -> None:
    with pytest.raises(ValueError, match="unknown reliability grade key"):
        _region_row(reliability_distribution={"bogus": 1})


def test_region_row_rejects_negative_distribution_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _region_row(reliability_distribution={"high": -1})


def test_region_row_allows_none_grade_and_empty_distribution() -> None:
    row = _region_row(reliability_grade=None, reliability_distribution={})
    assert row.reliability_grade is None
    assert row.reliability_distribution == {}


def test_region_row_rejects_top_region_share_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        _region_row(top_region_share=1.5)


def test_region_row_rejects_high_rate_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        _region_row(high_rate=-0.1)


def test_region_row_top_region_share_and_high_rate_default_to_none() -> None:
    row = RegionRow(
        region_key="grid::0",
        region_kind="grid",
        intended_area_px=64,
        effective_area_px=60,
        mean_degradation=0.3,
        flip_rate=0.2,
        n_valid=10,
        reliability_grade=None,
        reliability_distribution={},
    )
    assert row.top_region_share is None
    assert row.high_rate is None


# --- SpatialConcentration ---------------------------------------------------------


def test_spatial_concentration_accepts_valid_fields() -> None:
    concentration = _spatial_concentration()
    assert concentration.dominant_region_share == 0.5


def test_spatial_concentration_allows_all_none_with_zero_samples() -> None:
    concentration = _spatial_concentration(
        dominant_region_key=None,
        dominant_region_share=None,
        spatial_entropy=None,
        n_scored_samples=0,
    )
    assert concentration.dominant_region_key is None
    assert concentration.spatial_entropy is None


def test_spatial_concentration_rejects_key_without_share() -> None:
    with pytest.raises(ValueError, match="both present or both None"):
        _spatial_concentration(dominant_region_share=None)


def test_spatial_concentration_rejects_share_without_key() -> None:
    with pytest.raises(ValueError, match="both present or both None"):
        _spatial_concentration(dominant_region_key=None)


def test_spatial_concentration_rejects_share_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        _spatial_concentration(dominant_region_share=1.5)


def test_spatial_concentration_rejects_entropy_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        _spatial_concentration(spatial_entropy=-0.1)


def test_spatial_concentration_rejects_negative_n_scored_samples() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _spatial_concentration(n_scored_samples=-1)


# --- FlaggedItem -----------------------------------------------------------------


def test_flagged_item_rejects_empty_anchor_key_repr() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _flagged_item(anchor_key_repr="")


def test_flagged_item_rejects_empty_reason_summary() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _flagged_item(reason_summary="")


# --- RunSummary ------------------------------------------------------------------


def test_run_summary_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _run_summary(duration_seconds=-1.0)


def test_run_summary_allows_none_duration_and_failure_rate() -> None:
    summary = _run_summary(duration_seconds=None, failure_rate=None)
    assert summary.duration_seconds is None
    assert summary.failure_rate is None


def test_run_summary_rejects_empty_dataset_name() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _run_summary(dataset_name="")


# --- ReportModel -----------------------------------------------------------------


def test_report_model_rejects_wrong_scorecard_element_type() -> None:
    with pytest.raises(TypeError, match="MetricCard"):
        _report_model(scorecard=("not-a-card",))


def test_report_model_rejects_wrong_meta_type() -> None:
    with pytest.raises(TypeError, match="ReportMeta"):
        _report_model(meta="not-a-meta")


def test_report_model_rejects_wrong_spatial_concentration_type() -> None:
    with pytest.raises(TypeError, match="SpatialConcentration"):
        _report_model(spatial_concentration="not-a-spatial-concentration")
