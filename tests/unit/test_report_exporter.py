"""Unit tests for ssat.report.exporter's R1 JSON/CSV export (design §R1).

Builds small, hand-constructed ``ReportModel``/``SampleCard`` fixtures
directly — no dump/metrics/analysis pipeline is needed since :func:`export`
only ever reads what :class:`AssembledReportLike` already carries in memory
(IMPLE_PLAN_REPORTING_v1.md §5 단계 3's work is pure serialization, not a new
join/aggregation policy).
"""

from __future__ import annotations

import ast
import csv
import dataclasses
import json
from pathlib import Path

import pytest

from ssat.report.assembler import AssembledReport
from ssat.report.exporter import ExportedPaths, export
from ssat.report.types import (
    ClassSemanticRow,
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
    SemanticConcentration,
    SemanticGroupRow,
    SpatialConcentration,
    TaskKind,
    TopRegionEntry,
    VulnerabilityDistribution,
    VulnerabilitySummaryStats,
)

# --- builders ----------------------------------------------------------------


def _sample_card(**overrides: object) -> SampleCard:
    defaults: dict[str, object] = {
        "sample_id": "s0",
        "gt_label": 0,
        "clean_correct": True,
        "vulnerability_score": 0.8,
        "reliability_grade": ReportGrade.HIGH,
        "heatmap_asset_ref": None,
        "thumbnail_asset_ref": None,
        "top_regions": (
            TopRegionEntry(region_key="grid::0", degradation=0.4, reliability_grade=ReportGrade.HIGH),
        ),
        "task_extra": {},
    }
    defaults.update(overrides)
    return SampleCard(**defaults)  # type: ignore[arg-type]


def _region_row(**overrides: object) -> RegionRow:
    defaults: dict[str, object] = {
        "region_key": "grid::0",
        "region_id": "grid",
        "region_kind": "grid",
        "intended_area_px": 64,
        "effective_area_px": 60,
        "mean_degradation": 0.3,
        "flip_rate": 0.2,
        "n_valid": 10,
        "reliability_grade": ReportGrade.UNRELIABLE,
        "reliability_distribution": {"high": 1, "unreliable": 1},
        "top_region_share": 0.25,
        "high_rate": 0.5,
    }
    defaults.update(overrides)
    return RegionRow(**defaults)  # type: ignore[arg-type]


def _semantic_group_row(**overrides: object) -> SemanticGroupRow:
    defaults: dict[str, object] = {
        "semantic_group": "upper_limb",
        "region_ids": ("left_arm", "right_arm"),
        "n_samples": 4,
        "mean_degradation": 0.35,
        "high_rate": 0.5,
        "flip_rate": 0.25,
    }
    defaults.update(overrides)
    return SemanticGroupRow(**defaults)  # type: ignore[arg-type]


def _class_semantic_row(**overrides: object) -> ClassSemanticRow:
    defaults: dict[str, object] = {
        "gt_label": 0,
        "semantic_group": "upper_limb",
        "n_samples": 2,
        "mean_degradation": 0.1,
        "flip_rate": None,
    }
    defaults.update(overrides)
    return ClassSemanticRow(**defaults)  # type: ignore[arg-type]


def _flagged_item(**overrides: object) -> FlaggedItem:
    defaults: dict[str, object] = {
        "anchor_key_repr": "s0::grid::0::False",
        "reason_summary": "sign flips across fill strategies",
        "reliability_reasons": ("blur:+0.1", "mean_fill:-0.2"),
    }
    defaults.update(overrides)
    return FlaggedItem(**defaults)  # type: ignore[arg-type]


def _report_model(
    *,
    full_sample_ids: tuple[str, ...] = ("s0", "s1", "s2"),
    semantic_summary: tuple[SemanticGroupRow, ...] = (),
    class_semantic_matrix: tuple[ClassSemanticRow, ...] = (),
) -> ReportModel:
    return ReportModel(
        meta=ReportMeta(
            run_id="shortcut_A",
            generated_at="2026-08-14T00:00:00+00:00",
            tool_version="1.0.0",
            schema_versions=ReportSchemaVersions(
                dump="1.0.0", metrics="1.0.0", analysis="1.0.0", report="1.0.0"
            ),
            task_kind=TaskKind.CLASSIFICATION,
        ),
        run_summary=RunSummary(
            dataset_name="shortcut_A",
            n_samples=len(full_sample_ids),
            n_regions_per_sample=4,
            n_conditions=5,
            duration_seconds=120.5,
            failure_rate=0.01,
            model_id="resnet18",
            preprocessing_desc="224x224 center crop",
        ),
        scorecard=(
            MetricCard(key="accuracy", label="Clean Accuracy", value=0.9, unit="%", higher_is_better=True),
        ),
        vulnerability_distribution=VulnerabilityDistribution(
            histogram_asset_ref=None,
            summary_stats=VulnerabilitySummaryStats(mean=0.3, median=0.25, p90=0.6, p99=0.9),
        ),
        sample_rankings=SampleRankings(
            most_vulnerable=(_sample_card(sample_id=full_sample_ids[0]),), most_robust=()
        ),
        region_summary=RegionSummary(
            rows=(_region_row(),),
            reliability_distribution={"high": 1, "unreliable": 1},
            chart_asset_ref=None,
        ),
        spatial_concentration=SpatialConcentration(
            dominant_region_key="grid::0",
            dominant_region_share=0.25,
            spatial_entropy=0.9,
            n_scored_samples=len(full_sample_ids),
        ),
        # Defaults to the §1 격차#6 graceful-degradation marker (empty
        # tuples); tests exercising semantic_summary.csv/
        # class_semantic_matrix.csv pass populated tuples via the
        # semantic_summary/class_semantic_matrix parameters instead.
        semantic_summary=semantic_summary,
        class_semantic_matrix=class_semantic_matrix,
        semantic_concentration=SemanticConcentration(
            dominant_semantic_group=None,
            dominant_semantic_group_share=None,
            semantic_group_entropy=None,
            n_semantic_groups=0,
            n_scored_samples=0,
        ),
        fill_strategy_correlation_asset_ref=None,
        reliability_spotlight=ReliabilitySpotlight(flagged_examples=(_flagged_item(),)),
        provenance=ProvenanceInfo(
            dump_path="/data/dump",
            metrics_dir="/data/dump/metrics",
            analysis_dir="/data/dump/analysis",
            run_manifest_hash="c" * 64,
            metrics_manifest_hash="a" * 64,
            analysis_manifest_hash="b" * 64,
            thresholds={"z_vs_control_threshold": 2.0},
        ),
    )


def _assembled_report(
    *,
    full_sample_ids: tuple[str, ...] = ("s0", "s1", "s2"),
    semantic_summary: tuple[SemanticGroupRow, ...] = (),
    class_semantic_matrix: tuple[ClassSemanticRow, ...] = (),
    sample_semantic_degradation: dict[tuple[str, str], float] | None = None,
) -> AssembledReport:
    model = _report_model(
        full_sample_ids=full_sample_ids,
        semantic_summary=semantic_summary,
        class_semantic_matrix=class_semantic_matrix,
    )
    # full_sample_rankings intentionally outnumbers model.sample_rankings — the
    # whole reason it exists (§1 격차#5) — so tests can tell "full population"
    # apart from "top-K/bottom-K only" if the exporter accidentally conflates them.
    full_rankings = tuple(
        _sample_card(
            sample_id=sample_id,
            reliability_grade=ReportGrade.HIGH if index == 0 else None,
            top_regions=() if index != 0 else _sample_card().top_regions,
        )
        for index, sample_id in enumerate(full_sample_ids)
    )
    return AssembledReport(
        model=model,
        full_sample_rankings=full_rankings,
        sample_semantic_degradation=sample_semantic_degradation or {},
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# --- report_model.json ---------------------------------------------------------


def test_export_creates_report_model_json_that_round_trips(tmp_path: Path) -> None:
    assembled = _assembled_report()

    paths = export(assembled, tmp_path)

    payload = json.loads(paths.report_model_json.read_text(encoding="utf-8"))
    rebuilt = ReportModel.from_dict(payload)
    assert rebuilt == assembled.model


def test_export_report_model_json_is_sorted_and_newline_terminated(tmp_path: Path) -> None:
    paths = export(_assembled_report(), tmp_path)

    raw = paths.report_model_json.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    payload = json.loads(raw)
    # write_json_atomic sorts keys; a top-level dict's own key order in the
    # source text should already be alphabetical.
    assert list(payload.keys()) == sorted(payload.keys())


# --- sample_rankings.csv --------------------------------------------------------


def test_sample_rankings_csv_covers_full_population_not_top_k(tmp_path: Path) -> None:
    assembled = _assembled_report(full_sample_ids=("s0", "s1", "s2", "s3", "s4"))

    paths = export(assembled, tmp_path)

    rows = _read_csv_rows(paths.sample_rankings_csv)
    assert len(rows) == 5 == len(assembled.full_sample_rankings)
    assert len(rows) > len(assembled.model.sample_rankings.most_vulnerable)
    assert [row["sample_id"] for row in rows] == ["s0", "s1", "s2", "s3", "s4"]


def test_sample_rankings_csv_encodes_none_as_empty_string(tmp_path: Path) -> None:
    assembled = _assembled_report(full_sample_ids=("s0", "s1"))

    paths = export(assembled, tmp_path)

    rows = _read_csv_rows(paths.sample_rankings_csv)
    unscored_row = rows[1]
    assert unscored_row["reliability_grade"] == ""
    assert unscored_row["heatmap_asset_ref"] == ""


def test_sample_rankings_csv_nested_fields_survive_as_json_columns(tmp_path: Path) -> None:
    assembled = _assembled_report(full_sample_ids=("s0", "s1"))

    paths = export(assembled, tmp_path)

    rows = _read_csv_rows(paths.sample_rankings_csv)
    top_regions = json.loads(rows[0]["top_regions_json"])
    assert top_regions == [{"region_key": "grid::0", "degradation": 0.4, "reliability_grade": "high"}]
    assert json.loads(rows[1]["top_regions_json"]) == []
    assert json.loads(rows[0]["task_extra_json"]) == {}


def test_sample_rankings_csv_semantic_degradation_json_groups_by_sample(tmp_path: Path) -> None:
    # IMPLE_PLAN_SEMANTIC_VULNERABILITY_v1.md §3.5: this column is
    # report.labels's disk source for AssembledReport.sample_semantic_
    # degradation, so it must group strictly by sample_id -- s0's pairs must
    # never leak into s1's cell.
    assembled = _assembled_report(
        full_sample_ids=("s0", "s1"),
        sample_semantic_degradation={
            ("s0", "upper_limb"): 0.4,
            ("s0", "lower_limb"): 0.1,
            ("s1", "upper_limb"): 0.9,
        },
    )

    paths = export(assembled, tmp_path)

    rows = _read_csv_rows(paths.sample_rankings_csv)
    assert json.loads(rows[0]["semantic_degradation_json"]) == {
        "upper_limb": 0.4,
        "lower_limb": 0.1,
    }
    assert json.loads(rows[1]["semantic_degradation_json"]) == {"upper_limb": 0.9}


def test_sample_rankings_csv_semantic_degradation_json_empty_when_no_pairs(tmp_path: Path) -> None:
    assembled = _assembled_report(full_sample_ids=("s0",))

    paths = export(assembled, tmp_path)

    rows = _read_csv_rows(paths.sample_rankings_csv)
    assert json.loads(rows[0]["semantic_degradation_json"]) == {}


# --- region_summary.csv ---------------------------------------------------------


def test_region_summary_csv_flattens_reliability_distribution_into_count_columns(
    tmp_path: Path,
) -> None:
    paths = export(_assembled_report(), tmp_path)

    rows = _read_csv_rows(paths.region_summary_csv)
    assert len(rows) == 1
    row = rows[0]
    assert row["region_key"] == "grid::0"
    assert row["reliability_grade"] == "unreliable"
    assert row["high_count"] == "1"
    assert row["unreliable_count"] == "1"
    assert row["moderate_count"] == "0"
    assert row["low_count"] == "0"


def test_region_summary_csv_includes_top_region_share_and_high_rate(tmp_path: Path) -> None:
    paths = export(_assembled_report(), tmp_path)

    row = _read_csv_rows(paths.region_summary_csv)[0]
    assert row["top_region_share"] == "0.25"
    assert row["high_rate"] == "0.5"


def test_region_summary_csv_encodes_none_top_region_share_and_high_rate_as_empty(
    tmp_path: Path,
) -> None:
    model = _report_model()
    model = dataclasses.replace(
        model,
        region_summary=RegionSummary(
            rows=(_region_row(top_region_share=None, high_rate=None),),
            reliability_distribution={"high": 1, "unreliable": 1},
            chart_asset_ref=None,
        ),
    )
    assembled = AssembledReport(model=model, full_sample_rankings=(_sample_card(),))

    paths = export(assembled, tmp_path)

    row = _read_csv_rows(paths.region_summary_csv)[0]
    assert row["top_region_share"] == ""
    assert row["high_rate"] == ""


def test_region_summary_csv_empty_distribution_yields_zero_counts(tmp_path: Path) -> None:
    model = _report_model()
    model = dataclasses.replace(
        model,
        region_summary=RegionSummary(
            rows=(_region_row(reliability_grade=None, reliability_distribution={}),),
            reliability_distribution={},
            chart_asset_ref=None,
        ),
    )
    assembled = AssembledReport(model=model, full_sample_rankings=(_sample_card(),))

    paths = export(assembled, tmp_path)

    row = _read_csv_rows(paths.region_summary_csv)[0]
    assert row["reliability_grade"] == ""
    assert row["high_count"] == row["moderate_count"] == row["low_count"] == row["unreliable_count"] == "0"


# --- semantic_summary.csv / class_semantic_matrix.csv (§3.4) --------------------


def test_semantic_summary_csv_flattens_region_ids_with_semicolons(tmp_path: Path) -> None:
    assembled = _assembled_report(
        semantic_summary=(
            _semantic_group_row(semantic_group="upper_limb", region_ids=("left_arm", "right_arm")),
        )
    )

    paths = export(assembled, tmp_path)

    row = _read_csv_rows(paths.semantic_summary_csv)[0]
    assert row["semantic_group"] == "upper_limb"
    assert row["region_ids"] == "left_arm;right_arm"
    assert row["n_samples"] == "4"
    assert row["mean_degradation"] == "0.35"
    assert row["high_rate"] == "0.5"
    assert row["flip_rate"] == "0.25"


def test_semantic_summary_csv_encodes_none_flip_rate_and_high_rate_as_empty(tmp_path: Path) -> None:
    assembled = _assembled_report(
        semantic_summary=(_semantic_group_row(high_rate=None, flip_rate=None),)
    )

    paths = export(assembled, tmp_path)

    row = _read_csv_rows(paths.semantic_summary_csv)[0]
    assert row["high_rate"] == ""
    assert row["flip_rate"] == ""


def test_semantic_summary_csv_empty_when_no_semantic_groups(tmp_path: Path) -> None:
    paths = export(_assembled_report(), tmp_path)

    assert _read_csv_rows(paths.semantic_summary_csv) == []


def test_class_semantic_matrix_csv_reproduces_foot_action_class_pattern(tmp_path: Path) -> None:
    assembled = _assembled_report(
        class_semantic_matrix=(
            _class_semantic_row(gt_label=0, semantic_group="lower_limb", mean_degradation=0.85),
            _class_semantic_row(gt_label=0, semantic_group="upper_limb", mean_degradation=0.1),
            _class_semantic_row(gt_label=1, semantic_group="lower_limb", mean_degradation=0.15),
            _class_semantic_row(gt_label=1, semantic_group="upper_limb", mean_degradation=0.65),
        )
    )

    paths = export(assembled, tmp_path)

    rows = _read_csv_rows(paths.class_semantic_matrix_csv)
    assert len(rows) == 4
    by_key = {(row["gt_label"], row["semantic_group"]): row for row in rows}
    assert by_key[("0", "lower_limb")]["mean_degradation"] == "0.85"
    assert by_key[("1", "upper_limb")]["mean_degradation"] == "0.65"
    # flip_rate is always None for this plan (IMPLE_PLAN_SEMANTIC_VULNERABILITY_v1.md
    # §1 격차#3(b) — no (gt_label × semantic_group)-grain flip signal in N3).
    assert all(row["flip_rate"] == "" for row in rows)


def test_class_semantic_matrix_csv_empty_when_no_semantic_groups(tmp_path: Path) -> None:
    paths = export(_assembled_report(), tmp_path)

    assert _read_csv_rows(paths.class_semantic_matrix_csv) == []


# --- flagged_items.csv ----------------------------------------------------------


def test_flagged_items_csv_matches_reliability_spotlight_exactly(tmp_path: Path) -> None:
    assembled = _assembled_report()

    paths = export(assembled, tmp_path)

    rows = _read_csv_rows(paths.flagged_items_csv)
    assert len(rows) == len(assembled.model.reliability_spotlight.flagged_examples)
    row = rows[0]
    expected = assembled.model.reliability_spotlight.flagged_examples[0]
    assert row["anchor_key_repr"] == expected.anchor_key_repr
    assert row["reason_summary"] == expected.reason_summary
    assert json.loads(row["reliability_reasons_json"]) == list(expected.reliability_reasons)


def test_flagged_items_csv_empty_when_spotlight_empty(tmp_path: Path) -> None:
    model = _report_model()
    model = dataclasses.replace(
        model, reliability_spotlight=ReliabilitySpotlight(flagged_examples=())
    )
    assembled = AssembledReport(model=model, full_sample_rankings=(_sample_card(),))

    paths = export(assembled, tmp_path)

    assert _read_csv_rows(paths.flagged_items_csv) == []


# --- determinism / output directory ---------------------------------------------


def test_export_is_byte_identical_across_repeated_calls(tmp_path: Path) -> None:
    assembled = _assembled_report(
        semantic_summary=(_semantic_group_row(),),
        class_semantic_matrix=(_class_semantic_row(),),
    )

    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first = export(assembled, first_dir)
    second = export(assembled, second_dir)

    for name in (
        "report_model_json",
        "sample_rankings_csv",
        "region_summary_csv",
        "semantic_summary_csv",
        "class_semantic_matrix_csv",
        "flagged_items_csv",
    ):
        first_bytes = getattr(first, name).read_bytes()
        second_bytes = getattr(second, name).read_bytes()
        assert first_bytes == second_bytes, name


def test_export_creates_output_dir_if_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "report" / "data"
    assert not output_dir.exists()

    export(_assembled_report(), output_dir)

    assert output_dir.is_dir()


def test_export_returns_paths_inside_output_dir(tmp_path: Path) -> None:
    paths = export(_assembled_report(), tmp_path)

    assert isinstance(paths, ExportedPaths)
    for path in (
        paths.report_model_json,
        paths.sample_rankings_csv,
        paths.region_summary_csv,
        paths.semantic_summary_csv,
        paths.class_semantic_matrix_csv,
        paths.flagged_items_csv,
    ):
        assert path.parent == tmp_path
        assert path.is_file()


# --- structural typing (§3.3 "report.exporter → report.types, ssat.utils" only) -----


class _DuckTypedAssembledReport:
    """Satisfies AssembledReportLike structurally without inheriting from AssembledReport."""

    def __init__(self, model: ReportModel, full_sample_rankings: tuple[SampleCard, ...]) -> None:
        self.model = model
        self.full_sample_rankings = full_sample_rankings
        self.sample_semantic_degradation: dict[tuple[str, str], float] = {}


def test_export_accepts_any_structurally_matching_object(tmp_path: Path) -> None:
    model = _report_model()
    duck = _DuckTypedAssembledReport(model=model, full_sample_rankings=(_sample_card(),))

    paths = export(duck, tmp_path)  # type: ignore[arg-type]

    assert paths.report_model_json.exists()


def test_report_exporter_module_has_no_assembler_metrics_or_analysis_imports() -> None:
    """Statically enforce §3.3: report.exporter → report.types, ssat.utils (no report.assembler)."""

    source_path = Path(__file__).resolve().parents[2] / "ssat" / "report" / "exporter.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_prefixes = (
        "ssat.report.assembler",
        "ssat.report.adapters",
        "ssat.analysis",
        "ssat.metrics",
        "ssat.core",
        "ssat.application",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(forbidden_prefixes), alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith(forbidden_prefixes), module
