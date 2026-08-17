"""Unit tests for ssat.report.html_renderer's R4 HTML rendering (design §R4, §6.2 C1, §6.3 C2).

Builds small, hand-constructed ``ReportModel`` fixtures directly — no dump/
metrics/analysis pipeline is needed since :func:`render_report`/
:func:`render_secondary_report` only ever template whatever an already-
assembled ``ReportModel`` already carries (mirrors ``test_report_exporter.
py``/``test_report_charts.py``'s precedent: R1/R2/R4 are all pure
transformers of already-typed data).

Covers both the main report (report layout redesign's "A + C" combination,
docs/report_layout_improve/AGENTS_OPINION_1.md) and the auxiliary
"Question Driven" report (:func:`render_secondary_report`).
"""

from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path

import pytest

from ssat.report.html_renderer import (
    ReportManifestPaths,
    _grade_distribution_percentages,
    _grid_layout,
    _heat_color,
    render_report,
    render_secondary_report,
)
from ssat.report.static import ENHANCE_JS, STYLE_CSS
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


def _sample_card(**overrides: object) -> SampleCard:
    defaults: dict[str, object] = {
        "sample_id": "s0",
        "gt_label": 0,
        "clean_correct": True,
        "vulnerability_score": 0.8,
        "reliability_grade": ReportGrade.HIGH,
        "heatmap_asset_ref": "assets/img/heatmaps/sample_s0.png",
        "thumbnail_asset_ref": "assets/img/thumbnails/sample_s0.png",
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


def _flagged_item(**overrides: object) -> FlaggedItem:
    defaults: dict[str, object] = {
        "anchor_key_repr": "s0::grid::0::False",
        "reason_summary": "sign flips across fill strategies",
        "reliability_reasons": ("blur:+0.1", "mean_fill:-0.2"),
    }
    defaults.update(overrides)
    return FlaggedItem(**defaults)  # type: ignore[arg-type]


def _spatial_concentration(**overrides: object) -> SpatialConcentration:
    defaults: dict[str, object] = {
        "dominant_region_key": "grid::0",
        "dominant_region_share": 0.6,
        "spatial_entropy": 0.4,
        "n_scored_samples": 100,
    }
    defaults.update(overrides)
    return SpatialConcentration(**defaults)  # type: ignore[arg-type]


def _report_model(**overrides: object) -> ReportModel:
    defaults: dict[str, object] = {
        "meta": ReportMeta(
            run_id="shortcut_A_all_ops",
            generated_at="2026-08-14T00:00:00+00:00",
            tool_version="1.0.0",
            schema_versions=ReportSchemaVersions(
                dump="1.0.0", metrics="1.0.0", analysis="1.0.0", report="1.0.0"
            ),
            task_kind=TaskKind.CLASSIFICATION,
        ),
        "run_summary": RunSummary(
            dataset_name="shortcut_A",
            n_samples=100,
            n_regions_per_sample=4,
            n_conditions=5,
            duration_seconds=125.4,
            failure_rate=0.01,
            model_id="resnet18",
            preprocessing_desc="224x224 center crop",
        ),
        "scorecard": (
            MetricCard(key="accuracy", label="Clean Accuracy", value=0.9, unit="%", higher_is_better=True),
            MetricCard(
                key="mean_gt_logit_drop",
                label="Mean gt_logit_drop Degradation",
                value=0.35,
                unit="",
                higher_is_better=False,
            ),
            MetricCard(
                key="flip_rate",
                label="Flip Rate",
                value=None,
                unit="%",
                higher_is_better=False,
                note="해당 없음: continuous 지표라 flip 개념이 없습니다.",
            ),
            MetricCard(
                key="control_comparison",
                label="Mean Z vs Control",
                value=3.2,
                unit="",
                higher_is_better=False,
            ),
        ),
        "vulnerability_distribution": VulnerabilityDistribution(
            histogram_asset_ref="assets/img/charts/histogram.svg",
            summary_stats=VulnerabilitySummaryStats(mean=0.3, median=0.25, p90=0.6, p99=0.9),
        ),
        "sample_rankings": SampleRankings(
            most_vulnerable=(_sample_card(),), most_robust=(_sample_card(sample_id="s1"),)
        ),
        "region_summary": RegionSummary(
            rows=(_region_row(),),
            reliability_distribution={"high": 1, "unreliable": 1},
            chart_asset_ref="assets/img/charts/region_bar.svg",
        ),
        "spatial_concentration": _spatial_concentration(),
        "fill_strategy_correlation_asset_ref": "assets/img/charts/fill_strategy_correlation.svg",
        "reliability_spotlight": ReliabilitySpotlight(flagged_examples=(_flagged_item(),)),
        "provenance": ProvenanceInfo(
            dump_path="/data/dump",
            metrics_dir="/data/dump/metrics",
            analysis_dir="/data/dump/analysis",
            run_manifest_hash="c" * 64,
            metrics_manifest_hash="a" * 64,
            analysis_manifest_hash="b" * 64,
            thresholds={"z_vs_control_threshold": 2.0},
        ),
    }
    defaults.update(overrides)
    return ReportModel(**defaults)  # type: ignore[arg-type]


def _no_analysis_model() -> ReportModel:
    """A ReportModel matching what R0 assembles when analysis_dir=None (design §6.2 C1).

    ``spatial_concentration`` stays populated even without an analysis run
    (assembler.py: it is derived from ``spatial_profile`` alone, never
    ``AnalysisStore``) — only the reliability-grade-derived fields below go
    to their explicit "unavailable" markers.
    """

    return _report_model(
        meta=ReportMeta(
            run_id="shortcut_A_all_ops",
            generated_at="2026-08-14T00:00:00+00:00",
            tool_version="1.0.0",
            schema_versions=ReportSchemaVersions(
                dump="1.0.0", metrics="1.0.0", analysis=None, report="1.0.0"
            ),
            task_kind=TaskKind.CLASSIFICATION,
        ),
        scorecard=(
            MetricCard(key="accuracy", label="Clean Accuracy", value=0.9, unit="%", higher_is_better=True),
            MetricCard(
                key="control_comparison",
                label="Mean Z vs Control",
                value=None,
                unit="",
                higher_is_better=False,
                note="분석 미실행: 대조군 비교가 실행되지 않았습니다.",
            ),
        ),
        sample_rankings=SampleRankings(
            most_vulnerable=(_sample_card(reliability_grade=None, top_regions=()),), most_robust=()
        ),
        region_summary=RegionSummary(
            rows=(
                _region_row(
                    reliability_grade=None, reliability_distribution={}, high_rate=None
                ),
            ),
            reliability_distribution={},
            chart_asset_ref=None,
        ),
        fill_strategy_correlation_asset_ref=None,
        reliability_spotlight=ReliabilitySpotlight(flagged_examples=()),
        provenance=ProvenanceInfo(
            dump_path="/data/dump",
            metrics_dir="/data/dump/metrics",
            analysis_dir=None,
            run_manifest_hash="c" * 64,
            metrics_manifest_hash="a" * 64,
            analysis_manifest_hash=None,
            thresholds={},
        ),
    )


def _empty_gallery_model() -> ReportModel:
    """A ReportModel with no scored samples and no regions at all."""

    return _report_model(
        sample_rankings=SampleRankings(most_vulnerable=(), most_robust=()),
        region_summary=RegionSummary(rows=(), reliability_distribution={}, chart_asset_ref=None),
        vulnerability_distribution=VulnerabilityDistribution(
            histogram_asset_ref=None,
            summary_stats=VulnerabilitySummaryStats(mean=None, median=None, p90=None, p99=None),
        ),
        spatial_concentration=_spatial_concentration(
            dominant_region_key=None, dominant_region_share=None, spatial_entropy=None, n_scored_samples=0
        ),
    )


def _grid_region_row(row_index: int, col_index: int, **overrides: object) -> RegionRow:
    return _region_row(
        region_key=f"grid::grid/r{row_index}/c{col_index}",
        region_kind="grid",
        **overrides,
    )


_IMG_TAG_PATTERN = re.compile(r"<img\b[^>]*>")
_ALT_ATTR_PATTERN = re.compile(r'alt="[^"]+"')
_SCRIPT_TAG_PATTERN = re.compile(r"<script\b.*?</script>", re.DOTALL)
_EXTERNAL_REFERENCE_PATTERN = re.compile(r'(?:href|src)\s*=\s*"https?://', re.IGNORECASE)


# --- render_report: writes every file ----------------------------------------


def test_render_report_writes_all_four_files(tmp_path: Path) -> None:
    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)

    assert isinstance(paths, ReportManifestPaths)
    assert paths.report_html.is_file()
    assert paths.style_css.is_file()
    assert paths.enhance_js.is_file()
    assert paths.report_manifest_json.is_file()


def test_render_report_writes_static_assets_verbatim(tmp_path: Path) -> None:
    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)

    assert paths.style_css.read_text(encoding="utf-8") == STYLE_CSS
    assert paths.enhance_js.read_text(encoding="utf-8") == ENHANCE_JS


def test_render_report_creates_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "report"

    paths = render_report(_report_model(), output_dir, top_k=20, bottom_k=20)

    assert paths.report_html.parent == output_dir


# --- report_manifest.json (design §R4) ----------------------------------------


def test_report_manifest_json_has_every_design_field(tmp_path: Path) -> None:
    paths = render_report(_report_model(), tmp_path, top_k=7, bottom_k=3)

    manifest = json.loads(paths.report_manifest_json.read_text(encoding="utf-8"))

    assert manifest["report_schema_version"] == "1.0.0"
    assert manifest["source_manifest_hashes"] == {
        "run": "c" * 64,
        "metrics": "a" * 64,
        "analysis": "b" * 64,
    }
    assert manifest["top_k"] == 7
    assert manifest["bottom_k"] == 3
    assert manifest["generated_at"] == "2026-08-14T00:00:00+00:00"


def test_report_manifest_json_analysis_hash_is_none_without_analysis(tmp_path: Path) -> None:
    paths = render_report(_no_analysis_model(), tmp_path, top_k=20, bottom_k=20)

    manifest = json.loads(paths.report_manifest_json.read_text(encoding="utf-8"))

    assert manifest["source_manifest_hashes"]["analysis"] is None
    assert manifest["source_manifest_hashes"]["run"] == "c" * 64


# --- C1: every section renders without raising, and without silently vanishing --------


def test_render_report_full_model_renders_every_section(tmp_path: Path) -> None:
    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    for section_id in (
        "executive-interpretation",
        "fingerprint",
        "spatial-pattern",
        "region-summary",
        "gallery",
        "stability-controls",
        "flagged-anchors",
        "provenance",
    ):
        assert f'id="{section_id}"' in html
    assert "Clean Accuracy" in html
    assert "grid::0" in html
    assert "s0::grid::0::False" in html
    assert "assets/img/charts/region_bar.svg" in html
    assert "assets/img/charts/fill_strategy_correlation.svg" in html
    assert "assets/img/charts/histogram.svg" in html


def test_render_report_no_analysis_model_shows_explicit_markers_not_silence(tmp_path: Path) -> None:
    """§6.2 C1: missing analysis must render "해당 없음"/badges, never vanish (design §6.2)."""

    paths = render_report(_no_analysis_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert "해당 없음" in html  # note on the control_comparison scorecard entry
    assert "플래그된 항목 없음" in html  # empty reliability spotlight, explicitly labeled
    # a sample with reliability_grade=None still gets a "no grade" badge,
    # not an omitted badge:
    assert "해당 없음</span>" in html
    # a region with an empty reliability_distribution shows the composition
    # bar's own explicit "not evaluated" marker, not a blank cell:
    assert "평가 없음" in html
    # spatial_concentration is unaffected by analysis_dir=None (assembler.py:
    # derived from spatial_profile alone) -- the Executive Interpretation
    # sentence must still use its real dominant_region_key.
    assert "grid::0" in html


def test_scorecard_percent_unit_card_renders_percent_not_raw_fraction(tmp_path: Path) -> None:
    """The fixture's accuracy card (value=0.9, unit="%") must render 90.00%, not 0.9%/90%."""

    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert "90.00%" in html
    assert "0.9%" not in html


def test_scorecard_full_accuracy_renders_100_percent_not_1_percent(tmp_path: Path) -> None:
    """Real observed bug: 200/200 correct (value=1.0) must not render as '1%'."""

    model = _report_model(
        scorecard=(
            MetricCard(key="accuracy", label="Clean Accuracy", value=1.0, unit="%", higher_is_better=True),
        )
    )
    paths = render_report(model, tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert "100.00%" in html
    assert ">1%<" not in html


def test_scorecard_non_percent_unit_card_still_uses_fmt(tmp_path: Path) -> None:
    """unit="" cards (mean_<primary_metric>, control_comparison) must stay unaffected by the % fix."""

    model = _report_model(
        scorecard=(
            MetricCard(
                key="mean_x", label="Mean X Degradation", value=0.123456, unit="", higher_is_better=False
            ),
        )
    )
    paths = render_report(model, tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert "0.1235" in html
    assert "0.1235%" not in html


# --- executive interpretation (report layout redesign) -----------------------


def test_executive_interpretation_concentrated_wording_above_half_share(tmp_path: Path) -> None:
    model = _report_model(
        spatial_concentration=_spatial_concentration(
            dominant_region_key="grid::grid/r0/c0", dominant_region_share=0.82
        )
    )
    paths = render_report(model, tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert "집중되어 있습니다" in html
    assert "grid::grid/r0/c0" in html
    assert "82.00%" in html


def test_executive_interpretation_distributed_wording_below_half_share(tmp_path: Path) -> None:
    model = _report_model(
        spatial_concentration=_spatial_concentration(
            dominant_region_key="grid::grid/r2/c1", dominant_region_share=0.18
        )
    )
    paths = render_report(model, tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert "집중되지 않습니다" in html
    assert "18.00%" in html


def test_executive_interpretation_handles_no_determinable_top_region(tmp_path: Path) -> None:
    model = _report_model(
        spatial_concentration=_spatial_concentration(
            dominant_region_key=None, dominant_region_share=None, spatial_entropy=None, n_scored_samples=0
        )
    )
    paths = render_report(model, tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert "공간 집중도를 판단할 데이터가 없습니다." in html


def test_executive_interpretation_always_includes_non_causal_caveat(tmp_path: Path) -> None:
    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert "자동으로 판정하지" in html


# --- behavioral fingerprint ----------------------------------------------------


def test_fingerprint_section_renders_dominant_share_and_entropy_tiles(tmp_path: Path) -> None:
    model = _report_model(
        spatial_concentration=_spatial_concentration(
            dominant_region_key="grid::grid/r0/c0", dominant_region_share=0.6, spatial_entropy=0.4
        )
    )
    paths = render_report(model, tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert "Dominant-region Share" in html
    assert "Spatial Entropy" in html
    assert "60.00%" in html


# --- dataset spatial pattern (heat grid) --------------------------------------


def test_spatial_pattern_renders_heat_grid_for_grid_shaped_regions(tmp_path: Path) -> None:
    rows = tuple(
        _grid_region_row(r, c, top_region_share=0.1 * (r * 2 + c + 1), high_rate=0.2)
        for r in range(2)
        for c in range(2)
    )
    model = _report_model(
        region_summary=RegionSummary(
            rows=rows, reliability_distribution={"high": 4}, chart_asset_ref=None
        )
    )
    paths = render_report(model, tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert 'class="heat"' in html
    assert "grid::grid/r0/c0" in html
    assert "Top-region share" in html
    assert "HIGH-graded anchor rate" in html


def test_spatial_pattern_falls_back_to_message_for_non_grid_regions(tmp_path: Path) -> None:
    """The default fixture's single region_key ("grid::0") has no /r../c.. suffix."""

    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert "히트그리드를 생략합니다" in html
    assert 'class="heat"' not in html


# --- region summary: composition bars replace the worst-case badge -----------


def test_region_summary_renders_grade_composition_not_single_worst_case_badge(tmp_path: Path) -> None:
    model = _report_model(
        region_summary=RegionSummary(
            rows=(
                _region_row(
                    reliability_grade=ReportGrade.UNRELIABLE,
                    reliability_distribution={"high": 1, "unreliable": 1},
                ),
            ),
            reliability_distribution={"high": 1, "unreliable": 1},
            chart_asset_ref=None,
        )
    )
    paths = render_report(model, tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert "HIGH 50.00%" in html
    assert "UNRELIABLE 50.00%" in html
    # worst-case is preserved, but demoted to a hover title rather than a
    # standalone visible badge (report layout redesign):
    assert 'title="worst-case anchor: UNRELIABLE"' in html


def test_region_summary_legend_and_worst_case_scope_note_present(tmp_path: Path) -> None:
    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert 'class="grade-legend"' in html
    for grade_label in ("HIGH", "MODERATE", "LOW", "UNRELIABLE"):
        assert f'class="grade-legend-label">{grade_label}<' in html
    assert "sample/anchor 단위" in html


def test_render_report_gallery_points_to_region_summary(tmp_path: Path) -> None:
    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert 'href="#region-summary"' in html


def test_render_report_empty_gallery_and_region_summary_shows_no_data_markers(tmp_path: Path) -> None:
    paths = render_report(_empty_gallery_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert html.count('<p class="no-data">데이터 없음</p>') >= 2  # both galleries
    assert '<td colspan="5" class="no-data">데이터 없음</td>' in html
    assert '<p class="no-data">히스토그램 자산 없음</p>' in html


def test_render_report_missing_optional_asset_refs_omit_only_that_element(tmp_path: Path) -> None:
    model = _report_model(
        region_summary=RegionSummary(
            rows=(_region_row(),),
            reliability_distribution={"high": 1, "unreliable": 1},
            chart_asset_ref=None,
        ),
        fill_strategy_correlation_asset_ref=None,
    )

    paths = render_report(model, tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert "assets/img/charts/region_bar.svg" not in html
    assert "assets/img/charts/fill_strategy_correlation.svg" not in html
    assert "grid::0" in html  # the table row itself still renders


# --- stability / controls + detailed tables -----------------------------------


def test_stability_controls_section_reports_flagged_count(tmp_path: Path) -> None:
    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert "1개</strong> anchor" in html


def test_flagged_anchors_section_lists_spotlight_items(tmp_path: Path) -> None:
    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert "s0::grid::0::False" in html
    assert "sign flips across fill strategies" in html


# --- badges/colors -------------------------------------------------------------


def test_render_report_badge_color_matches_grade_colors(tmp_path: Path) -> None:
    from ssat.report.types import GRADE_COLORS

    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert f"background-color: {GRADE_COLORS[ReportGrade.HIGH]};" in html
    assert f"background-color: {GRADE_COLORS[ReportGrade.UNRELIABLE]};" in html


def test_render_report_flagged_item_badge_title_carries_reliability_reasons(tmp_path: Path) -> None:
    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert 'title="blur:+0.1; mean_fill:-0.2"' in html


# --- C2: offline / no-JS ---------------------------------------------------------


def test_provenance_details_is_collapsed_by_default(tmp_path: Path) -> None:
    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert '<details id="provenance">' in html  # no `open` attribute


def test_content_survives_removing_every_script_tag(tmp_path: Path) -> None:
    """§6.3 C2: strip <script>...</script> and confirm core content is still there."""

    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    stripped = _SCRIPT_TAG_PATTERN.sub("", html)

    assert "<script" not in stripped
    assert "Clean Accuracy" in stripped
    assert "grid::0" in stripped
    assert "<img" in stripped
    assert "s0::grid::0::False" in stripped


def test_every_img_tag_has_non_empty_alt_text(tmp_path: Path) -> None:
    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    img_tags = _IMG_TAG_PATTERN.findall(html)
    assert img_tags  # the fixture model has at least one image
    for tag in img_tags:
        assert _ALT_ATTR_PATTERN.search(tag), tag


def test_no_external_http_references_anywhere_in_output(tmp_path: Path) -> None:
    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)

    for path in (paths.report_html, paths.style_css, paths.enhance_js):
        content = path.read_text(encoding="utf-8")
        assert not _EXTERNAL_REFERENCE_PATTERN.search(content), path
        assert "http://" not in content
        assert "https://" not in content


def test_report_folder_survives_being_moved(tmp_path: Path) -> None:
    """§6.3 C2 "폴더 이동 후 상대경로 보존", exercised at the whole-report-folder level."""

    output_dir = tmp_path / "original" / "report"
    render_report(_report_model(), output_dir, top_k=20, bottom_k=20)
    render_secondary_report(_report_model(), output_dir, top_k=20, bottom_k=20)

    moved_dir = tmp_path / "elsewhere" / "moved_report"
    moved_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output_dir, moved_dir)

    html = (moved_dir / "report.html").read_text(encoding="utf-8")
    secondary_html = (moved_dir / "report_question_driven.html").read_text(encoding="utf-8")
    assert (moved_dir / "assets" / "css" / "style.css").is_file()
    assert (moved_dir / "assets" / "js" / "enhance.js").is_file()
    assert 'href="assets/css/style.css"' in html
    assert 'src="assets/js/enhance.js"' in html
    assert 'href="assets/css/style.css"' in secondary_html


# --- render_secondary_report (Layout B, auxiliary "Question Driven" report) --


def test_render_secondary_report_writes_report_question_driven_html(tmp_path: Path) -> None:
    render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)

    path = render_secondary_report(_report_model(), tmp_path, top_k=20, bottom_k=20)

    assert path == tmp_path / "report_question_driven.html"
    assert path.is_file()


def test_render_secondary_report_covers_five_questions(tmp_path: Path) -> None:
    render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    path = render_secondary_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = path.read_text(encoding="utf-8")

    assert "이 모델은 교란 전에는 얼마나 잘 작동하는가?" in html
    assert "영역을 가리면 전반적으로 얼마나 흔들리는가?" in html
    assert '모든 샘플이 반복적으로 의존하는 "고정 위치"가 있는가?' in html
    assert "보이는 효과는 우연한 마스크 효과보다 강하고 반복 가능한가?" in html
    assert "그렇다면 실제로 어떤 샘플을 먼저 조사해야 하는가?" in html
    assert "90.00%" in html  # accuracy card, same underlying data as the main report


def test_render_secondary_report_cross_links_to_main_report(tmp_path: Path) -> None:
    render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    path = render_secondary_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = path.read_text(encoding="utf-8")

    assert 'href="report.html"' in html


def test_main_report_cross_links_to_secondary_report(tmp_path: Path) -> None:
    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert 'href="report_question_driven.html"' in html


def test_render_secondary_report_does_not_claim_no_high_region_from_low_share(tmp_path: Path) -> None:
    """The review's explicit caution: never state "no HIGH region" from a low dominant share alone."""

    model = _report_model(
        spatial_concentration=_spatial_concentration(dominant_region_share=0.1)
    )
    render_report(model, tmp_path, top_k=20, bottom_k=20)
    path = render_secondary_report(model, tmp_path, top_k=20, bottom_k=20)
    html = path.read_text(encoding="utf-8")

    assert "HIGH region이 없다" in html  # the caveat names the misreading it forbids
    assert "결론짓지 않습니다" in html


def test_render_secondary_report_no_script_content_survives(tmp_path: Path) -> None:
    render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    path = render_secondary_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = path.read_text(encoding="utf-8")

    stripped = _SCRIPT_TAG_PATTERN.sub("", html)
    assert "이 모델은 교란 전에는 얼마나 잘 작동하는가?" in stripped


def test_render_secondary_report_has_no_external_references(tmp_path: Path) -> None:
    render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    path = render_secondary_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    content = path.read_text(encoding="utf-8")

    assert not _EXTERNAL_REFERENCE_PATTERN.search(content)
    assert "http://" not in content
    assert "https://" not in content


# --- pure helper functions -----------------------------------------------------


def test_grade_distribution_percentages_orders_best_to_worst_and_skips_zero_counts() -> None:
    result = _grade_distribution_percentages({"high": 1, "moderate": 0, "low": 0, "unreliable": 3})

    assert result == [(ReportGrade.HIGH, 0.25), (ReportGrade.UNRELIABLE, 0.75)]


def test_grade_distribution_percentages_empty_distribution_yields_empty_list() -> None:
    assert _grade_distribution_percentages({}) == []


def test_grid_layout_parses_row_col_from_grid_region_keys() -> None:
    rows = tuple(_grid_region_row(r, c) for r in range(2) for c in range(3))

    layout = _grid_layout(rows)

    assert layout is not None
    assert layout["rows"] == 2
    assert layout["cols"] == 3
    assert layout["cells"][0][0].region_key == "grid::grid/r0/c0"
    assert layout["cells"][1][2].region_key == "grid::grid/r1/c2"


def test_grid_layout_returns_none_for_non_grid_region_kind() -> None:
    rows = (_region_row(region_key="explicit::mask-1", region_kind="explicit"),)

    assert _grid_layout(rows) is None


def test_grid_layout_returns_none_for_unparseable_region_key() -> None:
    rows = (_region_row(region_key="grid::whole", region_kind="grid"),)

    assert _grid_layout(rows) is None


def test_grid_layout_returns_none_for_empty_rows() -> None:
    assert _grid_layout(()) is None


def test_heat_color_returns_neutral_for_none_value_or_max() -> None:
    neutral = _heat_color(None, 1.0)
    assert neutral == _heat_color(0.5, None)
    assert neutral == _heat_color(0.5, 0.0)


def test_heat_color_scales_toward_accent_with_higher_value() -> None:
    low = _heat_color(0.0, 1.0)
    high = _heat_color(1.0, 1.0)

    assert low != high
    assert re.fullmatch(r"#[0-9a-f]{6}", low)
    assert re.fullmatch(r"#[0-9a-f]{6}", high)


# --- dependency direction -------------------------------------------------------


def test_report_html_renderer_module_has_no_forbidden_package_imports() -> None:
    """Statically enforce §3.3: report.html_renderer → report.types, report.static (jinja2)."""

    source_path = Path(__file__).resolve().parents[2] / "ssat" / "report" / "html_renderer.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_prefixes = (
        "ssat.report.assembler",
        "ssat.report.adapters",
        "ssat.report.exporter",
        "ssat.report.assets",
        "ssat.report.charts",
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


def test_report_static_module_has_zero_dependencies() -> None:
    """Statically enforce §3.3: report.static → (없음)."""

    source_path = Path(__file__).resolve().parents[2] / "ssat" / "report" / "static.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue  # `from __future__ import annotations` carries no runtime dependency
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            pytest.fail(f"report.static must have zero imports, found: {ast.dump(node)}")
