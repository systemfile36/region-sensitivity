"""Unit tests for ssat.report.html_renderer's R4 HTML rendering (design §R4, §6.2 C1, §6.3 C2).

Builds small, hand-constructed ``ReportModel`` fixtures directly — no dump/
metrics/analysis pipeline is needed since :func:`render_report` only ever
templates whatever a already-assembled ``ReportModel`` already carries
(mirrors ``test_report_exporter.py``/``test_report_charts.py``'s precedent:
R1/R2/R4 are all pure transformers of already-typed data).
"""

from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path

import pytest

from ssat.report.html_renderer import ReportManifestPaths, render_report
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
                key="flip_rate",
                label="Flip Rate",
                value=None,
                unit="%",
                higher_is_better=False,
                note="해당 없음: continuous 지표라 flip 개념이 없습니다.",
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
    """A ReportModel matching what R0 assembles when analysis_dir=None (design §6.2 C1)."""

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
        sample_rankings=SampleRankings(
            most_vulnerable=(_sample_card(reliability_grade=None, top_regions=()),), most_robust=()
        ),
        region_summary=RegionSummary(
            rows=(_region_row(reliability_grade=None, reliability_distribution={}),),
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
        "report-guide",
        "scorecard",
        "vulnerability-distribution",
        "gallery",
        "region-summary",
        "reliability-spotlight",
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

    assert "해당 없음" in html  # note on the flip_rate scorecard entry
    assert "플래그된 항목 없음" in html  # empty reliability spotlight, explicitly labeled
    # a sample/region with reliability_grade=None still gets a "no grade" badge,
    # not an omitted badge:
    assert "해당 없음</span>" in html
    # region_reliability_overview: every RegionRow.reliability_grade is None
    # here, so the report-guide callout must show the explicit "no analysis"
    # message instead of a misleading 0/0 count:
    assert "해당 없음: 분석 미실행" in html


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


def test_render_report_has_report_guide_section_with_legend(tmp_path: Path) -> None:
    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert 'id="report-guide"' in html
    assert 'class="grade-legend"' in html
    for grade_label in ("HIGH", "MODERATE", "LOW", "UNRELIABLE"):
        assert f'class="grade-legend-label">{grade_label}<' in html


def test_render_report_gallery_has_rollup_rule_pointer_note(tmp_path: Path) -> None:
    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert 'href="#report-guide"' in html


def test_region_reliability_callout_names_high_region_and_unreliable_fraction(tmp_path: Path) -> None:
    """Mirrors the real §5 단계8 run shape: 1 HIGH region among 16, 15 UNRELIABLE."""

    high_row = _region_row(
        region_key="grid::grid/r0/c0",
        reliability_grade=ReportGrade.HIGH,
        reliability_distribution={"high": 1},
    )
    unreliable_rows = tuple(
        _region_row(
            region_key=f"grid::grid/r{i}/c{i}",
            reliability_grade=ReportGrade.UNRELIABLE,
            reliability_distribution={"unreliable": 1},
        )
        for i in range(15)
    )
    model = _report_model(
        region_summary=RegionSummary(
            rows=(high_row,) + unreliable_rows,
            reliability_distribution={"high": 1, "unreliable": 15},
            chart_asset_ref="assets/img/charts/region_bar.svg",
        )
    )

    paths = render_report(model, tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert "grid::grid/r0/c0" in html
    assert "93.75%" in html  # 15/16


def test_render_report_empty_gallery_and_region_summary_shows_no_data_markers(tmp_path: Path) -> None:
    paths = render_report(_empty_gallery_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert html.count('<p class="no-data">데이터 없음</p>') >= 2  # both galleries
    assert '<td colspan="8" class="no-data">데이터 없음</td>' in html
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


def test_render_report_region_row_badge_title_carries_distribution(tmp_path: Path) -> None:
    paths = render_report(_report_model(), tmp_path, top_k=20, bottom_k=20)
    html = paths.report_html.read_text(encoding="utf-8")

    assert 'title="high:1, unreliable:1"' in html


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

    moved_dir = tmp_path / "elsewhere" / "moved_report"
    moved_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output_dir, moved_dir)

    html = (moved_dir / "report.html").read_text(encoding="utf-8")
    assert (moved_dir / "assets" / "css" / "style.css").is_file()
    assert (moved_dir / "assets" / "js" / "enhance.js").is_file()
    assert 'href="assets/css/style.css"' in html
    assert 'src="assets/js/enhance.js"' in html


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
