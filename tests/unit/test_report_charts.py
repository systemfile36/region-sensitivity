"""Unit tests for ssat.report.charts's SVG chart rendering.

No dump/metrics/analysis pipeline needed — every render function is a pure
transform from already-typed ``report.types`` values (or, for the
correlation heatmap, anything structurally matching
:class:`ssat.report.charts.RankCorrelationRowLike`) to an SVG string.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from ssat.report.charts import (
    render_fill_strategy_correlation,
    render_region_bar,
    render_vulnerability_histogram,
)
from ssat.report.types import ReportGrade, SampleCard

# --- builders ----------------------------------------------------------------


def _sample_card(**overrides: object) -> SampleCard:
    defaults: dict[str, object] = {
        "sample_id": "s0",
        "gt_label": 0,
        "clean_correct": True,
        "vulnerability_score": 0.5,
        "reliability_grade": None,
        "heatmap_asset_ref": None,
        "thumbnail_asset_ref": None,
        "top_regions": (),
        "task_extra": {},
    }
    defaults.update(overrides)
    return SampleCard(**defaults)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class _RegionRowStub:
    region_key: str
    mean_degradation: float | None
    reliability_grade: ReportGrade | None


@dataclass(frozen=True, slots=True)
class _RankCorrelationRowStub:
    op_a: str
    op_b: str
    spearman: float | None


_EXTERNAL_REFERENCE_PATTERN = re.compile(r'(?:href|src)\s*=\s*"https?://', re.IGNORECASE)


def _no_external_references(svg: str) -> bool:
    return not _EXTERNAL_REFERENCE_PATTERN.search(svg) and "<script" not in svg


# --- render_vulnerability_histogram ---------------------------------------------


def test_histogram_is_reproducible_across_calls() -> None:
    cards = tuple(_sample_card(sample_id=f"s{i}", vulnerability_score=i / 10) for i in range(20))

    first = render_vulnerability_histogram(cards)
    second = render_vulnerability_histogram(cards)

    assert first == second


def test_histogram_has_no_external_references() -> None:
    cards = tuple(_sample_card(sample_id=f"s{i}", vulnerability_score=i / 10) for i in range(5))

    svg = render_vulnerability_histogram(cards)

    assert _no_external_references(svg)
    assert svg.startswith("<?xml")


def test_histogram_ignores_unscored_samples() -> None:
    cards = (
        _sample_card(sample_id="s0", vulnerability_score=0.5),
        _sample_card(sample_id="s1", vulnerability_score=None),
    )

    svg = render_vulnerability_histogram(cards)

    assert isinstance(svg, str) and svg


def test_histogram_handles_empty_input_without_raising() -> None:
    svg = render_vulnerability_histogram(())

    assert isinstance(svg, str) and svg


# --- render_region_bar --------------------------------------------------------


def test_region_bar_is_reproducible_across_calls() -> None:
    rows = (
        _RegionRowStub(region_key="grid::0", mean_degradation=0.3, reliability_grade=ReportGrade.HIGH),
        _RegionRowStub(region_key="grid::1", mean_degradation=0.6, reliability_grade=ReportGrade.UNRELIABLE),
    )

    first = render_region_bar(rows)
    second = render_region_bar(rows)

    assert first == second


def test_region_bar_has_no_external_references() -> None:
    rows = (
        _RegionRowStub(region_key="grid::0", mean_degradation=0.3, reliability_grade=ReportGrade.HIGH),
    )

    svg = render_region_bar(rows)

    assert _no_external_references(svg)


def test_region_bar_excludes_rows_with_no_mean_degradation() -> None:
    rows = (
        _RegionRowStub(
            region_key="grid::included", mean_degradation=0.3, reliability_grade=ReportGrade.HIGH
        ),
        _RegionRowStub(region_key="grid::excluded", mean_degradation=None, reliability_grade=None),
    )

    svg = render_region_bar(rows)

    assert "included" in svg
    assert "excluded" not in svg


def test_region_bar_xtick_labels_drop_redundant_region_id_prefix() -> None:
    """matplotlib embeds each tick label as a literal XML comment ahead of its

    vector glyph path (even under ``svg.fonttype="path"``), so the shortened
    label is directly searchable in the SVG text.
    """

    rows = (
        _RegionRowStub(
            region_key="grid_4x4::grid_4x4/r0/c0", mean_degradation=0.3, reliability_grade=ReportGrade.HIGH
        ),
    )

    svg = render_region_bar(rows)

    assert "grid_4x4::grid_4x4/r0/c0" not in svg
    assert "grid_4x4/r0/c0" in svg


def test_short_region_key_strips_leading_region_id_prefix() -> None:
    from ssat.report.charts import _short_region_key

    assert _short_region_key("grid_4x4::grid_4x4/r0/c0") == "grid_4x4/r0/c0"
    assert _short_region_key("grid::0") == "0"
    assert _short_region_key("left_arm") == "left_arm"  # no "::" -- unchanged


def test_region_bar_handles_empty_input_without_raising() -> None:
    svg = render_region_bar(())

    assert isinstance(svg, str) and svg


def test_region_bar_none_grade_uses_distinct_color_from_low() -> None:
    from ssat.report.charts import _grade_color
    from ssat.report.types import GRADE_COLORS

    assert _grade_color(None) != GRADE_COLORS[ReportGrade.LOW]
    assert _grade_color(ReportGrade.HIGH) == GRADE_COLORS[ReportGrade.HIGH]


# --- render_fill_strategy_correlation --------------------------------------------


def test_fill_strategy_correlation_returns_none_for_empty_input() -> None:
    assert render_fill_strategy_correlation(()) is None


def test_fill_strategy_correlation_reproducible_and_no_external_references() -> None:
    rows = (
        _RankCorrelationRowStub(op_a="blur", op_b="mean_fill", spearman=0.8),
        _RankCorrelationRowStub(op_a="blur", op_b="constant_fill", spearman=-0.2),
    )

    first = render_fill_strategy_correlation(rows)
    second = render_fill_strategy_correlation(rows)

    assert first is not None
    assert first == second
    assert _no_external_references(first)


def test_fill_strategy_correlation_handles_none_spearman_without_raising() -> None:
    rows = (_RankCorrelationRowStub(op_a="blur", op_b="mean_fill", spearman=None),)

    svg = render_fill_strategy_correlation(rows)

    assert isinstance(svg, str) and svg


# --- figure hygiene (no leaked matplotlib figures across calls) -----------------


def test_no_pyplot_figures_leak_across_repeated_calls() -> None:
    assert plt.get_fignums() == []

    cards = tuple(_sample_card(sample_id=f"s{i}", vulnerability_score=i / 100) for i in range(100))
    rows = tuple(
        _RegionRowStub(region_key=f"grid::{i}", mean_degradation=i / 100, reliability_grade=ReportGrade.HIGH)
        for i in range(20)
    )
    for _ in range(5):
        render_vulnerability_histogram(cards)
        render_region_bar(rows)

    assert plt.get_fignums() == []


def test_rc_context_does_not_leak_global_rcparams() -> None:
    original_fonttype = matplotlib.rcParams["svg.fonttype"]

    render_vulnerability_histogram((_sample_card(vulnerability_score=0.5),))

    assert matplotlib.rcParams["svg.fonttype"] == original_fonttype


# --- dependency direction -------------------------------------------------------


def test_report_charts_module_has_no_analysis_metrics_core_or_assembler_imports() -> None:
    """Statically enforce report.charts → report.types (matplotlib) as the only allowed dependency."""

    source_path = Path(__file__).resolve().parents[2] / "ssat" / "report" / "charts.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_prefixes = (
        "ssat.report.assembler",
        "ssat.report.adapters",
        "ssat.report.exporter",
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
