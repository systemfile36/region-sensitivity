"""R4 HTMLRenderer: assemble ``ReportModel`` + already-rendered assets into report.html (design §R4).

**Scope boundary (design §3.3 "report.html_renderer → report.types,
report.static (jinja2)").** This module never imports ``report.charts``,
``report.assets``, or ``report.assembler`` — it consumes a ``ReportModel``
that some earlier orchestration step (Stage 7's ``AuditApplication.
generate_report()``) has already fully populated: every ``SampleCard``'s
``heatmap_asset_ref``/``thumbnail_asset_ref`` (R3), the histogram's
``vulnerability_distribution.histogram_asset_ref``, the region bar chart's
``region_summary.chart_asset_ref``, and the optional
``fill_strategy_correlation_asset_ref`` (all R2) are assumed to already be
either a valid relative path or ``None`` by the time :func:`render_report`
runs. This mirrors how every other R-module here is a pure, single-purpose
transformer — ``ReportDataAssembler`` (R0) is deliberately the *only* module
that opens more than one upstream store at once
(IMPLE_PLAN_REPORTING_v1.md §3.1), and gluing R2/R3's rendering into R0's
model is exactly that kind of multi-module orchestration, which belongs to
Stage 7, not here.

**Two schema gaps closed in Stage 6 (confirmed with the user).** Before this
stage, ``ReportModel`` had no field to carry a rendered ``region_bar``
SVG's ref, and no data path at all for ``fill_strategy_correlation`` (R0
discarded ``rank_correlation.parquet`` rows outright) — even though
``ClassificationAdapter.applicable_charts()`` (Stage 1) already listed both
as real, renderable charts. Both gaps are closed as of this stage:
``RegionSummary.chart_asset_ref`` and ``ReportModel.
fill_strategy_correlation_asset_ref`` now exist (``report/types.py``), and
``ReportDataAssembler``/``AssembledReport`` now thread ``rank_correlation_
rows`` through as R0's extra, non-serialized data (``report/assembler.py``,
mirroring how ``full_sample_rankings`` already worked) so Stage 7 has
something to call ``report.charts.render_fill_strategy_correlation`` with.
Rendering those two charts and writing their SVG files is still Stage 7's
job, not this module's — the same boundary as the paragraph above.

**Templates are Python string constants, not ``.jinja`` files** (design
§3.1: this repository has never had a non-``.py`` source file under
``ssat/``, and adding one would need new packaging config this stage avoids).
``jinja2.Environment(loader=jinja2.DictLoader(...))`` is built from the
string constants below and never touches the filesystem for template lookup.

**Badge colors come from ``ssat.report.types.GRADE_COLORS``**, the same
palette R2's chart bars already use (that module's own docstring) — a
Jinja global function (:func:`_grade_color`) looks a grade up per-badge at
render time, so the color is computed once, in one place, for both views.

**Offline, no-JS, C2-compliant by construction.** Every asset reference the
template emits (``<img src=...>``, ``<link rel="stylesheet" href=...>``,
``<script src=...>``) is a plain relative path under ``output_dir`` — never
an absolute filesystem path, never an ``http(s)://`` URL. The Provenance
section is a bare HTML ``<details>`` element (design §7, confirmed:
"``<details>`` HTML 요소로 JS 없이 접이식") — expand/collapse is native
browser behavior, not a script. ``assets/js/enhance.js`` only adds
click-to-sort on the region table; every section is fully readable with
``<script>`` removed (design §R4 "JS 없이도 모든 콘텐츠가 보여야 한다").

**Report layout redesign (docs/report_layout_improve/AGENTS_OPINION_1.md).**
Real-data validation (C3, IMPLE_PLAN_REPORTING_v1.md §5 단계8) against
``experiments/synthetic_shortcut/results_crop_free`` surfaced two usability
problems an external review then confirmed: (1) the region table's worst-case
``reliability_grade`` badge collapsed a region with e.g. 67 HIGH-graded
anchors and 1 UNRELIABLE anchor down to a single alarming "UNRELIABLE" chip
— appropriate for a pass/fail QA gate, not for a report trying to show a
model's *behavior pattern*; and (2) the report had no dataset-level answer
to "does this model repeatedly depend on one fixed location, or is
sensitivity spread across many?" (``ReportModel.spatial_concentration``,
added in ``ssat.report.assembler``/``types``, answers exactly this). This
module was restructured to the review's recommended "A + C" combination of
its two HTML mockups (``docs/report_layout_improve/
report_layout_A_interpretation_first.html``/``report_layout_C_behavioral_
fingerprint.html``): an interpretation-first narrative (Executive
Interpretation → Behavioral Fingerprint → Dataset Spatial Pattern → Region
Summary → Vulnerable Samples → Stability/Controls → Detailed Tables →
Provenance) that leads with what the run's numbers mean before the raw
tables. The region-level worst-case badge is no longer the headline of the
Region Summary table — it is replaced by :func:`_grade_distribution_
percentages`/the ``grade_mix_bar`` macro, which render the *composition* of
grades among a region's anchors (matching mockup A's "HIGH 34% · LOW 12% ·
UNRELIABLE 54%"); the worst-case grade itself is preserved verbatim in
``RegionRow.reliability_grade`` (unchanged, still in every CSV/JSON export)
and surfaces only as a ``title`` attribute on the composition bar, so no
information is lost, only de-emphasized. Every helper below
(:func:`_grade_distribution_percentages`, :func:`_grid_layout`,
:func:`_heat_color`) is a pure reshaping/formatting function over values the
``ReportModel`` already carries — no new verdict is derived, matching this
module's role as a pure renderer (same precedent the removed pre-redesign
``_region_reliability_overview`` helper set).

A second, auxiliary template (:func:`render_secondary_report`, "Layout B" /
"Question Driven" from the same review) is written alongside the main
report as ``report_question_driven.html`` — always generated together with
``report.html`` (Application-layer orchestration, ``ssat.application.
application.AuditApplication.generate_report``), but explicitly a secondary,
supplementary view: it reuses the exact same ``ReportModel`` data as the
main report (no new computation), just organized as five plain-language
questions (baseline → sensitivity magnitude → spatial concentration →
control & stability → actionable examples) for a first-time reader, and
cross-links back to the main report's detailed sections rather than
duplicating them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import jinja2

from ssat.report.static import ENHANCE_JS, STYLE_CSS
from ssat.report.types import GRADE_COLORS, RegionRow, ReportGrade, ReportModel

# Distinct from every real GRADE_COLORS value — a badge for "no analysis run"
# must never read as any specific grade (mirrors report.charts._NO_GRADE_COLOR;
# duplicated rather than imported, since this module cannot depend on
# report.charts — module docstring).
_NO_GRADE_COLOR = "#9e9e9e"

# Fixed display order for every grade-composition rendering in this module
# (region table mix bars, region_summary CSV precedent) — most-trusted first,
# matching the severity ordering already established elsewhere in the report
# layer (ssat.report.assembler._GRADE_SEVERITY_ORDER is worst-first; this is
# its reverse, since a composition bar reads left-to-right as "best to worst"
# in every mockup this redesign follows).
_GRADE_DISPLAY_ORDER = (
    ReportGrade.HIGH,
    ReportGrade.MODERATE,
    ReportGrade.LOW,
    ReportGrade.UNRELIABLE,
)

# Matches a grid region_key's trailing "/r<row>/c<col>" (GridRegionExpander's
# region_instance_id format, ssat/core/plan/region_expanders.py) — the only
# geometry :func:`_grid_layout` knows how to lay out as a 2D heat grid.
_GRID_CELL_PATTERN = re.compile(r"/r(\d+)/c(\d+)$")

# Heat-grid color interpolation endpoints: a light neutral (no signal) up to
# the same --accent blue static.py's tokens use elsewhere in this report, so
# the heat grid's palette does not introduce a third color language on top of
# GRADE_COLORS (badges) and --accent (everything else).
_HEAT_COLOR_LOW_RGB = (238, 241, 245)
_HEAT_COLOR_HIGH_RGB = (21, 101, 192)


@dataclass(frozen=True, slots=True)
class ReportManifestPaths:
    """Where :func:`render_report` wrote each of its four output files (design §R4).

    Attributes:
        report_html: The rendered report page.
        style_css: ``assets/css/style.css``, written verbatim from
            ``ssat.report.static.STYLE_CSS``.
        enhance_js: ``assets/js/enhance.js``, written verbatim from
            ``ssat.report.static.ENHANCE_JS``.
        report_manifest_json: ``report_manifest.json`` (design §R4:
            ``report_schema_version, source_manifest_hashes, top_k,
            bottom_k, generated_at``).
    """

    report_html: Path
    style_css: Path
    enhance_js: Path
    report_manifest_json: Path


def render_report(
    model: ReportModel, output_dir: Path, *, top_k: int, bottom_k: int
) -> ReportManifestPaths:
    """Write ``report.html`` + static assets + ``report_manifest.json`` (design §R4).

    Args:
        model: A ``ReportModel`` whose asset-ref fields are already filled
            in by earlier orchestration (module docstring) — this function
            never renders a chart or an image itself, only templates the
            refs it is given.
        output_dir: The report root (``<run_dir>/report/``, matching R1/R3's
            already-established output layout); created if missing. Every
            path this function writes, and every ``href``/``src`` the
            rendered HTML emits, is relative to this directory.
        top_k: The top-K sample count R0 was configured with, for
            ``report_manifest.json`` and the gallery section headings —
            not recoverable from ``model`` alone when fewer samples were
            scored than ``top_k`` (``ReportDataAssembler._build_sample_
            rankings`` truncates silently in that case).
        bottom_k: Same, for the bottom-K gallery.

    Returns:
        The four files' paths, all inside ``output_dir``.
    """

    output_dir = Path(output_dir)
    css_dir = output_dir / "assets" / "css"
    js_dir = output_dir / "assets" / "js"
    css_dir.mkdir(parents=True, exist_ok=True)
    js_dir.mkdir(parents=True, exist_ok=True)

    paths = ReportManifestPaths(
        report_html=output_dir / "report.html",
        style_css=css_dir / "style.css",
        enhance_js=js_dir / "enhance.js",
        report_manifest_json=output_dir / "report_manifest.json",
    )

    environment = _build_environment()
    html = environment.get_template("report.html").render(model=model, top_k=top_k, bottom_k=bottom_k)

    paths.report_html.write_text(html, encoding="utf-8")
    paths.style_css.write_text(STYLE_CSS, encoding="utf-8")
    paths.enhance_js.write_text(ENHANCE_JS, encoding="utf-8")
    paths.report_manifest_json.write_text(
        json.dumps(_report_manifest_payload(model, top_k=top_k, bottom_k=bottom_k), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return paths


def render_secondary_report(
    model: ReportModel, output_dir: Path, *, top_k: int, bottom_k: int
) -> Path:
    """Write the auxiliary "Question Driven" report (module docstring, "Layout B").

    This writes only ``report_question_driven.html`` into ``output_dir`` —
    it assumes :func:`render_report` has already been called for the same
    ``output_dir`` and has already written ``assets/css/style.css``/
    ``assets/js/enhance.js`` there (the two reports intentionally share one
    static-asset bundle rather than duplicating it; ``ssat.application.
    application.AuditApplication.generate_report`` always calls
    :func:`render_report` immediately before this). Calling this function
    for an ``output_dir`` :func:`render_report` has not yet written to
    produces an HTML file whose ``<link>``/``<script>`` refs point at
    files that do not exist yet.

    Args:
        model: The same fully-populated ``ReportModel`` :func:`render_report`
            was given — this auxiliary report computes nothing new, it only
            reorganizes already-assembled data around five plain-language
            questions (module docstring).
        output_dir: The report root, matching :func:`render_report`'s
            ``output_dir``.
        top_k: Same meaning as :func:`render_report`'s ``top_k`` — how many
            of ``model.sample_rankings.most_vulnerable`` this report treats
            as "top" when describing the gallery it links back to.
        bottom_k: Unused by this template directly (it does not render a
            bottom-K gallery of its own — module docstring, "cross-links
            back... rather than duplicating them") but accepted for
            signature symmetry with :func:`render_report`, since both are
            always called together with the same ``ReportRequest`` values.

    Returns:
        The written file's path, inside ``output_dir``.
    """

    output_dir = Path(output_dir)
    environment = _build_environment()
    html = environment.get_template("report_b.html").render(model=model, top_k=top_k, bottom_k=bottom_k)
    path = output_dir / "report_question_driven.html"
    path.write_text(html, encoding="utf-8")
    return path


def _report_manifest_payload(model: ReportModel, *, top_k: int, bottom_k: int) -> dict[str, object]:
    """Build ``report_manifest.json``'s content (design §R4)."""

    return {
        "report_schema_version": model.meta.schema_versions.report,
        "source_manifest_hashes": {
            "run": model.provenance.run_manifest_hash,
            "metrics": model.provenance.metrics_manifest_hash,
            "analysis": model.provenance.analysis_manifest_hash,
        },
        "top_k": top_k,
        "bottom_k": bottom_k,
        "generated_at": model.meta.generated_at,
    }


# --- Jinja environment ----------------------------------------------------------


def _build_environment() -> jinja2.Environment:
    """Build the module's Jinja environment from in-memory template constants.

    ``autoescape=True`` (design §5 단계6: "사용자 제공 문자열이 섞여 들어가므로
    XSS/마크업 깨짐 방지, 오프라인 리포트라도 원칙적으로 켠다") — every value
    interpolated with ``{{ }}`` is HTML-escaped unless explicitly marked safe
    (nothing in these templates does that).
    """

    environment = jinja2.Environment(
        loader=jinja2.DictLoader(
            {
                "macros.html": _MACROS_TEMPLATE,
                "report.html": _REPORT_TEMPLATE,
                "report_b.html": _REPORT_TEMPLATE_B,
            }
        ),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.filters["fmt"] = _format_number
    environment.filters["fmt_pct"] = _format_percent
    environment.filters["fmt_duration"] = _format_duration
    environment.filters["fmt_distribution"] = _format_distribution
    environment.filters["grade_distribution_percentages"] = _grade_distribution_percentages
    environment.filters["grid_layout"] = _grid_layout
    environment.filters["heat_color"] = _heat_color
    environment.globals["grade_color"] = _grade_color
    environment.globals["unreliable_badge_color"] = GRADE_COLORS[ReportGrade.UNRELIABLE]
    environment.globals["ReportGrade"] = ReportGrade
    return environment


def _grade_color(grade: ReportGrade | None) -> str:
    """Look up one grade's shared badge color (module docstring), or the "no grade" gray."""

    if grade is None:
        return _NO_GRADE_COLOR
    return GRADE_COLORS[grade]


def _grade_distribution_percentages(
    distribution: Mapping[str, int],
) -> list[tuple[ReportGrade, float]]:
    """Turn a grade-count mapping into ``(grade, fraction)`` pairs for a composition bar.

    Pure arithmetic over a mapping the ``ReportModel`` already carries
    (``RegionRow.reliability_distribution``) — no new verdict, replacing the
    pre-redesign single worst-case badge as the region table's headline
    display (module docstring). Only grades with a non-zero count are
    included, in :data:`_GRADE_DISPLAY_ORDER` (best-to-worst, matching every
    mockup's reading order), so a template loop's ``loop.last`` lines up
    with the last entry actually rendered.

    Returns:
        An empty list when ``distribution`` is empty or sums to zero (no
        anchors evaluated for this region — the template shows "평가 없음"
        instead of an empty bar).
    """

    total = sum(distribution.values())
    if not total:
        return []
    return [
        (grade, distribution[grade.value] / total)
        for grade in _GRADE_DISPLAY_ORDER
        if distribution.get(grade.value, 0) > 0
    ]


def _grid_layout(rows: Sequence[RegionRow]) -> Mapping[str, object] | None:
    """Reshape ``region_summary.rows`` into a 2D grid, when the geometry allows it.

    Every row must be ``region_kind == "grid"`` and its ``region_key`` must
    end in ``/r<row>/c<col>`` (``GridRegionExpander``'s naming convention,
    ``ssat/core/plan/region_expanders.py``) — this is a structural
    re-layout of an identifier string, not a new statistic, staying inside
    this module's pure-renderer role. Regions that are not a grid (a single
    "whole" region, ``patch``, ``random_area_match`` controls, mixed
    families) cannot be laid out this way; the caller falls back to the
    plain Region Summary table rather than fabricating a shape that is not
    actually there (module docstring, same "no new statistic" boundary).

    Returns:
        ``None`` when ``rows`` is empty or any row fails the grid check
        above. Otherwise a mapping with ``"rows"``/``"cols"`` (grid extent),
        ``"cells"`` (row-major 2D list of ``RegionRow | None`` — ``None``
        for any cell the Cartesian product implies but ``rows`` did not
        actually contain), and ``"max_top_region_share"``/``"max_high_rate"``
        (the heat-color scale's denominators, ``None`` when every row's
        corresponding field is ``None``).
    """

    if not rows:
        return None

    parsed: list[tuple[int, int, RegionRow]] = []
    for row in rows:
        if row.region_kind != "grid":
            return None
        match = _GRID_CELL_PATTERN.search(row.region_key)
        if match is None:
            return None
        parsed.append((int(match.group(1)), int(match.group(2)), row))

    n_rows = max(row_index for row_index, _col_index, _row in parsed) + 1
    n_cols = max(col_index for _row_index, col_index, _row in parsed) + 1
    cells: list[list[RegionRow | None]] = [[None] * n_cols for _ in range(n_rows)]
    for row_index, col_index, row in parsed:
        cells[row_index][col_index] = row

    top_shares = [row.top_region_share for row in rows if row.top_region_share is not None]
    high_rates = [row.high_rate for row in rows if row.high_rate is not None]
    return {
        "rows": n_rows,
        "cols": n_cols,
        "cells": cells,
        "max_top_region_share": max(top_shares) if top_shares else None,
        "max_high_rate": max(high_rates) if high_rates else None,
    }


def _heat_color(value: float | None, max_value: float | None) -> str:
    """Interpolate one heat-grid cell's background color from ``value``/``max_value``.

    A pure display mapping (module docstring) — the color itself carries no
    new information beyond ``value``, it only makes the existing number
    scannable at a glance the way ``grade_color`` already does for grades.
    Reuses the two fixed RGB endpoints in :data:`_HEAT_COLOR_LOW_RGB`/
    :data:`_HEAT_COLOR_HIGH_RGB` rather than introducing a new palette.

    Returns:
        A ``"#rrggbb"`` string. The low/no-signal color when ``value`` is
        ``None`` or ``max_value`` is ``None``/``0`` (nothing to compare
        against).
    """

    if value is None or not max_value:
        low = _HEAT_COLOR_LOW_RGB
        return f"#{low[0]:02x}{low[1]:02x}{low[2]:02x}"
    intensity = min(1.0, max(0.0, value / max_value))
    blended = tuple(
        round(low + (high - low) * intensity)
        for low, high in zip(_HEAT_COLOR_LOW_RGB, _HEAT_COLOR_HIGH_RGB)
    )
    return f"#{blended[0]:02x}{blended[1]:02x}{blended[2]:02x}"


def _format_number(value: object, digits: int = 4) -> str:
    """Render one scalar for display: ``None`` becomes an em dash, floats get a digit cap."""

    if value is None:
        return "—"
    if isinstance(value, bool):
        return "예" if value else "아니오"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _format_percent(value: float | None) -> str:
    """Render a [0, 1] fraction as a percentage string, or an em dash when unavailable."""

    if value is None:
        return "—"
    return f"{value * 100:.2f}%"


def _format_duration(value: float | None) -> str:
    """Render a seconds count as a compact human-readable duration."""

    if value is None:
        return "—"
    minutes, seconds = divmod(value, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"
    if minutes:
        return f"{int(minutes)}m {seconds:.1f}s"
    return f"{seconds:.1f}s"


def _format_distribution(distribution: Mapping[str, int]) -> str:
    """Render a grade-count mapping as one title-attribute-friendly string."""

    if not distribution:
        return "분포 없음"
    return ", ".join(f"{key}:{count}" for key, count in sorted(distribution.items()))


# --- templates --------------------------------------------------------------


_MACROS_TEMPLATE = """
{% macro grade_badge(grade, title=none) %}
<span class="badge" style="background-color: {{ grade_color(grade) }};"
      title="{{ title if title else (grade.value if grade else '평가 없음') }}">{{
  grade.value | upper if grade else '해당 없음' }}</span>
{%- endmacro %}

{% macro grade_legend_item(grade, meaning) %}
<li class="grade-legend-item">
  <span class="grade-swatch" style="background-color: {{ grade_color(grade) }};" aria-hidden="true"></span>
  <span class="grade-legend-label">{{ grade.value | upper }}</span>
  <span class="grade-legend-meaning">{{ meaning }}</span>
</li>
{%- endmacro %}

{% macro grade_mix_bar(distribution, worst_grade=none) %}
{% set entries = distribution | grade_distribution_percentages %}
{% if entries %}
<div class="grade-mix"{% if worst_grade %} title="worst-case anchor: {{ worst_grade.value | upper }}"{% endif %}>
  <div class="stack">
    {% for grade, fraction in entries %}
    <span style="width: {{ (fraction * 100) | round(2) }}%; background-color: {{ grade_color(grade) }};"></span>
    {% endfor %}
  </div>
  <span class="grade-mix-text">
    {% for grade, fraction in entries %}{{ grade.value | upper }} {{ fraction | fmt_pct }}{% if not loop.last %} · {% endif %}{% endfor %}
  </span>
</div>
{% else %}
<span class="no-data">평가 없음</span>
{% endif %}
{% endmacro %}

{% macro sample_card(card) %}
<article class="sample-card">
  <div class="sample-card-images">
    {% if card.thumbnail_asset_ref %}
    <img src="{{ card.thumbnail_asset_ref }}" alt="{{ card.sample_id }} 원본 썸네일">
    {% else %}
    <div class="no-image">원본 없음</div>
    {% endif %}
    {% if card.heatmap_asset_ref %}
    <img src="{{ card.heatmap_asset_ref }}" alt="{{ card.sample_id }} 공간 히트맵 오버레이">
    {% else %}
    <div class="no-image">히트맵 없음</div>
    {% endif %}
  </div>
  <div class="sample-card-body">
    <div class="sample-card-header">
      <span class="sample-id">{{ card.sample_id }}</span>
      {{ grade_badge(card.reliability_grade) }}
    </div>
    <div class="sample-card-score">vulnerability_score: {{ card.vulnerability_score | fmt }}</div>
    {% if card.top_regions %}
    {% set top = card.top_regions[0] %}
    <div class="sample-card-top-region">
      최다 취약 region: {{ top.region_key }} ({{ top.degradation | fmt }})
      {{ grade_badge(top.reliability_grade) }}
    </div>
    {% endif %}
  </div>
</article>
{% endmacro %}
"""


_REPORT_TEMPLATE = """
{% import "macros.html" as macros %}
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ model.run_summary.dataset_name }} — SSAT Report</title>
<link rel="stylesheet" href="assets/css/style.css">
</head>
<body>

<header class="run-header">
  <h1>{{ model.run_summary.dataset_name }} — Spatial Sensitivity Audit Report</h1>
  <dl class="run-meta">
    <div><dt>모델</dt><dd>{{ model.run_summary.model_id }}</dd></div>
    <div><dt>전처리</dt><dd>{{ model.run_summary.preprocessing_desc }}</dd></div>
    <div><dt>샘플 수</dt><dd>{{ model.run_summary.n_samples }}</dd></div>
    <div><dt>영역 수(샘플당)</dt><dd>{{ model.run_summary.n_regions_per_sample }}</dd></div>
    <div><dt>조건 수</dt><dd>{{ model.run_summary.n_conditions }}</dd></div>
    <div><dt>소요 시간</dt><dd>{{ model.run_summary.duration_seconds | fmt_duration }}</dd></div>
    <div><dt>실패율</dt><dd>{{ model.run_summary.failure_rate | fmt_pct }}</dd></div>
    <div><dt>생성 시각</dt><dd>{{ model.meta.generated_at }}</dd></div>
    <div><dt>run_id</dt><dd>{{ model.meta.run_id }}</dd></div>
  </dl>
  <p class="section-note">
    질문 중심으로 요약된 <a href="report_question_driven.html">보조 리포트(Question Driven)</a>도
    함께 제공됩니다 — 이 페이지가 메인 리포트입니다.
  </p>
</header>

<section id="executive-interpretation" class="hero">
  <div class="hero-label">이 실행에서 가장 먼저 읽을 문장</div>
  {% set share = model.spatial_concentration.dominant_region_share %}
  {% if share is none %}
  <h2>공간 집중도를 판단할 데이터가 없습니다.</h2>
  <p class="hero-muted">
    유효한 degradation 값을 가진 샘플이 없어 dominant-region share/spatial entropy를 계산할 수 없습니다.
  </p>
  {% elif share >= 0.5 %}
  <h2>민감도가 <strong>{{ model.spatial_concentration.dominant_region_key }}</strong> 위치에 비교적 집중되어 있습니다.</h2>
  <p class="hero-muted">
    top region을 판별할 수 있었던 {{ model.spatial_concentration.n_scored_samples }}개 샘플 중
    {{ share | fmt_pct }}가 이 위치를 가장 취약한 region으로 꼽았습니다.
  </p>
  {% else %}
  <h2>민감도는 존재하지만 <strong>한 고정 위치에 집중되지 않습니다.</strong></h2>
  <p class="hero-muted">
    가장 자주 꼽히는 위치(<code>{{ model.spatial_concentration.dominant_region_key }}</code>)도
    {{ share | fmt_pct }}에 그쳐, 개별 샘플의 취약 위치가 서로 다릅니다.
  </p>
  {% endif %}
  <p class="callout">
    이 문장은 이번 실행에서 관측된 패턴을 서술할 뿐입니다 — shortcut 여부나 원인을 자동으로 판정하지
    않으며, 실제 원인 확인에는 후속 검증이 필요합니다.
  </p>
  <p class="hero-muted">
    아래 <code>vulnerability_score</code>는 이 샘플/영역이 얼마나 나빠지는지를 나타내는
    <strong>순위(ranking)</strong> 축이고, <code>reliability_grade</code>는 그 숫자를
    <strong>얼마나 믿을 수 있는지</strong>를 나타내는 별개의 <strong>신뢰도</strong> 축입니다 — 두 값은
    서로 독립적입니다.
  </p>
</section>

<section id="fingerprint">
  <h2>Behavioral Fingerprint</h2>
  <div class="card-grid">
    {% for card in model.scorecard %}
    <div class="metric-card">
      <div class="metric-label">{{ card.label }}</div>
      <div class="metric-value">
        {% if card.unit == '%' %}{{ card.value | fmt_pct }}{% else %}{{ card.value | fmt }}{{ card.unit }}{% endif %}
      </div>
      {% if card.note %}<div class="metric-note">{{ card.note }}</div>{% endif %}
    </div>
    {% endfor %}
    <div class="metric-card">
      <div class="metric-label">Dominant-region Share</div>
      <div class="metric-value">{{ model.spatial_concentration.dominant_region_share | fmt_pct }}</div>
      {% if model.spatial_concentration.dominant_region_key %}
      <div class="metric-note">{{ model.spatial_concentration.dominant_region_key }}</div>
      {% else %}
      <div class="metric-note">해당 없음: 판별 가능한 샘플이 없습니다.</div>
      {% endif %}
    </div>
    <div class="metric-card">
      <div class="metric-label">Spatial Entropy</div>
      <div class="metric-value">{{ model.spatial_concentration.spatial_entropy | fmt }}</div>
      <div class="metric-note">0 = 한 위치에 집중, 1 = 완전히 분산</div>
    </div>
  </div>

  <h3>평균 뒤에 숨은 변이</h3>
  {% if model.vulnerability_distribution.histogram_asset_ref %}
  <img class="chart" src="{{ model.vulnerability_distribution.histogram_asset_ref }}"
       alt="vulnerability_score 히스토그램">
  {% else %}
  <p class="no-data">히스토그램 자산 없음</p>
  {% endif %}
  {% set stats = model.vulnerability_distribution.summary_stats %}
  <p class="callout">
    평균 저하도는 {{ stats.mean | fmt }}이지만, 개별 샘플은 p90 {{ stats.p90 | fmt }} ~
    p99 {{ stats.p99 | fmt }} 범위까지 걸쳐 있습니다 — 아래 갤러리에서 극단값을 확인하세요.
  </p>
  <dl class="stats-list">
    <div><dt>평균</dt><dd>{{ stats.mean | fmt }}</dd></div>
    <div><dt>중앙값</dt><dd>{{ stats.median | fmt }}</dd></div>
    <div><dt>p90</dt><dd>{{ stats.p90 | fmt }}</dd></div>
    <div><dt>p99</dt><dd>{{ stats.p99 | fmt }}</dd></div>
  </dl>
</section>

<section id="spatial-pattern">
  <h2>Dataset Spatial Pattern</h2>
  {% set layout = model.region_summary.rows | grid_layout %}
  {% if layout %}
  <div class="grid2">
    <div>
      <h3>Top-region share</h3>
      <p class="section-note">
        색이 진할수록 그 위치가 더 많은 샘플의 최다 취약 region이 되었다는 뜻입니다 —
        reliability grade가 아닙니다.
      </p>
      <div class="heat" style="grid-template-columns: repeat({{ layout.cols }}, 1fr);">
        {% for grid_row in layout.cells %}
        {% for cell in grid_row %}
        {% if cell %}
        <div class="cell" style="background-color: {{ cell.top_region_share | heat_color(layout.max_top_region_share) }};"
             title="{{ cell.region_key }}">
          <b>{{ cell.top_region_share | fmt_pct }}</b>{{ cell.region_key }}
        </div>
        {% else %}
        <div class="cell no-data">—</div>
        {% endif %}
        {% endfor %}
        {% endfor %}
      </div>
    </div>
    <div>
      <h3>HIGH-graded anchor rate</h3>
      <p class="section-note">
        색이 진할수록 그 위치에서 HIGH 등급으로 판정된 anchor의 비율이 높다는 뜻입니다.
      </p>
      <div class="heat" style="grid-template-columns: repeat({{ layout.cols }}, 1fr);">
        {% for grid_row in layout.cells %}
        {% for cell in grid_row %}
        {% if cell %}
        <div class="cell" style="background-color: {{ cell.high_rate | heat_color(layout.max_high_rate) }};"
             title="{{ cell.region_key }}">
          <b>{{ cell.high_rate | fmt_pct }}</b>{{ cell.region_key }}
        </div>
        {% else %}
        <div class="cell no-data">—</div>
        {% endif %}
        {% endfor %}
        {% endfor %}
      </div>
    </div>
  </div>
  {% else %}
  <p class="no-data">
    이 실행의 region이 격자(grid) 형태로 배치되어 있지 않아 히트그리드를 생략합니다 — 아래
    "Region Summary" 표를 참고하세요.
  </p>
  {% endif %}
</section>

<section id="region-summary">
  <h2>Region Summary</h2>
  <p class="section-note">
    아래 막대는 이 region을 공유하는 anchor(sample × region × invert_mask) 전체의
    <strong>등급 구성비</strong>를 보여줍니다. "가장 나쁜 등급 하나"로 축약하는 worst-case 표시는
    sample/anchor 단위(아래 갤러리의 큰 배지)에서만 쓰고, 이 dataset 레벨 표에는 적용하지 않습니다 —
    HIGH가 다수인 region이 anchor 하나가 UNRELIABLE이라는 이유만으로 전체가 UNRELIABLE로 보이지
    않도록 하기 위함입니다. worst-case 값 자체는 각 막대에 마우스를 올리면(title) 확인할 수 있고,
    <a href="data/region_summary.csv">region_summary.csv</a>에는 그대로 남아 있습니다.
  </p>
  <ul class="grade-legend">
    {{ macros.grade_legend_item(ReportGrade.HIGH,
      "모든 신뢰도 검사(대조군 대비 초과, 2개 이상 fill 전략에서 재현, 부트스트랩 신뢰구간이 0을 배제)를 통과했습니다 — 방향과 크기 모두 확실합니다.") }}
    {{ macros.grade_legend_item(ReportGrade.MODERATE,
      "fill 전략 간 방향(부호)은 일치하지만, 핵심 검사(대조군 초과·다중 전략 재현·신뢰구간) 중 일부만 충족합니다.") }}
    {{ macros.grade_legend_item(ReportGrade.LOW,
      "fill 전략 간 방향은 일치하지만 핵심 검사를 거의 충족하지 못했거나 근거가 부족합니다 — 효과가 작거나 아직 충분히 검증되지 않았습니다.") }}
    {{ macros.grade_legend_item(ReportGrade.UNRELIABLE,
      "fill 전략(채우기 방식)마다 이 지점을 지웠을 때 성능이 좋아지는지 나빠지는지 방향 자체가 갈립니다 — 숫자의 크기가 아니라 부호가 불일치하므로 이 값은 신뢰할 수 없습니다.") }}
  </ul>
  {% if model.region_summary.chart_asset_ref %}
  <img class="chart" src="{{ model.region_summary.chart_asset_ref }}" alt="Region별 평균 저하도 막대그래프">
  {% endif %}
  <table id="region-table">
    <caption>영역(region)별 평균 저하도·구성비. 헤더를 클릭하면 정렬할 수 있습니다.</caption>
    <thead>
      <tr>
        <th data-sort="text">region_key</th>
        <th data-sort="num">mean_degradation</th>
        <th data-sort="num">top_region_share</th>
        <th data-sort="num">n_valid</th>
        <th data-sort="text">anchor reliability mix</th>
      </tr>
    </thead>
    <tbody>
      {% for row in model.region_summary.rows %}
      <tr>
        <td>{{ row.region_key }}</td>
        <td>{{ row.mean_degradation | fmt }}</td>
        <td>{{ row.top_region_share | fmt_pct }}</td>
        <td>{{ row.n_valid }}</td>
        <td>{{ macros.grade_mix_bar(row.reliability_distribution, row.reliability_grade) }}</td>
      </tr>
      {% else %}
      <tr><td colspan="5" class="no-data">데이터 없음</td></tr>
      {% endfor %}
    </tbody>
  </table>
</section>

<section id="gallery">
  <h2>Vulnerable Samples — 모델이 어디를 보는가</h2>
  <p class="section-note">
    카드의 큰 등급 배지는 그 샘플이 포괄하는 모든 region 중 최악의 등급입니다(worst-case) — 위
    Region Summary의 구성비 막대와는 축이 다릅니다. 자세한 설명은 <a href="#region-summary">Region
    Summary</a>를 참고하세요.
  </p>
  <h3>가장 취약한 샘플 (top-{{ top_k }})</h3>
  {% if model.sample_rankings.most_vulnerable %}
  <div class="gallery-grid">
    {% for card in model.sample_rankings.most_vulnerable %}{{ macros.sample_card(card) }}{% endfor %}
  </div>
  {% else %}
  <p class="no-data">데이터 없음</p>
  {% endif %}

  <h3>가장 강건한 샘플 (bottom-{{ bottom_k }})</h3>
  {% if model.sample_rankings.most_robust %}
  <div class="gallery-grid">
    {% for card in model.sample_rankings.most_robust %}{{ macros.sample_card(card) }}{% endfor %}
  </div>
  {% else %}
  <p class="no-data">데이터 없음</p>
  {% endif %}
</section>

<section id="stability-controls">
  <h2>Stability / Controls</h2>
  <p class="section-note">
    위 Behavioral Fingerprint의 "Mean Z vs Control" 카드가 대조군 대비 초과 정도를 요약합니다.
    여기서는 그 판단의 근거가 된 자산과 신뢰도 스포트라이트 요약을 보여줍니다.
  </p>
  {% if model.fill_strategy_correlation_asset_ref %}
  <h3>Fill Strategy Rank Correlation</h3>
  <img class="chart" src="{{ model.fill_strategy_correlation_asset_ref }}"
       alt="Fill strategy 쌍별 순위 상관관계 히트맵">
  {% else %}
  <p class="no-data">
    fill strategy 상관 분석 자산이 없습니다(다중 fill 전략 실행이 아니었거나 분석이 실행되지
    않았습니다).
  </p>
  {% endif %}
  <p>
    현재 <strong>{{ model.reliability_spotlight.flagged_examples | length }}개</strong> anchor가
    UNRELIABLE로 플래그되어 있습니다 — 전체 목록은 아래 "Detailed Tables / Flagged Anchors"를
    참고하세요.
  </p>
</section>

<section id="flagged-anchors">
  <h2>Detailed Tables / Flagged Anchors</h2>
  {% if model.reliability_spotlight.flagged_examples %}
  <ul class="flagged-list">
    {% for item in model.reliability_spotlight.flagged_examples %}
    <li>
      <span class="badge" style="background-color: {{ unreliable_badge_color }};"
            title="{{ item.reliability_reasons | join('; ') }}">UNRELIABLE</span>
      <span class="flagged-anchor">{{ item.anchor_key_repr }}</span>
      <span class="flagged-reason">{{ item.reason_summary }}</span>
    </li>
    {% endfor %}
  </ul>
  {% else %}
  <p class="no-data">플래그된 항목 없음</p>
  {% endif %}
</section>

<details id="provenance">
  <summary>Provenance (클릭하여 펼치기)</summary>
  <dl>
    <div><dt>dump_path</dt><dd>{{ model.provenance.dump_path }}</dd></div>
    <div><dt>metrics_dir</dt><dd>{{ model.provenance.metrics_dir }}</dd></div>
    <div><dt>analysis_dir</dt><dd>{{ model.provenance.analysis_dir or '해당 없음' }}</dd></div>
    <div><dt>run_manifest_hash</dt><dd><code>{{ model.provenance.run_manifest_hash }}</code></dd></div>
    <div><dt>metrics_manifest_hash</dt><dd><code>{{ model.provenance.metrics_manifest_hash }}</code></dd></div>
    <div><dt>analysis_manifest_hash</dt>
      <dd><code>{{ model.provenance.analysis_manifest_hash or '해당 없음' }}</code></dd></div>
    <div><dt>schema_versions</dt>
      <dd>dump {{ model.meta.schema_versions.dump }} / metrics {{ model.meta.schema_versions.metrics }} /
        analysis {{ model.meta.schema_versions.analysis or '해당 없음' }} /
        report {{ model.meta.schema_versions.report }}</dd></div>
  </dl>
  {% if model.provenance.thresholds %}
  <h4>Thresholds</h4>
  <ul>
    {% for key, value in model.provenance.thresholds.items() | sort %}
    <li>{{ key }}: {{ value }}</li>
    {% endfor %}
  </ul>
  {% endif %}
  <h4>원본 데이터 다운로드</h4>
  <ul class="download-links">
    <li><a href="data/report_model.json">report_model.json</a></li>
    <li><a href="data/sample_rankings.csv">sample_rankings.csv</a></li>
    <li><a href="data/region_summary.csv">region_summary.csv</a></li>
    <li><a href="data/flagged_items.csv">flagged_items.csv</a></li>
  </ul>
</details>

<script src="assets/js/enhance.js" defer></script>
</body>
</html>
"""


_REPORT_TEMPLATE_B = """
{% import "macros.html" as macros %}
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ model.run_summary.dataset_name }} — SSAT Report (Question Driven)</title>
<link rel="stylesheet" href="assets/css/style.css">
</head>
<body>

<header class="run-header">
  <h1>{{ model.run_summary.dataset_name }} — 결과를 질문 순서로 읽는 보조 리포트</h1>
  <p class="hero-muted">
    이 페이지는 <a href="report.html">메인 리포트(report.html)</a>의 보조 리포트입니다 — 같은 데이터를
    처음 보는 사람이 실제로 궁금해할 다섯 가지 질문 순서로 다시 배치했을 뿐, 새로 계산된 값은 없습니다.
  </p>
</header>

{% set accuracy_card = model.scorecard | selectattr("key", "equalto", "accuracy") | first %}
<section class="section-block">
  <div class="q">
    <div class="qnum">1</div>
    <div>
      <div class="hero-label">Baseline</div>
      <div class="q-answer">이 모델은 교란 전에는 얼마나 잘 작동하는가?</div>
      <p>
        <span class="pill">답</span> Clean accuracy는
        <b>{{ accuracy_card.value | fmt_pct if accuracy_card.value is not none else '해당 없음' }}</b>입니다.
        {% if accuracy_card.note %}({{ accuracy_card.note }}){% endif %}
      </p>
      <p class="hero-muted">N = {{ model.run_summary.n_samples }} · 실패율 {{ model.run_summary.failure_rate | fmt_pct }}</p>
    </div>
  </div>
</section>

{% set stats = model.vulnerability_distribution.summary_stats %}
<section class="section-block">
  <div class="q">
    <div class="qnum">2</div>
    <div>
      <div class="hero-label">Sensitivity magnitude</div>
      <div class="q-answer">영역을 가리면 전반적으로 얼마나 흔들리는가?</div>
      <p>
        <span class="pill">답</span> 평균 저하는 <b>{{ stats.mean | fmt }}</b>이지만,
        상위 샘플은 p90 {{ stats.p90 | fmt }} ~ p99 {{ stats.p99 | fmt }}까지 훨씬 크게 흔들립니다.
      </p>
      {% if model.vulnerability_distribution.histogram_asset_ref %}
      <img class="chart" src="{{ model.vulnerability_distribution.histogram_asset_ref }}"
           alt="vulnerability_score 히스토그램">
      {% endif %}
    </div>
  </div>
</section>

{% set share = model.spatial_concentration.dominant_region_share %}
<section class="section-block">
  <div class="q">
    <div class="qnum">3</div>
    <div>
      <div class="hero-label">Spatial concentration</div>
      <div class="q-answer">모든 샘플이 반복적으로 의존하는 "고정 위치"가 있는가?</div>
      {% if share is none %}
      <p><span class="pill">답: 해당 없음</span> 판별 가능한 샘플이 없어 결론을 내릴 수 없습니다.</p>
      {% elif share >= 0.5 %}
      <p>
        <span class="pill">답: 있음</span> <code>{{ model.spatial_concentration.dominant_region_key }}</code>가
        샘플의 {{ share | fmt_pct }}에서 top region으로 반복됩니다.
      </p>
      {% else %}
      <p>
        <span class="pill">답: 뚜렷하지 않음</span> 가장 자주 top-1이 되는 위치
        (<code>{{ model.spatial_concentration.dominant_region_key }}</code>)도 {{ share | fmt_pct }}
        수준이며, 취약 위치가 샘플마다 달라집니다.
      </p>
      {% endif %}
      <div class="grid2">
        <div><div class="metric-value">{{ share | fmt_pct }}</div><div class="hero-muted">Dominant-region share</div></div>
        <div><div class="metric-value">{{ model.spatial_concentration.spatial_entropy | fmt }}</div><div class="hero-muted">Spatial entropy (1에 가까울수록 분산)</div></div>
      </div>
      <p class="callout">
        <b>중요:</b> "HIGH region이 없다"라고 결론짓지 않습니다. 대신 dataset-wide 고정 위치 의존이
        약한지/강한지를 표현합니다 — 개별 샘플의 HIGH 근거는 아래 5번과 메인 리포트의 갤러리에 그대로
        남아 있습니다.
      </p>
    </div>
  </div>
</section>

{% set control_card = model.scorecard | selectattr("key", "equalto", "control_comparison") | first %}
<section class="section-block">
  <div class="q">
    <div class="qnum">4</div>
    <div>
      <div class="hero-label">Control &amp; stability</div>
      <div class="q-answer">보이는 효과는 우연한 마스크 효과보다 강하고 반복 가능한가?</div>
      <table>
        <thead><tr><th>검사</th><th>결과</th><th>의미</th></tr></thead>
        <tbody>
          <tr>
            <td>동일 면적 control</td>
            <td>{{ control_card.value | fmt if control_card.value is not none else '해당 없음' }}</td>
            <td>Mean Z vs Control — 클수록 대조군 대비 효과가 큼{% if control_card.note %} ({{ control_card.note }}){% endif %}</td>
          </tr>
          <tr>
            <td>Fill agreement</td>
            <td>{% if model.fill_strategy_correlation_asset_ref %}차트 있음{% else %}해당 없음{% endif %}</td>
            <td>fill strategy 간 순위 상관 — 아래 링크된 차트에서 직접 확인</td>
          </tr>
          <tr>
            <td>UNRELIABLE anchors</td>
            <td>{{ model.reliability_spotlight.flagged_examples | length }}개</td>
            <td>fill 전략마다 방향이 갈리는 anchor 수 — 메인 리포트의 상세 목록 참고</td>
          </tr>
        </tbody>
      </table>
      {% if model.fill_strategy_correlation_asset_ref %}
      <p><a href="report.html#stability-controls">메인 리포트에서 상관관계 차트 보기 →</a></p>
      {% endif %}
    </div>
  </div>
</section>

<section class="section-block">
  <div class="q">
    <div class="qnum">5</div>
    <div>
      <div class="hero-label">Actionable examples</div>
      <div class="q-answer">그렇다면 실제로 어떤 샘플을 먼저 조사해야 하는가?</div>
      {% set examples = model.sample_rankings.most_vulnerable[:5] %}
      {% if examples %}
      <table>
        <thead><tr><th>Sample</th><th>Top region</th><th>Score</th><th>Evidence</th></tr></thead>
        <tbody>
          {% for card in examples %}
          <tr>
            <td>{{ card.sample_id }}</td>
            <td>{{ card.top_regions[0].region_key if card.top_regions else '해당 없음' }}</td>
            <td>{{ card.vulnerability_score | fmt }}</td>
            <td>{{ macros.grade_badge(card.reliability_grade) }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% else %}
      <p class="no-data">데이터 없음</p>
      {% endif %}
      <p><a href="report.html#gallery">메인 리포트의 전체 갤러리 보기 →</a></p>
    </div>
  </div>
</section>

<section class="section-block">
  <details>
    <summary><b>전문가용 상세: region 표 / fill correlation / flagged anchors / provenance</b></summary>
    <p class="hero-muted">
      이 보조 리포트는 요약만 담습니다 — region별 구성비 표, 상세 상관관계 차트, 전체 flagged anchor
      목록, provenance/원본 CSV·JSON은 전부 <a href="report.html">메인 리포트(report.html)</a>에
      그대로 있습니다.
    </p>
  </details>
</section>

</body>
</html>
"""
