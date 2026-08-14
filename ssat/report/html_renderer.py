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
``jinja2.Environment(loader=jinja2.DictLoader(...))`` is built from the two
constants below and never touches the filesystem for template lookup.

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

**The ``report-guide`` section is computed entirely from data ``model.
region_summary.rows`` already carries** — real-data validation (C3,
IMPLE_PLAN_REPORTING_v1.md §5 단계8) against
``experiments/synthetic_shortcut/results_crop_free`` surfaced that a bare
worst-case ``UNRELIABLE`` badge, with no explanation, reads as broken rather
than as the intended "this region's degradation flips sign across fill
strategies, don't trust it" signal (design §6.2, §1 격차#3's worst-case
rollup). :func:`_region_reliability_overview` only counts/buckets
``RegionRow.reliability_grade`` values ``ReportDataAssembler`` already
assigned (``_worst_grade``) — it derives no new verdict, matching this
module's role as a pure renderer. It degrades to an explicit "해당 없음"
message, never silent omission, when every row's ``reliability_grade`` is
``None`` (design §6.2 C1, the same convention ``_format_distribution``/
``_NO_GRADE_COLOR`` already follow).
"""

from __future__ import annotations

import json
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
        loader=jinja2.DictLoader({"macros.html": _MACROS_TEMPLATE, "report.html": _REPORT_TEMPLATE}),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.filters["fmt"] = _format_number
    environment.filters["fmt_pct"] = _format_percent
    environment.filters["fmt_duration"] = _format_duration
    environment.filters["fmt_distribution"] = _format_distribution
    environment.filters["region_reliability_overview"] = _region_reliability_overview
    environment.globals["grade_color"] = _grade_color
    environment.globals["unreliable_badge_color"] = GRADE_COLORS[ReportGrade.UNRELIABLE]
    environment.globals["ReportGrade"] = ReportGrade
    return environment


def _grade_color(grade: ReportGrade | None) -> str:
    """Look up one grade's shared badge color (module docstring), or the "no grade" gray."""

    if grade is None:
        return _NO_GRADE_COLOR
    return GRADE_COLORS[grade]


def _region_reliability_overview(rows: Sequence[RegionRow]) -> Mapping[str, object]:
    """Summarize region_summary.rows into the report-guide callout's dynamic facts.

    Pure function over data already present in the ``ReportModel`` being
    rendered (module docstring) — this only counts/buckets ``RegionRow.
    reliability_grade`` values ``ReportDataAssembler`` already assigned via
    its worst-case rollup (§1 격차#3), the same way ``_format_distribution``
    already turns an existing mapping into display text; it never derives a
    new verdict.

    Returns:
        A mapping with ``"available"`` (``False`` when every row's
        ``reliability_grade`` is ``None`` — ``analysis_dir`` was
        unavailable for this run, or no anchor matched the primary metric
        for any region), ``"high_region_keys"`` (sorted tuple of
        region_keys graded HIGH — the run's actual, checks-passing
        vulnerability drivers), ``"n_graded"``/``"n_unreliable"`` (counts
        among graded rows), and ``"unreliable_fraction"`` (``None`` when
        unavailable).
    """

    graded = [row for row in rows if row.reliability_grade is not None]
    if not graded:
        return {
            "available": False,
            "high_region_keys": (),
            "n_graded": 0,
            "n_unreliable": 0,
            "unreliable_fraction": None,
        }
    high_region_keys = tuple(
        sorted(row.region_key for row in graded if row.reliability_grade is ReportGrade.HIGH)
    )
    n_unreliable = sum(1 for row in graded if row.reliability_grade is ReportGrade.UNRELIABLE)
    return {
        "available": True,
        "high_region_keys": high_region_keys,
        "n_graded": len(graded),
        "n_unreliable": n_unreliable,
        "unreliable_fraction": n_unreliable / len(graded),
    }


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
</header>

<section id="report-guide">
  <h2>리포트 읽는 법</h2>

  <h3>vulnerability_score vs. reliability_grade</h3>
  <p>
    <code>vulnerability_score</code>는 이 샘플/영역이 perturbation 이후 얼마나 성능이
    나빠지는지를 나타내는 <strong>순위(ranking)</strong> 축입니다. <code>reliability_grade</code>는
    그 숫자를 <strong>얼마나 믿을 수 있는지</strong>를 나타내는 완전히 별개의
    <strong>신뢰도(trustworthiness)</strong> 축입니다. 두 축은 독립적입니다 —
    vulnerability_score가 크다고 reliability_grade가 높다는 보장은 없고, 그 반대도
    마찬가지입니다.
  </p>

  <h3>배지는 "최악의 경우"를 보여줍니다</h3>
  <p>
    샘플 카드와 region 테이블의 등급 배지는 해당 샘플/영역이 포괄하는 <strong>모든</strong>
    anchor(sample × region × invert_mask 조합) 중 <strong>가장 나쁜 등급</strong>을
    표시합니다(worst-case rollup). 예를 들어 한 샘플이 여러 region을 갖고 그중 하나라도
    UNRELIABLE이면, 샘플 카드의 큰 배지는 UNRELIABLE로 표시됩니다 — 이는 그 샘플의
    <em>모든</em> region이 신뢰할 수 없다는 뜻이 아니라, 가장 신뢰할 수 없는 하나가
    존재한다는 뜻입니다. 아래 갤러리·region 테이블에서 카드의 큰 배지와 개별 top region의
    작은 배지가 다르게 보이는 것은 바로 이 규칙 때문입니다.
  </p>

  <h3>등급 범례</h3>
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

  <h3>이번 실행에서 실제로 무엇이 신뢰할 만한가</h3>
  {% set overview = model.region_summary.rows | region_reliability_overview %}
  <div class="callout" id="region-reliability-callout">
    {% if overview.available %}
      {% if overview.high_region_keys %}
      <p>
        이번 실행에서 실제로 <strong>HIGH</strong> 등급을 받은 region은
        {{ overview.high_region_keys | length }}개입니다:
        {% for key in overview.high_region_keys %}<code>{{ key }}</code>{% if not loop.last %}, {% endif %}{% endfor %}.
        이 region(들)이 이 리포트가 보여주는 취약성의 가장 신뢰할 수 있는 실제 원인입니다.
      </p>
      {% else %}
      <p>이번 실행에서 HIGH 등급을 받은 region은 없습니다 — 모든 신뢰도 검사를 통과한 region이 없다는 뜻입니다.</p>
      {% endif %}
      <p>
        분석된 {{ overview.n_graded }}개 region 중 {{ overview.n_unreliable }}개
        ({{ overview.unreliable_fraction | fmt_pct }})가 <strong>UNRELIABLE</strong>입니다 —
        fill 전략마다 방향이 갈려 숫자를 신뢰할 수 없는 region이라는 뜻입니다(위 범례의
        UNRELIABLE 설명 참고).
      </p>
    {% else %}
    <p class="no-data">해당 없음: 분석 미실행 (analysis_dir 없음) — reliability_grade를 계산할 근거가 없습니다.</p>
    {% endif %}
  </div>
</section>

<section id="scorecard">
  <h2>스코어카드</h2>
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
  </div>
</section>

<section id="vulnerability-distribution">
  <h2>취약도 분포</h2>
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

<section id="gallery">
  <h2>모델이 어디를 보는가 — 취약 샘플 갤러리</h2>
  <p class="section-note">
    카드의 큰 등급 배지는 그 샘플이 포괄하는 모든 region 중 최악의 등급입니다 — 아래 개별
    top region의 작은 배지와 다를 수 있습니다. 자세한 설명은
    <a href="#report-guide">리포트 읽는 법</a>을 참고하세요.
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

<section id="region-summary">
  <h2>Region 요약</h2>
  {% if model.region_summary.chart_asset_ref %}
  <img class="chart" src="{{ model.region_summary.chart_asset_ref }}" alt="Region별 평균 저하도 막대그래프">
  {% endif %}
  {% if model.fill_strategy_correlation_asset_ref %}
  <h3>Fill Strategy Rank Correlation</h3>
  <img class="chart" src="{{ model.fill_strategy_correlation_asset_ref }}"
       alt="Fill strategy 쌍별 순위 상관관계 히트맵">
  {% endif %}
  <table id="region-table">
    <caption>영역(region)별 평균 저하도·신뢰도. 헤더를 클릭하면 정렬할 수 있습니다.</caption>
    <thead>
      <tr>
        <th data-sort="text">region_key</th>
        <th data-sort="text">region_kind</th>
        <th data-sort="num">intended_area_px</th>
        <th data-sort="num">effective_area_px</th>
        <th data-sort="num">mean_degradation</th>
        <th data-sort="num">flip_rate</th>
        <th data-sort="num">n_valid</th>
        <th data-sort="text">reliability_grade</th>
      </tr>
    </thead>
    <tbody>
      {% for row in model.region_summary.rows %}
      <tr>
        <td>{{ row.region_key }}</td>
        <td>{{ row.region_kind }}</td>
        <td>{{ row.intended_area_px | fmt }}</td>
        <td>{{ row.effective_area_px | fmt }}</td>
        <td>{{ row.mean_degradation | fmt }}</td>
        <td>{{ row.flip_rate | fmt_pct }}</td>
        <td>{{ row.n_valid }}</td>
        <td>{{ macros.grade_badge(row.reliability_grade, row.reliability_distribution | fmt_distribution) }}</td>
      </tr>
      {% else %}
      <tr><td colspan="8" class="no-data">데이터 없음</td></tr>
      {% endfor %}
    </tbody>
  </table>
</section>

<section id="reliability-spotlight">
  <h2>신뢰도 스포트라이트 — "이 결과는 믿지 말라"</h2>
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
