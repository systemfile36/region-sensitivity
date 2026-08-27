"""HTMLRenderer: assemble ``ReportModel`` + already-rendered assets into report.html.

**Scope boundary.** This module never imports ``report.charts``,
``report.assets``, or ``report.assembler`` — it consumes a ``ReportModel``
that an earlier orchestration step (``AuditApplication.
generate_report()``) has already fully populated: every ``SampleCard``'s
``heatmap_asset_ref``/``thumbnail_asset_ref``, the histogram's
``vulnerability_distribution.histogram_asset_ref``, the region bar chart's
``region_summary.chart_asset_ref``, and the optional
``fill_strategy_correlation_asset_ref`` are assumed to already be
either a valid relative path or ``None`` by the time :func:`render_report`
runs. This mirrors how every other module in this package is a pure,
single-purpose transformer — ``ReportDataAssembler`` is deliberately the
*only* module that opens more than one upstream store at once, and gluing
chart/asset rendering into its model is exactly that kind of multi-module
orchestration, which belongs to the application layer, not here.

**Two ``ReportModel`` fields exist for charts this module never renders
itself.** ``RegionSummary.chart_asset_ref`` (for a rendered ``region_bar``
SVG) and ``ReportModel.fill_strategy_correlation_asset_ref`` (``report/
types.py``) are both populated only by upstream chart rendering;
``ReportDataAssembler``/``AssembledReport`` thread the underlying
``rank_correlation`` rows through as extra, non-serialized data
(``report/assembler.py``, mirroring how ``full_sample_rankings`` already
works) so that upstream step has something to call
``report.charts.render_fill_strategy_correlation`` with. Rendering those
two charts and writing their SVG files is still not this module's job —
the same boundary as the paragraph above.

**Templates are Python string constants, not ``.jinja`` files** — this
repository has never had a non-``.py`` source file under ``ssat/``, and
adding one would need new packaging config. ``jinja2.Environment(loader=
jinja2.DictLoader(...))`` is built from the string constants below and
never touches the filesystem for template lookup.

**Badge colors come from ``ssat.report.types.GRADE_COLORS``**, the same
palette ``report.charts``'s bars already use — a Jinja global function
(:func:`_grade_color`) looks a grade up per-badge at render time, so the
color is computed once, in one place, for both views.

**Offline and usable with no JavaScript, by construction.** Every asset
reference the template emits (``<img src=...>``, ``<link rel="stylesheet"
href=...>``, ``<script src=...>``) is a plain relative path under
``output_dir`` — never an absolute filesystem path, never an ``http(s)://``
URL. The Provenance section is a bare HTML ``<details>`` element, so
expand/collapse is native browser behavior, not a script.
``assets/js/enhance.js`` only adds click-to-sort on the region table; every
section is fully readable with ``<script>`` removed.

**Report layout: interpretation-first, not a raw pass/fail table.**
Validation against real data (``experiments/synthetic_shortcut/
results_crop_free``) surfaced two usability problems: (1) the region
table's worst-case ``reliability_grade`` badge collapsed a region with
e.g. 67 HIGH-graded anchors and 1 UNRELIABLE anchor down to a single
alarming "UNRELIABLE" chip — appropriate for a pass/fail QA gate, not for
a report trying to show a model's *behavior pattern*; and (2) the report
had no dataset-level answer to "does this model repeatedly depend on one
fixed location, or is sensitivity spread across many?"
(``ReportModel.spatial_concentration``, in ``ssat.report.assembler``/
``types``, answers exactly this). This module is structured as an
interpretation-first narrative (Executive Interpretation → Behavioral
Fingerprint → Dataset Spatial Pattern → Region Summary → Vulnerable
Samples → Stability/Controls → Detailed Tables → Provenance) that leads
with what the run's numbers mean before the raw tables. The region-level
worst-case badge is no longer the headline of the Region Summary table —
it is replaced by :func:`_grade_distribution_percentages`/the
``grade_mix_bar`` macro, which render the *composition* of grades among a
region's anchors (e.g. "HIGH 34% · LOW 12% · UNRELIABLE 54%"); the
worst-case grade itself is preserved verbatim in
``RegionRow.reliability_grade`` (unchanged, still in every CSV/JSON
export) and surfaces only as a ``title`` attribute on the composition bar,
so no information is lost, only de-emphasized. Every helper below
(:func:`_grade_distribution_percentages`, :func:`_grid_layout`,
:func:`_heat_color`) is a pure reshaping/formatting function over values
the ``ReportModel`` already carries — no new verdict is derived, matching
this module's role as a pure renderer (the same precedent the removed
pre-redesign ``_region_reliability_overview`` helper set).

A second, auxiliary template (:func:`render_secondary_report`, the
"Question Driven" report) is written alongside the main report as
``report_question_driven.html`` — always generated together with
``report.html`` (Application-layer orchestration, ``ssat.application.
application.AuditApplication.generate_report``), but explicitly a
secondary, supplementary view: it reuses the exact same ``ReportModel``
data as the main report (no new computation), just organized as five
plain-language questions (baseline → sensitivity magnitude → spatial
concentration → control & stability → actionable examples) for a
first-time reader, and cross-links back to the main report's detailed
sections rather than duplicating them.

**"Semantic Region Profile" — the 9th section.** Added after every
pre-existing section (including the Provenance appendix), so the earlier
8-section structure and the "Question Driven" report above are unchanged
byte-for-byte in their own content — this section only appends. The
section is gated on ``model.semantic_concentration.n_semantic_groups <=
1``: the common case (no ``regions[].semantic_group`` configured, e.g.
every existing grid run) collapses to a one-line "not applicable" notice
rather than a self-evidently trivial single-group table (mirrored at the
type level by ``SemanticConcentration.__post_init__``). When more than one
semantic group exists, the section renders the executive-style
dominant-group sentence (same non-causal caveat wording as
``#executive-interpretation``), then the ``(gt_label × semantic_group)``
cross-tabulation from ``model.class_semantic_matrix`` as a plain HTML
table whose cells are colored with the same :func:`_heat_color` helper
the Dataset Spatial Pattern heat grid already uses
(:func:`_class_semantic_grid` only reshapes already-computed rows into a
2D grid, the same "no new statistic" role :func:`_grid_layout` plays for
``region_summary``) — a table rather than a CSS grid because, unlike the
spatial-pattern heat grid, this one needs real row/column headers
(``gt_label`` values, ``semantic_group`` names). Each cell shows
``mean_degradation`` and, when the run's primary metric is binary,
``flip_rate``; the ``n_samples`` behind every cell is always available via
its ``title`` attribute so a sparse combination is never mistaken for a
well-evidenced one. The section ends with a plain-text pointer to the
separate ``ssat export-labels`` CLI command — label export is
deliberately not run automatically by ``ssat report``, so this module
never renders a real link to ``labels.jsonl``, only instructions for
generating it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import jinja2

from ssat.report.static import ENHANCE_JS, STYLE_CSS
from ssat.report.types import GRADE_COLORS, ClassSemanticRow, RegionRow, ReportGrade, ReportModel

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

# Max flagged-anchor rows rendered inline in the "Detailed Tables / Flagged
# Anchors" section. Real-dataset runs (e.g. ImageNet-scale) can flag anchors
# in numbers that scale with samples x regions x invert_mask; rendering one
# <li> per anchor with no cap was large enough to push report.html past
# 400MB on such runs — unopenable in VS Code and slow/memory-heavy even in a
# browser. Only this inline HTML list is capped: ReliabilitySpotlight.
# flagged_examples itself, the "N anchor(s) flagged" count (stability-
# controls section), and the full data/flagged_items.csv export (exporter.py)
# all stay untruncated, so no information is lost — it just isn't all
# inlined into the HTML.
_FLAGGED_ANCHORS_DISPLAY_LIMIT = 20


@dataclass(frozen=True, slots=True)
class ReportManifestPaths:
    """Where :func:`render_report` wrote each of its four output files.

    Attributes:
        report_html: The rendered report page.
        style_css: ``assets/css/style.css``, written verbatim from
            ``ssat.report.static.STYLE_CSS``.
        enhance_js: ``assets/js/enhance.js``, written verbatim from
            ``ssat.report.static.ENHANCE_JS``.
        report_manifest_json: ``report_manifest.json`` (``report_schema_version,
            source_manifest_hashes, top_k, bottom_k, generated_at``).
    """

    report_html: Path
    style_css: Path
    enhance_js: Path
    report_manifest_json: Path


def render_report(
    model: ReportModel, output_dir: Path, *, top_k: int, bottom_k: int
) -> ReportManifestPaths:
    """Write ``report.html`` + static assets + ``report_manifest.json``.

    Args:
        model: A ``ReportModel`` whose asset-ref fields are already filled
            in by earlier orchestration (module docstring) — this function
            never renders a chart or an image itself, only templates the
            refs it is given.
        output_dir: The report root (``<run_dir>/report/``, matching this
            package's already-established output layout); created if
            missing. Every path this function writes, and every
            ``href``/``src`` the rendered HTML emits, is relative to this
            directory.
        top_k: The top-K sample count the report was configured with, for
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
    """Build ``report_manifest.json``'s content."""

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

    ``autoescape=True`` — user-provided strings can end up interpolated
    into the templates, so escaping stays on even for an offline report, to
    prevent XSS/markup breakage. Every value interpolated with ``{{ }}`` is
    HTML-escaped unless explicitly marked safe (nothing in these templates
    does that).
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
    environment.filters["class_semantic_grid"] = _class_semantic_grid
    environment.filters["heat_color"] = _heat_color
    environment.globals["grade_color"] = _grade_color
    environment.globals["unreliable_badge_color"] = GRADE_COLORS[ReportGrade.UNRELIABLE]
    environment.globals["ReportGrade"] = ReportGrade
    environment.globals["flagged_anchors_display_limit"] = _FLAGGED_ANCHORS_DISPLAY_LIMIT
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
        anchors evaluated for this region — the template shows a "no
        evaluation" placeholder instead of an empty bar).
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


def _class_semantic_grid(
    rows: Sequence[ClassSemanticRow],
) -> Mapping[str, object] | None:
    """Reshape ``class_semantic_matrix`` into a ``(gt_label × semantic_group)`` grid.

    Same pure-reshaping role as :func:`_grid_layout` (module docstring): a
    ``ClassSemanticRow`` tuple is already every cell of this cross-tab
    (``ssat.report.assembler``) — this only lays it out as a 2D table a
    template can loop over by row/column, it does not compute anything the
    rows did not already carry.

    Returns:
        ``None`` when ``rows`` is empty. Otherwise a mapping with
        ``"gt_labels"``/``"semantic_groups"`` (sorted axis labels),
        ``"cells"`` (row-major 2D list of ``ClassSemanticRow | None`` —
        ``None`` for any ``(gt_label, semantic_group)`` combination absent
        from ``rows``, e.g. a class with zero samples in that group), and
        ``"max_mean_degradation"`` (the heat-color scale's denominator,
        ``None`` when every row's ``mean_degradation`` is ``None``).
    """

    if not rows:
        return None

    gt_labels = sorted({row.gt_label for row in rows})
    semantic_groups = sorted({row.semantic_group for row in rows})
    by_key = {(row.gt_label, row.semantic_group): row for row in rows}
    cells = [
        [by_key.get((gt_label, semantic_group)) for semantic_group in semantic_groups]
        for gt_label in gt_labels
    ]
    degradations = [row.mean_degradation for row in rows if row.mean_degradation is not None]
    return {
        "gt_labels": gt_labels,
        "semantic_groups": semantic_groups,
        "cells": cells,
        "max_mean_degradation": max(degradations) if degradations else None,
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
        return "Yes" if value else "No"
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
        return "No distribution"
    return ", ".join(f"{key}:{count}" for key, count in sorted(distribution.items()))


# --- templates --------------------------------------------------------------


_MACROS_TEMPLATE = """
{% macro grade_badge(grade, title=none) %}
<span class="badge" style="background-color: {{ grade_color(grade) }};"
      title="{{ title if title else (grade.value if grade else 'No grade') }}">{{
  grade.value | upper if grade else 'N/A' }}</span>
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
<span class="no-data">No grade</span>
{% endif %}
{% endmacro %}

{% macro sample_card(card) %}
<article class="sample-card">
  <div class="sample-card-images">
    {% if card.thumbnail_asset_ref %}
    <img src="{{ card.thumbnail_asset_ref }}" alt="{{ card.sample_id }} original thumbnail">
    {% else %}
    <div class="no-image">No original</div>
    {% endif %}
    {% if card.heatmap_asset_ref %}
    <img src="{{ card.heatmap_asset_ref }}" alt="{{ card.sample_id }} spatial heatmap overlay">
    {% else %}
    <div class="no-image">No heatmap</div>
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
      Most vulnerable region: {{ top.region_key }} ({{ top.degradation | fmt }})
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
<html lang="en">
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
    <div><dt>Model</dt><dd>{{ model.run_summary.model_id }}</dd></div>
    <div><dt>Preprocessing</dt><dd>{{ model.run_summary.preprocessing_desc }}</dd></div>
    <div><dt>Sample count</dt><dd>{{ model.run_summary.n_samples }}</dd></div>
    <div><dt>Regions per sample</dt><dd>{{ model.run_summary.n_regions_per_sample }}</dd></div>
    <div><dt>Condition count</dt><dd>{{ model.run_summary.n_conditions }}</dd></div>
    <div><dt>Duration</dt><dd>{{ model.run_summary.duration_seconds | fmt_duration }}</dd></div>
    <div><dt>Failure rate</dt><dd>{{ model.run_summary.failure_rate | fmt_pct }}</dd></div>
    <div><dt>Generated at</dt><dd>{{ model.meta.generated_at }}</dd></div>
    <div><dt>run_id</dt><dd>{{ model.meta.run_id }}</dd></div>
  </dl>
  <p class="section-note">
    A question-driven <a href="report_question_driven.html">secondary report</a> is
    also provided — this page is the main report.
  </p>
</header>

<section id="executive-interpretation" class="hero">
  <div class="hero-label">The first sentence to read from this run</div>
  {% set share = model.spatial_concentration.dominant_region_share %}
  {% if share is none %}
  <h2>No data available to judge spatial concentration.</h2>
  <p class="hero-muted">
    No samples have a valid degradation value, so dominant-region share/spatial entropy cannot be computed.
  </p>
  {% elif share >= 0.5 %}
  <h2>Sensitivity is relatively concentrated at <strong>{{ model.spatial_concentration.dominant_region_key }}</strong>.</h2>
  <p class="hero-muted">
    Of the {{ model.spatial_concentration.n_scored_samples }} samples for which a top region could be
    determined, {{ share | fmt_pct }} named this location as the most vulnerable region.
  </p>
  {% else %}
  <h2>Sensitivity exists, but is <strong>not concentrated at one fixed location</strong>.</h2>
  <p class="hero-muted">
    Even the most frequently named location (<code>{{ model.spatial_concentration.dominant_region_key }}</code>)
    only reaches {{ share | fmt_pct }} — individual samples' vulnerable locations differ from one another.
  </p>
  {% endif %}
  <p class="callout">
    This sentence only describes a pattern observed in this run — it does not automatically judge
    whether a shortcut exists or its cause; confirming the actual cause requires follow-up verification.
  </p>
  <p class="hero-muted">
    Below, <code>vulnerability_score</code> is a <strong>ranking</strong> axis expressing how much this
    sample/region degrades, while <code>reliability_grade</code> is a separate <strong>reliability</strong>
    axis expressing <strong>how much that number can be trusted</strong> — the two values are independent
    of each other.
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
      <div class="metric-note">N/A: no samples could be determined.</div>
      {% endif %}
    </div>
    <div class="metric-card">
      <div class="metric-label">Spatial Entropy</div>
      <div class="metric-value">{{ model.spatial_concentration.spatial_entropy | fmt }}</div>
      <div class="metric-note">0 = concentrated at one location, 1 = fully spread out</div>
    </div>
  </div>

  <h3>Variation hidden behind the average</h3>
  {% if model.vulnerability_distribution.histogram_asset_ref %}
  <img class="chart" src="{{ model.vulnerability_distribution.histogram_asset_ref }}"
       alt="vulnerability_score histogram">
  {% else %}
  <p class="no-data">No histogram asset</p>
  {% endif %}
  {% set stats = model.vulnerability_distribution.summary_stats %}
  <p class="callout">
    Mean degradation is {{ stats.mean | fmt }}, but individual samples range from p90
    {{ stats.p90 | fmt }} to p99 {{ stats.p99 | fmt }} — check the outliers in the gallery below.
  </p>
  <dl class="stats-list">
    <div><dt>Mean</dt><dd>{{ stats.mean | fmt }}</dd></div>
    <div><dt>Median</dt><dd>{{ stats.median | fmt }}</dd></div>
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
        Darker means that location was the most vulnerable region for more samples —
        this is not the reliability grade.
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
        Darker means a higher share of anchors at that location were graded HIGH.
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
    This run's regions are not laid out as a grid, so the heat-grid is skipped — see the
    "Region Summary" table below instead.
  </p>
  {% endif %}
</section>

<section id="region-summary">
  <h2>Region Summary</h2>
  <p class="section-note">
    The bars below show the <strong>grade mix</strong> across every anchor
    (sample × region × invert_mask) sharing this region. The worst-case-only display, which
    collapses to "one worst grade," is used only at the sample/anchor level (the large badges
    in the gallery below) and is not applied to this dataset-level table — so that a region
    where most anchors are HIGH doesn't look entirely UNRELIABLE just because one anchor is.
    The worst-case value itself is still visible by hovering over each bar (title), and remains
    in <a href="data/region_summary.csv">region_summary.csv</a>.
  </p>
  <ul class="grade-legend">
    {{ macros.grade_legend_item(ReportGrade.HIGH,
      "Passed every reliability check (exceeds control, reproduced across 2+ fill strategies, bootstrap CI excludes zero) — both direction and magnitude are confirmed.") }}
    {{ macros.grade_legend_item(ReportGrade.MODERATE,
      "Direction (sign) agrees across fill strategies, but only some of the core checks (exceeds control, multi-strategy reproduction, confidence interval) are met.") }}
    {{ macros.grade_legend_item(ReportGrade.LOW,
      "Direction agrees across fill strategies, but the core checks are barely met or the evidence is weak — the effect is small or not yet well verified.") }}
    {{ macros.grade_legend_item(ReportGrade.UNRELIABLE,
      "Different fill strategies disagree on whether removing this location makes performance better or worse — the sign, not the magnitude, is inconsistent, so this value cannot be trusted.") }}
  </ul>
  {% if model.region_summary.chart_asset_ref %}
  <img class="chart" src="{{ model.region_summary.chart_asset_ref }}" alt="Bar chart of mean degradation by region">
  {% endif %}
  <table id="region-table">
    <caption>Mean degradation and grade mix by region. Click a header to sort.</caption>
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
      <tr><td colspan="5" class="no-data">No data</td></tr>
      {% endfor %}
    </tbody>
  </table>
</section>

<section id="gallery">
  <h2>Vulnerable Samples — Where the model looks</h2>
  <p class="section-note">
    The large grade badge on each card is the worst grade across every region the sample
    covers (worst-case) — a different axis from the grade-mix bars in Region Summary above.
    See <a href="#region-summary">Region Summary</a> for details.
  </p>
  <h3>Most vulnerable samples (top-{{ top_k }})</h3>
  {% if model.sample_rankings.most_vulnerable %}
  <div class="gallery-grid">
    {% for card in model.sample_rankings.most_vulnerable %}{{ macros.sample_card(card) }}{% endfor %}
  </div>
  {% else %}
  <p class="no-data">No data</p>
  {% endif %}

  <h3>Most robust samples (bottom-{{ bottom_k }})</h3>
  {% if model.sample_rankings.most_robust %}
  <div class="gallery-grid">
    {% for card in model.sample_rankings.most_robust %}{{ macros.sample_card(card) }}{% endfor %}
  </div>
  {% else %}
  <p class="no-data">No data</p>
  {% endif %}
</section>

<section id="stability-controls">
  <h2>Stability / Controls</h2>
  <p class="section-note">
    The "Mean Z vs Control" card in Behavioral Fingerprint above summarizes the excess over
    control. This section shows the assets behind that judgment and a reliability spotlight
    summary.
  </p>
  {% if model.fill_strategy_correlation_asset_ref %}
  <h3>Fill Strategy Rank Correlation</h3>
  <img class="chart" src="{{ model.fill_strategy_correlation_asset_ref }}"
       alt="Rank correlation heatmap between fill strategy pairs">
  {% else %}
  <p class="no-data">
    No fill-strategy correlation asset is available (either the run did not exercise multiple
    fill strategies, or analysis was not run).
  </p>
  {% endif %}
  <p>
    <strong>{{ model.reliability_spotlight.flagged_examples | length }}</strong> anchor(s) are
    currently flagged UNRELIABLE — see "Detailed Tables / Flagged Anchors" below for the full
    list.
  </p>
</section>

<section id="flagged-anchors">
  <h2>Detailed Tables / Flagged Anchors</h2>
  {% if model.reliability_spotlight.flagged_examples %}
  {% set total_flagged = model.reliability_spotlight.flagged_examples | length %}
  {% set displayed_flagged = model.reliability_spotlight.flagged_examples[:flagged_anchors_display_limit] %}
  {% if total_flagged > displayed_flagged | length %}
  <p class="section-note">
    Showing the first {{ displayed_flagged | length }} of {{ total_flagged }} flagged
    anchors. The complete list is in
    <a href="data/flagged_items.csv">flagged_items.csv</a> — see
    <a href="#provenance">Provenance</a>.
  </p>
  {% endif %}
  <ul class="flagged-list">
    {% for item in displayed_flagged %}
    <li>
      <span class="badge" style="background-color: {{ unreliable_badge_color }};"
            title="{{ item.reliability_reasons | join('; ') }}">UNRELIABLE</span>
      <span class="flagged-anchor">{{ item.anchor_key_repr }}</span>
      <span class="flagged-reason">{{ item.reason_summary }}</span>
    </li>
    {% endfor %}
  </ul>
  {% else %}
  <p class="no-data">No flagged items</p>
  {% endif %}
</section>

<details id="provenance">
  <summary>Provenance (click to expand)</summary>
  <dl>
    <div><dt>dump_path</dt><dd>{{ model.provenance.dump_path }}</dd></div>
    <div><dt>metrics_dir</dt><dd>{{ model.provenance.metrics_dir }}</dd></div>
    <div><dt>analysis_dir</dt><dd>{{ model.provenance.analysis_dir or 'N/A' }}</dd></div>
    <div><dt>run_manifest_hash</dt><dd><code>{{ model.provenance.run_manifest_hash }}</code></dd></div>
    <div><dt>metrics_manifest_hash</dt><dd><code>{{ model.provenance.metrics_manifest_hash }}</code></dd></div>
    <div><dt>analysis_manifest_hash</dt>
      <dd><code>{{ model.provenance.analysis_manifest_hash or 'N/A' }}</code></dd></div>
    <div><dt>schema_versions</dt>
      <dd>dump {{ model.meta.schema_versions.dump }} / metrics {{ model.meta.schema_versions.metrics }} /
        analysis {{ model.meta.schema_versions.analysis or 'N/A' }} /
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
  <h4>Download raw data</h4>
  <ul class="download-links">
    <li><a href="data/report_model.json">report_model.json</a></li>
    <li><a href="data/sample_rankings.csv">sample_rankings.csv</a></li>
    <li><a href="data/region_summary.csv">region_summary.csv</a></li>
    <li><a href="data/flagged_items.csv">flagged_items.csv</a></li>
    <li><a href="data/semantic_summary.csv">semantic_summary.csv</a></li>
    <li><a href="data/class_semantic_matrix.csv">class_semantic_matrix.csv</a></li>
  </ul>
</details>

<section id="semantic-profile">
  <h2>Semantic Region Profile</h2>
  {% if model.semantic_concentration.n_semantic_groups <= 1 %}
  <p class="no-data">
    This run has no semantic region grouping configured beyond grid coordinates
    (<code>regions[].semantic_group</code> is unset) — this section has nothing to show.
  </p>
  {% else %}
  {% set semantic_share = model.semantic_concentration.dominant_semantic_group_share %}
  <p class="section-note">
    {% if semantic_share is none %}
    No data available to judge semantic region concentration.
    {% elif semantic_share >= 0.5 %}
    Sensitivity is relatively concentrated at the <strong>{{ model.semantic_concentration.dominant_semantic_group }}</strong>
    region ({{ semantic_share | fmt_pct }}, based on
    {{ model.semantic_concentration.n_scored_samples }} samples).
    {% else %}
    Sensitivity exists, but <strong>is not concentrated at one fixed region</strong> — even the
    most frequently named region
    (<code>{{ model.semantic_concentration.dominant_semantic_group }}</code>) only reaches
    {{ semantic_share | fmt_pct }}.
    {% endif %}
  </p>
  <p class="callout">
    This sentence only describes a pattern observed in this run — it does not automatically judge
    whether a shortcut exists or its cause; confirming the actual cause requires follow-up verification.
  </p>

  <h3>Class × Semantic-group</h3>
  <p class="section-note">
    Rows are the ground-truth label (gt_label), columns are the semantic region
    (semantic_group) — a darker cell means higher degradation (mean_degradation) when that
    region is masked for that label. Combinations with few samples should be read as
    reference only — hover over a cell (title) to see n_samples.
  </p>
  {% set grid = model.class_semantic_matrix | class_semantic_grid %}
  {% if grid %}
  <table>
    <caption>gt_label × semantic_group cross-tab (mean_degradation; flip_rate is also shown for binary metrics)</caption>
    <thead>
      <tr>
        <th>gt_label</th>
        {% for semantic_group in grid.semantic_groups %}
        <th>{{ semantic_group }}</th>
        {% endfor %}
      </tr>
    </thead>
    <tbody>
      {% for gt_label in grid.gt_labels %}
      <tr>
        <th>{{ gt_label }}</th>
        {% for cell in grid.cells[loop.index0] %}
        {% if cell %}
        <td style="background-color: {{ cell.mean_degradation | heat_color(grid.max_mean_degradation) }};"
            title="n_samples={{ cell.n_samples }}">
          {{ cell.mean_degradation | fmt }}{% if cell.flip_rate is not none %} (flip {{ cell.flip_rate | fmt_pct }}){% endif %}
        </td>
        {% else %}
        <td class="no-data">—</td>
        {% endif %}
        {% endfor %}
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% else %}
  <p class="no-data">No data</p>
  {% endif %}

  <p class="section-note">
    If you need a per-sample label file (for a downstream training pipeline), run
    <code>ssat export-labels &lt;report_dir&gt;</code> separately from this report —
    <code>ssat report</code> does not generate a label file automatically. Running it creates
    <code>labels.jsonl</code> and <code>labels_manifest.json</code> under this report directory.
  </p>
  {% endif %}
</section>

<script src="assets/js/enhance.js" defer></script>
</body>
</html>
"""


_REPORT_TEMPLATE_B = """
{% import "macros.html" as macros %}
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ model.run_summary.dataset_name }} — SSAT Report (Question Driven)</title>
<link rel="stylesheet" href="assets/css/style.css">
</head>
<body>

<header class="run-header">
  <h1>{{ model.run_summary.dataset_name }} — A question-driven secondary report</h1>
  <p class="hero-muted">
    This page is a secondary report alongside the <a href="report.html">main report
    (report.html)</a> — the same data, simply reordered around the five questions someone
    seeing it for the first time would actually ask. No values here are newly computed.
  </p>
</header>

{% set accuracy_card = model.scorecard | selectattr("key", "equalto", "accuracy") | first %}
<section class="section-block">
  <div class="q">
    <div class="qnum">1</div>
    <div>
      <div class="hero-label">Baseline</div>
      <div class="q-answer">How well does this model perform before any perturbation?</div>
      <p>
        <span class="pill">Answer</span> Clean accuracy is
        <b>{{ accuracy_card.value | fmt_pct if accuracy_card.value is not none else 'N/A' }}</b>.
        {% if accuracy_card.note %}({{ accuracy_card.note }}){% endif %}
      </p>
      <p class="hero-muted">N = {{ model.run_summary.n_samples }} · failure rate {{ model.run_summary.failure_rate | fmt_pct }}</p>
    </div>
  </div>
</section>

{% set stats = model.vulnerability_distribution.summary_stats %}
<section class="section-block">
  <div class="q">
    <div class="qnum">2</div>
    <div>
      <div class="hero-label">Sensitivity magnitude</div>
      <div class="q-answer">How much does masking a region shake things up overall?</div>
      <p>
        <span class="pill">Answer</span> Mean degradation is <b>{{ stats.mean | fmt }}</b>, but the
        top samples swing much harder, from p90 {{ stats.p90 | fmt }} to p99 {{ stats.p99 | fmt }}.
      </p>
      {% if model.vulnerability_distribution.histogram_asset_ref %}
      <img class="chart" src="{{ model.vulnerability_distribution.histogram_asset_ref }}"
           alt="vulnerability_score histogram">
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
      <div class="q-answer">Is there one "fixed location" that every sample repeatedly depends on?</div>
      {% if share is none %}
      <p><span class="pill">Answer: N/A</span> No conclusion can be drawn — no samples could be determined.</p>
      {% elif share >= 0.5 %}
      <p>
        <span class="pill">Answer: Yes</span> <code>{{ model.spatial_concentration.dominant_region_key }}</code>
        repeats as the top region in {{ share | fmt_pct }} of samples.
      </p>
      {% else %}
      <p>
        <span class="pill">Answer: Not distinct</span> Even the most frequent top-1 location
        (<code>{{ model.spatial_concentration.dominant_region_key }}</code>) only reaches
        {{ share | fmt_pct }}, and vulnerable locations differ from sample to sample.
      </p>
      {% endif %}
      <div class="grid2">
        <div><div class="metric-value">{{ share | fmt_pct }}</div><div class="hero-muted">Dominant-region share</div></div>
        <div><div class="metric-value">{{ model.spatial_concentration.spatial_entropy | fmt }}</div><div class="hero-muted">Spatial entropy (closer to 1 = more spread out)</div></div>
      </div>
      <p class="callout">
        <b>Important:</b> this does not conclude "there is no HIGH region." It only expresses
        how weak or strong the dataset-wide dependence on a fixed location is — the HIGH
        evidence for individual samples remains intact in question 5 below and in the main
        report's gallery.
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
      <div class="q-answer">Is the observed effect stronger than a chance masking effect, and reproducible?</div>
      <table>
        <thead><tr><th>Check</th><th>Result</th><th>Meaning</th></tr></thead>
        <tbody>
          <tr>
            <td>Same-area control</td>
            <td>{{ control_card.value | fmt if control_card.value is not none else 'N/A' }}</td>
            <td>Mean Z vs Control — higher means a larger effect over control{% if control_card.note %} ({{ control_card.note }}){% endif %}</td>
          </tr>
          <tr>
            <td>Fill agreement</td>
            <td>{% if model.fill_strategy_correlation_asset_ref %}Chart available{% else %}N/A{% endif %}</td>
            <td>Rank correlation across fill strategies — see the linked chart below</td>
          </tr>
          <tr>
            <td>UNRELIABLE anchors</td>
            <td>{{ model.reliability_spotlight.flagged_examples | length }}</td>
            <td>Number of anchors where fill strategies disagree on direction — see the main report's detailed list</td>
          </tr>
        </tbody>
      </table>
      {% if model.fill_strategy_correlation_asset_ref %}
      <p><a href="report.html#stability-controls">See the correlation chart in the main report →</a></p>
      {% endif %}
    </div>
  </div>
</section>

<section class="section-block">
  <div class="q">
    <div class="qnum">5</div>
    <div>
      <div class="hero-label">Actionable examples</div>
      <div class="q-answer">So which samples should actually be investigated first?</div>
      {% set examples = model.sample_rankings.most_vulnerable[:5] %}
      {% if examples %}
      <table>
        <thead><tr><th>Sample</th><th>Top region</th><th>Score</th><th>Evidence</th></tr></thead>
        <tbody>
          {% for card in examples %}
          <tr>
            <td>{{ card.sample_id }}</td>
            <td>{{ card.top_regions[0].region_key if card.top_regions else 'N/A' }}</td>
            <td>{{ card.vulnerability_score | fmt }}</td>
            <td>{{ macros.grade_badge(card.reliability_grade) }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% else %}
      <p class="no-data">No data</p>
      {% endif %}
      <p><a href="report.html#gallery">See the full gallery in the main report →</a></p>
    </div>
  </div>
</section>

<section class="section-block">
  <details>
    <summary><b>Expert detail: region table / fill correlation / flagged anchors / provenance</b></summary>
    <p class="hero-muted">
      This secondary report holds only a summary — the per-region grade-mix table, detailed
      correlation chart, full flagged-anchor list, and provenance/raw CSV·JSON all remain in the
      <a href="report.html">main report (report.html)</a>.
    </p>
  </details>
</section>

</body>
</html>
"""
