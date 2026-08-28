"""Chart renderer: server-side SVG charts for a ReportModel.

Three charts, each a pure function from already-assembled ``report.types``
data to an SVG string — no new statistics, only visualization of numbers
the assembler already computed: the vulnerability score histogram (full
population, not just top-K — this is *the* section that differentiates
this tool from a single-scalar benchmark table), the region degradation
bar chart (colored by :data:`ssat.report.types.GRADE_COLORS` so a bar's
color always agrees with the same region's badge in the rendered HTML),
and the optional fill-strategy rank correlation heatmap.

**No pyplot, ever.** Every function builds a bare ``matplotlib.figure.
Figure()`` directly and never touches ``matplotlib.pyplot`` — the same
choice ``ssat/metrics/viz/heatmap.py`` already made (its module docstring/
``_save_view_png``), which sidesteps the risk of a leaked,
never-``plt.close()``d global figure entirely rather than merely
mitigating it: a ``Figure()`` never registers with pyplot's global figure
registry, so there is nothing to leak or close.

**Reproducibility.** The same input must always produce the same SVG
bytes. matplotlib's default SVG output is *not* reproducible out of the
box — confirmed empirically before writing this module — for two
independent reasons, both neutralized via ``matplotlib.rc_context``
(scoped to each render call, so this module never mutates global
rcParams for any other matplotlib caller in the same process):

- ``rcParams["svg.hashsalt"]`` defaults to ``None``, under which clip-path/
  gradient element ids are derived from ``id(obj)`` (a memory address) and
  differ between calls in the *same* process. Pinning it to a fixed string
  makes those ids a deterministic hash instead.
- ``rcParams["svg.fonttype"]`` defaults to embedding text as ``<text>``
  elements that depend on system/embedded fonts being available to whatever
  later renders the SVG — a runtime dependency the project's fully-offline
  principle does not want. Setting it to ``"path"`` converts every glyph
  into a vector ``<path>`` at render time instead, so no font embedding is
  needed.

Separately, ``fig.savefig(..., metadata=...)`` is passed ``{"Date": None,
"Creator": None, "Format": None, "Type": None}`` to suppress matplotlib's
default embedded Dublin-Core RDF block — which otherwise embeds the current
wall-clock timestamp (breaking reproducibility on its own, independent of
the hashsalt issue above) and a literal ``https://matplotlib.org/`` credit
line.

**What "no external references" means here.** Even after every
suppression above, a spec-valid SVG 1.1 document still contains exactly
three ``http://`` occurrences: the DOCTYPE's DTD URI and the
``xmlns``/``xmlns:xlink`` namespace declarations mandated by the SVG/XML
specs themselves — no renderer ever fetches these over the network, they
are inert namespace identifiers, and no post-processing attempts to strip
them (doing so would produce a non-standard document for no functional
gain). "No external references" is enforced at the level that actually
matters: no attribute value that points at an external resource
(``href``/``src`` starting with ``http(s)://``, no ``<script>``/``<image>``
tags) — see ``tests/unit/test_report_charts.py``.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Protocol

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ssat.report.types import GRADE_COLORS, RegionRow, ReportGrade, SampleCard

# Fixed so ``xlink``/clip-path/gradient ids are a deterministic hash of this
# string plus an internal counter, never ``id(obj)`` (a memory address).
_SVG_RC_PARAMS = {"svg.fonttype": "path", "svg.hashsalt": "ssat-report-charts-v1"}

# Suppresses matplotlib's default embedded Dublin-Core metadata block (see
# module docstring) — every one of these four keys has a built-in default
# (a wall-clock timestamp, "Matplotlib vX.Y, https://matplotlib.org/", ...)
# that is only removed by explicitly setting it to ``None``.
_SVG_METADATA = {"Date": None, "Creator": None, "Format": None, "Type": None}

_HISTOGRAM_COLOR = "#455a64"
_NO_GRADE_COLOR = "#9e9e9e"  # distinct from GRADE_COLORS[LOW] — "no analysis run", not "LOW"

# English, unlike some user-facing notes elsewhere in this layer (e.g.
# assembler.py's MetricCard.note strings, which may contain non-Latin
# text): with svg.fonttype="path" (module docstring), matplotlib bakes
# this text into vector glyph outlines at render time using whatever font
# it has loaded locally (DejaVu Sans, confirmed empirically to have no
# Hangul coverage) — unlike the rendered HTML's notes, which a browser
# renders with its own font stack, so non-Latin text there costs nothing.
# Non-Latin text here would silently render as blank glyphs.
_NO_DATA_LABEL = "No data"


class RankCorrelationRowLike(Protocol):
    """Structural shape this module reads off one fill-strategy rank correlation row.

    Matches ``ssat.analysis.types.RankCorrelationRow`` field-for-field
    without importing it — the same duck-typing trade
    ``report.adapters.SampleMetricLike``/``report.exporter.
    AssembledReportLike`` already made for the same reason.

    Attributes:
        op_a: First perturb_op in the pair.
        op_b: Second perturb_op in the pair.
        spearman: Spearman correlation of per-region degradation ranks
            between op_a and op_b; ``None`` when not computable.
    """

    op_a: str
    op_b: str
    spearman: float | None


def render_vulnerability_histogram(
    full_sample_rankings: Sequence[SampleCard], *, bins: int = 20, seed: int = 0
) -> str:
    """Render the full vulnerability_score distribution as an SVG histogram.

    Takes ``full_sample_rankings`` (every sample) rather than
    ``ReportModel.sample_rankings`` (top-K/bottom-K only) — a histogram of
    only the extremes cannot represent a distribution.

    Args:
        full_sample_rankings: Every sample from :class:`ssat.report.
            assembler.AssembledReport`, in any order.
        bins: Histogram bin count.
        seed: Reserved for future jittered/randomized visualizations; this
            histogram's binning has no random component today, so ``seed``
            is currently unused.

    Returns:
        A standalone, reproducible SVG document as ``str``.
    """

    del seed
    scores = [
        card.vulnerability_score
        for card in full_sample_rankings
        if card.vulnerability_score is not None
    ]

    with matplotlib.rc_context(_SVG_RC_PARAMS):
        figure = Figure(figsize=(8, 5))
        axis = figure.add_subplot(111)
        if scores:
            axis.hist(scores, bins=bins, color=_HISTOGRAM_COLOR, edgecolor="white")
        else:
            _draw_no_data(axis)
        axis.set_xlabel("vulnerability_score")
        axis.set_ylabel("sample count")
        axis.set_title("Vulnerability Score Distribution")
        figure.tight_layout()
        return _figure_to_svg(figure)


def render_region_bar(region_summary_rows: Sequence[RegionRow]) -> str:
    """Render each region's mean degradation as a grade-colored bar chart.

    Bar color matches :data:`ssat.report.types.GRADE_COLORS` — the exact
    palette the rendered HTML's badges use — so a region's bar and its
    table row badge never disagree. A row whose ``mean_degradation`` is
    ``None`` (no valid items for that region) is omitted from the plotted
    bars entirely rather than drawn at height 0, which would misrepresent
    "no data" as "zero degradation" (the same unavailable-≠-false principle
    every other part of this layer follows).

    Args:
        region_summary_rows: One row per region, e.g. ``ReportModel.
            region_summary.rows``.

    Returns:
        A standalone, reproducible SVG document as ``str``.
    """

    plottable = [row for row in region_summary_rows if row.mean_degradation is not None]

    with matplotlib.rc_context(_SVG_RC_PARAMS):
        width = max(8.0, 0.4 * len(plottable))
        figure = Figure(figsize=(width, 5))
        axis = figure.add_subplot(111)
        if plottable:
            positions = range(len(plottable))
            axis.bar(
                positions,
                [row.mean_degradation for row in plottable],
                color=[_grade_color(row.reliability_grade) for row in plottable],
            )
            axis.set_xticks(list(positions))
            axis.set_xticklabels([_short_region_key(row.region_key) for row in plottable], rotation=90)
        else:
            _draw_no_data(axis)
        axis.set_ylabel("mean_degradation")
        axis.set_title("Region Mean Degradation")
        figure.tight_layout()
        return _figure_to_svg(figure)


def render_fill_strategy_correlation(
    rank_correlation_rows: Sequence[RankCorrelationRowLike],
) -> str | None:
    """Render an op×op Spearman rank-correlation heatmap, when available.

    Only offered when the adapter's ``applicable_charts()`` includes
    ``"fill_strategy_correlation_heatmap"`` — the empty input this
    function then sees when fill-strategy stability was not computed is
    handled here too, by returning ``None`` before any matplotlib object
    is created.

    Args:
        rank_correlation_rows: ``rank_correlation.parquet`` rows, unfiltered
            (this is the raw op-pair data, not a summary).

    Returns:
        A standalone, reproducible SVG document as ``str``, or ``None`` if
        ``rank_correlation_rows`` is empty.
    """

    if not rank_correlation_rows:
        return None

    ops = sorted({row.op_a for row in rank_correlation_rows} | {row.op_b for row in rank_correlation_rows})
    index = {op: position for position, op in enumerate(ops)}
    matrix = np.full((len(ops), len(ops)), np.nan)
    np.fill_diagonal(matrix, 1.0)
    for row in rank_correlation_rows:
        if row.spearman is None:
            continue
        i, j = index[row.op_a], index[row.op_b]
        matrix[i, j] = row.spearman
        matrix[j, i] = row.spearman

    with matplotlib.rc_context(_SVG_RC_PARAMS):
        size = max(4.0, 0.6 * len(ops))
        figure = Figure(figsize=(size, size))
        axis = figure.add_subplot(111)
        image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.set_xticks(range(len(ops)))
        axis.set_xticklabels(ops, rotation=90)
        axis.set_yticks(range(len(ops)))
        axis.set_yticklabels(ops)
        figure.colorbar(image, ax=axis, label="Spearman rank correlation")
        figure.tight_layout()
        return _figure_to_svg(figure)


# --- shared helpers -----------------------------------------------------------


def _grade_color(grade: ReportGrade | None) -> str:
    """Look up a grade's shared color, or a distinct "no analysis" gray for ``None``."""

    if grade is None:
        return _NO_GRADE_COLOR
    return GRADE_COLORS[grade]


def _short_region_key(region_key: str) -> str:
    """Drop the redundant leading ``"<region_id>::"`` for display only.

    Duplicated from ``report.html_renderer._short_region_key`` (mirrors
    ``_grade_color``/``_NO_GRADE_COLOR`` above) rather than imported, since
    this module's charts are consumed by ``html_renderer`` and not the
    other way around. See that function's docstring for why the split is
    always safe. A key with no ``"::"`` is returned unchanged.
    """

    return region_key.split("::", 1)[-1] if "::" in region_key else region_key


def _draw_no_data(axis: Axes) -> None:
    """Render an explicit "no data" placeholder rather than an empty, unexplained chart."""

    axis.text(0.5, 0.5, _NO_DATA_LABEL, ha="center", va="center", transform=axis.transAxes)


def _figure_to_svg(figure: Figure) -> str:
    """Serialize one figure to a reproducible, offline-safe SVG string (module docstring)."""

    buffer = io.BytesIO()
    figure.savefig(buffer, format="svg", metadata=_SVG_METADATA)
    return buffer.getvalue().decode("utf-8")
