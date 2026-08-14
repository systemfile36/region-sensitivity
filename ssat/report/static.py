"""R4 static assets: CSS/JS bundled as Python string constants (design §R4, §3.1).

Two constants, ``STYLE_CSS``/``ENHANCE_JS``, written verbatim to
``<output_dir>/assets/{css/style.css,js/enhance.js}`` by
``ssat.report.html_renderer``. They live here — rather than as real
``.css``/``.js`` files under ``ssat/report/`` — because the repository has
never had a non-``.py`` source file (IMPLE_PLAN_REPORTING_v1.md §3.1: `find
ssat -type f ! -name "*.py"` is empty besides ``__pycache__``), and adding
one would require new packaging config (``package-data``/``MANIFEST.in``)
this stage's plan deliberately avoids. The *rendered report folder* is, of
course, real ``.css``/``.js`` files — this module only holds their source.

**Zero dependencies (design §3.3 "report.static → (없음)").** Neither
constant references anything outside itself: no ``@import url(...)``, no
``<script src="http...">``-equivalent, no web fonts — every rule uses the
OS's own font stack, keeping the rendered report fully offline (design §0,
§R4 "폰트·아이콘·CSS 전부 로컬 번들. 외부 요청 0건").

**Badge colors are not hardcoded here.** ``ssat.report.types.GRADE_COLORS``
is the single source of truth for reliability-grade colors (that module's
own docstring), shared between R2's chart bars and R4's HTML badges so the
two never drift apart. Rather than duplicating those hex values into this
dependency-free module, every badge gets its color from an inline
``style="background-color: ..."`` attribute that ``html_renderer.py``
computes per-badge from ``GRADE_COLORS`` at render time (see that module's
``_grade_color`` Jinja global) — this stylesheet only defines the badge's
*shape* (padding, radius, font), not its color.
"""

from __future__ import annotations


STYLE_CSS = """
:root {
  --fg: #1b1f23;
  --muted: #5f6b7a;
  --border: #d9dee3;
  --bg: #ffffff;
  --bg-alt: #f5f7f9;
  --accent: #1565c0;
  --font-stack: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans KR",
    "Apple SD Gothic Neo", "Malgun Gothic", Roboto, Helvetica, Arial, sans-serif;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  padding: 0 1.5rem 3rem;
  font-family: var(--font-stack);
  color: var(--fg);
  background: var(--bg);
  line-height: 1.5;
}

h1, h2, h3, h4 { line-height: 1.25; }

a { color: var(--accent); }

.no-data {
  color: var(--muted);
  font-style: italic;
}

/* --- header ------------------------------------------------------------ */

.run-header {
  padding: 1.5rem 0 1rem;
  border-bottom: 1px solid var(--border);
}

.run-meta {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 0.75rem 1.5rem;
  margin: 1rem 0 0;
}

.run-meta > div {
  display: flex;
  flex-direction: column;
}

.run-meta dt {
  font-size: 0.75rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

.run-meta dd {
  margin: 0.15rem 0 0;
  font-weight: 600;
}

/* --- section shell ------------------------------------------------------ */

section, details {
  margin: 2rem 0;
}

.callout {
  background: var(--bg-alt);
  border-left: 4px solid var(--accent);
  padding: 0.75rem 1rem;
  margin: 1rem 0;
}

.chart {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 1rem 0;
  border: 1px solid var(--border);
}

.stats-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 0.75rem 1.5rem;
}

.stats-list dt { font-size: 0.75rem; color: var(--muted); }
.stats-list dd { margin: 0.15rem 0 0; font-weight: 600; }

.section-note {
  color: var(--muted);
  font-size: 0.85rem;
  margin: 0.25rem 0 1rem;
}

/* --- report guide (legend) ------------------------------------------------ */

.grade-legend {
  list-style: none;
  padding: 0;
  margin: 1rem 0;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 0.75rem 1.5rem;
}

.grade-legend-item {
  display: grid;
  grid-template-columns: 0.9rem 1fr;
  column-gap: 0.6rem;
}

.grade-swatch {
  width: 0.9rem;
  height: 0.9rem;
  border-radius: 999px;
  margin-top: 0.25rem;
}

.grade-legend-label {
  grid-column: 2;
  font-weight: 700;
  font-size: 0.85rem;
}

.grade-legend-meaning {
  grid-column: 2;
  color: var(--muted);
  font-size: 0.85rem;
}

/* --- scorecard ----------------------------------------------------------- */

.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1rem;
}

.metric-card {
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.9rem 1rem;
  background: var(--bg-alt);
}

.metric-label { font-size: 0.8rem; color: var(--muted); }
.metric-value { font-size: 1.5rem; font-weight: 700; margin-top: 0.2rem; }
.metric-note { font-size: 0.8rem; color: var(--muted); margin-top: 0.3rem; }

/* --- sample gallery ------------------------------------------------------- */

.gallery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 1rem;
}

.sample-card {
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
  break-inside: avoid;
}

.sample-card-images {
  display: flex;
  gap: 2px;
  background: var(--border);
}

.sample-card-images img, .sample-card-images .no-image {
  flex: 1 1 50%;
  width: 50%;
  height: 130px;
  object-fit: contain;
  background: var(--bg);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  color: var(--muted);
}

.sample-card-body { padding: 0.6rem 0.8rem; }

.sample-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}

.sample-id { font-weight: 700; font-family: monospace; }
.sample-card-score, .sample-card-top-region {
  font-size: 0.85rem;
  margin-top: 0.35rem;
  color: var(--muted);
}

/* --- badges --------------------------------------------------------------- */

.badge {
  display: inline-block;
  color: #fff;
  padding: 0.1em 0.6em;
  border-radius: 999px;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  white-space: nowrap;
}

/* --- tables ---------------------------------------------------------------- */

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}

caption { text-align: left; color: var(--muted); margin-bottom: 0.4rem; }

th, td {
  text-align: left;
  padding: 0.4rem 0.6rem;
  border-bottom: 1px solid var(--border);
}

thead th {
  background: var(--bg-alt);
  position: sticky;
  top: 0;
}

th[data-sort] { user-select: none; }

/* --- reliability spotlight -------------------------------------------------- */

.flagged-list { list-style: none; padding: 0; margin: 0; }

.flagged-list li {
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
  padding: 0.5rem 0;
  border-bottom: 1px solid var(--border);
}

.flagged-anchor { font-family: monospace; font-size: 0.85rem; }
.flagged-reason { color: var(--muted); font-size: 0.85rem; }

/* --- provenance -------------------------------------------------------------- */

details > summary {
  cursor: pointer;
  font-weight: 700;
}

details dl {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 0.3rem 1rem;
  margin: 0.75rem 0;
}

details dt { color: var(--muted); }
details code { word-break: break-all; }

.download-links { padding-left: 1.2rem; }

/* --- print ------------------------------------------------------------------- */

@media print {
  body { padding: 0; }
  a { color: inherit; text-decoration: none; }
  .sample-card, .metric-card { break-inside: avoid; }
  thead th { position: static; }
  /* Force <details> content visible on paper even when collapsed on
     screen — a well-known CSS-only override of the native <details>
     collapse behavior; no JS involved (design §7 "JS 없이 접이식"). */
  details:not([open]) > *:not(summary) { display: block !important; }
  details > summary { list-style: none; }
  details > summary::-webkit-details-marker { display: none; }
}
""".strip("\n")


ENHANCE_JS = """
(function () {
  "use strict";

  // Progressive enhancement only (design §R4 "JS 없이도 모든 콘텐츠가 보여야
  // 한다"): the region table already renders in a sensible server-decided
  // order without this script; this only adds click-to-sort on top of it.
  var table = document.getElementById("region-table");
  if (!table) {
    return;
  }
  var tbody = table.querySelector("tbody");
  var headers = Array.prototype.slice.call(table.querySelectorAll("thead th[data-sort]"));

  headers.forEach(function (header, columnIndex) {
    header.style.cursor = "pointer";
    header.addEventListener("click", function () {
      var kind = header.getAttribute("data-sort");
      var ascending = header.getAttribute("data-sort-dir") !== "asc";
      var rows = Array.prototype.slice.call(tbody.querySelectorAll("tr"));

      rows.sort(function (rowA, rowB) {
        var textA = (rowA.children[columnIndex].textContent || "").trim();
        var textB = (rowB.children[columnIndex].textContent || "").trim();
        if (kind === "num") {
          var numA = parseFloat(textA.replace("%", ""));
          var numB = parseFloat(textB.replace("%", ""));
          if (isNaN(numA)) numA = ascending ? Infinity : -Infinity;
          if (isNaN(numB)) numB = ascending ? Infinity : -Infinity;
          return ascending ? numA - numB : numB - numA;
        }
        return ascending ? textA.localeCompare(textB) : textB.localeCompare(textA);
      });

      headers.forEach(function (other) { other.removeAttribute("data-sort-dir"); });
      header.setAttribute("data-sort-dir", ascending ? "asc" : "desc");
      rows.forEach(function (row) { tbody.appendChild(row); });
    });
  });
})();
""".strip("\n")
