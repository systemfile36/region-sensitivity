"""Asset linker: re-render top-K/bottom-K heatmap + thumbnail PNGs for the report gallery.

**No pre-existing asset to relocate.** A heatmap gallery could in principle
just relocate already-rendered images, but ``ssat metrics``/``ssat
analyze`` never persist a DebugViz heatmap PNG anywhere, so there is
nothing to "relocate" here. Instead this module calls
``ssat.metrics.viz.heatmap.select_spatial_profile_rows``/
``resolve_heatmap_view``/``render_heatmap_overlay`` — functions the metrics
engine already wrote, tested, and made public — limited to exactly the
``sample_id``\\s the assembler already chose (top-K ∪ bottom-K), never a
set this module decides for itself. No new visualization logic or new
statistics are introduced; only the already-tested rendering path is
invoked here.

**Why this module reaches past ``ssat.metrics.viz.heatmap`` into
``ssat.metrics.store``/``ssat.metrics.viz._shared``/``ssat.metrics.errors``.**
``resolve_heatmap_view`` needs an already-open ``ImageFolderSource`` (built
by ``ssat.metrics.viz._shared.open_image_source`` — a helper that module's
own docstring reserves for ``mask_check.py``/``heatmap.py``/``ranking.py``,
the three DebugViz *view* modules), and ``select_spatial_profile_rows``
needs the raw ``SpatialProfile`` rows (only reachable via
``ssat.metrics.store.load_metrics`` — the exact call the report assembler
already makes, but ``AssembledReport`` never carries these rows onward:
``TopRegionEntry`` drops each row's ``RegionGeometryRef``, since a
JSON-serializable ``ReportModel`` has no reason to carry
mask-reconstruction geometry). Both of these two extra imports, plus
``ssat.metrics.errors.DebugVizError`` (needed to catch per-sample
failures), are treated as a minimal necessary extension of the same
already-granted exception ``ssat.metrics.viz.heatmap`` itself represents,
rather than routing the fix through ``ssat/metrics/viz/heatmap.py``, whose
already-tested public surface would have to grow a new
per-sample-resilient entry point solely for this caller.

**Per-sample resilience.** ``select_spatial_profile_rows``/
``resolve_heatmap_view`` are called once per ``sample_id`` — not once for
the whole top-K/bottom-K set — specifically because
``select_spatial_profile_rows`` raises ``DebugVizError`` for its *entire*
call the moment even one requested ``sample_id`` has no eligible row, and
``resolve_heatmap_view`` can raise the same for one geometrically
unreproducible region (e.g. ``random_area_match``). One failing sample must
not blank out every other sample's assets, so each ``sample_id`` gets its
own try/except.

**Two separate images, two separate reuse paths.** ``heatmap_asset_ref``
reuses ``render_heatmap_overlay`` — the same overlay-drawing logic
``render_heatmap_panel`` itself calls for its own 2-panel composite — but
renders only the overlay, into its own single-axis figure, since the
gallery card already shows the original via ``thumbnail_asset_ref`` on the
card's other half; duplicating it into this image too would waste half of
each half's width. ``thumbnail_asset_ref`` is a plain Pillow resize of the
same ``HeatmapView.original`` array — no matplotlib, no overlay, just a
small preview for the gallery grid, cheap because it is a resize of
already-rendered pixels rather than a new computation. Both read the
identical source array, so neither can show a different sample than the
other.

**Asset refs are report-root-relative POSIX paths, never absolute**, so
that a copied or moved ``report/`` folder keeps working. ``output_dir`` is
the report root (``<run_dir>/report/``), and every ref this module returns
is relative to it (e.g. ``"assets/img/heatmaps/sample_s0.png"``), so
copying the whole ``report/`` folder elsewhere never breaks an
``<img src=...>`` reference.

**Why ``link_assets`` takes ``metrics_dir``.** Without a ``metrics_dir``
parameter this function would have no path to load ``SpatialProfile`` rows
from at all, and there is no alternative source for that data (see the
note above on why raw ``SpatialProfile`` rows cannot come from
``assembled``), so :func:`link_assets` takes ``metrics_dir`` positionally
alongside ``dump_dir``, mirroring ``ReportDataAssembler.__init__``'s own
``(dump_dir, metrics_dir, ...)`` ordering.

**Manifest application.** :class:`AssetManifest` alone cannot update
``ReportModel``/``SampleCard`` — every type in ``report.types`` is frozen
(module docstring there). This module therefore also owns
:func:`apply_asset_manifest`, which rebuilds every top-K/bottom-K
``SampleCard`` (in both ``ReportModel.sample_rankings`` and
``full_sample_rankings`` — the assembler constructs both from the *same*
card objects for any sample_id present in both, see
``assembler.py``'s ``_build_sample_rankings``, so both must be rebuilt
consistently) via ``dataclasses.replace``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PIL import Image

from ssat.metrics.dump_reader import DumpHandle
from ssat.metrics.errors import DebugVizError
from ssat.metrics.store import load_metrics, verify_source_dump
from ssat.metrics.viz._shared import open_image_source, open_skeleton_store
from ssat.metrics.viz.heatmap import (
    HeatmapView,
    render_heatmap_overlay,
    resolve_heatmap_view,
    select_spatial_profile_rows,
)
from ssat.report.types import ReportModel, SampleCard, SampleRankings

# Relative to the report root (``output_dir``) — see module docstring.
_HEATMAP_SUBDIR = Path("assets") / "img" / "heatmaps"
_THUMBNAIL_SUBDIR = Path("assets") / "img" / "thumbnails"

# Longest edge of a saved thumbnail, in pixels; aspect ratio preserved,
# never upscaled (``PIL.Image.thumbnail``'s own contract).
_THUMBNAIL_MAX_EDGE = 256


class AssembledReportLike(Protocol):
    """Structural shape this module reads off / rebuilds from the assembler's output.

    Matches ``ssat.report.assembler.AssembledReport`` field-for-field
    without importing it — the same duck-typing trade
    ``report.exporter.AssembledReportLike``/
    ``report.charts.RankCorrelationRowLike`` already made for the same
    reason.

    Attributes:
        model: The JSON-serializable ``ReportModel``.
        full_sample_rankings: Every sample the run produced (module
            docstring on why :func:`apply_asset_manifest` must rebuild this
            too, not just ``model.sample_rankings``).
        sample_semantic_degradation: Every ``(sample_id, semantic_group)``
            pair with a determinable mean degradation
            (``ssat.report.assembler.AssembledReport``'s field of the same
            name) — :func:`apply_asset_manifest` never touches it (no asset
            ref lives on it), but must still carry it through unchanged so
            the downstream exporter (``ssat.report.exporter``) still has it
            after this module's ``AssetLinkedReport`` replaces
            ``AssembledReport`` in ``AuditApplication.generate_report``'s
            pipeline.
    """

    model: ReportModel
    full_sample_rankings: Sequence[SampleCard]
    sample_semantic_degradation: Mapping[tuple[str, str], float]


@dataclasses.dataclass(frozen=True, slots=True)
class AssetManifest:
    """Per-sample asset references :func:`link_assets` produced.

    Attributes:
        assets_available: ``False`` only when the source dump has no
            ``source_provenance`` at all — the whole gallery section is
            unavailable, not just individual samples.
            ``True`` even when some *individual* samples still failed to
            render (see below); those are simply absent from the two
            mappings, the same "no key = unavailable" contract a lookup
            already gives for free.
        heatmap_refs: ``sample_id`` -> report-root-relative path to its
            rendered heatmap PNG. Omits any ``sample_id`` that could not be
            rendered (module docstring, "per-sample resilience").
        thumbnail_refs: Same shape as ``heatmap_refs``, for the standalone
            thumbnail.
    """

    assets_available: bool
    heatmap_refs: dict[str, str]
    thumbnail_refs: dict[str, str]


@dataclasses.dataclass(frozen=True, slots=True)
class AssetLinkedReport:
    """An :class:`AssembledReportLike` with asset refs filled in.

    :func:`apply_asset_manifest`'s return type. Not
    ``ssat.report.assembler.AssembledReport`` itself (this module cannot
    import ``report.assembler``), but structurally identical, so the
    exporter's ``export`` and the HTML renderer can consume it exactly as
    they already consume ``AssembledReport`` (both type their own
    parameter as an ``AssembledReportLike`` Protocol, not the concrete
    class).

    Attributes:
        model: ``ReportModel`` with every top-K/bottom-K ``SampleCard``'s
            asset refs filled in where :class:`AssetManifest` had one.
        full_sample_rankings: Every sample, with the same refs filled in.
        sample_semantic_degradation: Carried through unchanged from the
            input ``assembled`` (:class:`AssembledReportLike` docstring).
    """

    model: ReportModel
    full_sample_rankings: tuple[SampleCard, ...]
    sample_semantic_degradation: Mapping[tuple[str, str], float]


def link_assets(
    assembled: AssembledReportLike,
    dump_dir: Path,
    metrics_dir: Path,
    output_dir: Path,
    *,
    primary_metric: str,
) -> AssetManifest:
    """Render heatmap + thumbnail PNGs for every gallery sample.

    Args:
        assembled: The assembler's output; only ``model.sample_rankings``
            (top-K ∪ bottom-K sample_ids) is read.
        dump_dir: Root of the source dump, opened only through
            ``ssat.metrics.dump_reader.DumpHandle``/``ssat.metrics.viz.
            heatmap``'s image-source helper (module docstring).
        metrics_dir: MetricsStore directory the same dump/``assembled`` pair
            was aggregated into (module docstring on why this parameter
            exists).
        output_dir: Report root PNGs are written under (``<output_dir>/
            assets/img/{heatmaps,thumbnails}/``); created if missing.
        primary_metric: Registered metric name ``assembled`` was built from
            — must match, so a sample's ``SpatialProfile`` rows and its
            already-assembled ``vulnerability_score``/``top_regions`` agree.

    Returns:
        Every successfully rendered sample's asset refs, keyed by
        ``sample_id``; a sample absent from both mappings could not be
        rendered (module docstring, "per-sample resilience") — this never
        raises for an individual sample's failure.
    """

    dump_dir = Path(dump_dir)
    output_dir = Path(output_dir)

    handle = DumpHandle(dump_dir)
    if handle.manifest.resolved_config.source_provenance is None:
        return AssetManifest(assets_available=False, heatmap_refs={}, thumbnail_refs={})

    sample_ids = _gallery_sample_ids(assembled.model)
    if not sample_ids:
        return AssetManifest(assets_available=True, heatmap_refs={}, thumbnail_refs={})

    _, aggregation, metrics_manifest = load_metrics(metrics_dir)
    verify_source_dump(metrics_manifest, handle.manifest_path)
    spatial_rows = [
        row for row in aggregation.spatial_profile if row.metric_name == primary_metric
    ]
    source = open_image_source(dump_dir)
    skeleton_store = open_skeleton_store(dump_dir)

    heatmap_dir = output_dir / _HEATMAP_SUBDIR
    thumbnail_dir = output_dir / _THUMBNAIL_SUBDIR
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    heatmap_refs: dict[str, str] = {}
    thumbnail_refs: dict[str, str] = {}
    for sample_id in sample_ids:
        try:
            rows_by_sample = select_spatial_profile_rows(
                spatial_rows, metric_name=primary_metric, sample_ids=[sample_id]
            )
            view = resolve_heatmap_view(
                sample_id, rows_by_sample[sample_id], source=source, skeleton_store=skeleton_store
            )
        except DebugVizError:
            continue

        heatmap_path = heatmap_dir / f"sample_{sample_id}.png"
        _save_heatmap_png(view, heatmap_path)
        heatmap_refs[sample_id] = (_HEATMAP_SUBDIR / heatmap_path.name).as_posix()

        thumbnail_path = thumbnail_dir / f"sample_{sample_id}.png"
        _save_thumbnail_png(view.original, thumbnail_path)
        thumbnail_refs[sample_id] = (_THUMBNAIL_SUBDIR / thumbnail_path.name).as_posix()

    return AssetManifest(assets_available=True, heatmap_refs=heatmap_refs, thumbnail_refs=thumbnail_refs)


def apply_asset_manifest(assembled: AssembledReportLike, manifest: AssetManifest) -> AssetLinkedReport:
    """Fill in every gallery ``SampleCard``'s asset refs from ``manifest``.

    Args:
        assembled: The same assembler output ``manifest`` was built from
            (``link_assets``'s ``assembled`` argument).
        manifest: :func:`link_assets`'s return value.

    Returns:
        A rebuilt, otherwise-identical report: every ``SampleCard`` present
        in ``manifest`` has its ``heatmap_asset_ref``/``thumbnail_asset_ref``
        set; every other card (including every non-gallery sample in
        ``full_sample_rankings``) is unchanged.
    """

    def linked(card: SampleCard) -> SampleCard:
        heatmap_ref = manifest.heatmap_refs.get(card.sample_id)
        thumbnail_ref = manifest.thumbnail_refs.get(card.sample_id)
        if heatmap_ref is None and thumbnail_ref is None:
            return card
        return dataclasses.replace(
            card, heatmap_asset_ref=heatmap_ref, thumbnail_asset_ref=thumbnail_ref
        )

    model = assembled.model
    sample_rankings = SampleRankings(
        most_vulnerable=tuple(linked(card) for card in model.sample_rankings.most_vulnerable),
        most_robust=tuple(linked(card) for card in model.sample_rankings.most_robust),
    )
    updated_model = dataclasses.replace(model, sample_rankings=sample_rankings)
    full_rankings = tuple(linked(card) for card in assembled.full_sample_rankings)
    return AssetLinkedReport(
        model=updated_model,
        full_sample_rankings=full_rankings,
        sample_semantic_degradation=assembled.sample_semantic_degradation,
    )


# --- private helpers ----------------------------------------------------------


def _gallery_sample_ids(model: ReportModel) -> tuple[str, ...]:
    """Union of top-K/bottom-K sample_ids, de-duplicated, selection order preserved.

    A sample can appear in both galleries when the run has fewer scored
    samples than ``top_k + bottom_k`` (``ReportDataAssembler._build_sample_
    rankings`` slices the same sorted list from both ends) — de-duplication
    keeps this module from rendering (and DumpHandle/heatmap-resolving) the
    same sample_id twice.
    """

    seen: dict[str, None] = {}
    for card in (*model.sample_rankings.most_vulnerable, *model.sample_rankings.most_robust):
        seen.setdefault(card.sample_id, None)
    return tuple(seen)


def _save_heatmap_png(view: HeatmapView, path: Path) -> None:
    """Save one sample's heatmap-overlay-only figure.

    The original is deliberately not re-rendered here: the gallery card
    already shows it via the standalone thumbnail (``_save_thumbnail_png``)
    on the card's left side, so this image only needs the overlay
    (``render_heatmap_overlay``), left at full card width on the right side.
    """

    figure = Figure(figsize=(4, 4))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    render_heatmap_overlay(axis, view)
    caption = f"sample_id={view.sample_id}"
    if view.num_frames > 1:
        caption += f" frame={view.frame_index}/{view.num_frames}"
    figure.suptitle(caption)
    figure.savefig(path)


def _save_thumbnail_png(original: NDArray[np.uint8], path: Path) -> None:
    """Save a small, aspect-ratio-preserved resize of ``original``."""

    image = Image.fromarray(original)
    image.thumbnail((_THUMBNAIL_MAX_EDGE, _THUMBNAIL_MAX_EDGE), Image.Resampling.LANCZOS)
    image.save(path)
