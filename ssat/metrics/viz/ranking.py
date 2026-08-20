"""Vulnerability ranking contrast view (V3).

Lists the most- and least-vulnerable samples side by side, each with its
heatmap, so a developer can see at a glance whether the top and bottom of
the ranking actually look visually different — if they can't be told apart,
that's a sign the metric or the core itself may have a problem. Ranking is
driven by ``SampleMetrics``'s ``vulnerability_score``, computed from the
run's configured primary metric.

This module deliberately imports ``ssat.metrics.viz.heatmap`` — unlike V1
(``mask_check.py``), which stays fully independent, V3 is defined in terms
of V2: listing the top N and bottom N samples by ``vulnerability_score``
alongside their heatmaps is not an accidental coupling, it is the feature
itself. V1 stays reachable even if V2/V3 break; a broken ``heatmap.py``
legitimately takes ``ranking.py`` down with it, since ranking's panels *are*
heatmaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ssat.core.source import ImageFolderSource
from ssat.metrics.aggregate import DEFAULT_PRIMARY_METRIC
from ssat.metrics.dump_reader import DumpHandle
from ssat.metrics.errors import DebugVizError
from ssat.metrics.store import load_metrics, verify_source_dump
from ssat.metrics.types import SampleMetrics, SpatialProfile
from ssat.metrics.viz._shared import open_image_source
from ssat.metrics.viz.heatmap import (
    render_heatmap_panel,
    resolve_heatmap_view,
    select_spatial_profile_rows,
)

__all__ = ["RankedSample", "save_ranking_views", "select_ranked_samples"]


@dataclass(frozen=True, slots=True)
class RankedSample:
    """One sample's position in the vulnerability ranking.

    Attributes:
        sample_id: Ranked sample.
        vulnerability_score: This sample's score.
    """

    sample_id: str
    vulnerability_score: float


def select_ranked_samples(
    sample_metrics: Sequence[SampleMetrics],
    *,
    n_top: int = 5,
    n_bottom: int = 5,
) -> tuple[tuple[RankedSample, ...], tuple[RankedSample, ...]]:
    """Rank samples by vulnerability_score and cut the top/bottom N.

    ``vulnerability_score`` is denormalized across every ``metric_name`` row
    for a sample, so each sample_id is scored once here regardless of how
    many metric_name rows it has in ``sample_metrics``.

    Args:
        sample_metrics: Rows from ``AggregationResult.sample_metrics``.
        n_top: Maximum most-vulnerable samples to return (highest score
            first). Fewer are returned if fewer are available.
        n_bottom: Maximum least-vulnerable samples to return (lowest score
            first). Fewer are returned if fewer are available.

    Returns:
        ``(top, bottom)``: ``top`` sorted by descending vulnerability_score,
        ``bottom`` sorted by ascending vulnerability_score. The two may
        overlap when there are fewer distinct scored samples than
        ``n_top + n_bottom``.

    Raises:
        DebugVizError: If no row carries a vulnerability_score.
    """

    scores: dict[str, float] = {}
    for row in sample_metrics:
        if row.vulnerability_score is not None:
            scores.setdefault(row.sample_id, row.vulnerability_score)
    if not scores:
        raise DebugVizError("no sample_metrics row carries a vulnerability_score")

    ascending = sorted(scores.items(), key=lambda item: (item[1], item[0]))
    descending = list(reversed(ascending))

    top = tuple(RankedSample(sample_id, score) for sample_id, score in descending[:n_top])
    bottom = tuple(RankedSample(sample_id, score) for sample_id, score in ascending[:n_bottom])
    return top, bottom


def save_ranking_views(
    dump_root: str | Path,
    metrics_dir: str | Path,
    output_dir: str | Path,
    *,
    metric_name: str = DEFAULT_PRIMARY_METRIC,
    n_top: int = 5,
    n_bottom: int = 5,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Render and save one 2-panel PNG per ranked sample.

    Each panel pair reuses ``heatmap.resolve_heatmap_view``/
    ``render_heatmap_panel`` for ``metric_name``'s spatial pattern; the title
    additionally records the sample's rank and vulnerability_score.

    Args:
        dump_root: Root directory of the raw audit dump these metrics were
            computed from.
        metrics_dir: Directory holding a stored aggregation run.
        output_dir: Directory PNGs are written into (created if missing).
        metric_name: Metric whose spatial pattern is drawn in each panel
            (default matches ``vulnerability_score``'s own source metric).
        n_top: See ``select_ranked_samples``.
        n_bottom: See ``select_ranked_samples``.

    Returns:
        ``(top_paths, bottom_paths)``, one path per ranked sample, in
        ``select_ranked_samples``' order.

    Raises:
        DebugVizError: If no sample can be ranked, or a ranked sample's
            heatmap cannot be reconstructed.
        MetricsCorruptionError: If ``metrics_dir`` no longer matches
            ``dump_root`` (see ``ssat.metrics.store.verify_source_dump``).
    """

    source = open_image_source(dump_root)
    _, result, manifest = load_metrics(Path(metrics_dir))
    verify_source_dump(manifest, DumpHandle(dump_root).manifest_path)

    top, bottom = select_ranked_samples(result.sample_metrics, n_top=n_top, n_bottom=n_bottom)
    rows_by_sample = select_spatial_profile_rows(
        result.spatial_profile,
        metric_name=metric_name,
        sample_ids=tuple({ranked.sample_id for ranked in (*top, *bottom)}),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    top_paths = tuple(
        _save_ranked_png(ranked, rank, "top", rows_by_sample, source, output_path)
        for rank, ranked in enumerate(top, start=1)
    )
    bottom_paths = tuple(
        _save_ranked_png(ranked, rank, "bottom", rows_by_sample, source, output_path)
        for rank, ranked in enumerate(bottom, start=1)
    )
    return top_paths, bottom_paths


def _save_ranked_png(
    ranked: RankedSample,
    rank: int,
    prefix: str,
    rows_by_sample: dict[str, list[SpatialProfile]],
    source: ImageFolderSource,
    output_path: Path,
) -> Path:
    """Render and save one ranked sample's panel.

    Args:
        ranked: Ranked sample to render.
        rank: 1-based position within its top/bottom list.
        prefix: ``"top"`` or ``"bottom"``, used in the filename.
        rows_by_sample: This run's eligible spatial_profile rows by sample_id.
        source: Image source to load the original image from.
        output_path: Destination directory.

    Returns:
        The saved PNG path.
    """

    view = resolve_heatmap_view(ranked.sample_id, rows_by_sample[ranked.sample_id], source=source)
    path = output_path / f"{prefix}_{rank:02d}_{ranked.sample_id}.png"

    figure = Figure(figsize=(8, 4))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    render_heatmap_panel((axes[0], axes[1]), view)
    figure.suptitle(
        f"{prefix} #{rank} sample_id={ranked.sample_id} "
        f"vulnerability_score={ranked.vulnerability_score:.4f}"
    )
    figure.savefig(path)
    return path
