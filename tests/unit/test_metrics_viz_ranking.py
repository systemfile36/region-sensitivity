"""Tests for the metrics engine's DebugViz V3 vulnerability ranking view."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from synthetic_dump_builder import (
    build_resolved_config,
    clean_record,
    compute_and_save_metrics,
    image_manifest_source_provenance,
    perturbed_record,
    write_dump,
)

from ssat.core.config.schema import ResolvedRegionConfig
from ssat.core.types import RegionKind
from ssat.metrics.builtin_metrics.continuous import GtLogitDrop
from ssat.metrics.errors import DebugVizError
from ssat.metrics.registry import MetricRegistry
from ssat.metrics.store import load_metrics
from ssat.metrics.types import SampleMetrics
from ssat.metrics.viz.ranking import select_ranked_samples, save_ranking_views

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_classification"
_FIXTURE_MANIFEST = _FIXTURE_ROOT / "manifest.json"
_METRIC_NAME = "gt_logit_drop"
_WHOLE_IMAGE_REGIONS = (
    ResolvedRegionConfig(region_id="grid", kind=RegionKind.GRID, params={"rows": 1, "cols": 1}),
)
_CLEAN_LOGITS = np.array([1.0, 0.0])
# sample_id -> perturbed logits, chosen so gt_logit_drop degradation strictly
# differs per sample: 0.0, 0.5, 1.0, 2.0 (clean_logits[0] - perturbed_logits[0]).
_SAMPLE_PERTURBED_LOGITS = {
    "synthetic-000": np.array([1.0, 0.0]),
    "synthetic-001": np.array([0.5, 0.5]),
    "synthetic-002": np.array([0.0, 1.0]),
    "synthetic-003": np.array([-1.0, 2.0]),
}


def _registry() -> MetricRegistry:
    registry = MetricRegistry()
    registry.register(GtLogitDrop())
    return registry


def _build_ranking_metrics(tmp_path: Path) -> tuple[Path, Path]:
    """Write a synthetic dump with 4 samples of increasing degradation, then aggregate it."""

    config = build_resolved_config(
        tmp_path,
        regions=_WHOLE_IMAGE_REGIONS,
        source_provenance=image_manifest_source_provenance(_FIXTURE_MANIFEST),
    )
    clean_records = tuple(
        clean_record(sample_id, logits=_CLEAN_LOGITS) for sample_id in _SAMPLE_PERTURBED_LOGITS
    )
    perturbed_records = tuple(
        perturbed_record(
            index,
            sample_id=sample_id,
            region_id="grid",
            region_instance_id="grid/r0/c0",
            logits=logits,
            region_params={"rows": 1, "cols": 1, "row_index": 0, "col_index": 0},
        )
        for index, (sample_id, logits) in enumerate(_SAMPLE_PERTURBED_LOGITS.items())
    )
    dump_root = tmp_path / "dump"
    write_dump(dump_root, config, clean_records=clean_records, perturbed_records=perturbed_records)

    metrics_dir = tmp_path / "metrics"
    compute_and_save_metrics(
        dump_root, config, metrics_dir, registry=_registry(), primary_metric=_METRIC_NAME
    )
    return dump_root, metrics_dir


def test_select_ranked_samples_orders_top_descending_and_bottom_ascending(
    tmp_path: Path,
) -> None:
    _, metrics_dir = _build_ranking_metrics(tmp_path)
    _, result, _ = load_metrics(metrics_dir)

    top, bottom = select_ranked_samples(result.sample_metrics, n_top=2, n_bottom=2)

    expected_desc = sorted(
        {row.sample_id: row.vulnerability_score for row in result.sample_metrics}.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    expected_asc = list(reversed(expected_desc))

    assert [(ranked.sample_id, ranked.vulnerability_score) for ranked in top] == expected_desc[:2]
    assert [(ranked.sample_id, ranked.vulnerability_score) for ranked in bottom] == expected_asc[:2]


def test_select_ranked_samples_caps_without_error_when_fewer_available() -> None:
    sample_metrics = [
        SampleMetrics(
            sample_id="s1",
            metric_name=_METRIC_NAME,
            gt_label=0,
            clean_correct=True,
            n_items=1,
            n_valid=1,
            flip_rate=None,
            vulnerability_score=0.5,
            metric_mean=0.5,
            metric_max=0.5,
            metric_std=0.0,
        )
    ]

    top, bottom = select_ranked_samples(sample_metrics, n_top=5, n_bottom=5)

    assert [ranked.sample_id for ranked in top] == ["s1"]
    assert [ranked.sample_id for ranked in bottom] == ["s1"]


def test_select_ranked_samples_raises_without_any_vulnerability_score() -> None:
    sample_metrics = [
        SampleMetrics(
            sample_id="s1",
            metric_name=_METRIC_NAME,
            gt_label=0,
            clean_correct=True,
            n_items=1,
            n_valid=0,
            flip_rate=None,
            vulnerability_score=None,
            metric_mean=None,
            metric_max=None,
            metric_std=None,
        )
    ]

    with pytest.raises(DebugVizError, match="vulnerability_score"):
        select_ranked_samples(sample_metrics)


def test_save_ranking_views_writes_top_and_bottom_pngs(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_ranking_metrics(tmp_path)
    output_dir = tmp_path / "debug_viz" / "ranking"

    top_paths, bottom_paths = save_ranking_views(
        dump_root, metrics_dir, output_dir, metric_name=_METRIC_NAME, n_top=2, n_bottom=2
    )

    assert len(top_paths) == 2
    assert len(bottom_paths) == 2
    assert top_paths[0].name == "top_01_synthetic-003.png"
    assert bottom_paths[0].name == "bottom_01_synthetic-000.png"
    for path in (*top_paths, *bottom_paths):
        assert path.is_file()
        assert path.stat().st_size > 0
