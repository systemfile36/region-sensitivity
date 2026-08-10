"""Tests for the metrics engine's N4 MetricsStore (save_metrics/load_metrics)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from ssat.core.types import RegionKind
from ssat.metrics.aggregate import AggregationResult
from ssat.metrics.builtin_metrics.flips import DEFAULT_TOPK, TopkExit
from ssat.metrics.errors import MetricsCorruptionError, MetricsRegistryError, MetricsSchemaError
from ssat.metrics.registry import MetricRegistry
from ssat.metrics.schema import SAMPLE_METRICS_SCHEMA
from ssat.metrics.store import load_metrics, save_metrics, verify_source_dump
from ssat.metrics.types import (
    ClassMetrics,
    ExclusionReason,
    ItemMetrics,
    RegionGeometryRef,
    RegionMetrics,
    SampleMetrics,
    SpatialProfile,
)
from ssat.utils.io import sha256_file

from test_metrics_aggregate import _FakeMetric


def _registry(*, with_topk: bool = False) -> MetricRegistry:
    registry = MetricRegistry()
    registry.register(_FakeMetric("margin_drop", "continuous"))
    registry.register(_FakeMetric("flip_correct_to_wrong", "binary"))
    if with_topk:
        registry.register(TopkExit())
    return registry


def _item_metrics() -> list[ItemMetrics]:
    return [
        ItemMetrics(
            item_id="a" * 64,
            sample_id="s1",
            metric_name="margin_drop",
            clean_correct=True,
            value_clean=1.0,
            value_perturbed=0.5,
            degradation=0.5,
            available=True,
        ),
        ItemMetrics(
            item_id="b" * 64,
            sample_id="s1",
            metric_name="margin_drop",
            clean_correct=True,
            value_clean=None,
            value_perturbed=None,
            degradation=None,
            available=False,
            excluded_reason=ExclusionReason.PERTURBED_STATUS_NOT_OK,
        ),
    ]


def _result() -> AggregationResult:
    return AggregationResult(
        sample_metrics=[
            SampleMetrics(
                sample_id="s1",
                metric_name="margin_drop",
                gt_label=3,
                clean_correct=True,
                n_items=2,
                n_valid=1,
                flip_rate=None,
                vulnerability_score=0.5,
                metric_mean=0.5,
                metric_max=0.5,
                metric_std=0.0,
            )
        ],
        region_metrics=[
            RegionMetrics(
                region_key="grid::grid/r0/c0",
                metric_name="margin_drop",
                region_kind=RegionKind.GRID,
                intended_area_px=4,
                effective_area_px=4,
                n_samples=1,
                n_valid=1,
                flip_rate=None,
                metric_mean=0.5,
            )
        ],
        class_metrics=[
            ClassMetrics(
                gt_label=3,
                metric_name="margin_drop",
                n_samples=1,
                flip_rate=None,
                metric_mean=0.5,
            )
        ],
        spatial_profile=[
            SpatialProfile(
                sample_id="s1",
                region_key="grid::grid/r0/c0",
                metric_name="margin_drop",
                region_geometry_ref=RegionGeometryRef(
                    region_kind=RegionKind.GRID, region_params_json="{}"
                ),
                degradation=0.5,
            )
        ],
    )


def _source_manifest(tmp_path: Path, *, content: bytes = b'{"marker": "v1"}') -> Path:
    path = tmp_path / "run_manifest.json"
    path.write_bytes(content)
    return path


def test_save_then_load_round_trips_every_output_exactly(tmp_path: Path) -> None:
    registry = _registry()
    manifest_path = _source_manifest(tmp_path)

    written = save_metrics(
        tmp_path / "metrics",
        _item_metrics(),
        _result(),
        registry=registry,
        primary_metric="margin_drop",
        source_run_manifest_path=manifest_path,
        exclusion_summary={"samples_excluded_clean_failed": 0},
    )
    loaded_items, loaded_result, loaded_manifest = load_metrics(tmp_path / "metrics")

    assert loaded_items == _item_metrics()
    assert loaded_result == _result()
    assert loaded_manifest == written


def test_source_run_manifest_hash_matches_the_actual_file(tmp_path: Path) -> None:
    manifest_path = _source_manifest(tmp_path)

    written = save_metrics(
        tmp_path / "metrics",
        _item_metrics(),
        _result(),
        registry=_registry(),
        primary_metric="margin_drop",
        source_run_manifest_path=manifest_path,
        exclusion_summary={},
    )

    assert written.source_run_manifest_hash == sha256_file(manifest_path)


def test_verify_source_dump_raises_when_the_dump_changed(tmp_path: Path) -> None:
    manifest_path = _source_manifest(tmp_path)
    written = save_metrics(
        tmp_path / "metrics",
        _item_metrics(),
        _result(),
        registry=_registry(),
        primary_metric="margin_drop",
        source_run_manifest_path=manifest_path,
        exclusion_summary={},
    )

    manifest_path.write_bytes(b'{"marker": "v2-rerun"}')

    with pytest.raises(MetricsCorruptionError, match="source dump has changed"):
        verify_source_dump(written, manifest_path)


def test_load_metrics_rejects_a_metrics_schema_version_mismatch(tmp_path: Path) -> None:
    manifest_path = _source_manifest(tmp_path)
    save_metrics(
        tmp_path / "metrics",
        _item_metrics(),
        _result(),
        registry=_registry(),
        primary_metric="margin_drop",
        source_run_manifest_path=manifest_path,
        exclusion_summary={},
    )

    manifest_json_path = tmp_path / "metrics" / "metrics_manifest.json"
    payload = json.loads(manifest_json_path.read_text())
    payload["metrics_schema_version"] = "9.9.9"
    manifest_json_path.write_text(json.dumps(payload))

    with pytest.raises(MetricsSchemaError, match="metrics schema version mismatch"):
        load_metrics(tmp_path / "metrics")


def test_registered_metrics_are_sorted_with_kind_and_direction(tmp_path: Path) -> None:
    manifest_path = _source_manifest(tmp_path)

    written = save_metrics(
        tmp_path / "metrics",
        _item_metrics(),
        _result(),
        registry=_registry(),
        primary_metric="margin_drop",
        source_run_manifest_path=manifest_path,
        exclusion_summary={},
    )

    assert [metric.name for metric in written.registered_metrics] == [
        "flip_correct_to_wrong",
        "margin_drop",
    ]
    by_name = {metric.name: metric for metric in written.registered_metrics}
    assert by_name["flip_correct_to_wrong"].kind == "binary"
    assert by_name["margin_drop"].kind == "continuous"
    assert by_name["margin_drop"].higher_is_better is True


def test_metric_config_topk_reflects_topk_exit_registration(tmp_path: Path) -> None:
    manifest_path = _source_manifest(tmp_path)

    without_topk = save_metrics(
        tmp_path / "metrics-a",
        _item_metrics(),
        _result(),
        registry=_registry(with_topk=False),
        primary_metric="margin_drop",
        source_run_manifest_path=manifest_path,
        exclusion_summary={},
    )
    with_topk = save_metrics(
        tmp_path / "metrics-b",
        _item_metrics(),
        _result(),
        registry=_registry(with_topk=True),
        primary_metric="margin_drop",
        source_run_manifest_path=manifest_path,
        exclusion_summary={},
    )

    assert without_topk.metric_config.topk is None
    assert with_topk.metric_config.topk == DEFAULT_TOPK


def test_save_metrics_rejects_an_unregistered_primary_metric(tmp_path: Path) -> None:
    manifest_path = _source_manifest(tmp_path)

    with pytest.raises(MetricsRegistryError, match="primary_metric not registered"):
        save_metrics(
            tmp_path / "metrics",
            _item_metrics(),
            _result(),
            registry=_registry(),
            primary_metric="not_a_real_metric",
            source_run_manifest_path=manifest_path,
            exclusion_summary={},
        )


def test_load_metrics_rejects_a_corrupted_parquet_fragment_version(tmp_path: Path) -> None:
    manifest_path = _source_manifest(tmp_path)
    save_metrics(
        tmp_path / "metrics",
        _item_metrics(),
        _result(),
        registry=_registry(),
        primary_metric="margin_drop",
        source_run_manifest_path=manifest_path,
        exclusion_summary={},
    )

    stale_schema = SAMPLE_METRICS_SCHEMA.with_metadata({b"ssat.metrics_schema_version": b"0.0.1"})
    table = pq.read_table(tmp_path / "metrics" / "sample_metrics.parquet").cast(stale_schema)
    pq.write_table(table, tmp_path / "metrics" / "sample_metrics.parquet")

    with pytest.raises(MetricsSchemaError, match="version mismatch"):
        load_metrics(tmp_path / "metrics")
