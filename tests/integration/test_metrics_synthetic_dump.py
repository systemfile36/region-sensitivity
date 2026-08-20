"""L2: dump -> ItemMetrics -> aggregation, over dumps built without running the core.

Uses tests/fixtures/synthetic_dump_builder.py (DumpWriter called directly —
no PlanBuilder/runtime/adapter) to reproduce the 5 defensive-path scenarios
for the metrics engine. Passing this file lets any later failure be
attributed to the metrics engine rather than the core execution path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from synthetic_dump_builder import (
    build_resolved_config,
    clean_record,
    perturbed_record,
    write_dump,
)

from ssat.core.config.schema import ResolvedRegionConfig
from ssat.core.types import ItemStatus, RegionKind
from ssat.metrics.aggregate import DEFAULT_PRIMARY_METRIC, aggregate_item_metrics
from ssat.metrics.builtin_metrics import default_metric_registry
from ssat.metrics.dump_reader import DumpHandle
from ssat.metrics.store import load_metrics, save_metrics, verify_source_dump

CORRECT = np.array([2.0, 0.0])
WRONG = np.array([0.0, 2.0])

_REGIONS = (ResolvedRegionConfig(region_id="grid", kind=RegionKind.GRID, params={"rows": 1, "cols": 3}),)
_COLUMNS = ("grid/r0/c0", "grid/r0/c1", "grid/r0/c2")


def _run_pipeline(root: Path):
    handle = DumpHandle(root)
    joined = handle.joined()
    registry = default_metric_registry()
    item_metrics = registry.compute_item_metrics(
        joined, adapter_spec=handle.manifest.resolved_config.adapter_spec
    )
    result = aggregate_item_metrics(
        item_metrics, joined, registry, handle.manifest.resolved_config
    )
    return joined, item_metrics, result


def test_completely_insensitive_dump_shows_zero_degradation_everywhere(tmp_path: Path) -> None:
    config = build_resolved_config(tmp_path, regions=_REGIONS)
    clean_records = (clean_record("sample-a", logits=CORRECT), clean_record("sample-b", logits=CORRECT))
    perturbed_records = tuple(
        perturbed_record(
            index,
            sample_id=sample_id,
            region_id="grid",
            region_instance_id=column,
            logits=CORRECT,  # unchanged by perturbation
        )
        for index, (sample_id, column) in enumerate(
            (sample_id, column) for sample_id in ("sample-a", "sample-b") for column in _COLUMNS
        )
    )
    write_dump(
        tmp_path / "dump", config, clean_records=clean_records, perturbed_records=perturbed_records
    )

    _, item_metrics, result = _run_pipeline(tmp_path / "dump")

    assert all(item.degradation == pytest.approx(0.0) for item in item_metrics if item.available)
    assert all(row.metric_mean == pytest.approx(0.0) for row in result.region_metrics)
    binary_rows = [row for row in result.region_metrics if row.flip_rate is not None]
    assert binary_rows  # at least the flip-metric rows exist
    assert all(row.flip_rate == pytest.approx(0.0) for row in binary_rows)


def test_single_vulnerable_region_stands_out_in_region_metrics(tmp_path: Path) -> None:
    config = build_resolved_config(tmp_path, regions=_REGIONS)
    clean_records = (clean_record("sample-a", logits=CORRECT), clean_record("sample-b", logits=CORRECT))
    perturbed_records = []
    index = 0
    for sample_id in ("sample-a", "sample-b"):
        for column in _COLUMNS:
            vulnerable = column == "grid/r0/c2"
            perturbed_records.append(
                perturbed_record(
                    index,
                    sample_id=sample_id,
                    region_id="grid",
                    region_instance_id=column,
                    logits=WRONG if vulnerable else CORRECT,
                )
            )
            index += 1
    write_dump(
        tmp_path / "dump",
        config,
        clean_records=clean_records,
        perturbed_records=tuple(perturbed_records),
    )

    _, _, result = _run_pipeline(tmp_path / "dump")

    by_region = {(row.region_key, row.metric_name): row for row in result.region_metrics}
    vulnerable_flip = by_region[("grid::grid/r0/c2", "flip_correct_to_wrong")]
    stable_flip = by_region[("grid::grid/r0/c0", "flip_correct_to_wrong")]
    assert vulnerable_flip.flip_rate == pytest.approx(1.0)
    assert stable_flip.flip_rate == pytest.approx(0.0)


def test_single_vulnerable_sample_stands_out_in_vulnerability_score(tmp_path: Path) -> None:
    config = build_resolved_config(tmp_path, regions=_REGIONS)
    clean_records = (clean_record("sample-a", logits=CORRECT), clean_record("sample-b", logits=CORRECT))
    perturbed_records = []
    index = 0
    for sample_id in ("sample-a", "sample-b"):
        vulnerable = sample_id == "sample-a"
        for column in _COLUMNS:
            perturbed_records.append(
                perturbed_record(
                    index,
                    sample_id=sample_id,
                    region_id="grid",
                    region_instance_id=column,
                    logits=WRONG if vulnerable else CORRECT,
                )
            )
            index += 1
    write_dump(
        tmp_path / "dump",
        config,
        clean_records=clean_records,
        perturbed_records=tuple(perturbed_records),
    )

    _, _, result = _run_pipeline(tmp_path / "dump")

    vulnerability_by_sample = {row.sample_id: row.vulnerability_score for row in result.sample_metrics}
    assert vulnerability_by_sample["sample-a"] > 0
    assert vulnerability_by_sample["sample-b"] == pytest.approx(0.0)


def test_mixed_missing_perturbed_items_reduce_n_valid_without_dropping_the_item(
    tmp_path: Path,
) -> None:
    config = build_resolved_config(tmp_path, regions=_REGIONS)
    clean_records = (clean_record("sample-a", logits=CORRECT),)
    perturbed_records = (
        perturbed_record(0, sample_id="sample-a", region_id="grid", region_instance_id=_COLUMNS[0], logits=CORRECT),
        perturbed_record(1, sample_id="sample-a", region_id="grid", region_instance_id=_COLUMNS[1], logits=WRONG),
        perturbed_record(
            2,
            sample_id="sample-a",
            region_id="grid",
            region_instance_id=_COLUMNS[2],
            logits=None,
            status=ItemStatus.PREDICT_FAILED,
        ),
    )
    write_dump(
        tmp_path / "dump", config, clean_records=clean_records, perturbed_records=perturbed_records
    )

    joined, item_metrics, result = _run_pipeline(tmp_path / "dump")

    assert len(joined) == 3  # the failed item stays in joined(), just without perturbed logits
    failed_rows = [row for row in item_metrics if row.item_id == f"{2:064x}"]
    assert failed_rows and all(not row.available for row in failed_rows)

    sample_row = next(
        r for r in result.sample_metrics if r.metric_name == "flip_correct_to_wrong"
    )
    assert sample_row.n_items == 3
    assert sample_row.n_valid == 2


def test_clean_failure_excludes_the_whole_sample_from_every_aggregate(tmp_path: Path) -> None:
    config = build_resolved_config(tmp_path, regions=_REGIONS)
    clean_records = (
        clean_record("sample-ok", logits=CORRECT),
        clean_record("sample-bad", logits=None, status=ItemStatus.LOAD_FAILED),
    )
    perturbed_records = (
        perturbed_record(0, sample_id="sample-ok", region_id="grid", region_instance_id=_COLUMNS[0], logits=CORRECT),
        perturbed_record(1, sample_id="sample-bad", region_id="grid", region_instance_id=_COLUMNS[0], logits=CORRECT),
    )
    write_dump(
        tmp_path / "dump", config, clean_records=clean_records, perturbed_records=perturbed_records
    )

    handle = DumpHandle(tmp_path / "dump")
    summary = handle.summary()
    assert summary["samples_excluded_clean_failed"] == 1

    joined, item_metrics, result = _run_pipeline(tmp_path / "dump")
    assert set(joined["sample_id"]) == {"sample-ok"}
    assert all(item.sample_id == "sample-ok" for item in item_metrics)
    assert all(row.sample_id == "sample-ok" for row in result.sample_metrics)


def test_saved_metrics_reload_to_the_same_aggregation_result(tmp_path: Path) -> None:
    """End-to-end: dump -> ItemMetrics -> aggregation -> N4 store -> reload."""

    config = build_resolved_config(tmp_path, regions=_REGIONS)
    clean_records = (clean_record("sample-a", logits=CORRECT),)
    perturbed_records = tuple(
        perturbed_record(
            index, sample_id="sample-a", region_id="grid", region_instance_id=column, logits=CORRECT
        )
        for index, column in enumerate(_COLUMNS)
    )
    write_dump(
        tmp_path / "dump", config, clean_records=clean_records, perturbed_records=perturbed_records
    )

    handle = DumpHandle(tmp_path / "dump")
    registry = default_metric_registry()
    joined, item_metrics, result = _run_pipeline(tmp_path / "dump")

    manifest = save_metrics(
        tmp_path / "dump" / "metrics",
        item_metrics,
        result,
        registry=registry,
        primary_metric=DEFAULT_PRIMARY_METRIC,
        source_run_manifest_path=handle.manifest_path,
        exclusion_summary=handle.summary(),
    )
    verify_source_dump(manifest, handle.manifest_path)  # the dump has not changed since save

    loaded_items, loaded_result, loaded_manifest = load_metrics(tmp_path / "dump" / "metrics")
    assert loaded_items == item_metrics
    assert loaded_result == result
    assert loaded_manifest == manifest
