"""Tests for the metrics engine's N0 DumpHandle (clean/items/joined/summary)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ssat.core.adapter.types import AdapterSpec, RawOutput
from ssat.core.config.schema import (
    DumpConfig,
    PerturbationConfig,
    ResolvedConfig,
    ResolvedRegionConfig,
    RuntimeConfig,
)
from ssat.core.dump import CleanDumpRecord, DumpWriter, EnvironmentSpec, PerturbedDumpRecord
from ssat.core.plan.types import WorkItem
from ssat.core.region.types import RegionMeta, RegionSpec
from ssat.core.types import ItemStatus, PerturbationOp, RegionKind
from ssat.metrics.dump_reader import DumpHandle
from ssat.metrics.errors import MetricsCorruptionError

ENVIRONMENT = EnvironmentSpec(python_version="3.11.0", platform="test-platform")


def _config(tmp_path: Path) -> ResolvedConfig:
    return ResolvedConfig(
        config_base_dir=tmp_path.resolve(),
        regions=(
            ResolvedRegionConfig(
                region_id="grid",
                kind=RegionKind.GRID,
                params={"rows": 1, "cols": 2},
            ),
        ),
        perturbations=(
            PerturbationConfig(op=PerturbationOp.BLUR, params={"sigma": 1.0}),
        ),
        runtime=RuntimeConfig(),
        dump=DumpConfig(),
        adapter_spec=AdapterSpec(
            model_id="model",
            deterministic=True,
            adapter_kind="callable",
            model_name="tiny",
            weights_hash="a" * 64,
            preprocessing_fingerprint="b" * 64,
            mask_transform_available=True,
        ),
    )


def _output(*values: float) -> RawOutput:
    return RawOutput(np.asarray(values, dtype=np.float64))


def _clean(
    sample_id: str,
    *,
    status: ItemStatus = ItemStatus.OK,
    gt_label: int = 1,
) -> CleanDumpRecord:
    successful = status is ItemStatus.OK
    return CleanDumpRecord(
        sample_id=sample_id,
        status=status,
        model_id="model",
        logits=_output(0.25, 0.75) if successful else None,
        content_hash="c" * 64 if successful else None,
        gt_label=gt_label,
        original_shape=(1, 4, 4, 3) if successful else None,
    )


def _work_item(item_hex: str, *, sample_id: str, column: int = 0) -> WorkItem:
    return WorkItem(
        item_id=item_hex * 64,
        sample_id=sample_id,
        region_spec=RegionSpec(
            region_id="grid",
            region_instance_id=f"grid/r0/c{column}",
            kind=RegionKind.GRID,
            params={"rows": 1, "cols": 2},
        ),
        perturb_op=PerturbationOp.BLUR,
        perturb_params={"sigma": 1.0},
    )


def _perturbed(
    item_hex: str,
    *,
    sample_id: str,
    status: ItemStatus = ItemStatus.OK,
    column: int = 0,
) -> PerturbedDumpRecord:
    successful = status is ItemStatus.OK
    return PerturbedDumpRecord(
        work_item=_work_item(item_hex, sample_id=sample_id, column=column),
        status=status,
        seed_used=7,
        logits=_output(-1.0, 2.0) if successful else None,
        region_meta=RegionMeta(4, 0.25, "grid", "1") if successful else None,
        effective_area_px=4 if successful else None,
    )


def _writer(root: Path, config: ResolvedConfig) -> DumpWriter:
    return DumpWriter(root, config, code_version="test-code", mode="create", environment=ENVIRONMENT)


def _write_baseline_dump(root: Path, config: ResolvedConfig) -> None:
    """Write one dump covering every §N0 status-table row but the orphan one.

    - sample-a: clean ok, both perturbed items ok (normal case)
    - sample-b: clean ok, one perturbed item predict_failed (excluded value)
    - sample-c: clean load_failed (entire sample excluded from joined())
    """

    with _writer(root, config) as writer:
        writer.write_clean_many(
            (
                _clean("sample-a"),
                _clean("sample-b"),
                _clean("sample-c", status=ItemStatus.LOAD_FAILED),
            )
        )
        writer.write_perturbed_many(
            (
                _perturbed("a", sample_id="sample-a", column=0),
                _perturbed("b", sample_id="sample-a", column=1),
                _perturbed("c", sample_id="sample-b", column=0),
                _perturbed(
                    "d",
                    sample_id="sample-b",
                    column=1,
                    status=ItemStatus.PREDICT_FAILED,
                ),
                _perturbed("e", sample_id="sample-c", column=0),
            )
        )


def test_clean_and_items_return_every_authoritative_row(tmp_path: Path) -> None:
    root = tmp_path / "dump"
    _write_baseline_dump(root, _config(tmp_path))

    handle = DumpHandle(root)
    assert len(handle.clean()) == 3
    assert len(handle.items()) == 5


def test_manifest_matches_the_resolved_config(tmp_path: Path) -> None:
    root = tmp_path / "dump"
    config = _config(tmp_path)
    _write_baseline_dump(root, config)

    handle = DumpHandle(root)
    assert handle.manifest.resolved_config == config


def test_joined_excludes_samples_with_clean_failure(tmp_path: Path) -> None:
    root = tmp_path / "dump"
    _write_baseline_dump(root, _config(tmp_path))

    joined = DumpHandle(root).joined()
    assert set(joined["sample_id"]) == {"sample-a", "sample-b"}
    assert len(joined) == 4


def test_joined_keeps_items_whose_perturbed_status_failed(tmp_path: Path) -> None:
    root = tmp_path / "dump"
    _write_baseline_dump(root, _config(tmp_path))

    joined = DumpHandle(root).joined()
    failed_row = joined[joined["item_id"] == "d" * 64].iloc[0]
    assert failed_row["status_perturbed"] == ItemStatus.PREDICT_FAILED.value
    assert failed_row["logits_perturbed"] is None
    assert failed_row["status_clean"] == ItemStatus.OK.value
    assert failed_row["logits_clean"] is not None


def test_joined_carries_region_and_perturbation_metadata(tmp_path: Path) -> None:
    root = tmp_path / "dump"
    _write_baseline_dump(root, _config(tmp_path))

    joined = DumpHandle(root).joined()
    row = joined[joined["item_id"] == "a" * 64].iloc[0]
    assert row["region_id"] == "grid"
    assert row["region_instance_id"] == "grid/r0/c0"
    assert row["region_kind"] == RegionKind.GRID.value
    assert row["perturb_op"] == PerturbationOp.BLUR.value
    assert bool(row["is_control"]) is False
    assert row["gt_label"] == 1


def test_joined_rejects_items_with_no_matching_clean_sample(tmp_path: Path) -> None:
    root = tmp_path / "dump"
    config = _config(tmp_path)
    with _writer(root, config) as writer:
        writer.write_clean(_clean("sample-a"))
        writer.write_perturbed(_perturbed("a", sample_id="ghost"))

    with pytest.raises(MetricsCorruptionError, match="absent from clean data"):
        DumpHandle(root).joined()


def test_summary_reports_status_counts_and_exclusions(tmp_path: Path) -> None:
    root = tmp_path / "dump"
    _write_baseline_dump(root, _config(tmp_path))

    summary = DumpHandle(root).summary()
    assert summary["total_clean_samples"] == 3
    assert summary["total_perturbed_items"] == 5
    assert summary["clean_status_counts"][ItemStatus.OK] == 2
    assert summary["clean_status_counts"][ItemStatus.LOAD_FAILED] == 1
    assert summary["perturbed_status_counts"][ItemStatus.OK] == 4
    assert summary["perturbed_status_counts"][ItemStatus.PREDICT_FAILED] == 1
    assert summary["samples_excluded_clean_failed"] == 1
    assert summary["items_excluded_perturbed_failed"] == 1


def test_summary_rejects_items_with_no_matching_clean_sample(tmp_path: Path) -> None:
    root = tmp_path / "dump"
    config = _config(tmp_path)
    with _writer(root, config) as writer:
        writer.write_clean(_clean("sample-a"))
        writer.write_perturbed(_perturbed("a", sample_id="ghost"))

    with pytest.raises(MetricsCorruptionError, match="absent from clean data"):
        DumpHandle(root).summary()
