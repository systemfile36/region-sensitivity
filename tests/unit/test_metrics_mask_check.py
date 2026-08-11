"""Tests for the metrics engine's DebugViz V1 mask-verification view."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from synthetic_dump_builder import (
    build_resolved_config,
    clean_record,
    image_manifest_source_provenance,
    perturbed_record,
    write_dump,
)

from ssat.core.config.schema import ResolvedRegionConfig
from ssat.core.source import ImageFolderSource
from ssat.core.source.types import SampleMeta
from ssat.core.types import RegionKind
from ssat.metrics.dump_reader import DumpHandle
from ssat.metrics.errors import DebugVizError
from ssat.metrics.viz.mask_check import (
    resolve_mask_check_view,
    save_mask_check_views,
    select_mask_check_rows,
)

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_classification"
_FIXTURE_MANIFEST = _FIXTURE_ROOT / "manifest.json"
_REGIONS = (ResolvedRegionConfig(region_id="grid", kind=RegionKind.GRID, params={"rows": 1, "cols": 3}),)
_CLEAN_LOGITS = np.array([1.0, 0.0])
_PERTURBED_LOGITS = np.array([0.0, 1.0])


def _fixture_source() -> ImageFolderSource:
    """Build an ImageFolderSource over the real, committed fixture images."""

    document = json.loads(_FIXTURE_MANIFEST.read_text())
    samples = [
        SampleMeta(entry["sample_id"], _FIXTURE_ROOT / entry["path"], entry.get("gt_label"))
        for entry in document["samples"]
        if entry["expected_status"] == "ok"
    ]
    return ImageFolderSource(samples)


def _build_grid_dump(
    tmp_path: Path,
    *,
    sample_id: str = "synthetic-000",
    col_index: int = 1,
    fill_value: float = 200.0,
    with_source_provenance: bool = True,
) -> Path:
    """Write a synthetic dump with one grid-perturbed item over a real fixture image."""

    config = build_resolved_config(
        tmp_path,
        regions=_REGIONS,
        source_provenance=(
            image_manifest_source_provenance(_FIXTURE_MANIFEST) if with_source_provenance else None
        ),
    )
    clean_records = (clean_record(sample_id, logits=_CLEAN_LOGITS),)
    perturbed_records = (
        perturbed_record(
            0,
            sample_id=sample_id,
            region_id="grid",
            region_instance_id=f"grid/r0/c{col_index}",
            logits=_PERTURBED_LOGITS,
            region_params={"rows": 1, "cols": 3, "row_index": 0, "col_index": col_index},
            perturb_params={"value": fill_value},
        ),
    )
    dump_root = tmp_path / "dump"
    write_dump(dump_root, config, clean_records=clean_records, perturbed_records=perturbed_records)
    return dump_root


def test_resolved_mask_matches_independently_computed_grid_coordinates(tmp_path: Path) -> None:
    dump_root = _build_grid_dump(tmp_path, col_index=1)
    handle = DumpHandle(dump_root)
    rows = select_mask_check_rows(handle.joined(), n_samples=1)
    view = resolve_mask_check_view(rows.iloc[0], source=_fixture_source())

    height, width = view.original.shape[:2]
    row_start, row_end = 0 * height // 1, 1 * height // 1
    col_start, col_end = 1 * width // 3, 2 * width // 3
    expected_mask = np.zeros((height, width), dtype=bool)
    expected_mask[row_start:row_end, col_start:col_end] = True

    assert np.array_equal(view.mask, expected_mask)
    assert view.mask.sum() > 0


def test_perturbation_is_confined_to_the_resolved_mask(tmp_path: Path) -> None:
    dump_root = _build_grid_dump(tmp_path, col_index=0, fill_value=200.0)
    handle = DumpHandle(dump_root)
    rows = select_mask_check_rows(handle.joined(), n_samples=1)
    view = resolve_mask_check_view(rows.iloc[0], source=_fixture_source())

    assert np.array_equal(view.perturbed[~view.mask], view.original[~view.mask])
    assert np.all(view.perturbed[view.mask] == 200)


def test_explicit_region_rows_raise_debug_viz_error() -> None:
    row = {
        "sample_id": "synthetic-000",
        "item_id": "a" * 64,
        "region_id": "explicit-region",
        "region_instance_id": "explicit-region/mask-0",
        "region_kind": "explicit",
        "region_params_json": "{}",
        "seed_used": 1,
        "invert_mask": False,
        "perturb_op": "constant_fill",
        "perturb_params_json": '{"value": 0.0}',
    }

    with pytest.raises(DebugVizError, match="explicit"):
        resolve_mask_check_view(row, source=_fixture_source())


def test_save_mask_check_views_without_source_provenance_raises(tmp_path: Path) -> None:
    dump_root = _build_grid_dump(tmp_path, with_source_provenance=False)

    with pytest.raises(DebugVizError, match="source_provenance"):
        save_mask_check_views(dump_root, tmp_path / "debug_viz" / "mask_check")


def test_save_mask_check_views_writes_one_png_per_sample(tmp_path: Path) -> None:
    dump_root = _build_grid_dump(tmp_path)
    output_dir = tmp_path / "debug_viz" / "mask_check"

    saved = save_mask_check_views(dump_root, output_dir, n_samples=5)

    assert saved == (output_dir / "sample_synthetic-000.png",)
    assert saved[0].is_file()
    assert saved[0].stat().st_size > 0


def test_select_mask_check_rows_is_sorted_and_capped(tmp_path: Path) -> None:
    config = build_resolved_config(tmp_path, regions=_REGIONS)
    clean_records = tuple(
        clean_record(sample_id, logits=_CLEAN_LOGITS)
        for sample_id in ("synthetic-002", "synthetic-000", "synthetic-001")
    )
    perturbed_records = tuple(
        perturbed_record(
            index,
            sample_id=sample_id,
            region_id="grid",
            region_instance_id="grid/r0/c0",
            logits=_PERTURBED_LOGITS,
            region_params={"rows": 1, "cols": 3, "row_index": 0, "col_index": 0},
        )
        for index, sample_id in enumerate(("synthetic-002", "synthetic-000", "synthetic-001"))
    )
    dump_root = tmp_path / "dump"
    write_dump(dump_root, config, clean_records=clean_records, perturbed_records=perturbed_records)

    handle = DumpHandle(dump_root)
    rows = select_mask_check_rows(handle.joined(), n_samples=2)

    assert list(rows["sample_id"]) == ["synthetic-000", "synthetic-001"]
