"""Tests for the metrics engine's DebugViz V2 spatial-sensitivity heatmap view."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from synthetic_dump_builder import (
    build_resolved_config,
    clean_record,
    compute_and_save_metrics,
    image_manifest_source_provenance,
    perturbed_record,
    skeleton_bbox_source,
    video_manifest_source_provenance,
    write_dump,
)

from ssat.core.config.schema import ResolvedRegionConfig, SourceProvenance
from ssat.core.region.skeleton_store import load_skeleton_bbox_store
from ssat.core.source import ImageFolderSource, VideoFolderSource
from ssat.core.source.types import SampleMeta
from ssat.core.types import RegionKind
from ssat.metrics.builtin_metrics.continuous import GtLogitDrop
from ssat.metrics.errors import DebugVizError
from ssat.metrics.registry import MetricRegistry
from ssat.metrics.store import load_metrics
from ssat.metrics.types import RegionGeometryRef, SpatialProfile
from ssat.metrics.viz._shared import open_image_source, open_skeleton_store
from ssat.metrics.viz.heatmap import (
    _select_representative_frame,
    resolve_heatmap_view,
    save_heatmap_views,
    select_spatial_profile_rows,
)
from ssat.utils.io import sha256_file

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_classification"
_FIXTURE_MANIFEST = _FIXTURE_ROOT / "manifest.json"
_METRIC_NAME = "gt_logit_drop"

_VIDEO_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_video"
_VIDEO_FIXTURE_MANIFEST = _VIDEO_FIXTURE_ROOT / "manifest.json"
_VIDEO_FIXTURE_SKELETON_BBOX = _VIDEO_FIXTURE_ROOT / "skeleton_bbox.json"
_VIDEO_NUM_FRAMES = 8
_GRID_REGIONS = (
    ResolvedRegionConfig(region_id="grid", kind=RegionKind.GRID, params={"rows": 1, "cols": 3}),
)
_CLEAN_LOGITS = np.array([1.0, 0.0])
# One entry per grid column, chosen so the degradation strictly increases
# c0 -> c1 -> c2 (gt_logit_drop = clean_logits[gt] - perturbed_logits[gt]).
_COLUMN_PERTURBED_LOGITS = (
    np.array([1.0, 0.0]),  # col 0: no drop -> degradation 0.0
    np.array([0.5, 0.5]),  # col 1: mild drop -> degradation 0.5
    np.array([-1.0, 2.0]),  # col 2: severe drop -> degradation 2.0
)


def _fixture_source() -> ImageFolderSource:
    """Build an ImageFolderSource over the real, committed fixture images."""

    document = json.loads(_FIXTURE_MANIFEST.read_text())
    samples = [
        SampleMeta(entry["sample_id"], _FIXTURE_ROOT / entry["path"], entry.get("gt_label"))
        for entry in document["samples"]
        if entry["expected_status"] == "ok"
    ]
    return ImageFolderSource(samples)


def _registry() -> MetricRegistry:
    registry = MetricRegistry()
    registry.register(GtLogitDrop())
    return registry


def _build_grid_metrics(tmp_path: Path, *, sample_id: str = "synthetic-000") -> tuple[Path, Path]:
    """Write a synthetic dump with 3 grid columns of increasing degradation, then aggregate it."""

    config = build_resolved_config(
        tmp_path,
        regions=_GRID_REGIONS,
        source_provenance=image_manifest_source_provenance(_FIXTURE_MANIFEST),
    )
    clean_records = (clean_record(sample_id, logits=_CLEAN_LOGITS),)
    perturbed_records = tuple(
        perturbed_record(
            index,
            sample_id=sample_id,
            region_id="grid",
            region_instance_id=f"grid/r0/c{index}",
            logits=logits,
            region_params={"rows": 1, "cols": 3, "row_index": 0, "col_index": index},
        )
        for index, logits in enumerate(_COLUMN_PERTURBED_LOGITS)
    )
    dump_root = tmp_path / "dump"
    write_dump(dump_root, config, clean_records=clean_records, perturbed_records=perturbed_records)

    metrics_dir = tmp_path / "metrics"
    compute_and_save_metrics(
        dump_root, config, metrics_dir, registry=_registry(), primary_metric=_METRIC_NAME
    )
    return dump_root, metrics_dir


def _build_imagenet_dump(tmp_path: Path, *, legacy_provenance: bool) -> Path:
    """Write a dump whose source uses the ImageNet text-file-list format."""

    annotation_file = tmp_path / "imagenet.txt"
    annotation_file.write_text("images/sample_000.png 0\n", encoding="utf-8")
    loader_parameters = {} if legacy_provenance else {"root": str(_FIXTURE_ROOT)}
    provenance = SourceProvenance(
        kind="imagenet",
        manifest=annotation_file.resolve(),
        manifest_hash=sha256_file(annotation_file),
        loader_parameters=loader_parameters,
    )
    config = build_resolved_config(
        tmp_path,
        regions=_GRID_REGIONS,
        source_provenance=provenance,
    )
    if legacy_provenance:
        config_source = tmp_path / "audit.yaml"
        config_source.write_text(
            json.dumps(
                {
                    "source": {
                        "kind": "imagenet",
                        "root": str(_FIXTURE_ROOT),
                        "annotation_file": str(annotation_file),
                    }
                }
            ),
            encoding="utf-8",
        )
        config = config.model_copy(update={"config_source": config_source.resolve()})

    dump_root = tmp_path / "dump"
    write_dump(
        dump_root,
        config,
        clean_records=(clean_record("images/sample_000.png", logits=_CLEAN_LOGITS),),
        perturbed_records=(),
    )
    return dump_root


def test_resolved_heatmap_matches_grid_coordinates_and_degradation_order(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_grid_metrics(tmp_path)
    _, result, _ = load_metrics(metrics_dir)
    rows_by_sample = select_spatial_profile_rows(
        result.spatial_profile, metric_name=_METRIC_NAME, n_samples=1
    )
    view = resolve_heatmap_view(
        "synthetic-000", rows_by_sample["synthetic-000"], source=_fixture_source()
    )

    height, width = view.original.shape[:2]
    column_means = []
    for col_index, expected_degradation in enumerate((0.0, 0.5, 2.0)):
        col_start, col_end = col_index * width // 3, (col_index + 1) * width // 3
        column = view.intensity[:, col_start:col_end]
        assert np.all(column == expected_degradation)
        column_means.append(float(np.mean(column)))

    # Pixel intensity order must match the SpatialProfile degradation order.
    assert column_means == sorted(column_means)


def test_explicit_region_is_supported(tmp_path: Path) -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:, :32] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask, mode="L").save(mask_path)

    row = SpatialProfile(
        sample_id="synthetic-000",
        region_key="explicit::explicit/mask-0",
        metric_name=_METRIC_NAME,
        region_geometry_ref=RegionGeometryRef(
            region_kind=RegionKind.EXPLICIT,
            region_params_json="{}",
            ref=str(mask_path),
            ref_hash=sha256_file(mask_path),
        ),
        degradation=1.5,
    )

    view = resolve_heatmap_view("synthetic-000", [row], source=_fixture_source())

    assert np.all(view.intensity[:, :32] == 1.5)
    assert np.all(np.isnan(view.intensity[:, 32:]))


def test_random_area_match_region_raises_debug_viz_error() -> None:
    row = SpatialProfile(
        sample_id="synthetic-000",
        region_key="random::random/0",
        metric_name=_METRIC_NAME,
        region_geometry_ref=RegionGeometryRef(
            region_kind=RegionKind.RANDOM_AREA_MATCH,
            region_params_json='{"target_area_px": 100}',
        ),
        degradation=1.0,
    )

    with pytest.raises(DebugVizError, match="random_area_match"):
        resolve_heatmap_view("synthetic-000", [row], source=_fixture_source())


def test_save_heatmap_views_writes_one_png_per_sample(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_grid_metrics(tmp_path)
    output_dir = tmp_path / "debug_viz" / "heatmap"

    saved = save_heatmap_views(
        dump_root, metrics_dir, output_dir, metric_name=_METRIC_NAME, n_samples=5
    )

    assert saved == (output_dir / "sample_synthetic-000.png",)
    assert saved[0].is_file()
    assert saved[0].stat().st_size > 0


def test_select_spatial_profile_rows_raises_for_unknown_sample(tmp_path: Path) -> None:
    _, metrics_dir = _build_grid_metrics(tmp_path)
    _, result, _ = load_metrics(metrics_dir)

    with pytest.raises(DebugVizError, match="synthetic-999"):
        select_spatial_profile_rows(
            result.spatial_profile, metric_name=_METRIC_NAME, sample_ids=["synthetic-999"]
        )


def test_open_image_source_used_by_save_heatmap_views_matches_fixture(tmp_path: Path) -> None:
    dump_root, _ = _build_grid_metrics(tmp_path)
    source = open_image_source(dump_root)
    loaded = source.load("synthetic-000")
    assert loaded.original_shape == (1, 64, 64, 3)


def test_open_image_source_supports_imagenet_file_list(tmp_path: Path) -> None:
    dump_root = _build_imagenet_dump(tmp_path, legacy_provenance=False)

    loaded = open_image_source(dump_root).load("images/sample_000.png")

    assert loaded.original_shape == (1, 64, 64, 3)
    assert loaded.gt_label == 0


def test_open_image_source_supports_legacy_imagenet_provenance(tmp_path: Path) -> None:
    dump_root = _build_imagenet_dump(tmp_path, legacy_provenance=True)

    loaded = open_image_source(dump_root).load("images/sample_000.png")

    assert loaded.original_shape == (1, 64, 64, 3)
    assert loaded.gt_label == 0


# --- video / skeleton_parts -------------------------------------------------


def _video_source() -> VideoFolderSource:
    """Build a VideoFolderSource over the real, committed video fixture."""

    document = json.loads(_VIDEO_FIXTURE_MANIFEST.read_text())
    samples = [
        SampleMeta(entry["sample_id"], _VIDEO_FIXTURE_ROOT / entry["path"], entry.get("gt_label"))
        for entry in document["samples"]
        if entry["expected_status"] == "ok"
    ]
    return VideoFolderSource(samples, num_frames=_VIDEO_NUM_FRAMES, sampling="uniform")


def _skeleton_row(sample_id: str, *, degradation: float = 1.0) -> SpatialProfile:
    """Build a skeleton_parts SpatialProfile row for the committed video fixture's left_arm part."""

    return SpatialProfile(
        sample_id=sample_id,
        region_key=f"occlude_left_arm::occlude_left_arm/{sample_id}",
        metric_name=_METRIC_NAME,
        region_geometry_ref=RegionGeometryRef(
            region_kind=RegionKind.SKELETON_PARTS,
            region_params_json=json.dumps(
                {"sample_id": sample_id, "body_part": "left_arm", "bbox_scale": 1.0}
            ),
        ),
        degradation=degradation,
    )


def _build_video_dump(tmp_path: Path, *, with_skeleton_source: bool) -> Path:
    """Write a dump whose source is the committed video fixture, optionally with skeleton_source."""

    config = build_resolved_config(
        tmp_path,
        regions=_GRID_REGIONS,
        source_provenance=video_manifest_source_provenance(
            _VIDEO_FIXTURE_MANIFEST, num_frames=_VIDEO_NUM_FRAMES
        ),
        skeleton_source=(
            skeleton_bbox_source(_VIDEO_FIXTURE_SKELETON_BBOX) if with_skeleton_source else None
        ),
    )
    dump_root = tmp_path / "dump"
    write_dump(
        dump_root,
        config,
        clean_records=(clean_record("synthetic-video-000", logits=_CLEAN_LOGITS),),
        perturbed_records=(),
    )
    return dump_root


def test_open_image_source_supports_video_manifest_provenance(tmp_path: Path) -> None:
    dump_root = _build_video_dump(tmp_path, with_skeleton_source=False)

    source = open_image_source(dump_root)
    loaded = source.load("synthetic-video-000")

    assert isinstance(source, VideoFolderSource)
    assert loaded.original_shape == (_VIDEO_NUM_FRAMES, 64, 64, 3)


def test_open_image_source_raises_for_kinetics400_provenance(tmp_path: Path) -> None:
    provenance = SourceProvenance(
        kind="kinetics400",
        manifest=_VIDEO_FIXTURE_MANIFEST.resolve(),
        manifest_hash=sha256_file(_VIDEO_FIXTURE_MANIFEST),
    )
    config = build_resolved_config(tmp_path, regions=_GRID_REGIONS, source_provenance=provenance)
    dump_root = tmp_path / "dump"
    write_dump(
        dump_root,
        config,
        clean_records=(clean_record("synthetic-video-000", logits=_CLEAN_LOGITS),),
        perturbed_records=(),
    )

    with pytest.raises(DebugVizError, match="kinetics400"):
        open_image_source(dump_root)


def test_open_skeleton_store_loads_configured_bbox_data(tmp_path: Path) -> None:
    dump_root = _build_video_dump(tmp_path, with_skeleton_source=True)

    store = open_skeleton_store(dump_root)

    assert store is not None
    assert store.frame_size("synthetic-video-000") == (64, 64)
    assert len(store.get("synthetic-video-000", "left_arm")) == _VIDEO_NUM_FRAMES


def test_open_skeleton_store_returns_none_when_unconfigured(tmp_path: Path) -> None:
    dump_root, _ = _build_grid_metrics(tmp_path)

    assert open_skeleton_store(dump_root) is None


def test_resolve_heatmap_view_selects_middle_frame_and_slices_mask(tmp_path: Path) -> None:
    del tmp_path
    store = load_skeleton_bbox_store(_VIDEO_FIXTURE_SKELETON_BBOX)
    row = _skeleton_row("synthetic-video-000", degradation=2.5)

    view = resolve_heatmap_view(
        "synthetic-video-000", [row], source=_video_source(), skeleton_store=store
    )

    assert view.num_frames == _VIDEO_NUM_FRAMES
    assert view.frame_index == _VIDEO_NUM_FRAMES // 2
    assert view.original.shape == (64, 64, 3)
    assert view.intensity.shape == (64, 64)
    # The committed fixture's left_arm box is tracked on every frame, so the
    # painted region is exactly the frame_index-th box, not frame 0's.
    boxes = store.get("synthetic-video-000", "left_arm")
    x1, y1, x2, y2 = boxes[view.frame_index]
    assert np.all(view.intensity[int(y1) : int(y2), int(x1) : int(x2)] == 2.5)
    assert np.isnan(view.intensity[0, 0])


def test_resolve_heatmap_view_without_skeleton_store_raises_debug_viz_error(tmp_path: Path) -> None:
    del tmp_path
    row = _skeleton_row("synthetic-video-000")

    with pytest.raises(DebugVizError, match="synthetic-video-000"):
        resolve_heatmap_view("synthetic-video-000", [row], source=_video_source())


def test_resolve_heatmap_view_preserves_frame_zero_for_image_sources(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_grid_metrics(tmp_path)
    _, result, _ = load_metrics(metrics_dir)
    rows_by_sample = select_spatial_profile_rows(
        result.spatial_profile, metric_name=_METRIC_NAME, n_samples=1
    )

    view = resolve_heatmap_view(
        "synthetic-000", rows_by_sample["synthetic-000"], source=_fixture_source()
    )

    assert view.frame_index == 0
    assert view.num_frames == 1


# --- _select_representative_frame (unit-level, direct) ---------------------


def test_select_representative_frame_returns_zero_for_a_single_frame() -> None:
    mask = np.zeros((2, 2), dtype=np.bool_)
    assert _select_representative_frame(1, [mask]) == 0


def test_select_representative_frame_prefers_the_middle_frame_when_valid() -> None:
    mask = np.ones((5, 2, 2), dtype=np.bool_)
    assert _select_representative_frame(5, [mask]) == 2


def test_select_representative_frame_falls_back_to_nearest_valid_frame() -> None:
    mask = np.zeros((5, 2, 2), dtype=np.bool_)
    mask[3] = True  # middle=2 is empty; nearest valid frame is 3 (|3-2|=1)
    assert _select_representative_frame(5, [mask]) == 3


def test_select_representative_frame_ties_toward_the_lower_index() -> None:
    mask = np.zeros((5, 2, 2), dtype=np.bool_)
    mask[1] = True
    mask[3] = True  # both equidistant from middle=2; tie goes to the lower index
    assert _select_representative_frame(5, [mask]) == 1


def test_select_representative_frame_falls_back_to_middle_when_nothing_is_tracked() -> None:
    mask = np.zeros((5, 2, 2), dtype=np.bool_)
    assert _select_representative_frame(5, [mask]) == 2
