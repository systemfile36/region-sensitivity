"""Unit tests for ssat.report.assets's asset linker.

Builds one real dump+metrics pair through the synthetic pipeline (mirroring
tests/unit/test_metrics_viz_heatmap.py's own precedent for exercising
``ssat.metrics.viz.heatmap`` against real, committed fixture images) and one
real ``AssembledReport`` through ``ReportDataAssembler`` — the actual output
:func:`link_assets`/:func:`apply_asset_manifest` are meant to consume,
rather than a hand-built stand-in (``ReportModel`` has too many mandatory
nested sections to hand-build meaningfully here).
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
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
from ssat.metrics.registry import MetricRegistry
from ssat.report.adapters import ClassificationAdapter
from ssat.report.assembler import AssembledReport, ReportDataAssembler
from ssat.report.assets import (
    AssetManifest,
    _save_thumbnail_png,
    apply_asset_manifest,
    link_assets,
)

_METRIC_NAME = "gt_logit_drop"
_CLEAN_GT_LOGIT = 10.0
_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_classification"
_FIXTURE_MANIFEST = _FIXTURE_ROOT / "manifest.json"

# region_key -> degradation, chosen so top-K/bottom-K selection is unambiguous.
_GOOD_SAMPLES = {
    "synthetic-000": 1.0,
    "synthetic-001": 5.0,
    "synthetic-002": 9.0,
}
# Uses a random_area_match region — not reproducible by resolve_heatmap_view
# (ssat/metrics/viz/heatmap.py's own documented constraint) — and is always
# the most vulnerable sample, so it lands in top_k for any top_k >= 1.
_BAD_SAMPLE = "synthetic-003"
_BAD_DEGRADATION = 20.0


def _registry() -> MetricRegistry:
    registry = MetricRegistry()
    registry.register(GtLogitDrop())
    return registry


def _build_dump_and_metrics(
    tmp_path: Path, *, include_bad_sample: bool, with_images: bool = True
) -> tuple[Path, Path]:
    source_provenance = (
        image_manifest_source_provenance(_FIXTURE_MANIFEST) if with_images else None
    )
    config = build_resolved_config(
        tmp_path,
        regions=(
            ResolvedRegionConfig(region_id="grid", kind=RegionKind.GRID, params={"rows": 1, "cols": 1}),
            ResolvedRegionConfig(region_id="random", kind=RegionKind.RANDOM_AREA_MATCH, params={}),
        ),
        source_provenance=source_provenance,
    )

    sample_degradations = dict(_GOOD_SAMPLES)
    if include_bad_sample:
        sample_degradations[_BAD_SAMPLE] = _BAD_DEGRADATION

    clean_records = tuple(
        clean_record(sample_id, logits=np.array([_CLEAN_GT_LOGIT, 0.0]), gt_label=0)
        for sample_id in sample_degradations
    )
    perturbed_records = []
    for index, (sample_id, degradation) in enumerate(sample_degradations.items()):
        perturbed_gt_logit = _CLEAN_GT_LOGIT - degradation
        if sample_id == _BAD_SAMPLE:
            perturbed_records.append(
                perturbed_record(
                    index,
                    sample_id=sample_id,
                    region_id="random",
                    region_instance_id="random/0",
                    logits=np.array([perturbed_gt_logit, 0.0]),
                    region_kind=RegionKind.RANDOM_AREA_MATCH,
                    region_params={"target_area_px": 100},
                )
            )
        else:
            perturbed_records.append(
                perturbed_record(
                    index,
                    sample_id=sample_id,
                    region_id="grid",
                    region_instance_id="grid/r0/c0",
                    logits=np.array([perturbed_gt_logit, 0.0]),
                    region_params={"rows": 1, "cols": 1, "row_index": 0, "col_index": 0},
                )
            )

    dump_root = tmp_path / "dump"
    write_dump(dump_root, config, clean_records=clean_records, perturbed_records=tuple(perturbed_records))

    metrics_dir = tmp_path / "metrics"
    compute_and_save_metrics(
        dump_root, config, metrics_dir, registry=_registry(), primary_metric=_METRIC_NAME
    )
    return dump_root, metrics_dir


def _assemble(dump_root: Path, metrics_dir: Path, *, top_k: int, bottom_k: int) -> AssembledReport:
    adapter = ClassificationAdapter(primary_metric=_METRIC_NAME)
    assembler = ReportDataAssembler(
        dump_root, metrics_dir, adapter=adapter, top_k=top_k, bottom_k=bottom_k, region_top_k=5
    )
    return assembler.assemble(_METRIC_NAME)


def _gallery_ids(assembled: AssembledReport) -> set[str]:
    rankings = assembled.model.sample_rankings
    return {card.sample_id for card in (*rankings.most_vulnerable, *rankings.most_robust)}


# --- link_assets --------------------------------------------------------------


def test_link_assets_writes_one_png_pair_per_gallery_sample(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_dump_and_metrics(tmp_path, include_bad_sample=False)
    assembled = _assemble(dump_root, metrics_dir, top_k=1, bottom_k=1)
    output_dir = tmp_path / "report"

    manifest = link_assets(assembled, dump_root, metrics_dir, output_dir, primary_metric=_METRIC_NAME)

    gallery_ids = _gallery_ids(assembled)
    assert manifest.assets_available is True
    assert set(manifest.heatmap_refs) == gallery_ids
    assert set(manifest.thumbnail_refs) == gallery_ids
    for ref in (*manifest.heatmap_refs.values(), *manifest.thumbnail_refs.values()):
        path = output_dir / ref
        assert path.is_file()
        assert path.stat().st_size > 0
        assert not Path(ref).is_absolute()

    # Fixture images are 64x64, below the 256px thumbnail cap -- never upscaled.
    one_thumbnail = output_dir / next(iter(manifest.thumbnail_refs.values()))
    with Image.open(one_thumbnail) as image:
        assert image.size == (64, 64)


def test_link_assets_isolates_debug_viz_error_to_one_sample(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_dump_and_metrics(tmp_path, include_bad_sample=True)
    assembled = _assemble(dump_root, metrics_dir, top_k=2, bottom_k=2)
    output_dir = tmp_path / "report"

    manifest = link_assets(assembled, dump_root, metrics_dir, output_dir, primary_metric=_METRIC_NAME)

    assert _BAD_SAMPLE in _gallery_ids(assembled)  # sanity: it was actually attempted
    assert _BAD_SAMPLE not in manifest.heatmap_refs
    assert _BAD_SAMPLE not in manifest.thumbnail_refs
    assert set(manifest.heatmap_refs) == set(_GOOD_SAMPLES)
    assert set(manifest.thumbnail_refs) == set(_GOOD_SAMPLES)


def test_link_assets_reports_unavailable_without_source_provenance(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_dump_and_metrics(
        tmp_path, include_bad_sample=False, with_images=False
    )
    assembled = _assemble(dump_root, metrics_dir, top_k=1, bottom_k=1)
    output_dir = tmp_path / "report"

    manifest = link_assets(assembled, dump_root, metrics_dir, output_dir, primary_metric=_METRIC_NAME)

    assert manifest == AssetManifest(assets_available=False, heatmap_refs={}, thumbnail_refs={})


def test_link_assets_returns_available_empty_manifest_when_gallery_is_empty(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_dump_and_metrics(tmp_path, include_bad_sample=False)
    assembled = _assemble(dump_root, metrics_dir, top_k=0, bottom_k=0)
    output_dir = tmp_path / "report"

    manifest = link_assets(assembled, dump_root, metrics_dir, output_dir, primary_metric=_METRIC_NAME)

    assert manifest == AssetManifest(assets_available=True, heatmap_refs={}, thumbnail_refs={})


def test_asset_refs_survive_moving_the_report_folder(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_dump_and_metrics(tmp_path, include_bad_sample=False)
    assembled = _assemble(dump_root, metrics_dir, top_k=1, bottom_k=1)
    output_dir = tmp_path / "original_report"

    manifest = link_assets(assembled, dump_root, metrics_dir, output_dir, primary_metric=_METRIC_NAME)

    moved_dir = tmp_path / "elsewhere" / "moved_report"
    moved_dir.parent.mkdir(parents=True)
    shutil.copytree(output_dir, moved_dir)

    for ref in (*manifest.heatmap_refs.values(), *manifest.thumbnail_refs.values()):
        assert (moved_dir / ref).is_file()


# --- apply_asset_manifest ------------------------------------------------------


def test_apply_asset_manifest_fills_gallery_cards_and_leaves_others_untouched(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_dump_and_metrics(tmp_path, include_bad_sample=False)
    assembled = _assemble(dump_root, metrics_dir, top_k=1, bottom_k=1)
    output_dir = tmp_path / "report"
    manifest = link_assets(assembled, dump_root, metrics_dir, output_dir, primary_metric=_METRIC_NAME)

    linked = apply_asset_manifest(assembled, manifest)

    gallery_ids = set(manifest.heatmap_refs)
    for card in linked.full_sample_rankings:
        if card.sample_id in gallery_ids:
            assert card.heatmap_asset_ref == manifest.heatmap_refs[card.sample_id]
            assert card.thumbnail_asset_ref == manifest.thumbnail_refs[card.sample_id]
        else:
            assert card.heatmap_asset_ref is None
            assert card.thumbnail_asset_ref is None


def test_apply_asset_manifest_keeps_sample_rankings_and_full_rankings_consistent(
    tmp_path: Path,
) -> None:
    dump_root, metrics_dir = _build_dump_and_metrics(tmp_path, include_bad_sample=True)
    assembled = _assemble(dump_root, metrics_dir, top_k=2, bottom_k=2)
    output_dir = tmp_path / "report"
    manifest = link_assets(assembled, dump_root, metrics_dir, output_dir, primary_metric=_METRIC_NAME)

    linked = apply_asset_manifest(assembled, manifest)

    by_id = {card.sample_id: card for card in linked.full_sample_rankings}
    gallery_cards = (
        *linked.model.sample_rankings.most_vulnerable,
        *linked.model.sample_rankings.most_robust,
    )
    assert gallery_cards  # sanity: the fixture actually exercises this path
    for card in gallery_cards:
        full_card = by_id[card.sample_id]
        assert card.heatmap_asset_ref == full_card.heatmap_asset_ref
        assert card.thumbnail_asset_ref == full_card.thumbnail_asset_ref


def test_apply_asset_manifest_is_a_noop_for_an_unavailable_manifest(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_dump_and_metrics(tmp_path, include_bad_sample=False)
    assembled = _assemble(dump_root, metrics_dir, top_k=1, bottom_k=1)
    manifest = AssetManifest(assets_available=False, heatmap_refs={}, thumbnail_refs={})

    linked = apply_asset_manifest(assembled, manifest)

    assert linked.model == assembled.model
    assert linked.full_sample_rankings == assembled.full_sample_rankings


# --- thumbnail resizing (256px longest edge, confirmed with the user) ----------


def test_save_thumbnail_png_resizes_to_256px_longest_edge_preserving_aspect(tmp_path: Path) -> None:
    original = np.zeros((600, 300, 3), dtype=np.uint8)
    path = tmp_path / "thumb.png"

    _save_thumbnail_png(original, path)

    with Image.open(path) as image:
        size = image.size

    assert max(size) == 256
    assert size == (128, 256)  # (width, height) -- aspect ratio preserved


# --- dependency direction -------------------------------------------------------


def test_report_assets_module_has_no_forbidden_package_imports() -> None:
    """Statically enforce the dependency direction for report.assets.

    ``ssat.metrics`` itself is *not* blanket-forbidden here (unlike
    ``report.charts``'s equivalent test) -- ``ssat.metrics.dump_reader`` and
    ``ssat.metrics.viz.heatmap`` are explicitly on this module's allow list,
    and ``ssat.metrics.store``/``ssat.metrics.viz._shared``/``ssat.metrics.
    errors`` are the minimal necessary extension of that same exception
    (module docstring). What is forbidden: every sibling report module this
    file has no business depending on, plus core/analysis/application.
    """

    source_path = Path(__file__).resolve().parents[2] / "ssat" / "report" / "assets.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_prefixes = (
        "ssat.report.assembler",
        "ssat.report.adapters",
        "ssat.report.exporter",
        "ssat.report.charts",
        "ssat.analysis",
        "ssat.core",
        "ssat.application",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(forbidden_prefixes), alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith(forbidden_prefixes), module
