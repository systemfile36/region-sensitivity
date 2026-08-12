"""Round-trip tests wiring SkeletonRegionProvider + SkeletonPartsMaskGenerator
into planning, region resolution, and the perturbation pipeline via DI.

CLI/YAML wiring for skeleton config is a separate, not-yet-implemented stage
(see docs/VIDEO_SKELETON_EXTENSION_ANALYSIS_v1.md §5.4); these tests exercise
the components the way Python code (or a future Application-layer adapter)
would inject them, mirroring the pattern already used for per-frame masks in
tests/unit/test_runtime_pipeline.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ssat.core.adapter import CallableAdapter
from ssat.core.config.schema import (
    DumpConfig,
    PerturbationConfig,
    ResolvedConfig,
    ResolvedRegionConfig,
    RuntimeConfig,
)
from ssat.core.plan import PlanBuilder
from ssat.core.plan.region_expander import RegionExpander
from ssat.core.region.resolver import RegionResolver
from ssat.core.region.skeleton_provider import SkeletonRegionProvider
from ssat.core.region.skeleton_store import load_skeleton_bbox_store
from ssat.core.runtime import ChunkProcessor, PreparedChunk
from ssat.core.runtime.pipeline import iter_prepared_work_chunks
from ssat.core.source.types import LoadedSample, SampleMeta
from ssat.core.types import PerturbationOp, RegionKind

_WIDTH, _HEIGHT = 6, 4


class SkeletonClipSource:
    """Load one fixed three-frame clip whose sample_id matches skeleton data."""

    def __init__(self, sample_id: str = "clip_001", *, fill: int = 200) -> None:
        self.sample = SampleMeta(sample_id, Path("unused"))
        self._fill = fill

    def list_samples(self) -> list[SampleMeta]:
        return [self.sample]

    def load(self, sample_id: str) -> LoadedSample:
        array = np.full((3, _HEIGHT, _WIDTH, 3), self._fill, dtype=np.uint8)
        return LoadedSample(array, sample_id, array.shape, "a" * 64)


def _write_store(tmp_path: Path):
    payload = {
        "clip_001": {
            "frame_size": [_WIDTH, _HEIGHT],
            "parts": {
                "left_arm": [
                    [1.0, 1.0, 3.0, 3.0],  # frame 0: 2x2 box
                    None,  # frame 1: untracked
                    [2.0, 2.0, 5.0, 4.0],  # frame 2: clipped at bottom edge
                ]
            },
        }
    }
    path = tmp_path / "skeleton.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_skeleton_bbox_store(path)


def _resolver_with_store(store) -> RegionResolver:
    """Build a resolver whose default SkeletonPartsMaskGenerator sees real data."""

    return RegionResolver(skeleton_store=store)


def _config(tmp_path: Path, adapter: CallableAdapter) -> ResolvedConfig:
    return ResolvedConfig(
        config_base_dir=tmp_path.resolve(),
        regions=(
            ResolvedRegionConfig(
                region_id="occlude_left_arm",
                kind=RegionKind.SKELETON_PARTS,
                params={"body_part": "left_arm"},
            ),
        ),
        perturbations=(
            PerturbationConfig(op=PerturbationOp.CONSTANT_FILL, params={"value": 0}),
        ),
        runtime=RuntimeConfig(variants_per_chunk=1, target_batch_size=4),
        dump=DumpConfig(),
        adapter_spec=adapter.describe(),
    )


def test_chunk_processor_rasterizes_real_skeleton_data_into_per_frame_masks(
    tmp_path: Path,
) -> None:
    """A real store, provider, and generator produce the expected (T,H,W) mask."""

    store = _write_store(tmp_path)
    source = SkeletonClipSource()
    adapter = CallableAdapter(
        lambda batch: np.zeros((len(batch), 2), dtype=np.float32),
        model_id="pipeline",
        transform_mask_fn=lambda mask: mask,
    )
    config = _config(tmp_path, adapter)
    builder = PlanBuilder(
        config,
        source,
        region_expander=RegionExpander(SkeletonRegionProvider(store)),
    )
    chunks = builder.enumerate()
    processor = ChunkProcessor(
        chunks,
        builder,
        source,
        config.runtime.global_seed,
        region_resolver=_resolver_with_store(store),
    )

    result = processor[0]

    assert isinstance(result, PreparedChunk)
    assert len(result.masks) == 1
    mask = result.masks[0]
    assert mask.shape == (3, _HEIGHT, _WIDTH)

    expected = np.zeros((3, _HEIGHT, _WIDTH), dtype=np.bool_)
    expected[0, 1:3, 1:3] = True
    expected[2, 2:4, 2:5] = True  # y2=4 clipped to frame height
    assert np.array_equal(mask, expected)

    # mean_frame_area over frames [4, 0, 6] -> 10/3 -> rounds to 3.
    area = result.item_metas[0].region_meta.intended_area_px
    assert area == 3


def test_pipeline_perturbs_only_the_tracked_body_part_per_frame(
    tmp_path: Path,
) -> None:
    """The full pipeline zeroes only the rasterized region, frame by frame."""

    store = _write_store(tmp_path)
    source = SkeletonClipSource(fill=200)
    adapter = CallableAdapter(
        lambda batch: np.zeros((len(batch), 2), dtype=np.float32),
        model_id="pipeline",
        transform_mask_fn=lambda mask: mask,
    )
    config = _config(tmp_path, adapter)
    builder = PlanBuilder(
        config,
        source,
        region_expander=RegionExpander(SkeletonRegionProvider(store)),
    )
    chunk_meta = builder.enumerate()[0]

    prepared = list(
        iter_prepared_work_chunks(
            (chunk_meta,),
            builder,
            source,
            adapter,
            global_seed=config.runtime.global_seed,
            num_workers=0,
            fail_fast=False,
            region_resolver=_resolver_with_store(store),
        )
    )[0]

    assert len(prepared.items) == 1
    array = prepared.items[0].array  # (T, H, W, C) perturbed pixels

    expected_zeroed = np.zeros((3, _HEIGHT, _WIDTH), dtype=np.bool_)
    expected_zeroed[0, 1:3, 1:3] = True
    expected_zeroed[2, 2:4, 2:5] = True

    for frame in range(3):
        assert np.all(array[frame][expected_zeroed[frame]] == 0)
        assert np.all(array[frame][~expected_zeroed[frame]] == 200)
