"""Tests for bounded preprocessing/effective-area sanity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ssat.core.adapter import CallableAdapter, DeclarativePreprocessor
from ssat.core.config.schema import (
    DumpConfig,
    PerturbationConfig,
    ResolvedConfig,
    ResolvedRegionConfig,
    RuntimeConfig,
)
from ssat.core.estimate import AdvisoryCode, AreaSanityCheck
from ssat.core.plan import PlanBuilder
from ssat.core.region import RegionResolver
from ssat.core.region.types import RegionMeta
from ssat.core.source.types import LoadError, LoadedSample, SampleMeta
from ssat.core.types import PerturbationOp, RegionKind


class MemorySource:
    """Provide deterministic in-memory images and record bounded selection."""

    def __init__(self, *, n_samples: int = 1, frames: int = 1) -> None:
        self.samples = tuple(
            SampleMeta(f"sample-{index}", Path("unused"), 0)
            for index in range(n_samples)
        )
        self.frames = frames
        self.load_calls: list[str] = []
        self.failed_ids: set[str] = set()

    def list_samples(self) -> list[SampleMeta]:
        return list(self.samples)

    def load(self, sample_id: str) -> LoadedSample | LoadError:
        self.load_calls.append(sample_id)
        if sample_id in self.failed_ids:
            return LoadError(sample_id, "injected", "load failed")
        array = np.zeros((self.frames, 32, 32, 3), dtype=np.uint8)
        return LoadedSample(array, sample_id, array.shape, "a" * 64, 0)


def _adapter(*, preprocessing=None, transform_mask_fn=None) -> CallableAdapter:
    return CallableAdapter(
        lambda batch: np.zeros((len(batch), 2), dtype=np.float32),
        model_id="area-sanity-fixture",
        preprocessor=preprocessing,
        transform_mask_fn=transform_mask_fn,
    )


def _config(
    tmp_path: Path,
    adapter: CallableAdapter,
    *,
    rows: int = 4,
    cols: int = 4,
    invert_mask: bool = False,
    duplicate_variants: bool = False,
) -> ResolvedConfig:
    perturbations = [
        PerturbationConfig(
            op=PerturbationOp.CONSTANT_FILL,
            params={"value": 0},
            invert_mask=invert_mask,
            seed_salts=(0, 1) if duplicate_variants else (0,),
        )
    ]
    if duplicate_variants:
        perturbations.append(
            PerturbationConfig(
                op=PerturbationOp.GAUSSIAN_NOISE,
                params={"sigma": 1.0},
                invert_mask=invert_mask,
                seed_salts=(0, 1),
            )
        )
    return ResolvedConfig(
        config_base_dir=tmp_path.resolve(),
        regions=(
            ResolvedRegionConfig(
                region_id="grid",
                kind=RegionKind.GRID,
                params={"rows": rows, "cols": cols},
            ),
        ),
        perturbations=tuple(perturbations),
        runtime=RuntimeConfig(variants_per_chunk=17, num_workers=0),
        dump=DumpConfig(),
        adapter_spec=adapter.describe(),
    )


def _run(
    tmp_path: Path,
    source: MemorySource,
    adapter: CallableAdapter,
    *,
    rows: int = 4,
    cols: int = 4,
    invert_mask: bool = False,
    duplicate_variants: bool = False,
    check: AreaSanityCheck | None = None,
    region_resolver=None,
):
    config = _config(
        tmp_path,
        adapter,
        rows=rows,
        cols=cols,
        invert_mask=invert_mask,
        duplicate_variants=duplicate_variants,
    )
    builder = PlanBuilder(config, source)
    return (check or AreaSanityCheck()).run(
        config,
        source.samples,
        builder,
        source,
        adapter,
        region_resolver=region_resolver,
    )


def test_center_crop_detects_grid_area_confound(tmp_path: Path) -> None:
    source = MemorySource()
    adapter = _adapter(
        preprocessing=DeclarativePreprocessor(
            [
                {"op": "resize", "size": 256},
                {"op": "center_crop", "size": 224},
            ]
        )
    )

    result = _run(tmp_path, source, adapter)

    resolver = RegionResolver()
    config = _config(tmp_path, adapter)
    builder = PlanBuilder(config, source)
    areas = set()
    for chunk in builder.enumerate():
        for item in builder.materialize(chunk.chunk_id).items:
            mask, _ = resolver.resolve(
                (1, 32, 32, 3), item.region_spec, np.random.default_rng(0)
            )
            areas.add(int(adapter.transform_mask(mask).sum()))
    assert areas == {2304, 3072, 4096}
    assert result.passed is False
    assert result.candidate_regions == 16
    assert result.evaluated_regions == 16
    assert result.failed_regions == 8
    assert result.maximum_relative_deviation == pytest.approx(0.3061224489)
    assert AdvisoryCode.AREA_SANITY_DEVIATION_EXCEEDED in {
        advisory.code for advisory in result.advisories
    }


def test_crop_free_resize_preserves_all_grid_area_ratios(tmp_path: Path) -> None:
    source = MemorySource()
    adapter = _adapter(
        preprocessing=DeclarativePreprocessor(
            [{"op": "resize", "size": [224, 224]}]
        )
    )

    result = _run(tmp_path, source, adapter)

    assert result.passed is True
    assert result.failed_regions == 0
    assert result.maximum_relative_deviation == pytest.approx(0.0)
    assert result.worst_effective_area_ratio == pytest.approx(1 / 16)


def test_tolerance_is_inclusive_and_configurable(tmp_path: Path) -> None:
    source = MemorySource()

    def remove_one_quarter(mask: np.ndarray) -> np.ndarray:
        transformed = mask.copy()
        transformed[..., :8, :] = False
        return transformed

    adapter = _adapter(transform_mask_fn=remove_one_quarter)

    passing = _run(
        tmp_path,
        source,
        adapter,
        rows=1,
        cols=1,
        check=AreaSanityCheck(max_relative_deviation=0.25),
    )
    failing = _run(
        tmp_path,
        source,
        adapter,
        rows=1,
        cols=1,
        check=AreaSanityCheck(max_relative_deviation=0.249),
    )

    assert passing.maximum_relative_deviation == pytest.approx(0.25)
    assert passing.passed is True
    assert failing.passed is False


def test_invert_mask_and_temporal_mean_follow_runtime_area_rules(
    tmp_path: Path,
) -> None:
    source = MemorySource(frames=2)

    def alternating_frames(mask: np.ndarray) -> np.ndarray:
        return np.stack((mask, np.zeros_like(mask)))

    adapter = _adapter(transform_mask_fn=alternating_frames)
    result = _run(
        tmp_path,
        source,
        adapter,
        rows=1,
        cols=2,
        invert_mask=True,
    )

    assert result.passed is False
    assert result.maximum_relative_deviation == pytest.approx(0.5)
    assert result.worst_intended_area_ratio == pytest.approx(0.5)
    assert result.worst_effective_area_ratio == pytest.approx(0.25)


def test_zero_area_contract_passes_only_when_effective_area_is_also_zero(
    tmp_path: Path,
) -> None:
    source = MemorySource()
    adapter = _adapter(transform_mask_fn=lambda mask: mask.copy())

    class EmptyResolver:
        def resolve(self, shape, spec, rng):
            mask = np.zeros(shape[1:3], dtype=np.bool_)
            return mask, RegionMeta(0, 0.0, spec.kind.value, "test")

    result = _run(
        tmp_path,
        source,
        adapter,
        rows=1,
        cols=1,
        region_resolver=EmptyResolver(),
    )

    assert result.passed is True
    assert result.maximum_relative_deviation == 0.0


def test_deduplication_and_bounds_are_deterministic(tmp_path: Path) -> None:
    source = MemorySource(n_samples=5)
    adapter = _adapter(transform_mask_fn=lambda mask: mask.copy())
    result = _run(
        tmp_path,
        source,
        adapter,
        rows=1,
        cols=300,
        duplicate_variants=True,
        check=AreaSanityCheck(max_samples=3, max_regions_per_sample=256),
    )

    assert source.load_calls == ["sample-0", "sample-2", "sample-4"]
    assert result.selected_samples == 3
    assert result.candidate_regions == 900
    assert result.evaluated_regions == 768
    assert result.coverage_truncated is True
    assert AdvisoryCode.AREA_SANITY_COVERAGE_TRUNCATED in {
        advisory.code for advisory in result.advisories
    }


def test_unavailable_mask_geometry_does_not_fail(tmp_path: Path) -> None:
    source = MemorySource()
    adapter = _adapter()

    result = _run(tmp_path, source, adapter)

    assert result.passed is None
    assert result.evaluated_regions == 0
    assert source.load_calls == []
    assert tuple(advisory.code for advisory in result.advisories) == (
        AdvisoryCode.AREA_SANITY_UNAVAILABLE,
    )


def test_load_failure_is_a_failed_partial_check(tmp_path: Path) -> None:
    source = MemorySource()
    source.failed_ids.add("sample-0")
    adapter = _adapter(transform_mask_fn=lambda mask: mask.copy())

    result = _run(tmp_path, source, adapter, rows=1, cols=2)

    assert result.passed is False
    assert result.evaluated_regions == 0
    assert result.failed_regions == 2
    assert AdvisoryCode.AREA_SANITY_PARTIAL_FAILURES in {
        advisory.code for advisory in result.advisories
    }
