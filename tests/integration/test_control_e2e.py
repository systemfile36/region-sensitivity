"""End-to-end coverage of the area-matched control path over real samples.

No test previously ran controls through PlanBuilder -> runtime -> aggregate
across more than one sample, which is why two defects survived: controls
scattered their area as isolated pixels instead of placing the target's
shape, and the metrics/analysis geometry checks compared a control's area
across samples even though it is re-drawn per item. Both only surface once
several samples' controls are aggregated together, so that is what this
module does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ssat.analysis.indexer import ComparisonIndexer
from ssat.core.adapter import CallableAdapter
from ssat.core.config.schema import (
    ControlConfig,
    DumpConfig,
    PerturbationConfig,
    ResolvedConfig,
    ResolvedRegionConfig,
    RuntimeConfig,
)
from ssat.core.dump import DumpWriter, EnvironmentSpec
from ssat.core.plan import PlanBuilder
from ssat.core.resume import ResumeIndex
from ssat.core.runtime import run_audit
from ssat.core.source import ImageFolderSource
from ssat.core.source.types import SampleMeta
from ssat.core.types import PerturbationOp, RegionKind
from ssat.metrics.aggregate import aggregate_item_metrics
from ssat.metrics.builtin_metrics import default_metric_registry
from ssat.metrics.dump_reader import DumpHandle

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_classification"
ENVIRONMENT = EnvironmentSpec(python_version="3.11.0", platform="test")


def _source() -> ImageFolderSource:
    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    return ImageFolderSource(
        [
            SampleMeta(
                sample_id=row["sample_id"],
                path=FIXTURE / row["path"],
                gt_label=row["gt_label"],
            )
            for row in manifest["samples"]
        ]
    )


def _config(tmp_path: Path, spec, *, n_controls: int, seed_salts: tuple[int, ...]) -> ResolvedConfig:
    return ResolvedConfig(
        config_base_dir=tmp_path.resolve(),
        regions=(
            ResolvedRegionConfig(
                region_id="grid",
                kind=RegionKind.GRID,
                params={"rows": 2, "cols": 2},
            ),
        ),
        perturbations=(
            PerturbationConfig(
                op=PerturbationOp.CONSTANT_FILL,
                params={"value": 0},
                seed_salts=seed_salts,
            ),
        ),
        controls=(ControlConfig(match_area_of="grid", n_samples=n_controls),),
        runtime=RuntimeConfig(variants_per_chunk=8, target_batch_size=5, num_workers=0),
        dump=DumpConfig(flush_every=16),
        adapter_spec=spec,
    )


def _adapter(*, crop: bool) -> CallableAdapter:
    def predict(batch: np.ndarray) -> np.ndarray:
        means = batch.astype(np.float32).mean(axis=(1, 2, 3, 4))
        return np.stack((means, means / 2.0, -means), axis=1)

    def transform_mask(mask: np.ndarray) -> np.ndarray:
        # A center crop is what makes model-space area depend on *where* a
        # region sits, which is the only condition under which a control's
        # area varies between draws. Without it every placement keeps the
        # same area and the defect this module guards is unreachable.
        return mask[8:-8, 8:-8].copy() if crop else mask.copy()

    return CallableAdapter(
        predict,
        model_id="control-e2e",
        class_names=("gradient", "geometry", "texture"),
        transform_mask_fn=transform_mask,
    )


def _run(
    tmp_path: Path,
    *,
    n_controls: int = 2,
    seed_salts: tuple[int, ...] = (0,),
    crop: bool = True,
):
    adapter = _adapter(crop=crop)
    source = _source()
    config = _config(tmp_path, adapter.describe(), n_controls=n_controls, seed_salts=seed_salts)
    root = tmp_path / "dump"
    writer = DumpWriter(
        root, config, code_version="control-e2e", mode="create", environment=ENVIRONMENT
    )
    run_audit(config, PlanBuilder(config, source), source, adapter, writer, ResumeIndex.open(root))
    writer.close(success=True)
    return DumpHandle(root), config


def test_control_geometry_varies_across_samples_without_failing_aggregation(
    tmp_path: Path,
) -> None:
    """A control is re-drawn per sample, and metrics must accept that."""

    handle, config = _run(tmp_path)
    items = handle.items()
    controls = items[(items["is_control"]) & (items["status"] == "ok")]

    assert not controls.empty
    # The same control slot really does land somewhere different per sample --
    # this is the condition that used to raise MetricsCorruptionError.
    per_slot_areas = controls.groupby("region_instance_id")["effective_area_px"].nunique()
    assert (per_slot_areas > 1).any()

    registry = default_metric_registry()
    joined = handle.joined()
    item_metrics = registry.compute_item_metrics(joined, adapter_spec=config.adapter_spec)

    # The assertion is that this returns at all: it used to raise
    # MetricsCorruptionError the moment a second sample's control reported a
    # different area for the same slot.
    result = aggregate_item_metrics(item_metrics, joined, registry, config)

    # Controls are excluded from N3 aggregation by design, so they must not
    # appear here -- they are consumed by the analysis layer instead.
    assert not [r for r in result.region_metrics if r.region_key.startswith("control:")]
    assert {r.region_key for r in result.region_metrics} == {
        f"grid::grid/r{row}/c{col}" for row in range(2) for col in range(2)
    }


def test_control_area_matches_its_target_exactly(tmp_path: Path) -> None:
    """Placing the target's own shape preserves the area match by construction."""

    handle, _ = _run(tmp_path)
    items = handle.items()
    ok = items[items["status"] == "ok"]

    targets = ok[~ok["is_control"]]
    controls = ok[ok["is_control"]]

    assert not controls.empty
    # Source-space area is exact regardless of placement, because the control
    # is the target's own shape translated -- not a re-sampled pixel set.
    assert set(controls["intended_area_px"]) == set(targets["intended_area_px"])


def test_multiple_seed_salts_redraw_the_control_within_one_sample(tmp_path: Path) -> None:
    """Seed salts are repeat trials, so one anchor can hold several draws."""

    handle, config = _run(tmp_path, seed_salts=(0, 1, 2))
    items = handle.items()
    controls = items[(items["is_control"]) & (items["status"] == "ok")]

    per_anchor = controls.groupby(["sample_id", "region_instance_id"])["effective_area_px"]
    assert (per_anchor.nunique() > 1).any()

    # The analysis layer holds the second copy of the geometry check, so it
    # must tolerate the same per-item variation the metrics layer does.
    registry = default_metric_registry()
    joined = handle.joined()
    item_metrics = registry.compute_item_metrics(joined, adapter_spec=config.adapter_spec)
    aggregate_item_metrics(item_metrics, joined, registry, config)

    index = ComparisonIndexer(handle.items()[[
        "item_id",
        "sample_id",
        "region_id",
        "region_instance_id",
        "region_kind",
        "region_params_json",
        "intended_area_px",
        "effective_area_px",
        "perturb_op",
        "perturb_params_json",
        "invert_mask",
        "is_control",
        "seed_used",
    ]])

    assert index.control_pairs
    assert index.coverage_report.n_controls_unmatched == 0
