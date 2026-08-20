"""CPU-only preflight estimation over the committed image fixture."""

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
from ssat.core.dump import DumpWriter, EnvironmentSpec
from ssat.core.estimate import CostEstimator, EstimateOptions
from ssat.core.plan import PlanBuilder
from ssat.core.resume import ResumeIndex
from ssat.core.runtime import run_audit
from ssat.core.source import ImageFolderSource
from ssat.core.source.types import SampleMeta
from ssat.core.types import PerturbationOp, RegionKind


FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_classification"
ENVIRONMENT = EnvironmentSpec(python_version="3.11.0", platform="estimate-test")


def _source() -> ImageFolderSource:
    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        row for row in manifest["samples"] if row["expected_status"] == "ok"
    ][:8]
    return ImageFolderSource(
        [
            SampleMeta(
                sample_id=row["sample_id"],
                path=FIXTURE / row["path"],
                gt_label=row["gt_label"],
            )
            for row in rows
        ]
    )


def _predict(batch: np.ndarray) -> np.ndarray:
    pixels = batch.astype(np.int64)
    total = pixels.sum(axis=(1, 2, 3, 4))[:, np.newaxis]
    channels = pixels.sum(axis=(1, 2, 3))
    return np.concatenate((total, channels), axis=1).astype(np.float32)


def _adapter(predict_fn=_predict) -> CallableAdapter:
    return CallableAdapter(
        predict_fn,
        model_id="estimate-fixture",
        class_names=("total", "red", "green", "blue"),
        transform_mask_fn=lambda mask: mask,
    )


def _config(tmp_path: Path, adapter: CallableAdapter) -> ResolvedConfig:
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
            PerturbationConfig(
                op=PerturbationOp.GAUSSIAN_NOISE,
                params={"sigma": 8.0},
                seed_salts=(0, 1),
            ),
        ),
        runtime=RuntimeConfig(
            global_seed=20260807,
            variants_per_chunk=3,
            target_batch_size=5,
            num_workers=2,
        ),
        dump=DumpConfig(flush_every=7),
        adapter_spec=adapter.describe(),
    )


def test_estimator_profiles_fixture_without_writing_dump(tmp_path: Path) -> None:
    source = _source()
    adapter = _adapter()
    config = _config(tmp_path, adapter)
    report = CostEstimator(
        EstimateOptions(max_profile_chunks=5, max_sanity_samples=5)
    ).estimate(
        config,
        PlanBuilder(config, source),
        source,
        adapter,
    )

    assert report.total_clean_samples == 8
    assert report.total_chunks == 16
    assert report.total_perturbed_items == 32
    assert report.class_count == 4
    assert report.profile is not None
    assert report.profile.selected_chunks == 5
    assert report.profile.successful_predictions == report.profile.selected_items
    assert report.sanity is not None
    assert report.area_sanity is not None
    assert report.area_sanity.passed is True
    assert report.sanity.selected_samples == 5
    assert report.estimated_remaining_seconds > 0
    assert report.estimated_total_dump_bytes is not None
    assert report.confirmation_required is False
    assert list(tmp_path.iterdir()) == []


def test_estimator_uses_real_resume_index_to_skip_completed_run(
    tmp_path: Path,
) -> None:
    source = _source()
    adapter = _adapter()
    config = _config(tmp_path, adapter)
    builder = PlanBuilder(config, source)
    root = tmp_path / "dump"
    writer = DumpWriter(
        root,
        config,
        code_version="estimate-e2e",
        mode="create",
        environment=ENVIRONMENT,
    )
    run_audit(
        config,
        builder,
        source,
        adapter,
        writer,
        ResumeIndex.open(root),
    )
    writer.close(success=True)

    calls: list[int] = []

    def counting_predict(batch: np.ndarray) -> np.ndarray:
        calls.append(len(batch))
        return _predict(batch)

    resumed_adapter = _adapter(counting_predict)
    report = CostEstimator().estimate(
        config,
        builder,
        source,
        resumed_adapter,
        resume_index=ResumeIndex.open(root),
    )

    assert calls == []
    assert report.pending_clean_samples == 0
    assert report.pending_chunks == 0
    assert report.pending_perturbed_items == 0
    assert report.profile is None
    assert report.sanity is None
    assert report.area_sanity is None
    assert report.estimated_remaining_dump_bytes == 0
