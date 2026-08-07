"""End-to-end execution tests over the committed synthetic fixture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ssat.core.adapter import CallableAdapter, TorchvisionAdapter
from ssat.core.adapter.types import AdapterSpec
from ssat.core.config.schema import (
    DumpConfig,
    PerturbationConfig,
    ResolvedConfig,
    ResolvedRegionConfig,
    RuntimeConfig,
)
from ssat.core.dump import DumpReader, DumpWriter, EnvironmentSpec
from ssat.core.plan import PlanBuilder
from ssat.core.resume import ResumeIndex
from ssat.core.runtime import run_audit
from ssat.core.source import ImageFolderSource
from ssat.core.source.types import SampleMeta
from ssat.core.types import ItemStatus, PerturbationOp, RegionKind


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


def _config(
    tmp_path: Path,
    spec: AdapterSpec,
    *,
    columns: int,
    num_workers: int,
) -> ResolvedConfig:
    return ResolvedConfig(
        config_base_dir=tmp_path.resolve(),
        regions=(
            ResolvedRegionConfig(
                region_id="grid",
                kind=RegionKind.GRID,
                params={"rows": 1, "cols": columns},
            ),
        ),
        perturbations=(
            PerturbationConfig(
                op=PerturbationOp.CONSTANT_FILL,
                params={"value": 0},
            ),
        ),
        runtime=RuntimeConfig(
            variants_per_chunk=columns,
            target_batch_size=5,
            num_workers=num_workers,
        ),
        dump=DumpConfig(flush_every=7),
        adapter_spec=spec,
    )


def _run(
    tmp_path: Path,
    adapter: CallableAdapter | TorchvisionAdapter,
    *,
    columns: int,
    num_workers: int,
):
    source = _source()
    config = _config(
        tmp_path,
        adapter.describe(),
        columns=columns,
        num_workers=num_workers,
    )
    builder = PlanBuilder(config, source)
    root = tmp_path / "dump"
    writer = DumpWriter(
        root,
        config,
        code_version="runtime-e2e",
        mode="create",
        environment=ENVIRONMENT,
    )
    resume = ResumeIndex.open(root)
    summary = run_audit(config, builder, source, adapter, writer, resume)
    writer.close(success=True)
    reader = DumpReader(root)
    return summary, reader.read_clean(), reader.read_perturbed()


def test_callable_runtime_e2e_with_workers_and_corrupt_inputs(tmp_path: Path) -> None:
    def predict(batch: np.ndarray) -> np.ndarray:
        means = batch.astype(np.float32).mean(axis=(1, 2, 3, 4))
        return np.stack((means, means / 2.0, -means), axis=1)

    adapter = CallableAdapter(
        predict,
        model_id="fixture-callable",
        class_names=("gradient", "geometry", "texture"),
        transform_mask_fn=lambda mask: mask.copy(),
    )

    summary, clean, perturbed = _run(
        tmp_path,
        adapter,
        columns=2,
        num_workers=2,
    )

    assert summary.clean_records == 20
    assert summary.perturbed_records == 40
    assert summary.counts_by_status[ItemStatus.OK] == 54
    assert summary.counts_by_status[ItemStatus.LOAD_FAILED] == 6
    assert clean.num_rows == 20
    assert perturbed.num_rows == 40
    clean_rows = clean.to_pylist()
    perturbed_rows = perturbed.to_pylist()
    assert sum(row["status"] == "ok" for row in clean_rows) == 18
    assert sum(row["status"] == "load_failed" for row in clean_rows) == 2
    assert all(len(row["logits"]) == 3 for row in clean_rows if row["status"] == "ok")
    assert sum(row["status"] == "ok" for row in perturbed_rows) == 36
    assert sum(row["status"] == "load_failed" for row in perturbed_rows) == 4
    assert {
        row["effective_area_px"]
        for row in perturbed_rows
        if row["status"] == "ok"
    } == {2048}


def test_torchvision_runtime_e2e_without_downloads(tmp_path: Path) -> None:
    adapter = TorchvisionAdapter(
        "squeezenet1_0",
        weights=None,
        device="cpu",
        max_batch_size=5,
    )

    summary, clean, perturbed = _run(
        tmp_path,
        adapter,
        columns=1,
        num_workers=2,
    )

    assert summary.clean_records == 20
    assert summary.perturbed_records == 20
    assert summary.counts_by_status[ItemStatus.OK] == 36
    assert summary.counts_by_status[ItemStatus.LOAD_FAILED] == 4
    assert all(
        len(row["logits"]) == 1000
        for row in clean.to_pylist() + perturbed.to_pylist()
        if row["status"] == "ok"
    )
    assert {
        row["effective_area_px"]
        for row in perturbed.to_pylist()
        if row["status"] == "ok"
    } == {224 * 224}
