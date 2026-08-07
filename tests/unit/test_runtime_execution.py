"""Integration-loop tests for failures, fail-fast, and resume filtering."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ssat.core.adapter import CallableAdapter
from ssat.core.adapter.base import AdapterOutOfMemoryError
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
from ssat.core.runtime import RuntimeExecutionError, run_audit
from ssat.core.source.types import LoadedSample, SampleMeta
from ssat.core.types import ItemStatus, PerturbationOp, RegionKind


ENVIRONMENT = EnvironmentSpec(python_version="3.11.0", platform="test")


class MemorySource:
    def __init__(self) -> None:
        self.samples = [SampleMeta("a", "unused"), SampleMeta("b", "unused")]
        self.load_calls: list[str] = []

    def list_samples(self) -> list[SampleMeta]:
        return list(self.samples)

    def load(self, sample_id: str) -> LoadedSample:
        self.load_calls.append(sample_id)
        array = np.full((1, 4, 4, 3), ord(sample_id), dtype=np.uint8)
        return LoadedSample(array, sample_id, array.shape, sample_id * 64)


def _config(
    tmp_path: Path,
    adapter: CallableAdapter,
    *,
    fail_fast: bool = False,
) -> ResolvedConfig:
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
                op=PerturbationOp.CONSTANT_FILL,
                params={"value": 0},
            ),
        ),
        runtime=RuntimeConfig(
            variants_per_chunk=2,
            target_batch_size=4,
            fail_fast=fail_fast,
        ),
        dump=DumpConfig(flush_every=100),
        adapter_spec=adapter.describe(),
    )


def _writer(root: Path, config: ResolvedConfig, *, mode: str) -> DumpWriter:
    return DumpWriter(
        root,
        config,
        code_version="runtime-test",
        mode=mode,
        environment=ENVIRONMENT,
    )


def test_prediction_error_marks_whole_batches_without_splitting(tmp_path: Path) -> None:
    calls: list[int] = []

    def failing(batch: np.ndarray) -> np.ndarray:
        calls.append(len(batch))
        raise RuntimeError("failed")

    adapter = CallableAdapter(
        failing,
        model_id="failing",
        transform_mask_fn=lambda mask: mask,
    )
    config = _config(tmp_path, adapter)
    source = MemorySource()
    builder = PlanBuilder(config, source)
    root = tmp_path / "dump"
    writer = _writer(root, config, mode="create")

    summary = run_audit(
        config,
        builder,
        source,
        adapter,
        writer,
        ResumeIndex.open(root),
    )
    writer.close(success=True)

    assert calls == [2, 4]
    assert summary.counts_by_status[ItemStatus.PREDICT_FAILED] == 6
    assert DumpReader(root).read_clean().to_pylist()[0]["status"] == "predict_failed"


def test_fail_fast_flushes_failure_before_raising(tmp_path: Path) -> None:
    adapter = CallableAdapter(
        lambda batch: (_ for _ in ()).throw(RuntimeError("failed")),
        model_id="failing-fast",
        transform_mask_fn=lambda mask: mask,
    )
    config = _config(tmp_path, adapter, fail_fast=True)
    source = MemorySource()
    builder = PlanBuilder(config, source)
    root = tmp_path / "dump"
    writer = _writer(root, config, mode="create")

    with pytest.raises(RuntimeExecutionError, match="prediction failed"):
        run_audit(
            config,
            builder,
            source,
            adapter,
            writer,
            ResumeIndex.open(root),
        )

    assert DumpReader(root).read_clean().num_rows == 2
    assert {
        row["status"] for row in DumpReader(root).read_clean().to_pylist()
    } == {"predict_failed"}
    writer.close(success=False)


def test_resume_skips_all_authoritative_successes(tmp_path: Path) -> None:
    first_calls: list[int] = []

    def first_predict(batch: np.ndarray) -> np.ndarray:
        first_calls.append(len(batch))
        return np.zeros((len(batch), 2), dtype=np.float32)

    adapter = CallableAdapter(
        first_predict,
        model_id="resume",
        transform_mask_fn=lambda mask: mask,
    )
    config = _config(tmp_path, adapter)
    source = MemorySource()
    builder = PlanBuilder(config, source)
    root = tmp_path / "dump"
    writer = _writer(root, config, mode="create")
    first = run_audit(
        config,
        builder,
        source,
        adapter,
        writer,
        ResumeIndex.open(root),
    )
    writer.close(success=True)
    assert first.records_written == 6

    source.load_calls.clear()
    resumed_writer = _writer(root, config, mode="resume")
    second = run_audit(
        config,
        builder,
        source,
        adapter,
        resumed_writer,
        ResumeIndex.open(root),
    )
    resumed_writer.close(success=True)

    assert second.records_written == 0
    assert source.load_calls == []
    assert len(first_calls) == 2


def test_runtime_import_keeps_torch_lazy() -> None:
    code = "import sys; import ssat.core.runtime; assert 'torch' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_oom_reduction_persists_from_clean_into_perturbed(tmp_path: Path) -> None:
    calls: list[int] = []
    cleanup_calls: list[None] = []

    def predict(batch: np.ndarray) -> np.ndarray:
        calls.append(len(batch))
        if len(batch) > 1:
            raise AdapterOutOfMemoryError("injected")
        return np.zeros((len(batch), 2), dtype=np.float32)

    adapter = CallableAdapter(
        predict,
        model_id="oom-recovery",
        transform_mask_fn=lambda mask: mask,
        cleanup_after_oom_fn=lambda: cleanup_calls.append(None),
    )
    config = _config(tmp_path, adapter)
    source = MemorySource()
    builder = PlanBuilder(config, source)
    root = tmp_path / "dump"
    writer = _writer(root, config, mode="create")

    summary = run_audit(
        config,
        builder,
        source,
        adapter,
        writer,
        ResumeIndex.open(root),
    )
    writer.close(success=True)

    assert calls == [2, 1, 1, 1, 1, 1, 1]
    assert len(cleanup_calls) == 1
    assert summary.oom_events == 1
    assert summary.initial_batch_size == 4
    assert summary.final_batch_size == 1
    assert summary.counts_by_status[ItemStatus.OK] == 6
