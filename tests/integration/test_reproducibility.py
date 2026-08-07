"""Required CI regressions for execution-level reproducibility."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from ssat.core.adapter import CallableAdapter
from ssat.core.adapter.base import AdapterOutOfMemoryError
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
from ssat.core.runtime import RuntimeExecutionError, run_audit
from ssat.core.source import ImageFolderSource
from ssat.core.source.types import SampleMeta
from ssat.core.types import PerturbationOp, RegionKind


FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_classification"
ENVIRONMENT = EnvironmentSpec(python_version="3.11.0", platform="repro-test")
CODE_VERSION = "reproducibility-test"
MODEL_ID = "deterministic-checksum-model"


@dataclass(frozen=True)
class LogicalDump:
    """Authoritative rows with physical write time and ordering removed."""

    clean: tuple[dict[str, object], ...]
    perturbed: tuple[dict[str, object], ...]


def _source() -> ImageFolderSource:
    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    valid = [
        row for row in manifest["samples"] if row["expected_status"] == "ok"
    ][:8]
    return ImageFolderSource(
        [
            SampleMeta(
                sample_id=row["sample_id"],
                path=FIXTURE / row["path"],
                gt_label=row["gt_label"],
            )
            for row in valid
        ]
    )


def _checksum_logits(batch: np.ndarray) -> np.ndarray:
    """Return exact integer-derived features sensitive to perturbation pixels."""

    pixels = batch.astype(np.int64)
    total = pixels.sum(axis=(1, 2, 3, 4))[:, np.newaxis]
    channels = pixels.sum(axis=(1, 2, 3))
    return np.concatenate((total, channels), axis=1).astype(np.float32)


def _adapter(
    predict_fn: Callable[[np.ndarray], np.ndarray] = _checksum_logits,
    *,
    cleanup_after_oom_fn: Callable[[], None] | None = None,
) -> CallableAdapter:
    return CallableAdapter(
        predict_fn,
        model_id=MODEL_ID,
        class_names=("total", "red", "green", "blue"),
        transform_mask_fn=lambda mask: mask.copy(),
        cleanup_after_oom_fn=cleanup_after_oom_fn,
    )


def _config(
    spec: AdapterSpec,
    *,
    num_workers: int,
    target_batch_size: int,
    fail_fast: bool = False,
    retry_failed: bool = False,
) -> ResolvedConfig:
    return ResolvedConfig(
        config_base_dir=FIXTURE.resolve(),
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
                params={"sigma": 12.0},
                seed_salts=(0, 1),
            ),
        ),
        runtime=RuntimeConfig(
            global_seed=20260807,
            variants_per_chunk=3,
            target_batch_size=target_batch_size,
            num_workers=num_workers,
            retry_failed=retry_failed,
            fail_fast=fail_fast,
        ),
        dump=DumpConfig(flush_every=5),
        adapter_spec=spec,
    )


def _run_once(
    root: Path,
    *,
    adapter: CallableAdapter,
    num_workers: int = 1,
    target_batch_size: int = 7,
    fail_fast: bool = True,
    retry_failed: bool = True,
) -> LogicalDump:
    source = _source()
    config = _config(
        adapter.describe(),
        num_workers=num_workers,
        target_batch_size=target_batch_size,
        fail_fast=fail_fast,
        retry_failed=retry_failed,
    )
    builder = PlanBuilder(config, source)
    writer = DumpWriter(
        root,
        config,
        code_version=CODE_VERSION,
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
    return _logical_dump(root)


@pytest.fixture(scope="module")
def baseline_dump(tmp_path_factory: pytest.TempPathFactory) -> LogicalDump:
    root = tmp_path_factory.mktemp("repro-baseline") / "dump"
    dump = _run_once(root, adapter=_adapter())
    assert len(dump.clean) == 8
    assert len(dump.perturbed) == 32
    assert {row["status"] for row in dump.clean + dump.perturbed} == {"ok"}
    return dump


def test_num_workers_does_not_change_dump(
    tmp_path: Path,
    baseline_dump: LogicalDump,
) -> None:
    actual = _run_once(
        tmp_path / "workers-4",
        adapter=_adapter(),
        num_workers=4,
    )

    assert actual == baseline_dump


def test_interrupted_resume_matches_uninterrupted_run(
    tmp_path: Path,
    baseline_dump: LogicalDump,
) -> None:
    calls = 0

    def fail_on_second_batch(batch: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected interruption")
        return _checksum_logits(batch)

    root = tmp_path / "resumed"
    interrupted_adapter = _adapter(fail_on_second_batch)
    source = _source()
    config = _config(
        interrupted_adapter.describe(),
        num_workers=1,
        target_batch_size=7,
        fail_fast=True,
        retry_failed=True,
    )
    builder = PlanBuilder(config, source)
    writer = DumpWriter(
        root,
        config,
        code_version=CODE_VERSION,
        mode="create",
        environment=ENVIRONMENT,
    )
    with pytest.raises(RuntimeExecutionError, match="prediction failed"):
        run_audit(
            config,
            builder,
            source,
            interrupted_adapter,
            writer,
            ResumeIndex.open(root),
        )
    writer.close(success=False)

    partial = DumpReader(root).read_clean().to_pylist()
    assert sum(row["status"] == "ok" for row in partial) == 7
    assert sum(row["status"] == "predict_failed" for row in partial) == 1

    resumed_adapter = _adapter()
    resumed_writer = DumpWriter(
        root,
        config,
        code_version=CODE_VERSION,
        mode="resume",
        environment=ENVIRONMENT,
    )
    run_audit(
        config,
        builder,
        source,
        resumed_adapter,
        resumed_writer,
        ResumeIndex.open(root),
    )
    resumed_writer.close(success=True)

    assert _logical_dump(root) == baseline_dump


def test_target_batch_size_does_not_change_dump(
    tmp_path: Path,
    baseline_dump: LogicalDump,
) -> None:
    actual = _run_once(
        tmp_path / "batch-2",
        adapter=_adapter(),
        target_batch_size=2,
    )

    assert actual == baseline_dump


def test_oom_recovery_does_not_change_dump(
    tmp_path: Path,
    baseline_dump: LogicalDump,
) -> None:
    cleanup_calls = 0

    def predict_with_oom(batch: np.ndarray) -> np.ndarray:
        if len(batch) > 2:
            raise AdapterOutOfMemoryError("injected reproducibility OOM")
        return _checksum_logits(batch)

    def cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    actual = _run_once(
        tmp_path / "oom-recovery",
        adapter=_adapter(
            predict_with_oom,
            cleanup_after_oom_fn=cleanup,
        ),
    )

    assert cleanup_calls > 0
    assert actual == baseline_dump


def test_repeated_identical_run_has_same_item_ids_and_logits(
    tmp_path: Path,
    baseline_dump: LogicalDump,
) -> None:
    repeated = _run_once(tmp_path / "repeated", adapter=_adapter())

    assert repeated == baseline_dump
    assert tuple(row["item_id"] for row in repeated.perturbed) == tuple(
        row["item_id"] for row in baseline_dump.perturbed
    )
    assert tuple(row["logits"] for row in repeated.perturbed) == tuple(
        row["logits"] for row in baseline_dump.perturbed
    )


def _logical_dump(root: Path) -> LogicalDump:
    reader = DumpReader(root)
    return LogicalDump(
        clean=_canonical_rows(reader.read_clean().to_pylist(), key="sample_id"),
        perturbed=_canonical_rows(
            reader.read_perturbed().to_pylist(),
            key="item_id",
        ),
    )


def _canonical_rows(
    rows: list[dict[str, object]],
    *,
    key: str,
) -> tuple[dict[str, object], ...]:
    normalized = []
    for row in rows:
        logical = dict(row)
        logical.pop("written_at")
        normalized.append(logical)
    return tuple(sorted(normalized, key=lambda row: str(row[key])))
