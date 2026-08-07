"""Tests for durable dump writing, authoritative reading, and resume filtering."""

from __future__ import annotations

import io
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ssat.core.adapter.types import AdapterSpec, RawOutput
from ssat.core.config.schema import (
    DumpConfig,
    PerturbationConfig,
    ResolvedConfig,
    ResolvedRegionConfig,
    RuntimeConfig,
)
from ssat.core.dump import (
    CLEAN_SCHEMA,
    CleanDumpRecord,
    DumpCorruptionError,
    DumpError,
    DumpReader,
    DumpSchemaError,
    DumpWriter,
    EnvironmentSpec,
    PerturbedDumpRecord,
)
from ssat.core.dump.types import seed_to_hex
from ssat.core.plan.types import WorkChunkMeta, WorkItem
from ssat.core.region.types import RegionMeta, RegionSpec
from ssat.core.resume import ResumeIndex
from ssat.core.source.types import SampleMeta
from ssat.core.types import ItemStatus, PerturbationOp, RegionKind
from ssat.utils.io import load_json, write_json_atomic


ENVIRONMENT = EnvironmentSpec(
    python_version="3.11.0",
    platform="test-platform",
)


def _config(
    tmp_path: Path,
    *,
    flush_every: int = 2,
    logits_warning: int | None = 10_000,
    model_id: str = "model",
) -> ResolvedConfig:
    return ResolvedConfig(
        config_base_dir=tmp_path.resolve(),
        regions=(
            ResolvedRegionConfig(
                region_id="grid",
                kind=RegionKind.GRID,
                params={"rows": 1, "cols": 1},
            ),
        ),
        perturbations=(
            PerturbationConfig(
                op=PerturbationOp.BLUR,
                params={"sigma": 1.0},
            ),
        ),
        runtime=RuntimeConfig(),
        dump=DumpConfig(
            flush_every=flush_every,
            max_classes_for_full_logits=logits_warning,
        ),
        adapter_spec=AdapterSpec(
            model_id=model_id,
            deterministic=True,
            adapter_kind="callable",
            model_name="tiny",
            weights_hash="a" * 64,
            preprocessing_fingerprint="b" * 64,
            mask_transform_available=True,
        ),
    )


def _output(*values: float) -> RawOutput:
    return RawOutput(np.asarray(values, dtype=np.float64))


def _clean(
    sample_id: str,
    *,
    status: ItemStatus = ItemStatus.OK,
) -> CleanDumpRecord:
    successful = status is ItemStatus.OK
    return CleanDumpRecord(
        sample_id=sample_id,
        status=status,
        model_id="model",
        logits=_output(0.25, 0.75) if successful else None,
        content_hash="c" * 64 if successful else None,
        gt_label=1,
        original_shape=(1, 4, 4, 3) if successful else None,
    )


def _work_item(character: str, *, sample_id: str = "sample") -> WorkItem:
    return WorkItem(
        item_id=character * 64,
        sample_id=sample_id,
        region_spec=RegionSpec(
            region_id="grid",
            region_instance_id="grid/r0/c0",
            kind=RegionKind.GRID,
            params={"rows": 1, "cols": 1},
        ),
        perturb_op=PerturbationOp.BLUR,
        perturb_params={"sigma": 1.0},
    )


def _perturbed(
    character: str,
    *,
    status: ItemStatus = ItemStatus.OK,
    seed: int = 7,
) -> PerturbedDumpRecord:
    successful = status is ItemStatus.OK
    return PerturbedDumpRecord(
        work_item=_work_item(character),
        status=status,
        seed_used=seed,
        logits=_output(-1.0, 2.0) if successful else None,
        region_meta=(
            RegionMeta(4, 0.25, "grid", "1") if successful else None
        ),
        effective_area_px=4 if successful else None,
    )


def _writer(root: Path, config: ResolvedConfig, *, mode: str = "create") -> DumpWriter:
    return DumpWriter(
        root,
        config,
        code_version="test-code",
        mode=mode,
        environment=ENVIRONMENT,
    )


def test_dump_records_enforce_status_logits_and_128_bit_seed() -> None:
    with pytest.raises(ValueError, match="status=ok requires logits"):
        CleanDumpRecord("sample", ItemStatus.OK, "model")
    with pytest.raises(ValueError, match="failure statuses"):
        CleanDumpRecord(
            "sample",
            ItemStatus.LOAD_FAILED,
            "model",
            logits=_output(1.0),
        )
    with pytest.raises(ValueError, match="unsigned 128-bit"):
        PerturbedDumpRecord(_work_item("a"), ItemStatus.LOAD_FAILED, 2**128)

    assert seed_to_hex(2**127 + 3) == "80000000000000000000000000000003"


def test_writer_reader_round_trip_fragments_manifest_and_seed(tmp_path: Path) -> None:
    root = tmp_path / "dump"
    config = _config(tmp_path)
    large_seed = 2**100 + 9
    with _writer(root, config) as writer:
        writer.write_clean_many((_clean("sample-a"), _clean("sample-b")))
        writer.write_perturbed_many(
            (_perturbed("a", seed=large_seed), _perturbed("b"))
        )

    assert [path.name for path in (root / "clean").iterdir()] == [
        "part_00000000.parquet"
    ]
    assert [path.name for path in (root / "perturbed").iterdir()] == [
        "chunk_00000000.parquet"
    ]
    assert [path.name for path in (root / "index").iterdir()] == [
        "part_00000000.parquet"
    ]

    reader = DumpReader(root)
    clean = reader.read_clean().to_pylist()
    perturbed = reader.read_perturbed().to_pylist()
    manifest = reader.read_manifest()

    assert [row["sample_id"] for row in clean] == ["sample-a", "sample-b"]
    assert perturbed[0]["seed_used"] == seed_to_hex(large_seed)
    assert perturbed[0]["region_params_json"] == '{"cols":1,"rows":1}'
    assert perturbed[0]["perturb_params_json"] == '{"sigma":"1.000000000000"}'
    assert perturbed[0]["logits"] == pytest.approx([-1.0, 2.0])
    assert manifest.adapter_spec == config.adapter_spec
    assert manifest.resolved_config == config
    assert manifest.finished_at is not None
    assert manifest.counts_by_status[ItemStatus.OK] == 4


def test_latest_fragment_wins_for_clean_and_perturbed_rows(tmp_path: Path) -> None:
    root = tmp_path / "dump"
    config = _config(tmp_path, flush_every=1)
    writer = _writer(root, config)
    writer.write_clean(_clean("sample", status=ItemStatus.LOAD_FAILED))
    writer.write_perturbed(_perturbed("a", status=ItemStatus.PREDICT_FAILED))
    writer.close(success=False)

    writer = _writer(root, config, mode="resume")
    writer.write_clean(_clean("sample"))
    writer.write_perturbed(_perturbed("a"))
    writer.close(success=True)

    reader = DumpReader(root)
    assert reader.read_clean().column("status").to_pylist() == ["ok"]
    assert reader.read_perturbed().column("status").to_pylist() == ["ok"]
    assert reader.read_manifest().counts_by_status[ItemStatus.OK] == 2
    index = ResumeIndex.open(root)
    assert index.clean_status("sample") is ItemStatus.OK
    assert index.item_status("a" * 64) is ItemStatus.OK


def test_resume_filters_whole_chunks_and_retryable_failures(tmp_path: Path) -> None:
    root = tmp_path / "dump"
    config = _config(tmp_path, flush_every=1)
    writer = _writer(root, config)
    writer.write_clean(_clean("done"))
    writer.write_clean(_clean("failed", status=ItemStatus.LOAD_FAILED))
    writer.write_perturbed(_perturbed("a"))
    writer.write_perturbed(_perturbed("c", status=ItemStatus.SKIPPED_OOM))
    writer.close(success=False)

    complete = WorkChunkMeta("1" * 64, "sample", ("a" * 64,))
    partial = WorkChunkMeta("2" * 64, "sample", ("a" * 64, "b" * 64))
    failed = WorkChunkMeta("3" * 64, "sample", ("c" * 64,))
    samples = (
        SampleMeta("done", Path("done")),
        SampleMeta("failed", Path("failed")),
        SampleMeta("missing", Path("missing")),
    )
    index = ResumeIndex.open(root)

    assert index.pending_chunks(
        (complete, partial, failed), retry_failed=False
    ) == (partial,)
    assert index.pending_chunks(
        (complete, partial, failed), retry_failed=True
    ) == (partial, failed)
    assert [
        sample.sample_id
        for sample in index.pending_clean_samples(samples, retry_failed=False)
    ] == ["missing"]
    assert [
        sample.sample_id
        for sample in index.pending_clean_samples(samples, retry_failed=True)
    ] == ["failed", "missing"]


def test_data_is_committed_before_index_and_orphan_is_rebuildable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ssat.core.dump.writer as writer_module

    root = tmp_path / "dump"
    config = _config(tmp_path, flush_every=1)
    writer = _writer(root, config)
    real_write = writer_module.atomic_write_parquet
    destinations: list[Path] = []

    def fail_index(path: Path, table: pa.Table) -> None:
        destinations.append(path)
        if path.parent.name == "index":
            raise RuntimeError("simulated crash")
        real_write(path, table)

    monkeypatch.setattr(writer_module, "atomic_write_parquet", fail_index)
    with pytest.raises(RuntimeError, match="simulated crash"):
        writer.write_perturbed(_perturbed("a"))

    assert [path.parent.name for path in destinations] == ["perturbed", "index"]
    assert (root / "perturbed" / "chunk_00000000.parquet").is_file()
    assert not (root / "index" / "part_00000000.parquet").exists()
    assert DumpReader(root).read_perturbed().num_rows == 0

    monkeypatch.setattr(writer_module, "atomic_write_parquet", real_write)
    rebuilt = ResumeIndex.rebuild(root)
    assert rebuilt.item_status("a" * 64) is ItemStatus.OK
    assert DumpReader(root).read_perturbed().num_rows == 1
    assert DumpReader(root).read_manifest().counts_by_status[ItemStatus.OK] == 1


def test_rebuild_restores_index_and_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "dump"
    writer = _writer(root, _config(tmp_path, flush_every=1))
    writer.write_perturbed(_perturbed("a"))
    writer.write_perturbed(_perturbed("b"))
    writer.close(success=True)
    expected = ResumeIndex.open(root)
    for path in (root / "index").iterdir():
        path.unlink()

    first = ResumeIndex.rebuild(root)
    second = ResumeIndex.rebuild(root)
    for item_id in ("a" * 64, "b" * 64):
        assert first.item_status(item_id) == expected.item_status(item_id)
        assert second.item_status(item_id) == expected.item_status(item_id)


def test_resume_requires_matching_config_and_code_and_records_environment(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dump"
    config = _config(tmp_path)
    _writer(root, config).close(success=True)

    with pytest.raises(DumpError, match="code_version"):
        DumpWriter(
            root,
            config,
            code_version="different",
            mode="resume",
            environment=ENVIRONMENT,
        )
    with pytest.raises(DumpError, match="resolved_config"):
        _writer(root, _config(tmp_path, model_id="other"), mode="resume")

    stream = io.StringIO()
    logger = logging.Logger("dump-test")
    logger.addHandler(logging.StreamHandler(stream))
    changed = ENVIRONMENT.model_copy(update={"platform": "different"})
    resumed = DumpWriter(
        root,
        config,
        code_version="test-code",
        mode="resume",
        environment=changed,
        logger=logger,
    )
    resumed.close(success=False)
    manifest = DumpReader(root).read_manifest()
    assert "resume_environment_changed" in stream.getvalue()
    assert manifest.finished_at is None
    assert manifest.resume_events[-1].environment == changed


def test_reader_rejects_schema_mismatch_and_missing_index_target(tmp_path: Path) -> None:
    root = tmp_path / "dump"
    writer = _writer(root, _config(tmp_path, flush_every=1))
    writer.write_clean(_clean("sample"))
    writer.write_perturbed(_perturbed("a"))
    writer.close(success=True)

    clean_path = root / "clean" / "part_00000000.parquet"
    table = pq.read_table(clean_path).replace_schema_metadata(
        {b"ssat.schema_version": b"9.0.0"}
    )
    pq.write_table(table, clean_path)
    with pytest.raises(DumpSchemaError, match="schema version mismatch"):
        DumpReader(root).read_clean()

    clean_path.unlink()
    (root / "perturbed" / "chunk_00000000.parquet").unlink()
    with pytest.raises(DumpCorruptionError, match="missing data fragment"):
        ResumeIndex.open(root)

    manifest_path = root / "run_manifest.json"
    manifest_payload = load_json(manifest_path)
    manifest_payload["schema_version"] = "9.0.0"
    write_json_atomic(manifest_path, manifest_payload)
    with pytest.raises(DumpSchemaError, match="manifest schema version mismatch"):
        DumpReader(root)


def test_large_logits_warning_is_configurable_and_emitted_once(tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = logging.Logger("logits-test")
    logger.addHandler(logging.StreamHandler(stream))
    root = tmp_path / "dump"
    config = _config(tmp_path, flush_every=10, logits_warning=1)
    writer = DumpWriter(
        root,
        config,
        code_version="test-code",
        mode="create",
        environment=ENVIRONMENT,
        logger=logger,
    )
    writer.write_clean(_clean("one"))
    writer.write_clean(_clean("two"))
    writer.close(success=True)
    assert stream.getvalue().count("dump.large_logits") == 1

    disabled = _writer(
        tmp_path / "disabled",
        _config(tmp_path, logits_warning=None),
    )
    disabled.write_clean(_clean("sample"))
    disabled.close(success=True)


def test_dump_import_does_not_import_torch() -> None:
    script = "import sys; import ssat.core.dump; print('torch' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"


def test_empty_reader_tables_keep_versioned_schemas(tmp_path: Path) -> None:
    root = tmp_path / "dump"
    _writer(root, _config(tmp_path)).close(success=True)
    reader = DumpReader(root)

    assert reader.read_clean().schema == CLEAN_SCHEMA
    assert reader.read_perturbed().num_rows == 0
