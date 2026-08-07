"""Crash-safe buffered writer for raw audit dump fragments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa

from ssat.core.config.schema import ResolvedConfig
from ssat.core.dump._index import load_clean_locations, load_index_locations
from ssat.core.dump._storage import atomic_write_parquet, next_fragment_ordinal
from ssat.core.dump.errors import DumpError, DumpSchemaError
from ssat.core.dump.manifest import (
    load_manifest,
    new_manifest,
    normalize_environment,
    write_manifest,
)
from ssat.core.dump.schema import CLEAN_SCHEMA, INDEX_SCHEMA, PERTURBED_SCHEMA
from ssat.core.dump.types import (
    CleanDumpRecord,
    EnvironmentSpec,
    PerturbedDumpRecord,
    ResumeEvent,
    RunManifest,
    seed_to_hex,
)
from ssat.core.plan.hashing import canonical_json
from ssat.core.types import ItemStatus, SCHEMA_VERSION
from ssat.utils.logger_factory import get_logger


class DumpWriter:
    """Persist bounded record buffers using data-before-index commit order."""

    def __init__(
        self,
        root: str | Path,
        resolved_config: ResolvedConfig,
        *,
        code_version: str,
        mode: Literal["create", "resume"],
        environment: EnvironmentSpec | Mapping[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if mode not in {"create", "resume"}:
            raise ValueError("mode must be 'create' or 'resume'")
        if not code_version:
            raise ValueError("code_version must not be empty")
        self._root = Path(root)
        self._manifest_path = self._root / "run_manifest.json"
        self._config = resolved_config
        self._code_version = code_version
        self._environment = normalize_environment(environment)
        self._logger = logger or get_logger(__name__)
        self._clean_buffer: list[CleanDumpRecord] = []
        self._perturbed_buffer: list[PerturbedDumpRecord] = []
        self._closed = False
        self._logits_warning_emitted = False

        if mode == "create":
            self._initialize_new_run()
        else:
            self._initialize_resumed_run()

    @property
    def manifest(self) -> RunManifest:
        """Return the writer's current in-memory manifest."""

        return self._manifest

    def write_clean(self, record: CleanDumpRecord) -> None:
        """Buffer one clean result and flush at the configured threshold."""

        self._ensure_open()
        if not isinstance(record, CleanDumpRecord):
            raise TypeError("record must be a CleanDumpRecord")
        self._warn_for_logits(record.logits)
        self._clean_buffer.append(record)
        if len(self._clean_buffer) >= self._config.dump.flush_every:
            self._flush_clean()

    def write_clean_many(self, records: Iterable[CleanDumpRecord]) -> None:
        """Buffer clean results in input order."""

        for record in records:
            self.write_clean(record)

    def write_perturbed(self, record: PerturbedDumpRecord) -> None:
        """Buffer one perturbed result and flush at the configured threshold."""

        self._ensure_open()
        if not isinstance(record, PerturbedDumpRecord):
            raise TypeError("record must be a PerturbedDumpRecord")
        self._warn_for_logits(record.logits)
        self._perturbed_buffer.append(record)
        if len(self._perturbed_buffer) >= self._config.dump.flush_every:
            self._flush_perturbed()

    def write_perturbed_many(self, records: Iterable[PerturbedDumpRecord]) -> None:
        """Buffer perturbed results in input order."""

        for record in records:
            self.write_perturbed(record)

    def flush(self) -> None:
        """Durably commit both buffers without marking the run complete."""

        self._ensure_open()
        self._flush_clean()
        self._flush_perturbed()

    def close(self, *, success: bool) -> None:
        """Flush pending rows and optionally mark the run as finished."""

        if self._closed:
            return
        self.flush()
        if success:
            self._manifest = self._manifest.model_copy(
                update={"finished_at": _now()}
            )
        else:
            self._manifest = self._manifest.model_copy(update={"finished_at": None})
        write_manifest(self._manifest_path, self._manifest)
        self._closed = True

    def __enter__(self) -> DumpWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if exc_type is None:
            self.close(success=True)
        else:
            try:
                self.close(success=False)
            except Exception:
                self._logger.exception("dump.close_failed_during_exception")
        return False

    def _initialize_new_run(self) -> None:
        if self._root.exists():
            if not self._root.is_dir():
                raise DumpError(f"dump root is not a directory: {self._root}")
            if any(self._root.iterdir()):
                raise DumpError(f"create mode requires an empty dump root: {self._root}")
        self._root.mkdir(parents=True, exist_ok=True)
        for name in ("clean", "perturbed", "index"):
            (self._root / name).mkdir()
        self._manifest = new_manifest(
            self._config,
            code_version=self._code_version,
            environment=self._environment,
            now=_now(),
        )
        self._clean_statuses: dict[str, ItemStatus] = {}
        self._item_statuses: dict[str, ItemStatus] = {}
        self._clean_ordinal = 0
        self._perturbed_ordinal = 0
        write_manifest(self._manifest_path, self._manifest)

    def _initialize_resumed_run(self) -> None:
        manifest = load_manifest(self._manifest_path)
        if manifest.schema_version != SCHEMA_VERSION:  # pragma: no cover - loader checks
            raise DumpSchemaError("cannot resume a dump with a different schema")
        if manifest.resolved_config != self._config:
            raise DumpError("resolved_config does not match the existing run manifest")
        if manifest.adapter_spec != self._config.adapter_spec:
            raise DumpError("adapter_spec does not match the existing run manifest")
        if manifest.code_version != self._code_version:
            raise DumpError("code_version does not match the existing run manifest")

        clean_locations = load_clean_locations(
            self._root,
            expected_version=SCHEMA_VERSION,
        )
        item_locations = load_index_locations(
            self._root,
            expected_version=SCHEMA_VERSION,
        )
        self._clean_statuses = {
            sample_id: location.status
            for sample_id, location in clean_locations.items()
        }
        self._item_statuses = {
            item_id: location.status for item_id, location in item_locations.items()
        }
        self._clean_ordinal = next_fragment_ordinal(self._root / "clean", "part")
        self._perturbed_ordinal = next_fragment_ordinal(
            self._root / "perturbed", "chunk"
        )
        if manifest.environment != self._environment:
            self._logger.warning("dump.resume_environment_changed")
        self._manifest = manifest.model_copy(
            update={
                "finished_at": None,
                "counts_by_status": self._authoritative_counts(),
                "resume_events": manifest.resume_events
                + (ResumeEvent(resumed_at=_now(), environment=self._environment),),
            }
        )
        write_manifest(self._manifest_path, self._manifest)

    def _flush_clean(self) -> None:
        if not self._clean_buffer:
            return
        records = tuple(self._clean_buffer)
        sample_ids = [record.sample_id for record in records]
        if len(set(sample_ids)) != len(sample_ids):
            raise DumpError("one clean fragment cannot contain duplicate sample_id values")
        written_at = _now()
        rows = [self._clean_row(record, written_at) for record in records]
        table = pa.Table.from_pylist(rows, schema=CLEAN_SCHEMA)
        ordinal = self._clean_ordinal
        path = self._root / "clean" / f"part_{ordinal:08d}.parquet"
        atomic_write_parquet(path, table)
        self._clean_ordinal += 1
        self._clean_buffer.clear()
        for record in records:
            self._clean_statuses[record.sample_id] = record.status
        self._write_counts()

    def _flush_perturbed(self) -> None:
        if not self._perturbed_buffer:
            return
        records = tuple(self._perturbed_buffer)
        item_ids = [record.work_item.item_id for record in records]
        if len(set(item_ids)) != len(item_ids):
            raise DumpError("one perturbed fragment cannot contain duplicate item_id values")
        written_at = _now()
        rows = [self._perturbed_row(record, written_at) for record in records]
        data_table = pa.Table.from_pylist(rows, schema=PERTURBED_SCHEMA)
        ordinal = self._perturbed_ordinal
        chunk_file = f"perturbed/chunk_{ordinal:08d}.parquet"
        data_path = self._root / chunk_file
        atomic_write_parquet(data_path, data_table)

        # Reserve the ordinal as soon as the data file is durable. If index
        # publication fails, retrying must not overwrite the orphan.
        self._perturbed_ordinal += 1
        index_rows = [
            {
                "item_id": record.work_item.item_id,
                "status": record.status.value,
                "chunk_file": chunk_file,
                "written_at": written_at,
            }
            for record in records
        ]
        index_table = pa.Table.from_pylist(index_rows, schema=INDEX_SCHEMA)
        index_path = self._root / "index" / f"part_{ordinal:08d}.parquet"
        atomic_write_parquet(index_path, index_table)

        self._perturbed_buffer.clear()
        for record in records:
            self._item_statuses[record.work_item.item_id] = record.status
        self._write_counts()

    def _write_counts(self) -> None:
        self._manifest = self._manifest.model_copy(
            update={"counts_by_status": self._authoritative_counts()}
        )
        write_manifest(self._manifest_path, self._manifest)

    def _authoritative_counts(self) -> dict[ItemStatus, int]:
        counts: Counter[ItemStatus] = Counter(self._clean_statuses.values())
        counts.update(self._item_statuses.values())
        return {status: counts.get(status, 0) for status in ItemStatus}

    def _warn_for_logits(self, output: object) -> None:
        if output is None or self._logits_warning_emitted:
            return
        threshold = self._config.dump.max_classes_for_full_logits
        if threshold is not None and output.logits.size > threshold:
            self._logger.warning(
                "dump.large_logits classes=%d threshold=%d",
                output.logits.size,
                threshold,
            )
            self._logits_warning_emitted = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise DumpError("dump writer is closed")

    @staticmethod
    def _clean_row(record: CleanDumpRecord, written_at: datetime) -> dict[str, object]:
        return {
            "sample_id": record.sample_id,
            "content_hash": record.content_hash,
            "gt_label": record.gt_label,
            "status": record.status.value,
            "logits": _logits_list(record.logits),
            "original_shape": record.original_shape,
            "model_id": record.model_id,
            "written_at": written_at,
        }

    @staticmethod
    def _perturbed_row(
        record: PerturbedDumpRecord,
        written_at: datetime,
    ) -> dict[str, object]:
        work_item = record.work_item
        region = work_item.region_spec
        meta = record.region_meta
        return {
            "item_id": work_item.item_id,
            "sample_id": work_item.sample_id,
            "status": record.status.value,
            "region_id": region.region_id,
            "region_instance_id": region.region_instance_id,
            "region_kind": region.kind.value,
            "region_params_json": canonical_json(region.params),
            "intended_area_px": None if meta is None else meta.intended_area_px,
            "intended_area_ratio": None if meta is None else meta.intended_area_ratio,
            "generator_kind": None if meta is None else meta.generator_kind,
            "generator_version": None if meta is None else meta.generator_version,
            "generator_confidence": None if meta is None else meta.confidence,
            "effective_area_px": record.effective_area_px,
            "effective_area_available": record.effective_area_px is not None,
            "perturb_op": work_item.perturb_op.value,
            "perturb_params_json": canonical_json(work_item.perturb_params),
            "invert_mask": work_item.invert_mask,
            "is_control": work_item.is_control,
            "seed_used": seed_to_hex(record.seed_used),
            "logits": _logits_list(record.logits),
            "written_at": written_at,
        }


def _logits_list(output: object) -> list[float] | None:
    if output is None:
        return None
    return np.asarray(output.logits, dtype=np.float32).tolist()


def _now() -> datetime:
    return datetime.now(timezone.utc)
