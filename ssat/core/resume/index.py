"""Chunk-level resume filtering backed by durable Parquet fragments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa

from ssat.core.dump._index import (
    CleanLocation,
    IndexLocation,
    load_clean_locations,
    load_index_locations,
)
from ssat.core.dump._storage import (
    atomic_write_parquet,
    fragment_files,
    fsync_directory,
    read_validated_table,
)
from ssat.core.dump.errors import DumpCorruptionError
from ssat.core.dump.manifest import load_manifest, write_manifest
from ssat.core.dump.schema import INDEX_SCHEMA, PERTURBED_SCHEMA
from ssat.core.plan.types import WorkChunkMeta
from ssat.core.source.types import SampleMeta
from ssat.core.types import ItemStatus, SCHEMA_VERSION


class ResumeIndex:
    """Expose authoritative statuses and whole-chunk pending filters."""

    def __init__(
        self,
        root: Path,
        item_locations: dict[str, IndexLocation],
        clean_locations: dict[str, CleanLocation],
        *,
        schema_version: str,
    ) -> None:
        self._root = root
        self._item_locations = dict(item_locations)
        self._clean_locations = dict(clean_locations)
        self._schema_version = schema_version

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        expected_schema_version: str = SCHEMA_VERSION,
    ) -> ResumeIndex:
        """Load and validate manifest, clean fragments, and resume index."""

        resolved_root = Path(root)
        load_manifest(
            resolved_root / "run_manifest.json",
            expected_version=expected_schema_version,
        )
        return cls(
            resolved_root,
            load_index_locations(
                resolved_root,
                expected_version=expected_schema_version,
            ),
            load_clean_locations(
                resolved_root,
                expected_version=expected_schema_version,
            ),
            schema_version=expected_schema_version,
        )

    def item_status(self, item_id: str) -> ItemStatus | None:
        """Return the latest indexed status for an item, if present."""

        location = self._item_locations.get(item_id)
        return None if location is None else location.status

    def clean_status(self, sample_id: str) -> ItemStatus | None:
        """Return the latest clean status for a sample, if present."""

        location = self._clean_locations.get(sample_id)
        return None if location is None else location.status

    def pending_chunks(
        self,
        chunks: Iterable[WorkChunkMeta],
        *,
        retry_failed: bool,
    ) -> tuple[WorkChunkMeta, ...]:
        """Return whole chunks containing absent or retryable item results."""

        pending: list[WorkChunkMeta] = []
        for chunk in chunks:
            statuses = [self.item_status(item_id) for item_id in chunk.item_ids]
            if any(status is None for status in statuses):
                pending.append(chunk)
            elif retry_failed and any(status is not ItemStatus.OK for status in statuses):
                pending.append(chunk)
        return tuple(pending)

    def pending_clean_samples(
        self,
        samples: Iterable[SampleMeta],
        *,
        retry_failed: bool,
    ) -> tuple[SampleMeta, ...]:
        """Return clean samples that are absent or have retryable failures."""

        pending: list[SampleMeta] = []
        for sample in samples:
            status = self.clean_status(sample.sample_id)
            if status is None or (retry_failed and status is not ItemStatus.OK):
                pending.append(sample)
        return tuple(pending)

    @classmethod
    def rebuild(
        cls,
        root: str | Path,
        *,
        expected_schema_version: str = SCHEMA_VERSION,
    ) -> ResumeIndex:
        """Recreate index fragments from all valid perturbed data fragments."""

        resolved_root = Path(root)
        manifest = load_manifest(
            resolved_root / "run_manifest.json",
            expected_version=expected_schema_version,
        )
        planned: list[tuple[int, pa.Table]] = []
        for ordinal, path in fragment_files(resolved_root / "perturbed", "chunk"):
            data = read_validated_table(
                path,
                PERTURBED_SCHEMA,
                expected_version=expected_schema_version,
            )
            rows = data.to_pylist()
            item_ids = [row["item_id"] for row in rows]
            if len(set(item_ids)) != len(item_ids):
                raise DumpCorruptionError(
                    f"duplicate item_id within perturbed fragment: {path}"
                )
            chunk_file = f"perturbed/chunk_{ordinal:08d}.parquet"
            index_rows = [
                {
                    "item_id": row["item_id"],
                    "status": row["status"],
                    "chunk_file": chunk_file,
                    "written_at": row["written_at"],
                }
                for row in rows
            ]
            planned.append(
                (ordinal, pa.Table.from_pylist(index_rows, schema=INDEX_SCHEMA))
            )

        index_directory = resolved_root / "index"
        expected_ordinals = {ordinal for ordinal, _ in planned}
        for ordinal, table in planned:
            atomic_write_parquet(
                index_directory / f"part_{ordinal:08d}.parquet",
                table,
            )
        for ordinal, path in fragment_files(index_directory, "part"):
            if ordinal not in expected_ordinals:
                path.unlink()
                fsync_directory(index_directory)
        rebuilt = cls.open(
            resolved_root,
            expected_schema_version=expected_schema_version,
        )
        counts: Counter[ItemStatus] = Counter(
            location.status for location in rebuilt._clean_locations.values()
        )
        counts.update(location.status for location in rebuilt._item_locations.values())
        updated_manifest = manifest.model_copy(
            update={
                "counts_by_status": {
                    status: counts.get(status, 0) for status in ItemStatus
                }
            }
        )
        write_manifest(resolved_root / "run_manifest.json", updated_manifest)
        return rebuilt
