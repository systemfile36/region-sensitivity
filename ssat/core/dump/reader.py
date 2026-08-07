"""Strict, authoritative Reader API for raw audit dumps."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa

from ssat.core.dump._index import load_clean_locations, load_index_locations
from ssat.core.dump._storage import fragment_files, read_validated_table
from ssat.core.dump.errors import DumpCorruptionError
from ssat.core.dump.manifest import load_manifest
from ssat.core.dump.schema import CLEAN_SCHEMA, PERTURBED_SCHEMA
from ssat.core.dump.types import RunManifest
from ssat.core.types import SCHEMA_VERSION


class DumpReader:
    """Expose schema-validated latest rows without leaking storage layout rules."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_schema_version: str = SCHEMA_VERSION,
    ) -> None:
        self._root = Path(root)
        self._expected_version = expected_schema_version
        self._manifest = load_manifest(
            self._root / "run_manifest.json",
            expected_version=expected_schema_version,
        )

    def read_manifest(self) -> RunManifest:
        """Return the validated run manifest."""

        return self._manifest

    def iter_clean_chunks(self) -> Iterator[pa.Table]:
        """Yield nonempty clean fragment views containing latest sample rows."""

        locations = load_clean_locations(
            self._root,
            expected_version=self._expected_version,
        )
        seen: set[str] = set()
        for ordinal, path in fragment_files(self._root / "clean", "part"):
            table = read_validated_table(
                path,
                CLEAN_SCHEMA,
                expected_version=self._expected_version,
            )
            relative = f"clean/part_{ordinal:08d}.parquet"
            selected = []
            for row in table.to_pylist():
                location = locations.get(row["sample_id"])
                if location is not None and location.fragment_file == relative:
                    selected.append(row)
                    seen.add(row["sample_id"])
            if selected:
                yield pa.Table.from_pylist(selected, schema=CLEAN_SCHEMA)
        if seen != set(locations):  # pragma: no cover - defensive invariant
            raise DumpCorruptionError("clean latest-row map is inconsistent")

    def iter_perturbed_chunks(self) -> Iterator[pa.Table]:
        """Yield indexed latest rows and ignore unindexed orphan fragments."""

        locations = load_index_locations(
            self._root,
            expected_version=self._expected_version,
        )
        seen: set[str] = set()
        for ordinal, path in fragment_files(self._root / "perturbed", "chunk"):
            table = read_validated_table(
                path,
                PERTURBED_SCHEMA,
                expected_version=self._expected_version,
            )
            relative = f"perturbed/chunk_{ordinal:08d}.parquet"
            selected = []
            item_ids_in_fragment: set[str] = set()
            for row in table.to_pylist():
                item_id = row["item_id"]
                if item_id in item_ids_in_fragment:
                    raise DumpCorruptionError(
                        f"duplicate item_id within perturbed fragment: {path}"
                    )
                item_ids_in_fragment.add(item_id)
                location = locations.get(item_id)
                if location is not None and location.chunk_file == relative:
                    if location.status.value != row["status"]:
                        raise DumpCorruptionError(
                            f"index status disagrees with data fragment for {item_id}"
                        )
                    selected.append(row)
                    seen.add(item_id)
            if selected:
                yield pa.Table.from_pylist(selected, schema=PERTURBED_SCHEMA)
        missing = set(locations) - seen
        if missing:
            raise DumpCorruptionError(
                f"index entries are absent from referenced data: {len(missing)}"
            )

    def read_clean(self) -> pa.Table:
        """Combine all authoritative clean chunks for small analyses."""

        return _combine(self.iter_clean_chunks(), CLEAN_SCHEMA)

    def read_perturbed(self) -> pa.Table:
        """Combine all authoritative perturbed chunks for small analyses."""

        return _combine(self.iter_perturbed_chunks(), PERTURBED_SCHEMA)


def _combine(tables: Iterator[pa.Table], schema: pa.Schema) -> pa.Table:
    resolved = list(tables)
    if not resolved:
        return pa.Table.from_pylist([], schema=schema)
    return pa.concat_tables(resolved)
