"""Low-level durable storage helpers shared by dump components."""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from ssat.core.dump.errors import DumpCorruptionError, DumpSchemaError


def fragment_files(directory: Path, prefix: str) -> tuple[tuple[int, Path], ...]:
    """Return recognized fragment paths sorted by their numeric ordinal."""

    if not directory.exists():
        return ()
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{8,}})\.parquet$")
    fragments: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match is not None and path.is_file():
            fragments.append((int(match.group(1)), path))
        elif path.suffix == ".parquet":
            raise DumpCorruptionError(f"unexpected parquet fragment name: {path}")
    fragments.sort(key=lambda item: item[0])
    ordinals = [ordinal for ordinal, _ in fragments]
    if len(set(ordinals)) != len(ordinals):  # pragma: no cover - filesystem invariant
        raise DumpCorruptionError(f"duplicate fragment ordinals in {directory}")
    return tuple(fragments)


def next_fragment_ordinal(directory: Path, prefix: str) -> int:
    """Return an ordinal that never overwrites an existing fragment."""

    fragments = fragment_files(directory, prefix)
    return 0 if not fragments else fragments[-1][0] + 1


def atomic_write_parquet(path: Path, table: pa.Table) -> None:
    """Write, fsync, and atomically publish one independent Parquet file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        pq.write_table(table, temporary_path, compression="zstd")
        with temporary_path.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
        fsync_directory(path.parent)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def read_validated_table(
    path: Path,
    expected_schema: pa.Schema,
    *,
    expected_version: str,
) -> pa.Table:
    """Read one fragment only after strict version and Arrow-schema checks."""

    try:
        parquet_file = pq.ParquetFile(path)
        actual_schema = parquet_file.schema_arrow
    except Exception as error:
        raise DumpCorruptionError(f"cannot read parquet fragment: {path}") from error

    metadata = actual_schema.metadata or {}
    raw_version = metadata.get(b"ssat.schema_version")
    if raw_version is None:
        raise DumpSchemaError(f"parquet fragment has no schema version: {path}")
    try:
        actual_version = raw_version.decode("ascii")
    except UnicodeDecodeError as error:
        raise DumpSchemaError(f"parquet fragment has invalid schema version: {path}") from error
    if actual_version != expected_version:
        raise DumpSchemaError(
            f"schema version mismatch for {path}: "
            f"expected {expected_version}, found {actual_version}"
        )
    if not actual_schema.equals(expected_schema, check_metadata=True):
        raise DumpSchemaError(f"parquet schema mismatch: {path}")
    try:
        return parquet_file.read()
    except Exception as error:
        raise DumpCorruptionError(f"cannot read parquet rows: {path}") from error


def fsync_directory(directory: Path) -> None:
    """Persist a directory entry update on the Unix deployment target."""

    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
