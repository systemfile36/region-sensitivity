"""Low-level durable storage helpers for the control/stability analysis module.

Re-implements the same atomic-write/fsync pattern as
``ssat.metrics._storage`` (which itself re-implements ``ssat.core.dump._storage``'s
pattern), rather than importing it directly — ``analysis.store`` is restricted
to depending on ``analysis.types`` and ``ssat.utils`` only, not
``ssat.metrics``. Duplicating this well-tested, package-boundary-local helper
is the same trade-off ``ssat.metrics._storage`` itself already made one layer
down.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from ssat.analysis.errors import AnalysisCorruptionError, AnalysisSchemaError


def fsync_directory(directory: Path) -> None:
    """Persist a directory entry update on the Unix deployment target."""

    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    """Read one Parquet file only after strict version and schema checks.

    Raises:
        AnalysisCorruptionError: If the file is missing or cannot be read.
        AnalysisSchemaError: If the file's version metadata or Arrow schema
            does not match what was expected.
    """

    if not path.is_file():
        raise AnalysisCorruptionError(f"analysis fragment does not exist: {path}")
    try:
        parquet_file = pq.ParquetFile(path)
        actual_schema = parquet_file.schema_arrow
    except Exception as error:
        raise AnalysisCorruptionError(f"cannot read analysis fragment: {path}") from error

    metadata = actual_schema.metadata or {}
    raw_version = metadata.get(b"ssat.analysis_schema_version")
    if raw_version is None:
        raise AnalysisSchemaError(f"analysis fragment has no schema version: {path}")
    try:
        actual_version = raw_version.decode("ascii")
    except UnicodeDecodeError as error:
        raise AnalysisSchemaError(f"analysis fragment has invalid schema version: {path}") from error
    if actual_version != expected_version:
        raise AnalysisSchemaError(
            f"analysis schema version mismatch for {path}: "
            f"expected {expected_version}, found {actual_version}"
        )
    if not actual_schema.equals(expected_schema, check_metadata=True):
        raise AnalysisSchemaError(f"analysis schema mismatch: {path}")
    try:
        return parquet_file.read()
    except Exception as error:
        raise AnalysisCorruptionError(f"cannot read analysis rows: {path}") from error
