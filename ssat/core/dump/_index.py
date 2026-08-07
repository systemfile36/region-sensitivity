"""Authoritative latest-row maps shared by readers and resume filtering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

from ssat.core.dump._storage import fragment_files, read_validated_table
from ssat.core.dump.errors import DumpCorruptionError
from ssat.core.dump.schema import CLEAN_SCHEMA, INDEX_SCHEMA
from ssat.core.types import ItemStatus


@dataclass(frozen=True, slots=True)
class IndexLocation:
    """Locate the authoritative perturbed row for one item."""

    status: ItemStatus
    chunk_file: str
    written_at: datetime
    ordinal: int


@dataclass(frozen=True, slots=True)
class CleanLocation:
    """Locate the authoritative clean row for one sample."""

    status: ItemStatus
    fragment_file: str
    ordinal: int


def load_index_locations(
    root: Path,
    *,
    expected_version: str,
) -> dict[str, IndexLocation]:
    """Load index fragments in commit order, keeping the latest item entry."""

    locations: dict[str, IndexLocation] = {}
    for ordinal, path in fragment_files(root / "index", "part"):
        table = read_validated_table(
            path,
            INDEX_SCHEMA,
            expected_version=expected_version,
        )
        rows = table.to_pylist()
        item_ids = [row["item_id"] for row in rows]
        if len(set(item_ids)) != len(item_ids):
            raise DumpCorruptionError(f"duplicate item_id within index fragment: {path}")
        expected_chunk = f"perturbed/chunk_{ordinal:08d}.parquet"
        for row in rows:
            item_id = _validate_identifier(row["item_id"], field_name="item_id")
            chunk_file = _validate_chunk_file(row["chunk_file"])
            if chunk_file != expected_chunk:
                raise DumpCorruptionError(
                    f"index fragment {path} must reference {expected_chunk}"
                )
            chunk_path = root / PurePosixPath(chunk_file)
            if not chunk_path.is_file():
                raise DumpCorruptionError(
                    f"index references a missing data fragment: {chunk_file}"
                )
            locations[item_id] = IndexLocation(
                status=_parse_status(row["status"]),
                chunk_file=chunk_file,
                written_at=row["written_at"],
                ordinal=ordinal,
            )
    return locations


def load_clean_locations(
    root: Path,
    *,
    expected_version: str,
) -> dict[str, CleanLocation]:
    """Scan clean fragments in order, keeping the latest sample entry."""

    locations: dict[str, CleanLocation] = {}
    for ordinal, path in fragment_files(root / "clean", "part"):
        table = read_validated_table(
            path,
            CLEAN_SCHEMA,
            expected_version=expected_version,
        )
        rows = table.to_pylist()
        sample_ids = [row["sample_id"] for row in rows]
        if len(set(sample_ids)) != len(sample_ids):
            raise DumpCorruptionError(f"duplicate sample_id within clean fragment: {path}")
        relative = f"clean/part_{ordinal:08d}.parquet"
        for row in rows:
            sample_id = row["sample_id"]
            if not isinstance(sample_id, str) or not sample_id:
                raise DumpCorruptionError(f"invalid sample_id in clean fragment: {path}")
            locations[sample_id] = CleanLocation(
                status=_parse_status(row["status"]),
                fragment_file=relative,
                ordinal=ordinal,
            )
    return locations


def _parse_status(value: object) -> ItemStatus:
    try:
        return ItemStatus(value)
    except (TypeError, ValueError) as error:
        raise DumpCorruptionError(f"unknown dump status: {value!r}") from error


def _validate_identifier(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DumpCorruptionError(f"invalid {field_name} in index fragment")
    return value


def _validate_chunk_file(value: object) -> str:
    if not isinstance(value, str):
        raise DumpCorruptionError("index chunk_file must be a string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 2:
        raise DumpCorruptionError(f"unsafe index chunk_file: {value!r}")
    return path.as_posix()
