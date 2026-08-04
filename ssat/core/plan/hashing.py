"""Schema-versioned canonical serialization and item ID generation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from ssat.core.types import SCHEMA_VERSION


def _canonicalize(value: Any, *, omit_none: bool = False) -> Any:
    """Normalize supported values into the schema-versioned JSON form."""

    if value is None:
        return None
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, BaseModel):
        return _canonicalize(value.model_dump(mode="python"), omit_none=True)
    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(
            {field.name: getattr(value, field.name) for field in fields(value)},
            omit_none=True,
        )
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical mappings require string keys")
            if omit_none and child is None:
                continue
            normalized[key] = _canonicalize(child, omit_none=True)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(child) for child in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical serialization rejects NaN and infinity")
        if value == 0.0:
            value = 0.0
        return format(value, ".12f")
    if isinstance(value, str):
        return value
    raise TypeError(f"unsupported canonical value type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize a value using the v1 canonical JSON contract.

    Args:
        value: Supported mapping, sequence, model, dataclass, or scalar value.

    Returns:
        Compact UTF-8-compatible JSON with deterministic ordering and floats.

    Raises:
        TypeError: If the value contains an unsupported type or mapping key.
        ValueError: If the value contains NaN or infinity.
    """

    return json.dumps(
        _canonicalize(value, omit_none=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Encode canonical JSON as UTF-8 bytes for hashing or persistence.

    Args:
        value: Value accepted by :func:`canonical_json`.

    Returns:
        Canonical JSON encoded as UTF-8.
    """

    return canonical_json(value).encode("utf-8")


def compute_item_id(value: Any, *, schema_version: str = SCHEMA_VERSION) -> str:
    """Compute the stable identifier used for resume and result joins.

    Args:
        value: Work item, identity payload, or mapping containing ``item_id``.
        schema_version: Namespace for the canonical identity contract.

    Returns:
        A 64-character lowercase SHA-256 hexadecimal digest.
    """

    if hasattr(value, "identity_payload"):
        payload = value.identity_payload()
    elif isinstance(value, Mapping):
        payload = {key: child for key, child in value.items() if key != "item_id"}
    else:
        payload = value
    envelope = {"schema_version": schema_version, "work_item": payload}
    return hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()


def compute_chunk_id(
    sample_id: str,
    chunk_ordinal: int,
    item_ids: tuple[str, ...],
    *,
    schema_version: str = SCHEMA_VERSION,
) -> str:
    """Compute the stable identifier for one ordered sample chunk.

    Args:
        sample_id: Source sample shared by every item in the chunk.
        chunk_ordinal: Zero-based chunk position within the sample.
        item_ids: Ordered item identifiers contained in the chunk.
        schema_version: Namespace for the canonical chunk contract.

    Returns:
        A 64-character lowercase SHA-256 hexadecimal digest.

    Raises:
        ValueError: If the sample, ordinal, or item list is invalid.
    """

    if not sample_id:
        raise ValueError("sample_id must not be empty")
    if chunk_ordinal < 0:
        raise ValueError("chunk_ordinal must be non-negative")
    if not item_ids:
        raise ValueError("item_ids must not be empty")

    envelope = {
        "schema_version": schema_version,
        "work_chunk": {
            "sample_id": sample_id,
            "chunk_ordinal": chunk_ordinal,
            "item_ids": item_ids,
        },
    }
    return hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()
