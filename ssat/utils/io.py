"""Small, reusable helpers for configuration and manifest file I/O."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> Any:
    """Load one UTF-8 YAML document with safe constructors.

    Args:
        path: YAML file to read.

    Returns:
        The Python value represented by the YAML document.
    """

    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_json(path: str | Path) -> Any:
    """Load one UTF-8 JSON document.

    Args:
        path: JSON file to read.

    Returns:
        The decoded JSON-compatible value.
    """

    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a file's SHA-256 digest without loading it fully into memory.

    Args:
        path: File whose contents should be hashed.
        chunk_size: Positive number of bytes read per iteration.

    Returns:
        A 64-character lowercase hexadecimal digest.

    Raises:
        ValueError: If ``chunk_size`` is not positive.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(
    path: str | Path,
    value: Any,
    *,
    indent: int | None = 2,
) -> None:
    """Atomically replace a JSON file with an fsynced UTF-8 document.

    The caller is responsible for converting models and dataclasses to
    JSON-compatible values before calling this helper.

    Args:
        path: Destination JSON file.
        value: JSON-compatible value to serialize.
        indent: Optional indentation passed to ``json.dump``.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(
                value,
                file,
                ensure_ascii=False,
                indent=indent,
                sort_keys=True,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

