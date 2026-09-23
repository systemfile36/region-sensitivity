"""Deterministic, outcome-blind sample subsets for revision-1 runs.

Every subset is fixed by ranking identifiers on ``sha256(f"{salt}:{id}")``
before any result is seen, so a subset can be regenerated exactly from its
salt and the source annotation file alone.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from ssat.utils.io import sha256_file


def hash_key(identifier: str, salt: str) -> str:
    """Return the ranking key ``sha256(f"{salt}:{identifier}")`` as hex."""

    return hashlib.sha256(f"{salt}:{identifier}".encode("utf-8")).hexdigest()


def hash_rank(ids: Iterable[str], salt: str) -> list[str]:
    """Order unique identifiers by ascending hash key (identifier breaks ties).

    Raises:
        ValueError: If ``ids`` contains duplicates.
    """

    ids = list(ids)
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be unique")
    return sorted(ids, key=lambda identifier: (hash_key(identifier, salt), identifier))


def read_annotation(annotation_file: Path) -> list[tuple[str, int]]:
    """Read an ImageNet-style ``<filename> <label>`` annotation file."""

    entries = []
    for line_number, line in enumerate(annotation_file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{annotation_file}:{line_number}: expected '<filename> <label>'")
        entries.append((parts[0], int(parts[1])))
    return entries


def class_balanced_subset(
    annotation_file: Path, per_class: int, salt: str
) -> list[tuple[str, int]]:
    """Pick the ``per_class`` lowest-hash files within every class.

    Returns:
        ``(filename, label)`` pairs ordered by label, then by hash rank.

    Raises:
        ValueError: If a class has fewer than ``per_class`` files.
    """

    if per_class < 1:
        raise ValueError("per_class must be positive")
    by_label: dict[int, list[str]] = defaultdict(list)
    for filename, label in read_annotation(annotation_file):
        by_label[label].append(filename)
    selected = []
    for label in sorted(by_label):
        ranked = hash_rank(by_label[label], salt)
        if len(ranked) < per_class:
            raise ValueError(f"class {label} has only {len(ranked)} files, need {per_class}")
        selected.extend((filename, label) for filename in ranked[:per_class])
    return selected


def top_n_subset(annotation_file: Path, n: int, salt: str) -> list[tuple[str, int]]:
    """Pick the ``n`` lowest-hash files overall; smaller ``n`` gives a nested prefix.

    Raises:
        ValueError: If the annotation file has fewer than ``n`` entries.
    """

    entries = dict(read_annotation(annotation_file))
    ranked = hash_rank(entries, salt)
    if len(ranked) < n:
        raise ValueError(f"annotation has only {len(ranked)} files, need {n}")
    return [(filename, entries[filename]) for filename in ranked[:n]]


def write_annotation(lines: Sequence[tuple[str, int]], path: Path) -> str:
    """Write ``(filename, label)`` pairs as an annotation file and return its SHA-256."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{filename} {label}\n" for filename, label in lines), encoding="utf-8")
    return sha256_file(path)
