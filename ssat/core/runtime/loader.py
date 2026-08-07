"""Shared lazy DataLoader boundary for runtime and preflight profiling."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def identity_collate(value: T) -> T:
    """Preserve NumPy arrays when automatic batching is disabled."""

    return value


def iter_worker_results(dataset: object, *, num_workers: int) -> Iterable[object]:
    """Iterate a dataset with the execution layer's fixed worker contract."""

    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=identity_collate,
    )
