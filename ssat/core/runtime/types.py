"""Values passed from workers to the main inference process."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ssat.core.region.types import RegionMeta
from ssat.core.types import ItemStatus


@dataclass(frozen=True, slots=True)
class ItemMeta:
    """Carry per-item region and status metadata alongside worker output.

    Attributes:
        item_id: Identifier joining arrays, logits, and dump records.
        region_meta: Materialized mask measurements, when available.
        status: Preparation status for the item.
    """

    item_id: str
    region_meta: RegionMeta | None = None
    status: ItemStatus = ItemStatus.OK


@dataclass(frozen=True, slots=True)
class PreparedChunk:
    """Transfer successful variants and failures from a worker process.

    Attributes:
        arrays: Successful variants in ``(K, T, H, W, C)`` uint8 layout.
        item_metas: Metadata aligned with the first array dimension.
        failed_items: Failed item metadata excluded from model inference.
    """

    arrays: NDArray[np.uint8]
    item_metas: tuple[ItemMeta, ...]
    failed_items: tuple[ItemMeta, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.arrays, np.ndarray):
            raise TypeError("arrays must be a numpy ndarray")
        if self.arrays.dtype != np.uint8:
            raise TypeError("arrays must have dtype uint8")
        if self.arrays.ndim != 5:
            raise ValueError("arrays must use (K, T, H, W, C) layout")
        if self.arrays.shape[0] != len(self.item_metas):
            raise ValueError("arrays and successful item_metas must be aligned")
        if any(meta.status is not ItemStatus.OK for meta in self.item_metas):
            raise ValueError("item_metas may contain successful items only")
        if any(meta.status is ItemStatus.OK for meta in self.failed_items):
            raise ValueError("failed_items must use a failure status")


@dataclass(frozen=True, slots=True)
class FailedChunk:
    """Report a chunk-wide failure without raising across worker IPC.

    Attributes:
        reason: Failure status applied to every item in the chunk.
        item_ids: Items that should receive failure dump records.
    """

    reason: ItemStatus
    item_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.reason is ItemStatus.OK:
            raise ValueError("a failed chunk cannot use ok status")
        if not self.item_ids:
            raise ValueError("item_ids must not be empty")
