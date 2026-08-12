"""Worker transport and main-process inference values."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ssat.core.adapter.types import RawOutput
from ssat.core.plan.types import WorkItem
from ssat.core.region.types import RegionMeta
from ssat.core.source.types import LoadedSample
from ssat.core.types import ItemStatus


def _validate_hash(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class ItemMeta:
    """Carry per-item preparation metadata alongside worker output."""

    item_id: str
    region_meta: RegionMeta | None = None
    status: ItemStatus = ItemStatus.OK

    def __post_init__(self) -> None:
        _validate_hash(self.item_id, field_name="item_id")
        if not isinstance(self.status, ItemStatus):
            raise TypeError("status must be an ItemStatus")
        if self.status is ItemStatus.OK and self.region_meta is None:
            raise ValueError("successful prepared items require region_meta")


@dataclass(frozen=True, slots=True)
class PreparedChunk:
    """Transfer prepared variants and source-space masks from one worker.

    ``masks`` is a tuple rather than one stacked array because each item's
    mask may independently be ``(H, W)`` (broadcast across every frame) or
    ``(T, H, W)`` (selected per frame); a single ndarray cannot hold both
    ranks.
    """

    chunk_id: str
    arrays: NDArray[np.uint8]
    masks: tuple[NDArray[np.bool_], ...]
    item_metas: tuple[ItemMeta, ...]
    failed_items: tuple[ItemMeta, ...] = ()

    def __post_init__(self) -> None:
        _validate_hash(self.chunk_id, field_name="chunk_id")
        if not isinstance(self.arrays, np.ndarray):
            raise TypeError("arrays must be a numpy ndarray")
        if self.arrays.dtype != np.uint8:
            raise TypeError("arrays must have dtype uint8")
        if self.arrays.ndim != 5:
            raise ValueError("arrays must use (K, T, H, W, C) layout")
        if not isinstance(self.masks, tuple):
            raise TypeError("masks must be a tuple of per-item mask arrays")
        if self.arrays.shape[0] != len(self.item_metas) or len(self.masks) != len(
            self.item_metas
        ):
            raise ValueError("arrays, masks, and successful item_metas must be aligned")
        frame_count, height, width = self.arrays.shape[1:4]
        for mask in self.masks:
            if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
                raise TypeError("each mask must be a boolean numpy ndarray")
            if mask.shape not in ((height, width), (frame_count, height, width)):
                raise ValueError(
                    "each mask must use (H, W) or (T, H, W) layout matching "
                    "prepared arrays"
                )
        if any(meta.status is not ItemStatus.OK for meta in self.item_metas):
            raise ValueError("item_metas may contain successful items only")
        if any(
            meta.status is not ItemStatus.PREPARE_FAILED
            for meta in self.failed_items
        ):
            raise ValueError("failed_items must use prepare_failed status")


@dataclass(frozen=True, slots=True)
class FailedChunk:
    """Report a chunk-wide load failure without raising across worker IPC."""

    chunk_id: str
    reason: ItemStatus
    item_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_hash(self.chunk_id, field_name="chunk_id")
        if self.reason is ItemStatus.OK:
            raise ValueError("a failed chunk cannot use ok status")
        if not self.item_ids:
            raise ValueError("item_ids must not be empty")
        for item_id in self.item_ids:
            _validate_hash(item_id, field_name="item_id")


@dataclass(frozen=True, slots=True)
class CleanInferenceItem:
    """Main-process clean inference input."""

    sample: LoadedSample

    @property
    def array(self) -> NDArray[np.uint8]:
        return self.sample.array


@dataclass(frozen=True, slots=True)
class PerturbedInferenceItem:
    """Main-process perturbed input hydrated from worker transport metadata."""

    work_item: WorkItem
    array: NDArray[np.uint8]
    mask: NDArray[np.bool_]
    region_meta: RegionMeta
    seed_used: int
    effective_area_px: int | None = None

    def __post_init__(self) -> None:
        if self.array.dtype != np.uint8 or self.array.ndim != 4:
            raise ValueError("array must be (T, H, W, C) uint8")
        frame_count, height, width = self.array.shape[:3]
        if self.mask.dtype != np.bool_ or self.mask.shape not in (
            (height, width),
            (frame_count, height, width),
        ):
            raise ValueError("mask must be (H, W) or (T, H, W) bool matching array")
        if not 0 <= self.seed_used < 2**128:
            raise ValueError("seed_used must be an unsigned 128-bit integer")
        if self.effective_area_px is not None and self.effective_area_px < 0:
            raise ValueError("effective_area_px must be non-negative")


InferenceItem: TypeAlias = CleanInferenceItem | PerturbedInferenceItem


@dataclass(frozen=True, slots=True)
class InferenceBatch:
    """One shape-homogeneous batch owned by the main process."""

    items: tuple[InferenceItem, ...]
    arrays: NDArray[np.uint8]

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("inference batch must not be empty")
        if self.arrays.dtype != np.uint8 or self.arrays.ndim != 5:
            raise ValueError("arrays must be (B, T, H, W, C) uint8")
        if self.arrays.shape[0] != len(self.items):
            raise ValueError("items and arrays must be aligned")
        if any(item.array.shape != self.arrays.shape[1:] for item in self.items):
            raise ValueError("every item array must match the batch shape")


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """One ordered inference outcome returned by BatchSplitter."""

    item: InferenceItem
    status: ItemStatus
    output: RawOutput | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            ItemStatus.OK,
            ItemStatus.PREDICT_FAILED,
            ItemStatus.SKIPPED_OOM,
        }:
            raise ValueError(
                "splitter results support only ok, predict_failed, or skipped_oom"
            )
        if (self.status is ItemStatus.OK) != (self.output is not None):
            raise ValueError("only successful predictions may carry output")


@dataclass(slots=True)
class BatchSizeState:
    """Mutable batch cap shared by clean and perturbed inference."""

    initial_size: int
    current_size: int = field(init=False)
    oom_events: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.initial_size, bool) or self.initial_size <= 0:
            raise ValueError("initial_size must be positive")
        self.current_size = self.initial_size

    def record_oom(self, failed_size: int) -> None:
        if failed_size <= 0:
            raise ValueError("failed_size must be positive")
        self.oom_events += 1
        self.current_size = min(self.current_size, max(1, failed_size // 2))


@dataclass(frozen=True, slots=True)
class ExecutionSummary:
    """Describe records emitted and adaptive batching during one invocation."""

    clean_pending: int
    chunks_pending: int
    clean_records: int
    perturbed_records: int
    counts_by_status: dict[ItemStatus, int]
    oom_events: int
    initial_batch_size: int
    final_batch_size: int

    @property
    def records_written(self) -> int:
        return self.clean_records + self.perturbed_records
