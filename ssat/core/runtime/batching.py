"""Main-process dynamic rebatching and OOM recovery."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from ssat.core.adapter.base import AdapterOutOfMemoryError, ModelAdapter
from ssat.core.runtime.errors import RuntimeContractError
from ssat.core.runtime.types import (
    BatchSizeState,
    InferenceBatch,
    InferenceItem,
    PredictionResult,
)
from ssat.core.types import ItemStatus


class Rebatcher:
    """Combine shape-compatible items up to a mutable batch-size cap."""

    def __init__(
        self,
        items: Iterable[InferenceItem],
        batch_size_state: BatchSizeState,
    ) -> None:
        self._items = items
        self._state = batch_size_state

    def __iter__(self) -> Iterator[InferenceBatch]:
        buffered: list[InferenceItem] = []
        shape: tuple[int, ...] | None = None
        for item in self._items:
            item_shape = tuple(item.array.shape)
            if buffered and item_shape != shape:
                yield self._make_batch(buffered)
                buffered = []
            shape = item_shape
            buffered.append(item)
            while len(buffered) >= self._state.current_size:
                size = self._state.current_size
                yield self._make_batch(buffered[:size])
                buffered = buffered[size:]
        if buffered:
            yield self._make_batch(buffered)

    @staticmethod
    def _make_batch(items: list[InferenceItem]) -> InferenceBatch:
        return InferenceBatch(
            items=tuple(items),
            arrays=np.stack([item.array for item in items]),
        )


class BatchSplitter:
    """Predict a batch, recursively splitting only framework-neutral OOMs."""

    def __init__(
        self,
        adapter: ModelAdapter,
        batch_size_state: BatchSizeState,
    ) -> None:
        self._adapter = adapter
        self._state = batch_size_state

    def predict(self, batch: InferenceBatch) -> tuple[PredictionResult, ...]:
        """Return ordered results; ordinary failures mark the current batch."""

        try:
            outputs = self._adapter.predict(batch.arrays)
        except AdapterOutOfMemoryError:
            self._adapter.cleanup_after_oom()
            self._state.record_oom(len(batch.items))
            if len(batch.items) == 1:
                return (
                    PredictionResult(
                        item=batch.items[0],
                        status=ItemStatus.SKIPPED_OOM,
                    ),
                )
            midpoint = len(batch.items) // 2
            left = self._slice(batch, 0, midpoint)
            right = self._slice(batch, midpoint, len(batch.items))
            return self.predict(left) + self.predict(right)
        except RuntimeContractError:
            raise
        except Exception:
            return tuple(
                PredictionResult(item=item, status=ItemStatus.PREDICT_FAILED)
                for item in batch.items
            )

        if len(outputs) != len(batch.items):
            raise RuntimeContractError(
                "adapter returned a different number of outputs than inputs"
            )
        return tuple(
            PredictionResult(item=item, status=ItemStatus.OK, output=output)
            for item, output in zip(batch.items, outputs, strict=True)
        )

    @staticmethod
    def _slice(batch: InferenceBatch, start: int, stop: int) -> InferenceBatch:
        return InferenceBatch(
            items=batch.items[start:stop],
            arrays=batch.arrays[start:stop],
        )
