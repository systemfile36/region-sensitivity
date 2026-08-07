"""Tests for adaptive main-process inference batching."""

from __future__ import annotations

import numpy as np

from ssat.core.adapter.base import AdapterOutOfMemoryError, ModelAdapter
from ssat.core.adapter.types import AdapterSpec, RawOutput
from ssat.core.runtime import (
    BatchSizeState,
    BatchSplitter,
    CleanInferenceItem,
    InferenceBatch,
    Rebatcher,
)
from ssat.core.source.types import LoadedSample
from ssat.core.types import ItemStatus


def _inference_item(sample_id: str, shape: tuple[int, int, int, int]) -> CleanInferenceItem:
    array = np.zeros(shape, dtype=np.uint8)
    return CleanInferenceItem(
        LoadedSample(array, sample_id, shape, "a" * 64)
    )


class ThresholdAdapter(ModelAdapter):
    def __init__(self, threshold: int, *, ordinary_error: bool = False) -> None:
        self.threshold = threshold
        self.ordinary_error = ordinary_error
        self.calls: list[int] = []
        self.cleanup_calls = 0
        self._spec = AdapterSpec(model_id="threshold", deterministic=True)

    def describe(self) -> AdapterSpec:
        return self._spec

    def predict(self, batch: np.ndarray) -> list[RawOutput]:
        self.calls.append(len(batch))
        if self.ordinary_error:
            raise RuntimeError("prediction failed")
        if len(batch) > self.threshold:
            raise AdapterOutOfMemoryError("oom")
        return [RawOutput(np.array([float(array.mean())], dtype=np.float32)) for array in batch]

    def cleanup_after_oom(self) -> None:
        self.cleanup_calls += 1


def test_rebatcher_combines_boundaries_and_flushes_shape_changes() -> None:
    state = BatchSizeState(3)
    items = [
        _inference_item("a", (1, 4, 4, 3)),
        _inference_item("b", (1, 4, 4, 3)),
        _inference_item("c", (1, 5, 4, 3)),
        _inference_item("d", (1, 5, 4, 3)),
    ]

    batches = list(Rebatcher(items, state))

    assert [len(batch.items) for batch in batches] == [2, 2]
    assert [tuple(batch.arrays.shape[1:]) for batch in batches] == [
        (1, 4, 4, 3),
        (1, 5, 4, 3),
    ]


def test_rebatcher_observes_cap_reduction_after_yield() -> None:
    state = BatchSizeState(3)
    iterator = iter(
        Rebatcher(
            [_inference_item(str(index), (1, 4, 4, 3)) for index in range(6)],
            state,
        )
    )

    assert len(next(iterator).items) == 3
    state.record_oom(3)
    assert [len(batch.items) for batch in iterator] == [1, 1, 1]


def test_batch_splitter_recovers_recursively_and_preserves_order() -> None:
    adapter = ThresholdAdapter(1)
    state = BatchSizeState(8)
    items = tuple(_inference_item(str(index), (1, 4, 4, 3)) for index in range(5))
    batch = InferenceBatch(items, np.stack([item.array for item in items]))

    results = BatchSplitter(adapter, state).predict(batch)

    assert [result.item.sample.sample_id for result in results] == [
        "0",
        "1",
        "2",
        "3",
        "4",
    ]
    assert all(result.status is ItemStatus.OK for result in results)
    assert adapter.cleanup_calls == 4
    assert state.current_size == 1
    assert state.oom_events == 4


def test_batch_splitter_marks_singleton_oom_and_does_not_split_other_errors() -> None:
    item = _inference_item("sample", (1, 4, 4, 3))
    batch = InferenceBatch((item,), np.stack([item.array]))
    oom_adapter = ThresholdAdapter(0)

    result = BatchSplitter(oom_adapter, BatchSizeState(2)).predict(batch)

    assert result[0].status is ItemStatus.SKIPPED_OOM
    assert oom_adapter.cleanup_calls == 1

    failing_adapter = ThresholdAdapter(10, ordinary_error=True)
    failed = BatchSplitter(failing_adapter, BatchSizeState(2)).predict(batch)
    assert failed[0].status is ItemStatus.PREDICT_FAILED
    assert failing_adapter.calls == [1]
