"""Shared measurement helpers for sanity checks and cost profiling."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import math
from typing import TypeVar

import numpy as np

from ssat.core.adapter.base import ModelAdapter
from ssat.core.adapter.types import AdapterSpec, RawOutput
from ssat.core.config.schema import ResolvedConfig
from ssat.core.estimate.errors import EstimationError
from ssat.core.runtime.pipeline import initial_batch_size
from ssat.core.runtime.types import BatchSizeState
from ssat.core.types import ItemStatus

T = TypeVar("T")


class _CountingAdapter(ModelAdapter):
    """Count adapter invocations while preserving the wrapped behavior.

    Args:
        adapter: Model adapter measured by an estimate pass.

    Attributes:
        adapter: Wrapped model adapter.
        inference_calls: Number of prediction calls, including OOM retries.
    """

    def __init__(self, adapter: ModelAdapter) -> None:
        """Initialize an invocation-counting adapter wrapper.

        Args:
            adapter: Model adapter measured by an estimate pass.
        """

        self.adapter = adapter
        self.inference_calls = 0

    def describe(self) -> AdapterSpec:
        """Return metadata from the wrapped adapter.

        Returns:
            The wrapped adapter specification.
        """

        return self.adapter.describe()

    def predict(self, batch: np.ndarray) -> list[RawOutput]:
        """Count and delegate one model prediction call.

        Args:
            batch: Validated model input batch.

        Returns:
            Raw outputs returned by the wrapped adapter.
        """

        self.inference_calls += 1
        return self.adapter.predict(batch)

    def transform_mask(self, mask: np.ndarray) -> np.ndarray | None:
        """Delegate model-space mask transformation.

        Args:
            mask: Source-space boolean mask.

        Returns:
            The transformed mask, or ``None`` when unavailable.
        """

        return self.adapter.transform_mask(mask)

    def cleanup_after_oom(self) -> None:
        """Delegate resource cleanup after a recoverable OOM."""

        self.adapter.cleanup_after_oom()


def _select_evenly(values: Sequence[T], limit: int) -> tuple[T, ...]:
    """Select deterministic representatives spanning a full sequence.

    Args:
        values: Ordered candidates to sample.
        limit: Positive maximum number of returned values.

    Returns:
        At most ``limit`` values distributed from the first to last position.

    Raises:
        ValueError: If ``limit`` is not a positive integer.
    """

    items = tuple(values)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if len(items) <= limit:
        return items
    if limit == 1:
        return (items[(len(items) - 1) // 2],)
    indices = [
        round(index * (len(items) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return tuple(items[index] for index in indices)


def _new_batch_size_state(
    config: ResolvedConfig,
    spec: AdapterSpec,
) -> BatchSizeState:
    """Create the adaptive batch-size state for an estimate pass.

    Args:
        config: Resolved configuration containing the target batch size.
        spec: Adapter metadata containing an optional batch-size cap.

    Returns:
        A new mutable batch-size state.
    """

    return BatchSizeState(
        initial_batch_size(config.runtime.target_batch_size, spec)
    )


def _validate_provenance(
    config: ResolvedConfig,
    adapter: ModelAdapter,
) -> None:
    """Validate estimate inputs against resolved adapter provenance.

    Args:
        config: Fully resolved audit configuration.
        adapter: Model adapter supplied for measurement.

    Raises:
        TypeError: If an input has an invalid type.
        EstimationError: If adapter metadata differs from the configuration.
    """

    if not isinstance(config, ResolvedConfig):
        raise TypeError("config must be a ResolvedConfig")
    if not isinstance(adapter, ModelAdapter):
        raise TypeError("adapter must be a ModelAdapter")
    if adapter.describe() != config.adapter_spec:
        raise EstimationError("adapter spec does not match resolved config")


def _merge_class_count(current: int | None, observed: int) -> int:
    """Merge one observed logit dimension into a measured class count.

    Args:
        current: Previously observed class count, when available.
        observed: Class count from one successful prediction.

    Returns:
        The consistent positive class count.

    Raises:
        EstimationError: If counts are empty or inconsistent.
    """

    if observed <= 0:
        raise EstimationError("successful logits must contain at least one class")
    if current is not None and current != observed:
        raise EstimationError(
            "successful logits class dimensions are inconsistent"
        )
    return observed


def _elapsed(value: float) -> float:
    """Validate a measured elapsed duration.

    Args:
        value: Duration reported by the injected clock.

    Returns:
        The positive finite duration.

    Raises:
        EstimationError: If the duration is nonpositive or nonfinite.
    """

    if not math.isfinite(value) or value <= 0.0:
        raise EstimationError(
            "profile clock must report positive finite elapsed time"
        )
    return value


def _complete_counts(
    counts: Counter[ItemStatus],
) -> dict[ItemStatus, int]:
    """Return a status mapping containing every terminal status.

    Args:
        counts: Sparse status counter collected during measurement.

    Returns:
        A complete mapping keyed by every ``ItemStatus`` value.
    """

    return {status: counts[status] for status in ItemStatus}
