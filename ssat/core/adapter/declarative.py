"""Callable adapter with one declared pixel and mask preprocessing pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from numpy.typing import NDArray

from ssat.core.adapter.callable_adapter import CallableAdapter
from ssat.core.adapter.output_decoder import OutputDecoder
from ssat.core.adapter.preprocessing import (
    DeclarativePreprocessor,
    OpInput,
)


class DeclarativeAdapter(CallableAdapter):
    """Apply a deterministic op list before invoking a user prediction callable."""

    def __init__(
        self,
        predict_fn: Callable[[NDArray[Any]], Any],
        *,
        model_id: str,
        preprocessing_ops: Sequence[OpInput],
        deterministic: bool = True,
        max_batch_size: int | None = None,
        class_names: tuple[str, ...] | None = None,
        output_decoder: OutputDecoder | None = None,
        model_name: str | None = None,
        weights_id: str | None = None,
        weights_hash: str | None = None,
        cleanup_after_oom_fn: Callable[[], None] | None = None,
    ) -> None:
        self._declarative_preprocessor = DeclarativePreprocessor(preprocessing_ops)
        super().__init__(
            predict_fn,
            model_id=model_id,
            deterministic=deterministic,
            max_batch_size=max_batch_size,
            class_names=class_names,
            preprocessor=self._declarative_preprocessor,
            output_decoder=output_decoder,
            adapter_kind="declarative",
            model_name=model_name,
            weights_id=weights_id,
            weights_hash=weights_hash,
            cleanup_after_oom_fn=cleanup_after_oom_fn,
        )

    @property
    def preprocessing_ops(self) -> tuple[Any, ...]:
        """Expose the immutable resolved pipeline for inspection."""

        return self._declarative_preprocessor.ops
