"""Framework-neutral raw model output decoding contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ssat.core.adapter.base import AdapterError
from ssat.core.adapter.types import RawOutput


class OutputDecoder(ABC):
    """Decode one model invocation into aligned v1 logits outputs."""

    @abstractmethod
    def decode(
        self,
        output: Any,
        *,
        batch_size: int,
        class_names: tuple[str, ...] | None = None,
    ) -> list[RawOutput]:
        """Return exactly one one-dimensional floating logits vector per input."""


class LogitsOutputDecoder(OutputDecoder):
    """Decode common tensor, ndarray, mapping, and sequence logits results."""

    def __init__(self, *, logits_key: str = "logits") -> None:
        if not logits_key:
            raise ValueError("logits_key must not be empty")
        self._logits_key = logits_key

    def decode(
        self,
        output: Any,
        *,
        batch_size: int,
        class_names: tuple[str, ...] | None = None,
    ) -> list[RawOutput]:
        """Normalize supported raw values and validate batch/class alignment."""

        if isinstance(output, Mapping):
            if self._logits_key not in output:
                raise AdapterError(
                    f"model output mapping has no {self._logits_key!r} key"
                )
            output = output[self._logits_key]
        output = _tensor_to_numpy(output)

        normalized: list[RawOutput]
        if isinstance(output, np.ndarray):
            if output.ndim != 2:
                raise AdapterError("model output ndarray must use (B, classes) layout")
            if not np.issubdtype(output.dtype, np.floating):
                raise AdapterError("model output ndarray must use a floating dtype")
            normalized = [RawOutput(np.array(row, copy=True)) for row in output]
        elif isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
            normalized = [
                self._decode_sequence_item(item, index)
                for index, item in enumerate(output)
            ]
        else:
            raise AdapterError(
                "model output must be logits tensor, ndarray, mapping, or sequence"
            )

        if len(normalized) != batch_size:
            raise AdapterError(
                f"adapter returned {len(normalized)} outputs for batch size {batch_size}"
            )
        class_counts = {item.logits.shape[0] for item in normalized}
        if len(class_counts) > 1:
            raise AdapterError("adapter outputs must have a consistent class dimension")
        if normalized and normalized[0].logits.size == 0:
            raise AdapterError("adapter logits must not be empty")
        if class_names is not None and normalized:
            if normalized[0].logits.shape[0] != len(class_names):
                raise AdapterError(
                    "adapter logits class dimension does not match class_names"
                )
        return normalized

    @staticmethod
    def _decode_sequence_item(item: Any, index: int) -> RawOutput:
        """Decode one explicit sequence element with index-aware errors."""

        if isinstance(item, RawOutput):
            return item
        item = _tensor_to_numpy(item)
        if not isinstance(item, np.ndarray):
            raise AdapterError(
                f"model output at index {index} must be RawOutput or ndarray"
            )
        try:
            return RawOutput(np.array(item, copy=True))
        except (TypeError, ValueError) as error:
            raise AdapterError(
                f"invalid model output at index {index}: {error}"
            ) from error


def _tensor_to_numpy(value: Any) -> Any:
    """Convert torch-like tensors without importing their framework eagerly."""

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return value
