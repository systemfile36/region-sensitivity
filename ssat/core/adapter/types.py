"""Adapter metadata and raw model outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    """Describe a model adapter before an audit run starts.

    ConfigResolver uses this metadata to reject nondeterministic preprocessing,
    while the runtime uses it to respect adapter batch limits.

    Attributes:
        model_id: Stable identifier recorded in run outputs.
        deterministic: Whether repeated inference is deterministic.
        input_layout: Array layout accepted by the adapter.
        max_batch_size: Optional adapter-declared batch-size ceiling.
        output_kind: Raw output representation; v1 accepts logits only.
        class_names: Optional class names ordered like the logits vector.
        preprocessing_desc: Human-readable preprocessing summary.
    """

    model_id: str
    deterministic: bool
    input_layout: Literal["THWC_uint8"] = "THWC_uint8"
    max_batch_size: int | None = None
    output_kind: Literal["logits"] = "logits"
    class_names: tuple[str, ...] | None = None
    preprocessing_desc: str = ""

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if self.max_batch_size is not None and self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive when provided")
        if self.class_names is not None and not self.class_names:
            raise ValueError("class_names must not be empty when provided")


@dataclass(frozen=True, slots=True)
class RawOutput:
    """Carry one sample's unmodified model logits to the dump layer.

    Attributes:
        logits: One-dimensional floating array ordered by model class.
    """

    logits: NDArray[np.floating]

    def __post_init__(self) -> None:
        if not isinstance(self.logits, np.ndarray):
            raise TypeError("logits must be a numpy ndarray")
        if self.logits.ndim != 1:
            raise ValueError("logits must be one-dimensional")
        if not np.issubdtype(self.logits.dtype, np.floating):
            raise TypeError("logits must use a floating dtype")
