"""Adapter metadata and raw model outputs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

import numpy as np
from numpy.typing import NDArray


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class PreprocessingSpec:
    """Describe one deterministic or caller-managed preprocessing pipeline.

    Attributes:
        kind: Stable implementation category such as ``declarative`` or ``timm``.
        deterministic: Whether repeated transformations are deterministic.
        description: Human-readable manifest summary.
        fingerprint: Optional canonical SHA-256 identity of the resolved pipeline.
        mask_transform_available: Whether model-space mask geometry is available.
    """

    kind: str
    deterministic: bool
    description: str
    fingerprint: str | None = None
    mask_transform_available: bool = False

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("preprocessing kind must not be empty")
        if not isinstance(self.deterministic, bool):
            raise TypeError("preprocessing deterministic must be bool")
        if not isinstance(self.description, str):
            raise TypeError("preprocessing description must be str")
        _validate_optional_hash(self.fingerprint, field_name="preprocessing fingerprint")
        if not isinstance(self.mask_transform_available, bool):
            raise TypeError("mask_transform_available must be bool")


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
        adapter_kind: Stable adapter implementation category.
        model_name: Optional framework model registry name.
        weights_id: Optional framework or user-provided weights identifier.
        weights_hash: Optional SHA-256 identity of a model artifact.
        preprocessing_fingerprint: Optional SHA-256 pipeline identity.
        mask_transform_available: Whether effective model-space area can be computed.
    """

    model_id: str
    deterministic: bool
    input_layout: Literal["THWC_uint8"] = "THWC_uint8"
    max_batch_size: int | None = None
    output_kind: Literal["logits"] = "logits"
    class_names: tuple[str, ...] | None = None
    preprocessing_desc: str = ""
    adapter_kind: str = "custom"
    model_name: str | None = None
    weights_id: str | None = None
    weights_hash: str | None = None
    preprocessing_fingerprint: str | None = None
    mask_transform_available: bool = False

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if self.max_batch_size is not None and self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive when provided")
        if self.class_names is not None and not self.class_names:
            raise ValueError("class_names must not be empty when provided")
        if self.class_names is not None and any(not name for name in self.class_names):
            raise ValueError("class_names entries must not be empty")
        if not self.adapter_kind:
            raise ValueError("adapter_kind must not be empty")
        for field_name, value in (
            ("model_name", self.model_name),
            ("weights_id", self.weights_id),
        ):
            if value is not None and not value:
                raise ValueError(f"{field_name} must not be empty when provided")
        _validate_optional_hash(self.weights_hash, field_name="weights_hash")
        _validate_optional_hash(
            self.preprocessing_fingerprint,
            field_name="preprocessing_fingerprint",
        )
        if not isinstance(self.mask_transform_available, bool):
            raise TypeError("mask_transform_available must be bool")


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


def _validate_optional_hash(value: str | None, *, field_name: str) -> None:
    """Require lowercase full SHA-256 hex when an identity is available."""

    if value is not None and _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
