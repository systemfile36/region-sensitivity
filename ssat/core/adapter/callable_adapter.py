"""Adapter for user-supplied numpy prediction callables."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ssat.core.adapter.base import AdapterError, ModelAdapter
from ssat.core.adapter.output_decoder import LogitsOutputDecoder, OutputDecoder
from ssat.core.adapter.preprocessor import CallablePreprocessor, Preprocessor
from ssat.core.adapter.types import AdapterSpec, RawOutput


class CallableAdapter(ModelAdapter):
    """Wrap a prediction callable with explicit, validated metadata."""

    def __init__(
        self,
        predict_fn: Callable[[Any], Any],
        *,
        model_id: str,
        deterministic: bool = True,
        max_batch_size: int | None = None,
        class_names: tuple[str, ...] | None = None,
        preprocessing_desc: str = "caller-managed",
        transform_mask_fn: Callable[[NDArray[np.bool_]], NDArray[np.bool_]] | None = None,
        preprocessor: Preprocessor | None = None,
        output_decoder: OutputDecoder | None = None,
        preprocessing_fingerprint: str | None = None,
        adapter_kind: str = "callable",
        model_name: str | None = None,
        weights_id: str | None = None,
        weights_hash: str | None = None,
        cleanup_after_oom_fn: Callable[[], None] | None = None,
    ) -> None:
        if not callable(predict_fn):
            raise TypeError("predict_fn must be callable")
        if transform_mask_fn is not None and not callable(transform_mask_fn):
            raise TypeError("transform_mask_fn must be callable when provided")
        if cleanup_after_oom_fn is not None and not callable(cleanup_after_oom_fn):
            raise TypeError("cleanup_after_oom_fn must be callable when provided")
        if preprocessor is not None and not isinstance(preprocessor, Preprocessor):
            raise TypeError("preprocessor must implement Preprocessor")
        if output_decoder is not None and not isinstance(output_decoder, OutputDecoder):
            raise TypeError("output_decoder must implement OutputDecoder")
        if preprocessor is not None and transform_mask_fn is not None:
            raise ValueError("preprocessor and transform_mask_fn cannot be used together")
        if preprocessor is not None and preprocessing_fingerprint is not None:
            raise ValueError(
                "preprocessor owns its fingerprint when explicitly provided"
            )
        self._predict_fn = predict_fn
        self._preprocessor = preprocessor or CallablePreprocessor(
            transform_mask_fn=transform_mask_fn,
            description=preprocessing_desc,
            fingerprint=preprocessing_fingerprint,
        )
        self._output_decoder = output_decoder or LogitsOutputDecoder()
        self._cleanup_after_oom_fn = cleanup_after_oom_fn
        preprocessing_spec = self._preprocessor.describe()
        self._spec = AdapterSpec(
            model_id=model_id,
            deterministic=deterministic and preprocessing_spec.deterministic,
            max_batch_size=max_batch_size,
            class_names=class_names,
            preprocessing_desc=preprocessing_spec.description,
            adapter_kind=adapter_kind,
            model_name=model_name,
            weights_id=weights_id,
            weights_hash=weights_hash,
            preprocessing_fingerprint=preprocessing_spec.fingerprint,
            mask_transform_available=preprocessing_spec.mask_transform_available,
        )

    def describe(self) -> AdapterSpec:
        """Return the immutable metadata supplied at construction."""

        return self._spec

    def predict(self, batch: NDArray[np.uint8]) -> list[RawOutput]:
        """Invoke the user callable and normalize its output."""

        self._validate_batch(batch, self._spec)
        if batch.shape[0] == 0:
            return []
        try:
            prepared = self._preprocessor.transform_batch(batch)
        except Exception as error:
            if isinstance(error, AdapterError):
                raise
            raise AdapterError("preprocessing failed") from error
        try:
            outputs = self._predict_fn(prepared)
        except Exception as error:
            if isinstance(error, AdapterError):
                raise
            raise AdapterError("model invocation failed") from error
        try:
            return self._output_decoder.decode(
                outputs,
                batch_size=batch.shape[0],
                class_names=self._spec.class_names,
            )
        except Exception as error:
            if isinstance(error, AdapterError):
                raise
            raise AdapterError("output decoding failed") from error

    def transform_mask(
        self,
        mask: NDArray[np.bool_],
    ) -> NDArray[np.bool_] | None:
        """Invoke the optional user mask transform with strict validation."""

        self._validate_mask(mask)
        try:
            transformed = self._preprocessor.transform_mask(mask)
        except Exception as error:
            if isinstance(error, AdapterError):
                raise
            raise AdapterError("mask preprocessing failed") from error
        if transformed is not None:
            self._validate_mask(transformed)
        return transformed

    def cleanup_after_oom(self) -> None:
        """Invoke the optional user-supplied resource cleanup callback."""

        if self._cleanup_after_oom_fn is not None:
            self._cleanup_after_oom_fn()
