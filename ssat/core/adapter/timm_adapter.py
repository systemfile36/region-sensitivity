"""timm model-zoo image classification adapter."""

from __future__ import annotations

import gc
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ssat.core.adapter._torch_helpers import (
    transform_mask_with_ops,
    validate_image_classifier_batch,
)
from ssat.core.adapter.base import (
    AdapterError,
    AdapterOutOfMemoryError,
    ModelAdapter,
)
from ssat.core.adapter.output_decoder import LogitsOutputDecoder, OutputDecoder
from ssat.core.adapter.preprocessor import (
    Preprocessor,
    fingerprint_payload,
    validate_mask,
)
from ssat.core.adapter.types import AdapterSpec, PreprocessingSpec, RawOutput


class TimmPreprocessor(Preprocessor):
    """Own timm evaluation pixel transforms and matching mask geometry."""

    def __init__(self, preprocessing: Any, data_config: dict[str, Any]) -> None:
        self._preprocessing = preprocessing
        self._geometry_ops = tuple(preprocessing.transforms)
        self._spec = PreprocessingSpec(
            kind="timm",
            deterministic=True,
            description=json.dumps(data_config, sort_keys=True),
            fingerprint=fingerprint_payload(data_config),
            mask_transform_available=True,
        )

    def describe(self) -> PreprocessingSpec:
        """Return the canonical timm preprocessing identity."""

        return self._spec

    def transform_batch(self, batch: NDArray[np.uint8]) -> Any:
        """Apply the timm evaluation transform to every RGB image on CPU."""

        import torch

        validate_image_classifier_batch(batch)
        return torch.stack(
            [
                self._preprocessing(Image.fromarray(np.ascontiguousarray(image)))
                for image in batch[:, 0]
            ]
        )

    def transform_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Apply timm evaluation geometry while ignoring photometric ops."""

        validate_mask(mask)
        return transform_mask_with_ops(mask, self._geometry_ops)


class TimmAdapter(ModelAdapter):
    """Instantiate and run a timm classifier by registered model name."""

    def __init__(
        self,
        model_name: str,
        *,
        pretrained: bool = False,
        device: Any = None,
        deterministic: bool = True,
        max_batch_size: int | None = None,
        model_id: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        init_seed: int = 0,
        output_decoder: OutputDecoder | None = None,
        weights_hash: str | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")
        if isinstance(init_seed, bool) or not 0 <= init_seed <= 2**63 - 1:
            raise ValueError("init_seed must be between 0 and 2**63 - 1")
        try:
            import timm
            import torch
            from timm.data import create_transform, resolve_model_data_config

            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(init_seed)
                model = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    **(model_kwargs or {}),
                )
            data_config = resolve_model_data_config(model)
            preprocessing = create_transform(**data_config, is_training=False)
        except Exception as error:
            raise AdapterError(f"failed to initialize timm model {model_name!r}") from error

        resolved_device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        try:
            self._model = model.eval().to(resolved_device)
        except Exception as error:
            raise AdapterError(f"failed to move model to device {resolved_device}") from error
        self._device = resolved_device
        self._preprocessor = TimmPreprocessor(preprocessing, data_config)
        self._output_decoder = output_decoder or LogitsOutputDecoder()
        if not isinstance(self._output_decoder, OutputDecoder):
            raise TypeError("output_decoder must implement OutputDecoder")
        preprocessing_spec = self._preprocessor.describe()
        default_model_id = f"timm:{model_name}:pretrained={str(pretrained).lower()}"
        if not pretrained:
            default_model_id += f":init_seed={init_seed}"
        weights_id = self._weights_id(model, model_name, pretrained, init_seed)
        self._spec = AdapterSpec(
            model_id=model_id or default_model_id,
            deterministic=deterministic and preprocessing_spec.deterministic,
            max_batch_size=max_batch_size,
            preprocessing_desc=preprocessing_spec.description,
            adapter_kind="timm",
            model_name=model_name,
            weights_id=weights_id,
            weights_hash=weights_hash,
            preprocessing_fingerprint=preprocessing_spec.fingerprint,
            mask_transform_available=preprocessing_spec.mask_transform_available,
        )

    @staticmethod
    def _weights_id(
        model: Any,
        model_name: str,
        pretrained: bool,
        init_seed: int,
    ) -> str:
        """Resolve a manifest identifier without downloading or hashing weights."""

        if not pretrained:
            return f"none:init_seed={init_seed}"
        config = getattr(model, "pretrained_cfg", {}) or {}
        for key in ("hf_hub_id", "url", "file"):
            value = config.get(key)
            if value:
                return str(value)
        return f"pretrained:{model_name}"

    def describe(self) -> AdapterSpec:
        """Return model, determinism, and preprocessing metadata."""

        return self._spec

    def predict(self, batch: NDArray[np.uint8]) -> list[RawOutput]:
        """Apply timm evaluation preprocessing and return CPU numpy logits."""

        import torch

        self._validate_batch(batch, self._spec)
        validate_image_classifier_batch(batch)
        if batch.shape[0] == 0:
            return []
        try:
            prepared = self._preprocessor.transform_batch(batch)
        except Exception as error:
            if isinstance(error, AdapterError):
                raise
            raise AdapterError("timm preprocessing failed") from error
        try:
            with torch.inference_mode():
                logits = self._model(prepared.to(self._device))
        except torch.cuda.OutOfMemoryError as error:
            raise AdapterOutOfMemoryError("timm prediction ran out of memory") from error
        except Exception as error:
            raise AdapterError("timm prediction failed") from error
        try:
            return self._output_decoder.decode(
                logits,
                batch_size=batch.shape[0],
                class_names=self._spec.class_names,
            )
        except Exception as error:
            if isinstance(error, AdapterError):
                raise
            raise AdapterError("timm output decoding failed") from error

    def transform_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Apply timm evaluation geometry while ignoring photometric ops."""

        self._validate_mask(mask)
        transformed = self._preprocessor.transform_mask(mask)
        if transformed is None:  # pragma: no cover - contract is always available
            raise AdapterError("timm mask transform is unavailable")
        return transformed

    def cleanup_after_oom(self) -> None:
        """Release Python and CUDA allocations before a smaller retry."""

        import torch

        gc.collect()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
