"""Torchvision model-zoo image classification adapter."""

from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ssat.core.adapter._torch_helpers import (
    transform_mask_with_ops,
    validate_image_classifier_batch,
)
from ssat.core.adapter.checkpoint import load_state_dict_checkpoint
from ssat.core.adapter.base import (
    AdapterError,
    AdapterOutOfMemoryError,
    ModelAdapter,
)
from ssat.core.adapter.output_decoder import LogitsOutputDecoder, OutputDecoder
from ssat.core.adapter.preprocessing import DeclarativePreprocessor, OpInput
from ssat.core.adapter.preprocessor import (
    Preprocessor,
    fingerprint_payload,
    validate_mask,
)
from ssat.core.adapter.transform_registry import TransformRegistry
from ssat.core.adapter.transforms import PipelinePreprocessor
from ssat.core.adapter.types import AdapterSpec, PreprocessingSpec, RawOutput


class TorchvisionPreprocessor(Preprocessor):
    """Own torchvision evaluation pixel transforms and matching mask geometry."""

    def __init__(self, preprocessing: Any) -> None:
        self._preprocessing = preprocessing
        self._geometry_ops = self._preset_geometry(preprocessing)
        payload = {
            "resize_size": preprocessing.resize_size,
            "crop_size": preprocessing.crop_size,
            "mean": preprocessing.mean,
            "std": preprocessing.std,
            "interpolation": preprocessing.interpolation,
            "antialias": preprocessing.antialias,
        }
        self._spec = PreprocessingSpec(
            kind="torchvision",
            deterministic=True,
            description=repr(preprocessing),
            fingerprint=fingerprint_payload(payload),
            mask_transform_available=True,
        )

    @staticmethod
    def _preset_geometry(preprocessing: Any) -> tuple[Any, ...]:
        """Convert a classification preset into explicit geometry."""

        from torchvision.transforms import CenterCrop, Resize

        required = (
            "resize_size",
            "crop_size",
            "mean",
            "std",
            "interpolation",
            "antialias",
        )
        if not all(hasattr(preprocessing, name) for name in required):
            raise AdapterError("torchvision weight preset has unsupported preprocessing")
        return (
            Resize(
                preprocessing.resize_size,
                interpolation=preprocessing.interpolation,
                antialias=preprocessing.antialias,
            ),
            CenterCrop(preprocessing.crop_size),
        )

    def describe(self) -> PreprocessingSpec:
        """Return the canonical torchvision preprocessing identity."""

        return self._spec

    def transform_batch(self, batch: NDArray[np.uint8]) -> Any:
        """Convert THWC RGB images through the weight preset on CPU."""

        import torch

        validate_image_classifier_batch(batch)
        images = torch.from_numpy(np.ascontiguousarray(batch[:, 0])).permute(0, 3, 1, 2)
        return self._preprocessing(images)

    def transform_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Apply the preset resize/crop geometry with nearest interpolation."""

        validate_mask(mask)
        return transform_mask_with_ops(mask, self._geometry_ops)


class TorchvisionAdapter(ModelAdapter):
    """Instantiate and run a torchvision classifier by registered model name.

    ``weights=None`` avoids implicit network access. Passing ``"DEFAULT"`` or
    a weight enum member enables pretrained weights and their matching metadata.
    Even without weights, the default weight preset supplies the architecture's
    standard deterministic evaluation preprocessing.
    """

    def __init__(
        self,
        model_name: str,
        *,
        weights: Any = None,
        device: Any = None,
        deterministic: bool = True,
        max_batch_size: int | None = None,
        model_id: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        init_seed: int = 0,
        output_decoder: OutputDecoder | None = None,
        weights_hash: str | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_state_dict_key: str | None = None,
        checkpoint_strict: bool = True,
        preprocessing_ops: Sequence[OpInput] | None = None,
        pipeline_config: Sequence[Mapping[str, Any]] | None = None,
        transform_registry: TransformRegistry | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")
        if isinstance(init_seed, bool) or not 0 <= init_seed <= 2**63 - 1:
            raise ValueError("init_seed must be between 0 and 2**63 - 1")
        if weights is not None and checkpoint_path is not None:
            raise ValueError("weights and checkpoint_path are mutually exclusive")
        if preprocessing_ops is not None and pipeline_config is not None:
            raise ValueError("preprocessing_ops and pipeline_config are mutually exclusive")
        try:
            import torch
            from torchvision import models

            weights_enum = models.get_model_weights(model_name)
            resolved_weights = self._resolve_weights(weights_enum, weights)
            preprocessing_weights = resolved_weights or weights_enum.DEFAULT
            preprocessing = preprocessing_weights.transforms()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(init_seed)
                model = models.get_model(
                    model_name,
                    weights=resolved_weights,
                    **(model_kwargs or {}),
                )
            if checkpoint_path is not None:
                load_state_dict_checkpoint(
                    model,
                    checkpoint_path,
                    state_dict_key=checkpoint_state_dict_key,
                    strict=checkpoint_strict,
                )
        except Exception as error:
            raise AdapterError(
                f"failed to initialize torchvision model {model_name!r}"
            ) from error

        resolved_device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        try:
            self._model = model.eval().to(resolved_device)
        except Exception as error:
            raise AdapterError(f"failed to move model to device {resolved_device}") from error
        self._device = resolved_device
        # An explicit op list or transform pipeline replaces the weight
        # preset entirely. The preset is only a sensible default when the
        # source images resemble what the weights were trained on; for
        # anything else its fixed Resize/CenterCrop geometry silently
        # reshapes -- and can partly crop away -- the very regions being
        # audited. pipeline_config takes priority over preprocessing_ops,
        # but the two are already rejected as mutually exclusive above.
        self._preprocessor: Preprocessor = (
            PipelinePreprocessor(pipeline_config, registry=transform_registry)
            if pipeline_config is not None
            else DeclarativePreprocessor(preprocessing_ops)
            if preprocessing_ops is not None
            else TorchvisionPreprocessor(preprocessing)
        )
        self._output_decoder = output_decoder or LogitsOutputDecoder()
        if not isinstance(self._output_decoder, OutputDecoder):
            raise TypeError("output_decoder must implement OutputDecoder")
        preprocessing_spec = self._preprocessor.describe()

        checkpoint_name = None if checkpoint_path is None else Path(checkpoint_path).name
        weight_name = (
            f"checkpoint:{checkpoint_name}"
            if checkpoint_name is not None
            else ("none" if resolved_weights is None else str(resolved_weights))
        )
        categories = (
            resolved_weights.meta.get("categories")
            if resolved_weights is not None
            else None
        )
        class_names = tuple(categories) if categories else None
        default_model_id = f"torchvision:{model_name}:weights={weight_name}"
        if resolved_weights is None and checkpoint_path is None:
            default_model_id += f":init_seed={init_seed}"
        weights_id = (
            (f"checkpoint:{checkpoint_name}" if checkpoint_name is not None else f"none:init_seed={init_seed}")
            if resolved_weights is None
            else str(resolved_weights)
        )
        self._spec = AdapterSpec(
            model_id=model_id or default_model_id,
            deterministic=deterministic and preprocessing_spec.deterministic,
            max_batch_size=max_batch_size,
            class_names=class_names,
            preprocessing_desc=preprocessing_spec.description,
            adapter_kind="torchvision",
            model_name=model_name,
            weights_id=weights_id,
            weights_hash=weights_hash,
            preprocessing_fingerprint=preprocessing_spec.fingerprint,
            mask_transform_available=preprocessing_spec.mask_transform_available,
        )

    @staticmethod
    def _resolve_weights(weights_enum: Any, weights: Any) -> Any:
        """Resolve a convenient string weight selector against one enum."""

        if weights is None:
            return None
        if isinstance(weights, str):
            if weights == "DEFAULT":
                return weights_enum.DEFAULT
            try:
                return weights_enum[weights]
            except KeyError as error:
                raise ValueError(f"unknown torchvision weights {weights!r}") from error
        if not isinstance(weights, weights_enum):
            raise TypeError(f"weights must be None, a string, or {weights_enum.__name__}")
        return weights

    def describe(self) -> AdapterSpec:
        """Return model, determinism, class, and preprocessing metadata."""

        return self._spec

    def predict(self, batch: NDArray[np.uint8]) -> list[RawOutput]:
        """Preprocess a THWC image batch and return CPU numpy logits."""

        import torch

        self._validate_batch(batch, self._spec)
        validate_image_classifier_batch(batch)
        if batch.shape[0] == 0:
            return []
        try:
            prepared = self._preprocessor.transform_batch(batch)
            # TorchvisionPreprocessor hands back a torch tensor; the
            # declarative pipeline stays in numpy, so adopt it here rather
            # than making every op torch-aware.
            if not isinstance(prepared, torch.Tensor):
                prepared = torch.from_numpy(np.ascontiguousarray(prepared))
        except Exception as error:
            if isinstance(error, AdapterError):
                raise
            raise AdapterError("torchvision preprocessing failed") from error
        try:
            with torch.inference_mode():
                logits = self._model(prepared.to(self._device))
        except torch.cuda.OutOfMemoryError as error:
            raise AdapterOutOfMemoryError("torchvision prediction ran out of memory") from error
        except Exception as error:
            raise AdapterError("torchvision prediction failed") from error
        try:
            return self._output_decoder.decode(
                logits,
                batch_size=batch.shape[0],
                class_names=self._spec.class_names,
            )
        except Exception as error:
            if isinstance(error, AdapterError):
                raise
            raise AdapterError("torchvision output decoding failed") from error

    def transform_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Apply the preset resize/crop geometry with nearest interpolation."""

        self._validate_mask(mask)
        transformed = self._preprocessor.transform_mask(mask)
        if transformed is None:  # pragma: no cover - contract is always available
            raise AdapterError("torchvision mask transform is unavailable")
        return transformed

    def cleanup_after_oom(self) -> None:
        """Release Python and CUDA allocations before a smaller retry."""

        import torch

        gc.collect()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
