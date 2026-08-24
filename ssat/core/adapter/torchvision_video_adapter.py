"""Torchvision model-zoo video action-recognition adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ssat.core.adapter._torch_adapter_mixin import _TorchAdapterMixin
from ssat.core.adapter._torch_helpers import resolve_torchvision_weights
from ssat.core.adapter.base import AdapterError, ModelAdapter
from ssat.core.adapter.checkpoint import load_state_dict_checkpoint
from ssat.core.adapter.output_decoder import LogitsOutputDecoder, OutputDecoder
from ssat.core.adapter.preprocessing import (
    CenterCrop,
    ChannelsFirst,
    DeclarativePreprocessor,
    Normalize,
    Resize,
    ToFloat,
)
from ssat.core.adapter.preprocessor import Preprocessor
from ssat.core.adapter.transform_registry import TransformRegistry
from ssat.core.adapter.transforms import PipelinePreprocessor
from ssat.core.adapter.types import AdapterSpec, RawOutput

# Kinetics-400 defaults shared by torchvision's r3d_18/mc3_18/s3d weight presets.
DEFAULT_RESIZE_SIZE = 128
DEFAULT_CROP_SIZE = 112
DEFAULT_MEAN = (0.43216, 0.394666, 0.37645)
DEFAULT_STD = (0.22803, 0.22145, 0.216989)


class TorchvisionVideoAdapter(_TorchAdapterMixin, ModelAdapter):
    """Instantiate and run a ``torchvision.models.video`` classifier.

    Unlike :class:`TorchvisionAdapter`, clips with ``T > 1`` are accepted.
    Pixels are resized/cropped/normalized in ``(B,T,H,W,C)`` through the same
    declarative pipeline used elsewhere in SSAT, then transposed to the
    ``(B,C,T,H,W)`` layout torchvision video models expect only inside
    :meth:`predict`, so the resize/crop geometry (and therefore mask
    transform) stays framework-agnostic and shared with image adapters.
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
        resize_size: int = DEFAULT_RESIZE_SIZE,
        crop_size: int = DEFAULT_CROP_SIZE,
        mean: tuple[float, ...] = DEFAULT_MEAN,
        std: tuple[float, ...] = DEFAULT_STD,
        output_decoder: OutputDecoder | None = None,
        weights_hash: str | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_state_dict_key: str | None = None,
        checkpoint_strict: bool = True,
        pipeline_config: Sequence[Mapping[str, Any]] | None = None,
        transform_registry: TransformRegistry | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")
        self._validate_init_seed(init_seed)
        if weights is not None and checkpoint_path is not None:
            raise ValueError("weights and checkpoint_path are mutually exclusive")
        try:
            from torchvision import models

            weights_enum = models.get_model_weights(model_name)
            resolved_weights = resolve_torchvision_weights(weights_enum, weights)
            model = self._seeded_model_init(
                init_seed,
                lambda: models.get_model(
                    model_name,
                    weights=resolved_weights,
                    **(model_kwargs or {}),
                ),
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
                f"failed to initialize torchvision video model {model_name!r}"
            ) from error

        self._place_on_device(model, device)
        # pipeline_config replaces the fixed resize/crop/normalize entirely
        # (resize_size/crop_size/mean/std are then unused, mirroring how
        # TorchvisionAdapter's preprocessing_ops overrides its weight
        # preset). A pipeline must end in FormatShape(input_format="NTCHW")
        # to match predict()'s hardcoded permute below -- ending in
        # "NCHW" instead fails clearly there rather than silently, since
        # the resulting 4D array cannot be permuted as if it were 5D.
        self._preprocessor: Preprocessor = (
            PipelinePreprocessor(pipeline_config, registry=transform_registry)
            if pipeline_config is not None
            else DeclarativePreprocessor(
                [
                    Resize(resize_size),
                    CenterCrop(crop_size),
                    ToFloat(),
                    Normalize(mean, std),
                    ChannelsFirst(),
                ]
            )
        )
        self._output_decoder = self._finalize_output_decoder(
            output_decoder, LogitsOutputDecoder()
        )
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
        default_model_id = f"torchvision_video:{model_name}:weights={weight_name}"
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
            adapter_kind="torchvision_video",
            model_name=model_name,
            weights_id=weights_id,
            weights_hash=weights_hash,
            preprocessing_fingerprint=preprocessing_spec.fingerprint,
            mask_transform_available=preprocessing_spec.mask_transform_available,
        )

    def describe(self) -> AdapterSpec:
        """Return model, determinism, class, and preprocessing metadata."""

        return self._spec

    def predict(self, batch: NDArray[np.uint8]) -> list[RawOutput]:
        """Preprocess a THWC video batch and return CPU numpy logits."""

        import torch

        self._validate_batch(batch, self._spec)
        if batch.shape[0] == 0:
            return []

        def _preprocess() -> Any:
            prepared = self._preprocessor.transform_batch(batch)
            return torch.from_numpy(prepared).permute(0, 2, 1, 3, 4).contiguous()

        clip = self._wrap_stage(_preprocess, "torchvision video preprocessing failed")

        def _infer() -> Any:
            with torch.inference_mode():
                return self._model(clip.to(self._device))

        logits = self._wrap_inference(
            _infer,
            oom_message="torchvision video prediction ran out of memory",
            error_message="torchvision video prediction failed",
        )

        def _decode() -> list[RawOutput]:
            return self._output_decoder.decode(
                logits,
                batch_size=batch.shape[0],
                class_names=self._spec.class_names,
            )

        return self._wrap_stage(_decode, "torchvision video output decoding failed")

    def transform_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Apply the shared resize/crop geometry with nearest interpolation."""

        self._validate_mask(mask)
        transformed = self._preprocessor.transform_mask(mask)
        if transformed is None:  # pragma: no cover - contract is always available
            raise AdapterError("torchvision video mask transform is unavailable")
        return transformed
