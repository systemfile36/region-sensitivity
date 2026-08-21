"""Native torchvision implementation of the eight-segment TSM-ResNet50."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ssat.core.adapter.base import AdapterError, AdapterOutOfMemoryError, ModelAdapter
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
from ssat.core.adapter.types import AdapterSpec, RawOutput

MMACTION_MEAN = (123.675, 116.28, 103.53)
MMACTION_STD = (58.395, 57.12, 57.375)


class TemporalShift:
    """Namespace for the parameter-free temporal shift operation."""

    @staticmethod
    def shift(x: Any, num_segments: int, shift_div: int) -> Any:
        """Shift two channel folds in opposite temporal directions."""

        import torch

        if x.ndim != 4:
            raise ValueError("temporal shift input must have shape (N*T,C,H,W)")
        if num_segments <= 0 or shift_div <= 0:
            raise ValueError("num_segments and shift_div must be positive")
        nt, channels, height, width = x.shape
        if nt % num_segments:
            raise ValueError("temporal shift batch is not divisible by num_segments")
        fold = channels // shift_div
        if fold == 0:
            return x
        view = x.reshape(nt // num_segments, num_segments, channels, height, width)
        out = torch.zeros_like(view)
        out[:, :-1, :fold] = view[:, 1:, :fold]
        out[:, 1:, fold : 2 * fold] = view[:, :-1, fold : 2 * fold]
        out[:, :, 2 * fold :] = view[:, :, 2 * fold :]
        return out.reshape(nt, channels, height, width)


def _temporal_shift_module(net: Any, *, num_segments: int, shift_div: int) -> Any:
    import torch.nn as nn

    class _TemporalShiftModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = net

        def forward(self, x: Any) -> Any:
            return self.net(TemporalShift.shift(x, num_segments, shift_div))

    return _TemporalShiftModule()


def build_tsm_resnet50(*, num_segments: int, num_classes: int, shift_div: int) -> Any:
    """Build a torchvision ResNet50 with block-residual temporal shifts."""

    from torchvision.models import resnet50

    backbone = resnet50(weights=None, num_classes=num_classes)
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        for block in getattr(backbone, layer_name):
            block.conv1 = _temporal_shift_module(
                block.conv1,
                num_segments=num_segments,
                shift_div=shift_div,
            )
    return backbone


class TSMResNet50Model:
    """Factory namespace retained for a stable, testable model constructor."""

    @staticmethod
    def create(*, num_segments: int, num_classes: int, shift_div: int) -> Any:
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = build_tsm_resnet50(
                    num_segments=num_segments,
                    num_classes=num_classes,
                    shift_div=shift_div,
                )

            def forward(self, clip: Any) -> Any:
                if clip.ndim != 5:
                    raise ValueError("TSM input must have shape (B,T,C,H,W)")
                batch_size, segments = clip.shape[:2]
                if segments != num_segments:
                    raise ValueError(
                        f"TSM requires {num_segments} segments, received {segments}"
                    )
                logits = self.backbone(clip.reshape((-1,) + tuple(clip.shape[2:])))
                return logits.reshape(batch_size, segments, -1).mean(dim=1)

        return _Model()


class TorchvisionTSMAdapter(ModelAdapter):
    """Run an NTU60 TSM checkpoint without a runtime MMAction2 dependency."""

    def __init__(
        self,
        *,
        num_segments: int = 8,
        num_classes: int = 60,
        shift_div: int = 8,
        preprocessing: Literal["mmaction2_val", "crop_free"] = "mmaction2_val",
        device: Any = None,
        deterministic: bool = True,
        max_batch_size: int | None = None,
        model_id: str | None = None,
        init_seed: int = 0,
        output_decoder: OutputDecoder | None = None,
        weights_hash: str | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_state_dict_key: str | None = None,
        checkpoint_strict: bool = True,
    ) -> None:
        if num_segments <= 0 or num_classes <= 0 or shift_div <= 0:
            raise ValueError("num_segments, num_classes, and shift_div must be positive")
        if preprocessing not in {"mmaction2_val", "crop_free"}:
            raise ValueError("preprocessing must be 'mmaction2_val' or 'crop_free'")
        if isinstance(init_seed, bool) or not 0 <= init_seed <= 2**63 - 1:
            raise ValueError("init_seed must be between 0 and 2**63 - 1")
        try:
            import torch

            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(init_seed)
                model = TSMResNet50Model.create(
                    num_segments=num_segments,
                    num_classes=num_classes,
                    shift_div=shift_div,
                )
            if checkpoint_path is not None:
                load_state_dict_checkpoint(
                    model,
                    checkpoint_path,
                    state_dict_key=checkpoint_state_dict_key,
                    strict=checkpoint_strict,
                )
        except Exception as error:
            raise AdapterError("failed to initialize torchvision TSM-ResNet50") from error

        resolved_device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        try:
            self._model = model.eval().to(resolved_device)
        except Exception as error:
            raise AdapterError(f"failed to move model to device {resolved_device}") from error
        self._device = resolved_device
        self._num_segments = num_segments
        geometry = (
            [Resize(256), CenterCrop(224)]
            if preprocessing == "mmaction2_val"
            else [Resize((224, 224))]
        )
        self._preprocessor = DeclarativePreprocessor(
            [
                *geometry,
                ToFloat(scale=1.0),
                Normalize(MMACTION_MEAN, MMACTION_STD),
                ChannelsFirst(),
            ]
        )
        self._output_decoder = output_decoder or LogitsOutputDecoder()
        if not isinstance(self._output_decoder, OutputDecoder):
            raise TypeError("output_decoder must implement OutputDecoder")
        preprocessing_spec = self._preprocessor.describe()
        checkpoint_name = None if checkpoint_path is None else Path(checkpoint_path).name
        weights_id = (
            f"checkpoint:{checkpoint_name}"
            if checkpoint_name is not None
            else f"none:init_seed={init_seed}"
        )
        default_id = (
            f"torchvision_tsm:tsm_resnet50:segments={num_segments}:"
            f"classes={num_classes}:weights={weights_id}"
        )
        self._spec = AdapterSpec(
            model_id=model_id or default_id,
            deterministic=deterministic and preprocessing_spec.deterministic,
            max_batch_size=max_batch_size,
            preprocessing_desc=preprocessing_spec.description,
            adapter_kind="torchvision_tsm",
            model_name="tsm_resnet50",
            weights_id=weights_id,
            weights_hash=weights_hash,
            preprocessing_fingerprint=preprocessing_spec.fingerprint,
            mask_transform_available=True,
        )

    def describe(self) -> AdapterSpec:
        return self._spec

    def predict(self, batch: NDArray[np.uint8]) -> list[RawOutput]:
        import torch

        self._validate_batch(batch, self._spec)
        if batch.shape[1] != self._num_segments:
            raise AdapterError(
                f"TSM requires T={self._num_segments}, received T={batch.shape[1]}"
            )
        if batch.shape[-1] != 3:
            raise AdapterError("TSM requires three RGB channels")
        if batch.shape[0] == 0:
            return []
        try:
            prepared = torch.from_numpy(self._preprocessor.transform_batch(batch))
            with torch.inference_mode():
                logits = self._model(prepared.to(self._device))
        except torch.cuda.OutOfMemoryError as error:
            raise AdapterOutOfMemoryError("TSM prediction ran out of memory") from error
        except Exception as error:
            if isinstance(error, AdapterError):
                raise
            raise AdapterError("TSM prediction failed") from error
        return self._output_decoder.decode(logits, batch_size=batch.shape[0])

    def transform_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        self._validate_mask(mask)
        transformed = self._preprocessor.transform_mask(mask)
        if transformed is None:  # pragma: no cover
            raise AdapterError("TSM mask transform is unavailable")
        return transformed

    def cleanup_after_oom(self) -> None:
        import torch

        gc.collect()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
