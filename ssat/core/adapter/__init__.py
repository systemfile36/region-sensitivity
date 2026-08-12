"""Model adapter contracts and built-in implementations."""

from __future__ import annotations

from typing import Any

from ssat.core.adapter.base import (
    AdapterError,
    AdapterOutOfMemoryError,
    ModelAdapter,
    SupportsDescribe,
)
from ssat.core.adapter.output_decoder import LogitsOutputDecoder, OutputDecoder
from ssat.core.adapter.preprocessor import (
    CallablePreprocessor,
    IdentityPreprocessor,
    Preprocessor,
)
from ssat.core.adapter.types import AdapterSpec, PreprocessingSpec, RawOutput
from ssat.core.adapter.provider import (
    AdapterProvider,
    AdapterProviderError,
    AdapterProviderRegistry,
    CheckpointConfig,
    ProviderConfig,
    TimmProviderConfig,
    TorchvisionProviderConfig,
    TorchvisionVideoProviderConfig,
    default_adapter_provider_registry,
)

__all__ = [
    "AdapterError",
    "AdapterOutOfMemoryError",
    "AdapterProvider",
    "AdapterProviderError",
    "AdapterProviderRegistry",
    "AdapterSpec",
    "CallableAdapter",
    "CallablePreprocessor",
    "CheckpointConfig",
    "CenterCrop",
    "ChannelsFirst",
    "DeclarativeAdapter",
    "DeclarativePreprocessor",
    "IdentityPreprocessor",
    "LogitsOutputDecoder",
    "ModelAdapter",
    "Normalize",
    "OutputDecoder",
    "PreprocessingSpec",
    "Preprocessor",
    "ProviderConfig",
    "RawOutput",
    "Resize",
    "SqueezeTime",
    "SupportsDescribe",
    "TimmAdapter",
    "TimmProviderConfig",
    "TimmPreprocessor",
    "ToFloat",
    "TorchvisionAdapter",
    "TorchvisionProviderConfig",
    "TorchvisionPreprocessor",
    "TorchvisionVideoAdapter",
    "TorchvisionVideoProviderConfig",
    "default_adapter_provider_registry",
]


def __getattr__(name: str) -> Any:
    """Load concrete adapters and preprocessing only when explicitly requested."""

    if name == "CallableAdapter":
        from ssat.core.adapter.callable_adapter import CallableAdapter

        return CallableAdapter
    if name == "DeclarativeAdapter":
        from ssat.core.adapter.declarative import DeclarativeAdapter

        return DeclarativeAdapter
    if name in {
        "CenterCrop",
        "ChannelsFirst",
        "DeclarativePreprocessor",
        "Normalize",
        "Resize",
        "SqueezeTime",
        "ToFloat",
    }:
        from ssat.core.adapter import preprocessing

        return getattr(preprocessing, name)

    if name == "TorchvisionAdapter":
        from ssat.core.adapter.torchvision_adapter import TorchvisionAdapter

        return TorchvisionAdapter
    if name == "TorchvisionPreprocessor":
        from ssat.core.adapter.torchvision_adapter import TorchvisionPreprocessor

        return TorchvisionPreprocessor
    if name == "TorchvisionVideoAdapter":
        from ssat.core.adapter.torchvision_video_adapter import TorchvisionVideoAdapter

        return TorchvisionVideoAdapter
    if name == "TimmAdapter":
        from ssat.core.adapter.timm_adapter import TimmAdapter

        return TimmAdapter
    if name == "TimmPreprocessor":
        from ssat.core.adapter.timm_adapter import TimmPreprocessor

        return TimmPreprocessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
