"""Name-based adapter configuration and provider registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ssat.core._provider_registry_support import validate_config_literal
from ssat.core.adapter.base import ModelAdapter
from ssat.utils.io import sha256_file


class AdapterProviderError(ValueError):
    """Indicate invalid provider registration, configuration, or construction."""


class ProviderConfig(BaseModel):
    """Strict base for provider-specific configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    provider: str


class CheckpointConfig(BaseModel):
    """Configure one trusted local torch state-dict checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    path: Path
    state_dict_key: str | None = None
    strict: bool = True


class TorchvisionProviderConfig(ProviderConfig):
    """Configuration accepted by :class:`TorchvisionProvider`."""

    provider: Literal["torchvision"] = "torchvision"
    model_name: str
    weights: str | None = None
    checkpoint: CheckpointConfig | None = None
    device: str = "auto"
    deterministic: bool = True
    max_batch_size: int | None = Field(default=None, gt=0)
    model_id: str | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    init_seed: int = Field(default=0, ge=0, le=2**63 - 1)
    weights_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    preprocessing: tuple[dict[str, Any], ...] | None = None
    pipeline_config: tuple[dict[str, Any], ...] | None = None

    @model_validator(mode="after")
    def validate_weights(self) -> TorchvisionProviderConfig:
        if self.weights is not None and self.checkpoint is not None:
            raise ValueError("weights and checkpoint are mutually exclusive")
        if self.checkpoint is not None and self.weights_hash is not None:
            raise ValueError("checkpoint hash is computed and cannot be supplied")
        return self

    @model_validator(mode="after")
    def validate_preprocessing(self) -> TorchvisionProviderConfig:
        """Reject a malformed op list at config-load time, not mid-run.

        Without this the adapter's weight preset is the only preprocessing
        available, which silently applies the model's stock ImageNet
        Resize/CenterCrop to whatever the source images are -- for inputs
        far from that geometry the crop can trim regions unevenly and
        confound any per-region comparison.
        """

        if self.preprocessing is not None:
            from ssat.core.adapter.preprocessing import parse_preprocessing_ops

            if not self.preprocessing:
                raise ValueError("preprocessing must not be empty when provided")
            parse_preprocessing_ops(self.preprocessing)
        return self

    @model_validator(mode="after")
    def validate_pipeline_config(self) -> TorchvisionProviderConfig:
        """Reject a malformed pipeline_config at config-load time, not mid-run.

        Mirrors validate_preprocessing's fail-fast rationale for the newer
        registry-based transform pipeline. preprocessing and pipeline_config
        are mutually exclusive so a config can never silently mix the
        flat-op engine and the registry engine.
        """

        if self.pipeline_config is not None:
            if self.preprocessing is not None:
                raise ValueError("preprocessing and pipeline_config are mutually exclusive")
            if not self.pipeline_config:
                raise ValueError("pipeline_config must not be empty when provided")
            from ssat.core.adapter.transform_registry import build_pipeline
            from ssat.core.adapter.transforms import default_transform_registry

            build_pipeline(self.pipeline_config, default_transform_registry())
        return self


class TorchvisionVideoProviderConfig(ProviderConfig):
    """Configuration accepted by :class:`TorchvisionVideoProvider`."""

    provider: Literal["torchvision_video"] = "torchvision_video"
    model_name: str
    weights: str | None = None
    checkpoint: CheckpointConfig | None = None
    device: str = "auto"
    deterministic: bool = True
    max_batch_size: int | None = Field(default=None, gt=0)
    model_id: str | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    init_seed: int = Field(default=0, ge=0, le=2**63 - 1)
    weights_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    resize_size: int = Field(default=128, gt=0)
    crop_size: int = Field(default=112, gt=0)
    mean: tuple[float, ...] = (0.43216, 0.394666, 0.37645)
    std: tuple[float, ...] = (0.22803, 0.22145, 0.216989)
    pipeline_config: tuple[dict[str, Any], ...] | None = None

    @model_validator(mode="after")
    def validate_weights(self) -> TorchvisionVideoProviderConfig:
        if self.weights is not None and self.checkpoint is not None:
            raise ValueError("weights and checkpoint are mutually exclusive")
        if self.checkpoint is not None and self.weights_hash is not None:
            raise ValueError("checkpoint hash is computed and cannot be supplied")
        return self

    @model_validator(mode="after")
    def validate_pipeline_config(self) -> TorchvisionVideoProviderConfig:
        """Reject a malformed pipeline_config at config-load time, not mid-run.

        This is the first preprocessing override this config accepts --
        resize_size/crop_size/mean/std only ever fed the adapter's one fixed
        DeclarativePreprocessor. When a pipeline is used with the video
        adapter it must end in FormatShape(input_format="NTCHW"), since
        TorchvisionVideoAdapter.predict() hardcodes a permute that assumes
        that layout; this validator does not check that constraint itself
        (it has no adapter to consult) -- a pipeline ending in
        FormatShape("NCHW") still passes validation here and instead fails
        clearly inside predict() (see the implementation plan's risk table).
        """

        if self.pipeline_config is not None:
            if not self.pipeline_config:
                raise ValueError("pipeline_config must not be empty when provided")
            from ssat.core.adapter.transform_registry import build_pipeline
            from ssat.core.adapter.transforms import default_transform_registry

            build_pipeline(self.pipeline_config, default_transform_registry())
        return self


class TimmProviderConfig(ProviderConfig):
    """Configuration accepted by :class:`TimmProvider`."""

    provider: Literal["timm"] = "timm"
    model_name: str
    pretrained: bool = False
    checkpoint: CheckpointConfig | None = None
    device: str = "auto"
    deterministic: bool = True
    max_batch_size: int | None = Field(default=None, gt=0)
    model_id: str | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    init_seed: int = Field(default=0, ge=0, le=2**63 - 1)
    weights_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    geometry_mode: Literal["model_default", "squash"] = "model_default"

    @model_validator(mode="after")
    def validate_weights(self) -> TimmProviderConfig:
        if self.pretrained and self.checkpoint is not None:
            raise ValueError("pretrained and checkpoint are mutually exclusive")
        if self.checkpoint is not None and self.weights_hash is not None:
            raise ValueError("checkpoint hash is computed and cannot be supplied")
        return self


class TorchvisionTSMProviderConfig(ProviderConfig):
    """Configuration for the native MMAction-compatible TSM adapter."""

    provider: Literal["torchvision_tsm"] = "torchvision_tsm"
    model_name: Literal["tsm_resnet50"] = "tsm_resnet50"
    num_segments: int = Field(default=8, gt=0)
    num_classes: int = Field(default=60, gt=0)
    shift_div: int = Field(default=8, gt=0)
    preprocessing: Literal["mmaction2_val", "crop_free"] = "mmaction2_val"
    checkpoint: CheckpointConfig | None = None
    device: str = "auto"
    deterministic: bool = True
    max_batch_size: int | None = Field(default=None, gt=0)
    model_id: str | None = None
    init_seed: int = Field(default=0, ge=0, le=2**63 - 1)
    weights_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_checkpoint_hash(self) -> TorchvisionTSMProviderConfig:
        if self.checkpoint is not None and self.weights_hash is not None:
            raise ValueError("checkpoint hash is computed and cannot be supplied")
        return self


class AdapterProvider(ABC):
    """Build adapters from one explicitly registered configuration model."""

    name: ClassVar[str]
    config_model: ClassVar[type[ProviderConfig]]

    @abstractmethod
    def build(self, config: ProviderConfig, *, base_dir: Path) -> ModelAdapter:
        """Build one adapter from a validated provider configuration."""


class TorchvisionProvider(AdapterProvider):
    """Build torchvision classification adapters."""

    name = "torchvision"
    config_model = TorchvisionProviderConfig

    def build(self, config: ProviderConfig, *, base_dir: Path) -> ModelAdapter:
        from ssat.core.adapter.torchvision_adapter import TorchvisionAdapter

        if not isinstance(config, TorchvisionProviderConfig):
            raise TypeError("config must be TorchvisionProviderConfig")
        checkpoint, checkpoint_hash = _resolve_checkpoint(config.checkpoint, base_dir)
        return TorchvisionAdapter(
            config.model_name,
            weights=config.weights,
            device=None if config.device == "auto" else config.device,
            deterministic=config.deterministic,
            max_batch_size=config.max_batch_size,
            model_id=config.model_id,
            model_kwargs=config.model_kwargs,
            init_seed=config.init_seed,
            weights_hash=checkpoint_hash or config.weights_hash,
            checkpoint_path=checkpoint,
            checkpoint_state_dict_key=(
                None if config.checkpoint is None else config.checkpoint.state_dict_key
            ),
            checkpoint_strict=(
                True if config.checkpoint is None else config.checkpoint.strict
            ),
            preprocessing_ops=config.preprocessing,
            pipeline_config=config.pipeline_config,
        )


class TorchvisionVideoProvider(AdapterProvider):
    """Build torchvision.models.video action-recognition adapters."""

    name = "torchvision_video"
    config_model = TorchvisionVideoProviderConfig

    def build(self, config: ProviderConfig, *, base_dir: Path) -> ModelAdapter:
        from ssat.core.adapter.torchvision_video_adapter import TorchvisionVideoAdapter

        if not isinstance(config, TorchvisionVideoProviderConfig):
            raise TypeError("config must be TorchvisionVideoProviderConfig")
        checkpoint, checkpoint_hash = _resolve_checkpoint(config.checkpoint, base_dir)
        return TorchvisionVideoAdapter(
            config.model_name,
            weights=config.weights,
            device=None if config.device == "auto" else config.device,
            deterministic=config.deterministic,
            max_batch_size=config.max_batch_size,
            model_id=config.model_id,
            model_kwargs=config.model_kwargs,
            init_seed=config.init_seed,
            resize_size=config.resize_size,
            crop_size=config.crop_size,
            mean=config.mean,
            std=config.std,
            weights_hash=checkpoint_hash or config.weights_hash,
            checkpoint_path=checkpoint,
            checkpoint_state_dict_key=(
                None if config.checkpoint is None else config.checkpoint.state_dict_key
            ),
            checkpoint_strict=(
                True if config.checkpoint is None else config.checkpoint.strict
            ),
            pipeline_config=config.pipeline_config,
        )


class TimmProvider(AdapterProvider):
    """Build timm classification adapters."""

    name = "timm"
    config_model = TimmProviderConfig

    def build(self, config: ProviderConfig, *, base_dir: Path) -> ModelAdapter:
        from ssat.core.adapter.timm_adapter import TimmAdapter

        if not isinstance(config, TimmProviderConfig):
            raise TypeError("config must be TimmProviderConfig")
        checkpoint, checkpoint_hash = _resolve_checkpoint(config.checkpoint, base_dir)
        return TimmAdapter(
            config.model_name,
            pretrained=config.pretrained,
            device=None if config.device == "auto" else config.device,
            deterministic=config.deterministic,
            max_batch_size=config.max_batch_size,
            model_id=config.model_id,
            model_kwargs=config.model_kwargs,
            init_seed=config.init_seed,
            weights_hash=checkpoint_hash or config.weights_hash,
            checkpoint_path=checkpoint,
            checkpoint_state_dict_key=(
                None if config.checkpoint is None else config.checkpoint.state_dict_key
            ),
            checkpoint_strict=(
                True if config.checkpoint is None else config.checkpoint.strict
            ),
            geometry_mode=config.geometry_mode,
        )


class TorchvisionTSMProvider(AdapterProvider):
    """Build the native TSM-ResNet50 action-recognition adapter."""

    name = "torchvision_tsm"
    config_model = TorchvisionTSMProviderConfig

    def build(self, config: ProviderConfig, *, base_dir: Path) -> ModelAdapter:
        from ssat.core.adapter.torchvision_tsm_adapter import TorchvisionTSMAdapter

        if not isinstance(config, TorchvisionTSMProviderConfig):
            raise TypeError("config must be TorchvisionTSMProviderConfig")
        checkpoint, checkpoint_hash = _resolve_checkpoint(config.checkpoint, base_dir)
        return TorchvisionTSMAdapter(
            num_segments=config.num_segments,
            num_classes=config.num_classes,
            shift_div=config.shift_div,
            preprocessing=config.preprocessing,
            device=None if config.device == "auto" else config.device,
            deterministic=config.deterministic,
            max_batch_size=config.max_batch_size,
            model_id=config.model_id,
            init_seed=config.init_seed,
            weights_hash=checkpoint_hash or config.weights_hash,
            checkpoint_path=checkpoint,
            checkpoint_state_dict_key=(
                None if config.checkpoint is None else config.checkpoint.state_dict_key
            ),
            checkpoint_strict=True if config.checkpoint is None else config.checkpoint.strict,
        )


class AdapterProviderRegistry:
    """Instance-local name registry with explicit provider registration."""

    def __init__(self) -> None:
        self._providers: dict[str, AdapterProvider] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._providers)

    def register(self, provider: AdapterProvider) -> None:
        if not isinstance(provider, AdapterProvider):
            raise TypeError("provider must be an AdapterProvider")
        name = provider.name
        if not name:
            raise AdapterProviderError("provider name must not be empty")
        if name in self._providers:
            raise AdapterProviderError(f"adapter provider already registered: {name}")
        if not issubclass(provider.config_model, ProviderConfig):
            raise AdapterProviderError("provider config_model must extend ProviderConfig")
        validate_config_literal(
            provider.config_model, "provider", name, error=AdapterProviderError
        )
        self._providers[name] = provider

    def parse(self, value: Any) -> ProviderConfig:
        if not isinstance(value, Mapping):
            raise AdapterProviderError("adapter configuration must be a mapping")
        raw = dict(value)
        name = raw.get("provider")
        if not isinstance(name, str) or not name:
            raise AdapterProviderError("adapter.provider must be a non-empty string")
        provider = self._providers.get(name)
        if provider is None:
            known = ", ".join(self._providers) or "none"
            raise AdapterProviderError(
                f"unknown adapter provider {name!r}; registered providers: {known}"
            )
        try:
            return provider.config_model.model_validate(raw)
        except Exception as error:
            raise AdapterProviderError(
                f"invalid {name!r} adapter configuration: {error}"
            ) from error

    def build(self, config: ProviderConfig, *, base_dir: str | Path) -> ModelAdapter:
        if not isinstance(config, ProviderConfig):
            raise TypeError("config must be a ProviderConfig")
        provider = self._providers.get(config.provider)
        if provider is None:
            raise AdapterProviderError(
                f"adapter provider is not registered: {config.provider}"
            )
        try:
            return provider.build(config, base_dir=Path(base_dir).resolve())
        except AdapterProviderError:
            raise
        except Exception as error:
            raise AdapterProviderError(
                f"failed to build adapter provider {config.provider!r}: {error}"
            ) from error


def default_adapter_provider_registry() -> AdapterProviderRegistry:
    """Return a fresh registry containing only v1 built-in providers."""

    registry = AdapterProviderRegistry()
    registry.register(TorchvisionProvider())
    registry.register(TorchvisionVideoProvider())
    registry.register(TimmProvider())
    registry.register(TorchvisionTSMProvider())
    return registry


def _resolve_checkpoint(
    checkpoint: CheckpointConfig | None,
    base_dir: Path,
) -> tuple[Path | None, str | None]:
    if checkpoint is None:
        return None, None
    path = checkpoint.path.expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve(strict=True)
    if not path.is_file():
        raise AdapterProviderError(f"checkpoint is not a file: {path}")
    return path, sha256_file(path)
