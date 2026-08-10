"""Name-based adapter configuration and provider registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

    @model_validator(mode="after")
    def validate_weights(self) -> TorchvisionProviderConfig:
        if self.weights is not None and self.checkpoint is not None:
            raise ValueError("weights and checkpoint are mutually exclusive")
        if self.checkpoint is not None and self.weights_hash is not None:
            raise ValueError("checkpoint hash is computed and cannot be supplied")
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

    @model_validator(mode="after")
    def validate_weights(self) -> TimmProviderConfig:
        if self.pretrained and self.checkpoint is not None:
            raise ValueError("pretrained and checkpoint are mutually exclusive")
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
    registry.register(TimmProvider())
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
