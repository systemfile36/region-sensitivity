"""Name-based sample source configuration and provider registry.

Mirrors :mod:`ssat.core.adapter.provider` on the source side: each concrete
``source.kind`` is registered as one :class:`SourceProvider` in a
:class:`SourceProviderRegistry`, and callers (currently only
``ssat.application.config``) resolve a raw ``source`` mapping through the
registry instead of a fixed dispatch table. Every provider must still return
a file-backed :class:`~ssat.core.config.schema.SourceProvenance` -- this
registry only opens *how* a ``SampleSource`` is built, not the reproducibility
contract that its provenance must satisfy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from ssat.core.config.schema import SourceProvenance
from ssat.core.source.base import SampleSource
from ssat.core.source.image_folder import ImageFolderSource
from ssat.core.source.types import SampleMeta
from ssat.core.source.video_folder import VideoFolderSource
from ssat.utils.io import load_json, sha256_file


class SourceProviderError(ValueError):
    """Indicate invalid provider registration, configuration, or construction."""


class SourceProviderConfig(BaseModel):
    """Strict base for provider-specific source configuration.

    Uses ``kind`` rather than ``provider`` as the discriminator field name
    to preserve the existing ``source.kind: ...`` YAML key that configs
    already use.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: str


class ImageManifestSourceConfig(SourceProviderConfig):
    """Configure an image source backed by an explicit sample manifest."""

    kind: Literal["image_manifest"] = "image_manifest"
    manifest: Path


class VideoManifestSourceConfig(SourceProviderConfig):
    """Configure a decord-backed video source with uniform frame sampling."""

    kind: Literal["video_manifest"] = "video_manifest"
    manifest: Path
    num_frames: int = Field(default=16, gt=0)


class _ManifestSample(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    sample_id: str
    path: Path
    gt_label: int | None = None


class _SampleManifest(BaseModel):
    """Shared manifest document shape for both image and video sources."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    samples: tuple[_ManifestSample, ...]


def _load_manifest_samples(
    manifest: Path,
    base_dir: Path,
) -> tuple[Path, list[SampleMeta]]:
    """Load and resolve every ``SampleMeta`` referenced by a manifest file."""

    manifest_path = manifest.expanduser()
    if not manifest_path.is_absolute():
        manifest_path = base_dir / manifest_path
    manifest_path = manifest_path.resolve(strict=True)
    if not manifest_path.is_file():
        raise SourceProviderError(f"source manifest is not a file: {manifest_path}")
    document = _SampleManifest.model_validate(load_json(manifest_path))
    if not document.samples:
        raise SourceProviderError("source manifest samples must not be empty")
    sample_ids = [sample.sample_id for sample in document.samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise SourceProviderError("source manifest contains duplicate sample_id values")
    samples = []
    for sample in document.samples:
        path = sample.path.expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        samples.append(SampleMeta(sample.sample_id, path.resolve(), sample.gt_label))
    return manifest_path, samples


class SourceProvider(ABC):
    """Build one sample source and its provenance from one registered config."""

    name: ClassVar[str]
    config_model: ClassVar[type[SourceProviderConfig]]

    @abstractmethod
    def build(
        self, config: SourceProviderConfig, *, base_dir: Path
    ) -> tuple[SampleSource, SourceProvenance]:
        """Build one sample source and a file-backed provenance record."""


class ImageManifestProvider(SourceProvider):
    """Build an :class:`ImageFolderSource` from an explicit manifest."""

    name = "image_manifest"
    config_model = ImageManifestSourceConfig

    def build(
        self, config: SourceProviderConfig, *, base_dir: Path
    ) -> tuple[SampleSource, SourceProvenance]:
        if not isinstance(config, ImageManifestSourceConfig):
            raise TypeError("config must be ImageManifestSourceConfig")
        manifest_path, samples = _load_manifest_samples(config.manifest, base_dir)
        provenance = SourceProvenance(
            kind=config.kind,
            manifest=manifest_path,
            manifest_hash=sha256_file(manifest_path),
        )
        return ImageFolderSource(samples), provenance


class VideoManifestProvider(SourceProvider):
    """Build a :class:`VideoFolderSource` from an explicit manifest."""

    name = "video_manifest"
    config_model = VideoManifestSourceConfig

    def build(
        self, config: SourceProviderConfig, *, base_dir: Path
    ) -> tuple[SampleSource, SourceProvenance]:
        if not isinstance(config, VideoManifestSourceConfig):
            raise TypeError("config must be VideoManifestSourceConfig")
        manifest_path, samples = _load_manifest_samples(config.manifest, base_dir)
        provenance = SourceProvenance(
            kind=config.kind,
            manifest=manifest_path,
            manifest_hash=sha256_file(manifest_path),
        )
        return VideoFolderSource(samples, num_frames=config.num_frames), provenance


class SourceProviderRegistry:
    """Instance-local name registry with explicit provider registration."""

    def __init__(self) -> None:
        self._providers: dict[str, SourceProvider] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._providers)

    def register(self, provider: SourceProvider) -> None:
        if not isinstance(provider, SourceProvider):
            raise TypeError("provider must be a SourceProvider")
        name = provider.name
        if not name:
            raise SourceProviderError("provider name must not be empty")
        if name in self._providers:
            raise SourceProviderError(f"source provider already registered: {name}")
        if not issubclass(provider.config_model, SourceProviderConfig):
            raise SourceProviderError("provider config_model must extend SourceProviderConfig")
        self._providers[name] = provider

    def parse(self, value: Any) -> SourceProviderConfig:
        if not isinstance(value, Mapping):
            raise SourceProviderError("source configuration must be a mapping")
        raw = dict(value)
        kind = raw.get("kind", "image_manifest")
        if not isinstance(kind, str) or not kind:
            raise SourceProviderError("source.kind must be a non-empty string")
        provider = self._providers.get(kind)
        if provider is None:
            known = ", ".join(self._providers) or "none"
            raise SourceProviderError(
                f"unknown source.kind {kind!r}; registered providers: {known}"
            )
        try:
            return provider.config_model.model_validate(raw)
        except Exception as error:
            raise SourceProviderError(f"invalid {kind!r} source configuration: {error}") from error

    def build(
        self, config: SourceProviderConfig, *, base_dir: str | Path
    ) -> tuple[SampleSource, SourceProvenance]:
        if not isinstance(config, SourceProviderConfig):
            raise TypeError("config must be a SourceProviderConfig")
        provider = self._providers.get(config.kind)
        if provider is None:
            raise SourceProviderError(f"source provider is not registered: {config.kind}")
        try:
            return provider.build(config, base_dir=Path(base_dir).resolve())
        except SourceProviderError:
            raise
        except Exception as error:
            raise SourceProviderError(
                f"failed to build source provider {config.kind!r}: {error}"
            ) from error


def default_source_provider_registry() -> SourceProviderRegistry:
    """Return a fresh registry containing only v1 built-in providers."""

    # Imported here rather than at module scope: both modules import
    # SourceProvider/SourceProviderConfig/SourceProviderError from this
    # module, so a top-level import back into this module would be circular.
    from ssat.core.source.imagenet import ImageNetSourceProvider
    from ssat.core.source.kinetics import KineticsSourceProvider

    registry = SourceProviderRegistry()
    registry.register(ImageManifestProvider())
    registry.register(VideoManifestProvider())
    registry.register(ImageNetSourceProvider())
    registry.register(KineticsSourceProvider())
    return registry
