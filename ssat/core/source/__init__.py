"""Sample source contracts."""

from ssat.core.source.base import SampleSource
from ssat.core.source.image_folder import ImageFolderSource
from ssat.core.source.provider import (
    ImageManifestProvider,
    ImageManifestSourceConfig,
    SourceProvider,
    SourceProviderConfig,
    SourceProviderError,
    SourceProviderRegistry,
    VideoManifestProvider,
    VideoManifestSourceConfig,
    default_source_provider_registry,
)
from ssat.core.source.types import LoadError, LoadedSample, SampleMeta
from ssat.core.source.video_folder import VideoFolderSource

__all__ = [
    "ImageFolderSource",
    "ImageManifestProvider",
    "ImageManifestSourceConfig",
    "LoadError",
    "LoadedSample",
    "SampleMeta",
    "SampleSource",
    "SourceProvider",
    "SourceProviderConfig",
    "SourceProviderError",
    "SourceProviderRegistry",
    "VideoFolderSource",
    "VideoManifestProvider",
    "VideoManifestSourceConfig",
    "default_source_provider_registry",
]
