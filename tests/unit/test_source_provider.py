from pathlib import Path

import pytest

from ssat.core.source import (
    ImageFolderSource,
    ImageManifestProvider,
    ImageManifestSourceConfig,
    SourceProvider,
    SourceProviderConfig,
    SourceProviderError,
    SourceProviderRegistry,
    VideoFolderSource,
    VideoManifestProvider,
    default_source_provider_registry,
)

_IMAGE_MANIFEST = Path(__file__).parents[1] / "fixtures" / "synthetic_classification" / "manifest.json"
_VIDEO_MANIFEST = Path(__file__).parents[1] / "fixtures" / "synthetic_video" / "manifest.json"


def test_default_registry_is_fresh_and_contains_only_builtins() -> None:
    first = default_source_provider_registry()
    second = default_source_provider_registry()
    assert first is not second
    assert first.names == ("image_manifest", "video_manifest")


def test_register_rejects_duplicate_or_invalid_providers() -> None:
    registry = SourceProviderRegistry()
    registry.register(ImageManifestProvider())
    with pytest.raises(SourceProviderError, match="already registered"):
        registry.register(ImageManifestProvider())
    with pytest.raises(TypeError, match="must be a SourceProvider"):
        registry.register(object())  # type: ignore[arg-type]


def test_parse_defaults_kind_to_image_manifest_for_backward_compatibility() -> None:
    """Existing configs that omit ``source.kind`` must keep resolving to image_manifest."""

    registry = default_source_provider_registry()
    config = registry.parse({"manifest": str(_IMAGE_MANIFEST)})
    assert isinstance(config, ImageManifestSourceConfig)


def test_parse_unknown_kind_lists_registered_provider_names() -> None:
    registry = default_source_provider_registry()
    with pytest.raises(SourceProviderError, match="image_manifest, video_manifest"):
        registry.parse({"kind": "nope", "manifest": str(_IMAGE_MANIFEST)})


def test_default_registry_builds_image_manifest_source(tmp_path: Path) -> None:
    registry = default_source_provider_registry()
    config = registry.parse({"kind": "image_manifest", "manifest": str(_IMAGE_MANIFEST)})
    source, provenance = registry.build(config, base_dir=tmp_path)

    assert isinstance(source, ImageFolderSource)
    assert provenance.kind == "image_manifest"
    assert provenance.manifest == _IMAGE_MANIFEST.resolve()
    assert len(provenance.manifest_hash) == 64
    assert len(source.list_samples()) > 0


def test_default_registry_builds_video_manifest_source(tmp_path: Path) -> None:
    registry = default_source_provider_registry()
    config = registry.parse(
        {"kind": "video_manifest", "manifest": str(_VIDEO_MANIFEST), "num_frames": 4}
    )
    source, provenance = registry.build(config, base_dir=tmp_path)

    assert isinstance(source, VideoFolderSource)
    assert provenance.kind == "video_manifest"
    assert provenance.manifest == _VIDEO_MANIFEST.resolve()
    assert len(provenance.manifest_hash) == 64
    assert len(source.list_samples()) > 0


def test_custom_provider_requires_explicit_registration() -> None:
    class _EchoConfig(SourceProviderConfig):
        kind: str = "test_echo"

    class _EchoProvider(SourceProvider):
        name = "test_echo"
        config_model = _EchoConfig

        def build(self, config, *, base_dir):  # noqa: ANN001
            registry = default_source_provider_registry()
            image_config = registry.parse({"kind": "image_manifest", "manifest": str(_IMAGE_MANIFEST)})
            return registry.build(image_config, base_dir=base_dir)

    registry = SourceProviderRegistry()
    with pytest.raises(SourceProviderError, match="unknown source.kind"):
        registry.parse({"kind": "test_echo"})

    registry.register(_EchoProvider())
    config = registry.parse({"kind": "test_echo"})
    source, provenance = registry.build(config, base_dir=Path.cwd())
    assert isinstance(source, ImageFolderSource)
    assert provenance.kind == "image_manifest"


def test_build_rejects_unregistered_provider_config(tmp_path: Path) -> None:
    class _OrphanConfig(SourceProviderConfig):
        kind: str = "orphan"

    registry = SourceProviderRegistry()
    with pytest.raises(SourceProviderError, match="not registered"):
        registry.build(_OrphanConfig(), base_dir=tmp_path)
