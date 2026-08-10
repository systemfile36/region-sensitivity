"""Load one application configuration without depending on a UI framework."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict

from ssat.core.adapter import (
    AdapterProviderError,
    AdapterProviderRegistry,
    ProviderConfig,
)
from ssat.core.config import AuditConfig, SourceProvenance
from ssat.core.source import ImageFolderSource
from ssat.core.source.types import SampleMeta
from ssat.utils.io import load_json, load_yaml, sha256_file


class ApplicationConfigError(ValueError):
    """Indicate an invalid application configuration document."""


class ImageManifestSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["image_manifest"] = "image_manifest"
    manifest: Path


class _ManifestSample(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    sample_id: str
    path: Path
    gt_label: int | None = None


class _ImageManifest(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    samples: tuple[_ManifestSample, ...]


@dataclass(frozen=True, slots=True)
class LoadedApplicationConfig:
    audit: AuditConfig
    adapter_config: ProviderConfig
    sample_source: ImageFolderSource
    source_provenance: SourceProvenance
    base_dir: Path
    config_source: Path | None
    config_hash: str | None


def load_application_config(
    config: str | Path | Mapping[str, Any],
    registry: AdapterProviderRegistry,
    *,
    base_dir: str | Path | None = None,
) -> LoadedApplicationConfig:
    """Load top-level source/adapter sections and the existing audit schema."""

    try:
        raw, config_source, resolved_base, config_hash = _load_document(
            config,
            base_dir=base_dir,
        )
        source_raw = raw.pop("source", None)
        adapter_raw = raw.pop("adapter", None)
        if source_raw is None:
            raise ApplicationConfigError("configuration requires a source section")
        if adapter_raw is None:
            raise ApplicationConfigError("configuration requires an adapter section")
        source_config = ImageManifestSourceConfig.model_validate(source_raw)
        adapter_config = registry.parse(adapter_raw)
        audit = AuditConfig.model_validate(raw)
        source, provenance = _load_source(source_config, resolved_base)
        return LoadedApplicationConfig(
            audit=audit,
            adapter_config=adapter_config,
            sample_source=source,
            source_provenance=provenance,
            base_dir=resolved_base,
            config_source=config_source,
            config_hash=config_hash,
        )
    except (ApplicationConfigError, AdapterProviderError):
        raise
    except Exception as error:
        raise ApplicationConfigError(f"invalid application configuration: {error}") from error


def _load_document(
    config: str | Path | Mapping[str, Any],
    *,
    base_dir: str | Path | None,
) -> tuple[dict[str, Any], Path | None, Path, str | None]:
    if isinstance(config, (str, Path)):
        path = Path(config).expanduser().resolve(strict=True)
        value = load_yaml(path)
        if not isinstance(value, dict):
            raise ApplicationConfigError("configuration must contain a mapping")
        return dict(value), path, path.parent, sha256_file(path)
    if not isinstance(config, Mapping):
        raise ApplicationConfigError("config must be a path or mapping")
    resolved_base = Path(base_dir or Path.cwd()).expanduser().resolve()
    return dict(config), None, resolved_base, None


def _load_source(
    config: ImageManifestSourceConfig,
    base_dir: Path,
) -> tuple[ImageFolderSource, SourceProvenance]:
    manifest_path = config.manifest.expanduser()
    if not manifest_path.is_absolute():
        manifest_path = base_dir / manifest_path
    manifest_path = manifest_path.resolve(strict=True)
    if not manifest_path.is_file():
        raise ApplicationConfigError(f"source manifest is not a file: {manifest_path}")
    manifest = _ImageManifest.model_validate(load_json(manifest_path))
    if not manifest.samples:
        raise ApplicationConfigError("source manifest samples must not be empty")
    sample_ids = [sample.sample_id for sample in manifest.samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ApplicationConfigError("source manifest contains duplicate sample_id values")
    samples = []
    for sample in manifest.samples:
        path = sample.path.expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        samples.append(SampleMeta(sample.sample_id, path.resolve(), sample.gt_label))
    provenance = SourceProvenance(
        kind=config.kind,
        manifest=manifest_path,
        manifest_hash=sha256_file(manifest_path),
    )
    return ImageFolderSource(samples), provenance
