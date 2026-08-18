"""Load one application configuration without depending on a UI framework."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ssat.core.adapter import (
    AdapterProviderError,
    AdapterProviderRegistry,
    ProviderConfig,
)
from ssat.core.config import AuditConfig, SourceProvenance
from ssat.core.source import (
    SampleSource,
    SourceProviderError,
    SourceProviderRegistry,
    default_source_provider_registry,
)
from ssat.utils.io import load_yaml, sha256_file


class ApplicationConfigError(ValueError):
    """Indicate an invalid application configuration document."""


@dataclass(frozen=True, slots=True)
class LoadedApplicationConfig:
    audit: AuditConfig
    adapter_config: ProviderConfig
    sample_source: SampleSource
    source_provenance: SourceProvenance
    base_dir: Path
    config_source: Path | None
    config_hash: str | None


def load_application_config(
    config: str | Path | Mapping[str, Any],
    registry: AdapterProviderRegistry,
    *,
    source_registry: SourceProviderRegistry | None = None,
    base_dir: str | Path | None = None,
) -> LoadedApplicationConfig:
    """Load top-level source/adapter sections and the existing audit schema."""

    resolved_source_registry = source_registry or default_source_provider_registry()
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
        source_config = resolved_source_registry.parse(source_raw)
        adapter_config = registry.parse(adapter_raw)
        audit = AuditConfig.model_validate(raw)
        source, provenance = resolved_source_registry.build(source_config, base_dir=resolved_base)
        return LoadedApplicationConfig(
            audit=audit,
            adapter_config=adapter_config,
            sample_source=source,
            source_provenance=provenance,
            base_dir=resolved_base,
            config_source=config_source,
            config_hash=config_hash,
        )
    except (ApplicationConfigError, AdapterProviderError, SourceProviderError):
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
