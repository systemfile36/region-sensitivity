"""Run-manifest creation, validation, and atomic persistence."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import platform
from pathlib import Path
import sys
from typing import Any

from pydantic import ValidationError

from ssat.core.config.schema import ResolvedConfig
from ssat.core.dump._storage import fsync_directory
from ssat.core.dump.errors import DumpCorruptionError, DumpSchemaError
from ssat.core.dump.types import EnvironmentSpec, RunManifest
from ssat.core.types import ItemStatus, SCHEMA_VERSION
from ssat.utils.io import load_json, write_json_atomic


def capture_environment() -> EnvironmentSpec:
    """Capture stdlib details and inspect torch only when already imported."""

    torch = sys.modules.get("torch")
    torch_version: str | None = None
    cuda_version: str | None = None
    gpu_names: tuple[str, ...] = ()
    if torch is not None:
        torch_version = str(getattr(torch, "__version__", "unknown"))
        version_module = getattr(torch, "version", None)
        cuda_value = getattr(version_module, "cuda", None)
        cuda_version = None if cuda_value is None else str(cuda_value)
        cuda_module = getattr(torch, "cuda", None)
        try:
            if cuda_module is not None and cuda_module.is_available():
                gpu_names = tuple(
                    str(cuda_module.get_device_name(index))
                    for index in range(cuda_module.device_count())
                )
        except Exception:
            gpu_names = ()
    return EnvironmentSpec(
        python_version=platform.python_version(),
        platform=platform.platform(),
        torch_version=torch_version,
        cuda_version=cuda_version,
        gpu_names=gpu_names,
    )


def normalize_environment(
    value: EnvironmentSpec | Mapping[str, Any] | None,
) -> EnvironmentSpec:
    """Resolve an optional explicitly supplied environment description."""

    if value is None:
        return capture_environment()
    if isinstance(value, EnvironmentSpec):
        return value
    return EnvironmentSpec.model_validate(value)


def new_manifest(
    resolved_config: ResolvedConfig,
    *,
    code_version: str,
    environment: EnvironmentSpec,
    now: datetime | None = None,
) -> RunManifest:
    """Create the initial manifest for a new dump directory."""

    started_at = now or datetime.now(timezone.utc)
    counts = {status: 0 for status in ItemStatus}
    return RunManifest(
        resolved_config=resolved_config,
        code_version=code_version,
        adapter_spec=resolved_config.adapter_spec,
        dataset_stats=resolved_config.dataset_stats,
        environment=environment,
        started_at=started_at,
        counts_by_status=counts,
    )


def load_manifest(path: Path, *, expected_version: str = SCHEMA_VERSION) -> RunManifest:
    """Load a strictly versioned manifest with stable public errors."""

    if not path.is_file():
        raise DumpCorruptionError(f"run manifest does not exist: {path}")
    try:
        payload = load_json(path)
    except Exception as error:
        raise DumpCorruptionError(f"cannot read run manifest: {path}") from error
    if not isinstance(payload, dict):
        raise DumpCorruptionError("run manifest must contain a JSON object")
    version = payload.get("schema_version")
    if version != expected_version:
        raise DumpSchemaError(
            f"manifest schema version mismatch: expected {expected_version}, found {version}"
        )
    try:
        return RunManifest.model_validate(payload)
    except ValidationError as error:
        raise DumpCorruptionError("run manifest does not match its schema") from error


def write_manifest(path: Path, manifest: RunManifest) -> None:
    """Atomically publish one JSON manifest and persist its directory entry."""

    write_json_atomic(path, manifest.model_dump(mode="json"))
    fsync_directory(path.parent)
