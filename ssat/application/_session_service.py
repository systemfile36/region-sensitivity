"""Shared session-building plumbing behind ``prepare_run``/``execute_run``/``estimate``.

Extracted from ``AuditApplication`` as part of a facade-delegation refactor:
every function here takes the owning ``AuditApplication`` instance as an
explicit ``app`` parameter, mirroring what was previously an implicit
``self``. ``AuditApplication``'s public methods remain the only public
surface of the application layer; this module is a private implementation
detail shared between ``_run_service``/``_estimate_service``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ssat.application.config import (
    ApplicationConfigError,
    LoadedApplicationConfig,
    load_application_config,
)
from ssat.application.types import (
    ApplicationError,
    ApplicationErrorCode,
    ApplicationEvent,
    EventSink,
)
from ssat.core.adapter import AdapterProviderError, ModelAdapter
from ssat.core.config import ConfigResolver, ResolvedConfig, ResolvedSkeletonSourceConfig
from ssat.core.dump.manifest import load_manifest
from ssat.core.plan import PlanBuilder, RegionExpander
from ssat.core.plan.hashing import canonical_json
from ssat.core.region import RegionResolver
from ssat.core.region.skeleton_provider import SkeletonRegionProvider
from ssat.core.region.skeleton_store import (
    SkeletonBBoxStore,
    SkeletonDataError,
    load_skeleton_bbox_store,
)
from ssat.core.resume import ResumeIndex
from ssat.core.source import SampleSource, SourceProviderError
from ssat.utils.io import sha256_bytes, sha256_file

if TYPE_CHECKING:
    from ssat.application.application import AuditApplication


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    loaded: LoadedApplicationConfig
    adapter: ModelAdapter
    resolved: ResolvedConfig
    source: SampleSource
    builder: PlanBuilder
    region_resolver: RegionResolver


def build_context(
    app: AuditApplication,
    config: object,
    base_dir: Path | None,
) -> _ExecutionContext:
    try:
        loaded = load_application_config(
            config,  # type: ignore[arg-type]
            app._registry,
            source_registry=app._source_registry,
            base_dir=base_dir,
        )
        adapter = app._registry.build(
            loaded.adapter_config,
            base_dir=loaded.base_dir,
        )
        resolved = ConfigResolver().resolve(
            loaded.audit,
            adapter,
            loaded.sample_source,
            base_dir=loaded.base_dir,
            config_source=loaded.config_source,
            source_provenance=loaded.source_provenance,
        )
        skeleton_store = load_skeleton_store(resolved.skeleton_source)
        region_expander = RegionExpander(
            SkeletonRegionProvider(skeleton_store)
            if skeleton_store is not None
            else None
        )
        return _ExecutionContext(
            loaded,
            adapter,
            resolved,
            loaded.sample_source,
            PlanBuilder(resolved, loaded.sample_source, region_expander=region_expander),
            RegionResolver(skeleton_store=skeleton_store),
        )
    except ApplicationConfigError as error:
        raise ApplicationError(ApplicationErrorCode.CONFIG, str(error)) from error
    except (AdapterProviderError, SourceProviderError) as error:
        raise ApplicationError(ApplicationErrorCode.PROVIDER, str(error)) from error
    except Exception as error:
        raise ApplicationError(
            ApplicationErrorCode.CONFIG,
            f"failed to resolve audit configuration: {error}",
        ) from error


def load_skeleton_store(
    skeleton_source: ResolvedSkeletonSourceConfig | None,
) -> SkeletonBBoxStore | None:
    """Load the skeleton bbox store referenced by a resolved config.

    Args:
        skeleton_source: Resolved, hash-verified reference, or ``None``.

    Returns:
        The loaded store, or ``None`` if no ``skeleton_source`` was
        configured.

    Raises:
        ApplicationConfigError: If the referenced file is unreadable or
            fails re-verification against its resolved hash.
    """

    if skeleton_source is None:
        return None
    try:
        return load_skeleton_bbox_store(
            skeleton_source.bbox_data,
            expected_hash=skeleton_source.bbox_data_hash,
        )
    except SkeletonDataError as error:
        raise ApplicationConfigError(
            f"failed to load skeleton_source data: {error}"
        ) from error


def resume_for_context(
    app: AuditApplication,
    output: Path,
    context: _ExecutionContext,
) -> ResumeIndex:
    try:
        manifest = load_manifest(output / "run_manifest.json")
        if manifest.resolved_config != context.resolved:
            raise ApplicationError(
                ApplicationErrorCode.OUTPUT,
                "existing dump resolved_config does not match the request",
            )
        if manifest.adapter_spec != context.adapter.describe():
            raise ApplicationError(
                ApplicationErrorCode.OUTPUT,
                "existing dump adapter does not match the request",
            )
        if manifest.code_version != app._code_version:
            raise ApplicationError(
                ApplicationErrorCode.OUTPUT,
                "existing dump code_version does not match this application",
            )
        return ResumeIndex.open(output)
    except ApplicationError:
        raise
    except Exception as error:
        raise ApplicationError(
            ApplicationErrorCode.OUTPUT,
            f"cannot resume existing dump: {error}",
        ) from error


def output_mode(output: Path) -> Literal["create", "resume"]:
    if not output.exists():
        return "create"
    if not output.is_dir():
        raise ApplicationError(
            ApplicationErrorCode.OUTPUT,
            f"output is not a directory: {output}",
        )
    if not any(output.iterdir()):
        return "create"
    if (output / "run_manifest.json").is_file():
        try:
            load_manifest(output / "run_manifest.json")
        except Exception as error:
            raise ApplicationError(
                ApplicationErrorCode.OUTPUT,
                f"output is not a valid dump: {error}",
            ) from error
        return "resume"
    raise ApplicationError(
        ApplicationErrorCode.OUTPUT,
        f"non-empty output is not an SSAT dump: {output}",
    )


def preflight_fingerprint(
    context: _ExecutionContext,
    output: Path,
    expected_mode: Literal["create", "resume"],
) -> str:
    loaded = context.loaded
    if sha256_file(loaded.source_provenance.manifest) != loaded.source_provenance.manifest_hash:
        return "source-manifest-changed"
    if loaded.config_source is not None and loaded.config_hash is not None:
        if sha256_file(loaded.config_source) != loaded.config_hash:
            return "configuration-changed"
    current_mode = output_mode(output)
    payload = {
        "expected_mode": expected_mode,
        "current_mode": current_mode,
        "source_hash": loaded.source_provenance.manifest_hash,
        "config_hash": loaded.config_hash,
        "resolved_config": context.resolved,
        "adapter_spec": context.adapter.describe(),
        "output": output_snapshot(output, current_mode),
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def output_snapshot(
    output: Path,
    mode: Literal["create", "resume"],
) -> list[tuple[str, int, int]]:
    if mode == "create":
        return []
    rows = []
    for directory in ("clean", "perturbed", "index"):
        for path in sorted((output / directory).glob("*.parquet")):
            stat = path.stat()
            rows.append((path.relative_to(output).as_posix(), stat.st_size, stat.st_mtime_ns))
    manifest = output / "run_manifest.json"
    stat = manifest.stat()
    rows.append(("run_manifest.json", stat.st_size, stat.st_mtime_ns))
    return rows


def emit(
    app: AuditApplication,
    sink: EventSink | None,
    kind: str,
    phase: str,
    *,
    completed: int | None = None,
    total: int | None = None,
) -> None:
    if sink is None:
        return
    try:
        sink(ApplicationEvent(kind, phase, completed, total))
    except Exception:
        app._logger.warning("application.event_sink_failed phase=%s", phase)
