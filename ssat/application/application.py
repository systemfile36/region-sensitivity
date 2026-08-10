"""Reusable application service orchestrating the SSAT core."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ssat import __version__
from ssat.application.config import (
    ApplicationConfigError,
    LoadedApplicationConfig,
    load_application_config,
)
from ssat.application.locking import OutputLockedError, output_lock
from ssat.application.types import (
    ApplicationError,
    ApplicationErrorCode,
    ApplicationEvent,
    CancellationToken,
    DumpSectionSummary,
    DumpSummary,
    EstimateRequest,
    EstimateResult,
    EventSink,
    IndexRebuildResult,
    InspectRequest,
    RebuildIndexRequest,
    RunRequest,
    RunResult,
)
from ssat.core.adapter import (
    AdapterProviderError,
    AdapterProviderRegistry,
    ModelAdapter,
    default_adapter_provider_registry,
)
from ssat.core.config import ConfigResolver, ResolvedConfig
from ssat.core.dump import DumpReader, DumpWriter
from ssat.core.dump.manifest import load_manifest
from ssat.core.estimate import CostEstimator
from ssat.core.plan import PlanBuilder
from ssat.core.plan.hashing import canonical_json
from ssat.core.resume import ResumeIndex
from ssat.core.runtime import RuntimeCancelledError, run_audit
from ssat.core.source import ImageFolderSource
from ssat.core.types import ItemStatus
from ssat.utils.io import sha256_bytes, sha256_file
from ssat.utils.logger_factory import get_logger


CODE_VERSION = __version__


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    loaded: LoadedApplicationConfig
    adapter: ModelAdapter
    resolved: ResolvedConfig
    source: ImageFolderSource
    builder: PlanBuilder


class PreparedRun:
    """One-shot, process-local preflight session owned by an application."""

    def __init__(
        self,
        application: AuditApplication,
        request: RunRequest,
        context: _ExecutionContext,
        estimate: EstimateResult,
        mode: Literal["create", "resume"],
        fingerprint: str,
    ) -> None:
        self._application = application
        self.request = request
        self._context: _ExecutionContext | None = context
        self.estimate = estimate
        self.mode = mode
        self.fingerprint = fingerprint
        self._closed = False
        self._consumed = False

    @property
    def confirmation_required(self) -> bool:
        return self.estimate.report.confirmation_required

    @property
    def context(self) -> _ExecutionContext:
        if self._context is None:
            raise ApplicationError(
                ApplicationErrorCode.EXECUTION,
                "prepared run resources are closed",
            )
        return self._context

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        context = self._context
        self._context = None
        if context is not None:
            context.adapter.cleanup_after_oom()
        self._closed = True

    def __enter__(self) -> PreparedRun:
        if self._closed:
            raise ApplicationError(
                ApplicationErrorCode.EXECUTION,
                "prepared run is already closed",
            )
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False


class AuditApplication:
    """UI-independent facade for configuration, preflight, execution, and dumps."""

    def __init__(
        self,
        adapter_registry: AdapterProviderRegistry | None = None,
        *,
        code_version: str = CODE_VERSION,
    ) -> None:
        if not code_version:
            raise ValueError("code_version must not be empty")
        self._registry = adapter_registry or default_adapter_provider_registry()
        self._code_version = code_version
        self._logger = get_logger(__name__)

    def prepare_run(
        self,
        request: RunRequest,
        *,
        event_sink: EventSink | None = None,
    ) -> PreparedRun:
        """Resolve and profile an audit without creating a new dump."""

        if not isinstance(request, RunRequest):
            raise TypeError("request must be a RunRequest")
        self._emit(event_sink, "started", "config")
        context = self._build_context(request.config, request.base_dir)
        output = request.output.expanduser().resolve()
        mode = self._output_mode(output)
        resume = self._resume_for_context(output, context) if mode == "resume" else None
        self._emit(event_sink, "completed", "config")
        self._emit(event_sink, "started", "preflight")
        report = CostEstimator(request.estimate_options).estimate(
            context.resolved,
            context.builder,
            context.source,
            context.adapter,
            resume_index=resume,
        )
        result = EstimateResult(report=report, dump_mode=mode)
        fingerprint = self._preflight_fingerprint(context, output, mode)
        self._emit(
            event_sink,
            "completed",
            "preflight",
            completed=report.pending_perturbed_items,
            total=report.total_perturbed_items,
        )
        return PreparedRun(self, request, context, result, mode, fingerprint)

    def execute_run(
        self,
        prepared: PreparedRun,
        *,
        confirmation_granted: bool = False,
        event_sink: EventSink | None = None,
        cancellation: CancellationToken | None = None,
    ) -> RunResult:
        """Execute one prepared audit after confirmation and stale-state checks."""

        if not isinstance(prepared, PreparedRun) or prepared._application is not self:
            raise ApplicationError(
                ApplicationErrorCode.EXECUTION,
                "prepared run belongs to a different application",
            )
        if prepared._closed or prepared._consumed:
            raise ApplicationError(
                ApplicationErrorCode.EXECUTION,
                "prepared run is closed or already consumed",
            )
        if prepared.confirmation_required and not confirmation_granted:
            raise ApplicationError(
                ApplicationErrorCode.CONFIRMATION_REQUIRED,
                "audit preflight requires explicit confirmation",
            )
        output = prepared.request.output.expanduser().resolve()
        if cancellation is not None and cancellation.is_cancelled:
            prepared._consumed = True
            self._emit(event_sink, "cancelled", "run")
            return RunResult("cancelled", output, prepared.estimate, None)

        try:
            with output_lock(output):
                try:
                    current = self._preflight_fingerprint(
                        prepared.context,
                        output,
                        prepared.mode,
                    )
                except Exception as error:
                    raise ApplicationError(
                        ApplicationErrorCode.STALE_PREFLIGHT,
                        "configuration, source, adapter, or dump is no longer available",
                    ) from error
                if current != prepared.fingerprint:
                    raise ApplicationError(
                        ApplicationErrorCode.STALE_PREFLIGHT,
                        "configuration, source, or dump changed after preflight",
                    )
                self._emit(event_sink, "started", "run")
                prepared._consumed = True
                writer = DumpWriter(
                    output,
                    prepared.context.resolved,
                    code_version=self._code_version,
                    mode=prepared.mode,
                )
                try:
                    resume = ResumeIndex.open(output)
                    summary = run_audit(
                        prepared.context.resolved,
                        prepared.context.builder,
                        prepared.context.source,
                        prepared.context.adapter,
                        writer,
                        resume,
                        progress_callback=lambda completed, total: self._emit(
                            event_sink,
                            "progress",
                            "run",
                            completed=completed,
                            total=total,
                        ),
                        cancel_requested=(
                            None
                            if cancellation is None
                            else lambda: cancellation.is_cancelled
                        ),
                    )
                except RuntimeCancelledError:
                    writer.close(success=False)
                    self._emit(event_sink, "cancelled", "run")
                    return RunResult("cancelled", output, prepared.estimate, None)
                except Exception:
                    writer.close(success=False)
                    raise
                else:
                    writer.close(success=True)
        except ApplicationError:
            raise
        except OutputLockedError as error:
            raise ApplicationError(ApplicationErrorCode.LOCKED, str(error)) from error
        except Exception as error:
            raise ApplicationError(
                ApplicationErrorCode.EXECUTION,
                f"audit execution failed: {error}",
            ) from error

        self._emit(
            event_sink,
            "completed",
            "run",
            completed=summary.records_written,
            total=summary.records_written,
        )
        return RunResult("completed", output, prepared.estimate, summary)

    def estimate(
        self,
        request: EstimateRequest,
        *,
        event_sink: EventSink | None = None,
    ) -> EstimateResult:
        """Return a structured standalone estimate without prompting."""

        if not isinstance(request, EstimateRequest):
            raise TypeError("request must be an EstimateRequest")
        self._emit(event_sink, "started", "config")
        context = self._build_context(request.config, request.base_dir)
        mode: Literal["create", "resume", "none"] = "none"
        resume = None
        if request.dump is not None:
            dump = request.dump.expanduser().resolve()
            mode = self._output_mode(dump)
            if mode != "resume":
                raise ApplicationError(
                    ApplicationErrorCode.OUTPUT,
                    "estimate --dump requires an existing valid dump",
                )
            resume = self._resume_for_context(dump, context)
        self._emit(event_sink, "completed", "config")
        self._emit(event_sink, "started", "preflight")
        report = CostEstimator(request.options).estimate(
            context.resolved,
            context.builder,
            context.source,
            context.adapter,
            resume_index=resume,
        )
        self._emit(event_sink, "completed", "preflight")
        return EstimateResult(report, mode)

    def inspect(self, request: InspectRequest) -> DumpSummary:
        """Summarize authoritative dump rows and manifest provenance."""

        try:
            dump = request.dump.expanduser().resolve(strict=True)
            reader = DumpReader(dump)
            manifest = reader.read_manifest()
            clean = reader.read_clean()
            perturbed = reader.read_perturbed()
            clean_observed = Counter(row["status"] for row in clean.to_pylist())
            perturbed_observed = Counter(row["status"] for row in perturbed.to_pylist())
            clean_counts = {
                status.value: clean_observed.get(status.value, 0) for status in ItemStatus
            }
            perturbed_counts = {
                status.value: perturbed_observed.get(status.value, 0)
                for status in ItemStatus
            }
            total_counts = {
                status.value: clean_counts[status.value] + perturbed_counts[status.value]
                for status in ItemStatus
            }
            manifest_counts = {
                status.value: count for status, count in manifest.counts_by_status.items()
            }
            return DumpSummary(
                dump=dump,
                schema_version=manifest.schema_version,
                code_version=manifest.code_version,
                model_id=manifest.adapter_spec.model_id,
                started_at=manifest.started_at.isoformat(),
                finished_at=(
                    None if manifest.finished_at is None else manifest.finished_at.isoformat()
                ),
                resume_count=len(manifest.resume_events),
                clean=DumpSectionSummary(clean.num_rows, clean_counts),
                perturbed=DumpSectionSummary(
                    perturbed.num_rows,
                    perturbed_counts,
                ),
                total_counts_by_status=total_counts,
                manifest_counts_match=manifest_counts == total_counts,
            )
        except Exception as error:
            raise ApplicationError(
                ApplicationErrorCode.CORRUPTION,
                f"cannot inspect dump: {error}",
            ) from error

    def rebuild_index(self, request: RebuildIndexRequest) -> IndexRebuildResult:
        """Rebuild the perturbed index and return the resulting summary."""

        dump = request.dump.expanduser().resolve()
        try:
            with output_lock(dump):
                ResumeIndex.rebuild(dump)
            summary = self.inspect(InspectRequest(dump))
            return IndexRebuildResult(dump, summary.perturbed.rows, summary)
        except OutputLockedError as error:
            raise ApplicationError(ApplicationErrorCode.LOCKED, str(error)) from error
        except ApplicationError:
            raise
        except Exception as error:
            raise ApplicationError(
                ApplicationErrorCode.CORRUPTION,
                f"cannot rebuild dump index: {error}",
            ) from error

    def _build_context(
        self,
        config: object,
        base_dir: Path | None,
    ) -> _ExecutionContext:
        try:
            loaded = load_application_config(
                config,  # type: ignore[arg-type]
                self._registry,
                base_dir=base_dir,
            )
            adapter = self._registry.build(
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
            return _ExecutionContext(
                loaded,
                adapter,
                resolved,
                loaded.sample_source,
                PlanBuilder(resolved, loaded.sample_source),
            )
        except ApplicationConfigError as error:
            raise ApplicationError(ApplicationErrorCode.CONFIG, str(error)) from error
        except AdapterProviderError as error:
            raise ApplicationError(ApplicationErrorCode.PROVIDER, str(error)) from error
        except Exception as error:
            raise ApplicationError(
                ApplicationErrorCode.CONFIG,
                f"failed to resolve audit configuration: {error}",
            ) from error

    def _resume_for_context(
        self,
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
            if manifest.code_version != self._code_version:
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

    @staticmethod
    def _output_mode(output: Path) -> Literal["create", "resume"]:
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

    def _preflight_fingerprint(
        self,
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
        current_mode = self._output_mode(output)
        payload = {
            "expected_mode": expected_mode,
            "current_mode": current_mode,
            "source_hash": loaded.source_provenance.manifest_hash,
            "config_hash": loaded.config_hash,
            "resolved_config": context.resolved,
            "adapter_spec": context.adapter.describe(),
            "output": self._output_snapshot(output, current_mode),
        }
        return sha256_bytes(canonical_json(payload).encode("utf-8"))

    @staticmethod
    def _output_snapshot(
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

    def _emit(
        self,
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
            self._logger.warning("application.event_sink_failed phase=%s", phase)
