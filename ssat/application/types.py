"""UI-independent requests, results, events, and errors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from threading import Event
from typing import Any, Callable, Literal, Mapping, TypeAlias

from ssat.core.estimate import EstimateOptions, EstimateReport
from ssat.core.runtime import ExecutionSummary


ConfigValue: TypeAlias = str | Path | Mapping[str, Any]


class ApplicationErrorCode(str, Enum):
    CONFIG = "config_error"
    PROVIDER = "provider_error"
    OUTPUT = "output_error"
    CONFIRMATION_REQUIRED = "confirmation_required"
    STALE_PREFLIGHT = "stale_preflight"
    LOCKED = "output_locked"
    CANCELLED = "cancelled"
    EXECUTION = "execution_error"
    CORRUPTION = "dump_corruption"


class ApplicationError(RuntimeError):
    """Stable application-boundary failure suitable for any presentation layer."""

    def __init__(self, code: ApplicationErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ApplicationEvent:
    """One non-sensitive lifecycle or progress event."""

    kind: str
    phase: str
    completed: int | None = None
    total: int | None = None
    message: str | None = None


EventSink: TypeAlias = Callable[[ApplicationEvent], None]


class CancellationToken:
    """Thread-safe cooperative cancellation token."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True, slots=True)
class RunRequest:
    config: ConfigValue
    output: Path
    base_dir: Path | None = None
    estimate_options: EstimateOptions = field(default_factory=EstimateOptions)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", Path(self.output))


@dataclass(frozen=True, slots=True)
class EstimateRequest:
    config: ConfigValue
    dump: Path | None = None
    base_dir: Path | None = None
    options: EstimateOptions = field(default_factory=EstimateOptions)

    def __post_init__(self) -> None:
        if self.dump is not None:
            object.__setattr__(self, "dump", Path(self.dump))


@dataclass(frozen=True, slots=True)
class InspectRequest:
    dump: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "dump", Path(self.dump))


@dataclass(frozen=True, slots=True)
class RebuildIndexRequest:
    dump: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "dump", Path(self.dump))


@dataclass(frozen=True, slots=True)
class EstimateResult:
    report: EstimateReport
    dump_mode: Literal["create", "resume", "none"]

    def to_dict(self) -> dict[str, Any]:
        return {"dump_mode": self.dump_mode, "report": to_primitive(self.report)}


@dataclass(frozen=True, slots=True)
class RunResult:
    status: Literal["completed", "cancelled"]
    output: Path
    estimate: EstimateResult
    summary: ExecutionSummary | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output": str(self.output),
            "estimate": self.estimate.to_dict(),
            "summary": to_primitive(self.summary),
        }


@dataclass(frozen=True, slots=True)
class DumpSectionSummary:
    rows: int
    counts_by_status: dict[str, int]


@dataclass(frozen=True, slots=True)
class DumpSummary:
    dump: Path
    schema_version: str
    code_version: str
    model_id: str
    started_at: str
    finished_at: str | None
    resume_count: int
    clean: DumpSectionSummary
    perturbed: DumpSectionSummary
    total_counts_by_status: dict[str, int]
    manifest_counts_match: bool

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class IndexRebuildResult:
    dump: Path
    indexed_items: int
    summary: DumpSummary

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def to_primitive(value: Any) -> Any:
    """Recursively convert public dataclasses, enums, and paths to JSON values."""

    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return to_primitive(asdict(value))
    if isinstance(value, Mapping):
        return {str(to_primitive(key)): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_primitive(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"cannot serialize application value {type(value).__name__}")
