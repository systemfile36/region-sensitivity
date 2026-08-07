"""Validated records and manifest models for raw audit dumps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ssat.core.adapter.types import AdapterSpec, RawOutput
from ssat.core.config.schema import DatasetStats, ResolvedConfig
from ssat.core.plan.types import WorkItem
from ssat.core.region.types import RegionMeta
from ssat.core.types import ItemStatus, SCHEMA_VERSION


class DumpModel(BaseModel):
    """Base model for immutable, strict dump metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EnvironmentSpec(DumpModel):
    """Record the execution environment without requiring a model framework."""

    python_version: str
    platform: str
    torch_version: str | None = None
    cuda_version: str | None = None
    gpu_names: tuple[str, ...] = ()

    @field_validator("python_version", "platform")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value:
            raise ValueError("environment text fields must not be empty")
        return value


class ResumeEvent(DumpModel):
    """Record one explicitly resumed execution session."""

    resumed_at: datetime
    environment: EnvironmentSpec

    @field_validator("resumed_at")
    @classmethod
    def validate_resumed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("resume timestamps must be timezone-aware")
        return value


class RunManifest(DumpModel):
    """Describe one resumable audit run and its authoritative dump contents."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    resolved_config: ResolvedConfig
    code_version: str
    adapter_spec: AdapterSpec
    dataset_stats: DatasetStats | None = None
    environment: EnvironmentSpec
    started_at: datetime
    finished_at: datetime | None = None
    counts_by_status: dict[ItemStatus, int] = Field(
        default_factory=lambda: _empty_counts()
    )
    resume_events: tuple[ResumeEvent, ...] = ()

    @field_validator("started_at", "finished_at")
    @classmethod
    def validate_utc_datetime(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("manifest timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_provenance(self) -> RunManifest:
        """Ensure duplicated convenience fields cannot diverge from config."""

        if not self.code_version:
            raise ValueError("code_version must not be empty")
        if self.adapter_spec != self.resolved_config.adapter_spec:
            raise ValueError("adapter_spec must match resolved_config.adapter_spec")
        if self.dataset_stats != self.resolved_config.dataset_stats:
            raise ValueError("dataset_stats must match resolved_config.dataset_stats")
        if set(self.counts_by_status) != set(ItemStatus):
            raise ValueError("counts_by_status must contain every ItemStatus")
        if any(count < 0 for count in self.counts_by_status.values()):
            raise ValueError("counts_by_status values must be non-negative")
        return self


@dataclass(frozen=True, slots=True)
class CleanDumpRecord:
    """Carry one validated clean-sample result before persistence."""

    sample_id: str
    status: ItemStatus
    model_id: str
    logits: RawOutput | None = None
    content_hash: str | None = None
    gt_label: int | None = None
    original_shape: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if not self.sample_id or not self.model_id:
            raise ValueError("sample_id and model_id must not be empty")
        _validate_status_and_logits(self.status, self.logits)
        if self.content_hash is not None:
            _validate_sha256(self.content_hash, field_name="content_hash")
        if self.original_shape is not None and (
            len(self.original_shape) != 4
            or any(dimension <= 0 for dimension in self.original_shape)
        ):
            raise ValueError("original_shape must contain four positive dimensions")
        if self.status is ItemStatus.OK:
            if self.content_hash is None or self.original_shape is None:
                raise ValueError(
                    "successful clean records require content_hash and original_shape"
                )


@dataclass(frozen=True, slots=True)
class PerturbedDumpRecord:
    """Carry one WorkItem result and its materialized region measurements."""

    work_item: WorkItem
    status: ItemStatus
    seed_used: int
    logits: RawOutput | None = None
    region_meta: RegionMeta | None = None
    effective_area_px: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.work_item, WorkItem):
            raise TypeError("work_item must be a WorkItem")
        _validate_status_and_logits(self.status, self.logits)
        if (
            isinstance(self.seed_used, bool)
            or not isinstance(self.seed_used, int)
            or not 0 <= self.seed_used < 2**128
        ):
            raise ValueError("seed_used must be an unsigned 128-bit integer")
        if self.effective_area_px is not None and self.effective_area_px < 0:
            raise ValueError("effective_area_px must be non-negative")
        if self.status is ItemStatus.OK and self.region_meta is None:
            raise ValueError("successful perturbed records require region_meta")


def seed_to_hex(seed: int) -> str:
    """Return the fixed-width representation persisted by the v1 schema."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**128:
        raise ValueError("seed must be an unsigned 128-bit integer")
    return f"{seed:032x}"


def _validate_status_and_logits(
    status: ItemStatus,
    logits: RawOutput | None,
) -> None:
    if not isinstance(status, ItemStatus):
        raise TypeError("status must be an ItemStatus")
    if logits is not None and not isinstance(logits, RawOutput):
        raise TypeError("logits must be a RawOutput or None")
    if status is ItemStatus.OK and logits is None:
        raise ValueError("status=ok requires logits")
    if status is not ItemStatus.OK and logits is not None:
        raise ValueError("failure statuses require logits=None")


def _validate_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


def _empty_counts() -> dict[ItemStatus, int]:
    return {status: 0 for status in ItemStatus}
