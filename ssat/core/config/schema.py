"""Pydantic v2 models for the audit-focused user configuration."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ssat.core.adapter.types import AdapterSpec
from ssat.core.types import (
    PerturbationOp,
    RegionKind,
    SCHEMA_VERSION,
    validate_json_value,
)

RegionId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]


class FrozenModel(BaseModel):
    """Base class for immutable configuration models with strict fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RegionConfig(FrozenModel):
    """Describe one user-configured region before path and hash resolution.

    Attributes:
        region_id: Stable name used by controls and dump records.
        kind: Strategy used to materialize the region mask.
        params: Strategy-specific JSON parameters.
        ref: Optional path to an explicit mask.
        ref_hash: Optional precomputed hash for the referenced mask.
    """

    region_id: RegionId
    kind: RegionKind
    params: dict[str, Any] = Field(default_factory=dict)
    ref: Path | None = None
    ref_hash: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")

    @field_validator("params")
    @classmethod
    def validate_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Ensure region parameters are canonical JSON-compatible values."""

        validate_json_value(value)
        return value

    @model_validator(mode="after")
    def validate_reference(self) -> RegionConfig:
        """Enforce reference fields according to the region kind."""

        if self.kind is RegionKind.EXPLICIT:
            if self.ref is None:
                raise ValueError("explicit regions require ref")
        elif self.ref is not None or self.ref_hash is not None:
            raise ValueError("procedural regions cannot define ref or ref_hash")
        return self


class PerturbationConfig(FrozenModel):
    """Configure one perturbation family expanded into work items.

    Attributes:
        op: Perturbation operation to apply inside the resolved mask.
        params: Operation-specific JSON parameters.
        invert_mask: Whether to perturb the complement of the region.
        seed_salts: Deterministic salts used to enumerate stochastic variants.
    """

    op: PerturbationOp
    params: dict[str, Any] = Field(default_factory=dict)
    invert_mask: bool = False
    seed_salts: tuple[NonNegativeInt, ...] = (0,)

    @field_validator("params")
    @classmethod
    def validate_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Ensure perturbation parameters are canonical JSON-compatible values."""

        validate_json_value(value)
        return value

    @field_validator("seed_salts")
    @classmethod
    def validate_seed_salts(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Require at least one unique seed salt."""

        if not value:
            raise ValueError("seed_salts must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("seed_salts must be unique")
        return value


class ControlConfig(FrozenModel):
    """Request random controls matched to the area of a named region.

    Attributes:
        match_area_of: Region ID whose intended area should be matched.
        n_samples: Number of control regions to generate per target region.
    """

    match_area_of: RegionId
    n_samples: PositiveInt


class RuntimeConfig(FrozenModel):
    """Collect execution, batching, retry, and determinism settings.

    These values are consumed by planning and runtime modules after resolution.
    """

    global_seed: Annotated[int, Field(ge=0, le=2**64 - 1)] = 0
    variants_per_chunk: PositiveInt = 16
    target_batch_size: PositiveInt = 32
    num_workers: NonNegativeInt = 0
    retry_failed: bool = False
    fail_fast: bool = False
    allow_nondeterministic: bool = False


class DumpConfig(FrozenModel):
    """Configure buffering behavior for the future dump writer."""

    flush_every: PositiveInt = 1000


class DatasetStats(FrozenModel):
    """Store dataset-level values resolved before perturbation begins.

    Attributes:
        channel_mean: Per-channel mean in the original uint8 pixel space.
    """

    channel_mean: tuple[float, ...]

    @field_validator("channel_mean")
    @classmethod
    def validate_channel_mean(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        """Validate channel means against the source pixel-space contract."""

        if not value:
            raise ValueError("channel_mean must not be empty")
        if any(not isfinite(item) or not 0.0 <= item <= 255.0 for item in value):
            raise ValueError("channel_mean values must be finite and within uint8 range")
        return value


class AuditConfig(FrozenModel):
    """Represent the complete audit specification loaded from YAML.

    Source and model adapter objects are injected separately by ConfigResolver.
    This model defines only the audit space and its execution policy.

    Attributes:
        schema_version: Version governing config, hashing, and dump contracts.
        regions: Regions to materialize for every selected sample.
        perturbations: Perturbations combined with the configured regions.
        controls: Optional area-matched random control requests.
        runtime: Worker, batching, retry, and determinism settings.
        dump: Raw dump buffering settings.
        dataset_stats: Optional precomputed statistics used by perturbations.
    """

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    regions: tuple[RegionConfig, ...]
    perturbations: tuple[PerturbationConfig, ...]
    controls: tuple[ControlConfig, ...] = ()
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    dump: DumpConfig = Field(default_factory=DumpConfig)
    dataset_stats: DatasetStats | None = None

    @model_validator(mode="after")
    def validate_audit_space(self) -> AuditConfig:
        """Validate cross-references and required audit dimensions."""

        if not self.regions:
            raise ValueError("regions must not be empty")
        if not self.perturbations:
            raise ValueError("perturbations must not be empty")

        region_ids = [region.region_id for region in self.regions]
        if len(set(region_ids)) != len(region_ids):
            raise ValueError("region_id values must be unique")

        known_regions = set(region_ids)
        unknown = {
            control.match_area_of
            for control in self.controls
            if control.match_area_of not in known_regions
        }
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"controls reference unknown region_id values: {names}")
        return self


class ResolvedRegionConfig(FrozenModel):
    """Describe a region after all filesystem references are resolved.

    Attributes:
        region_id: Stable name used by controls and dump records.
        kind: Strategy used to materialize the region mask.
        params: Strategy-specific JSON parameters.
        ref: Absolute explicit-mask path, when applicable.
        ref_hash: Verified SHA-256 digest of the explicit mask.
    """

    region_id: RegionId
    kind: RegionKind
    params: dict[str, Any] = Field(default_factory=dict)
    ref: Path | None = None
    ref_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("params")
    @classmethod
    def validate_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Ensure resolved parameters remain canonical JSON-compatible values."""

        validate_json_value(value)
        return value

    @model_validator(mode="after")
    def validate_resolved_reference(self) -> ResolvedRegionConfig:
        """Require complete absolute references for explicit regions."""

        if self.kind is RegionKind.EXPLICIT:
            if self.ref is None or not self.ref.is_absolute() or self.ref_hash is None:
                raise ValueError(
                    "resolved explicit regions require an absolute ref and ref_hash"
                )
        elif self.ref is not None or self.ref_hash is not None:
            raise ValueError("procedural regions cannot define ref or ref_hash")
        return self


class ResolvedConfig(FrozenModel):
    """Hold the fully resolved, manifest-ready audit configuration.

    No downstream module should reinterpret paths, hashes, statistics, adapter
    metadata, or perturbation defaults after this model is created.

    Attributes:
        schema_version: Version governing config, hashing, and dump contracts.
        config_source: Absolute YAML path, or None for in-memory input.
        config_base_dir: Absolute base used to resolve relative paths.
        regions: Region definitions with verified filesystem references.
        perturbations: Perturbations with complete runtime parameters.
        controls: Area-matched control requests.
        runtime: Worker, batching, retry, and determinism settings.
        dump: Raw dump buffering settings.
        dataset_stats: Optional resolved dataset statistics.
        adapter_spec: Adapter metadata verified before execution.
    """

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    config_source: Path | None = None
    config_base_dir: Path
    regions: tuple[ResolvedRegionConfig, ...]
    perturbations: tuple[PerturbationConfig, ...]
    controls: tuple[ControlConfig, ...] = ()
    runtime: RuntimeConfig
    dump: DumpConfig
    dataset_stats: DatasetStats | None = None
    adapter_spec: AdapterSpec

    @field_validator("config_source")
    @classmethod
    def validate_config_source(cls, value: Path | None) -> Path | None:
        """Require the optional source path to be absolute."""

        if value is not None and not value.is_absolute():
            raise ValueError("config_source must be absolute")
        return value

    @field_validator("config_base_dir")
    @classmethod
    def validate_config_base_dir(cls, value: Path) -> Path:
        """Require the path-resolution base directory to be absolute."""

        if not value.is_absolute():
            raise ValueError("config_base_dir must be absolute")
        return value
