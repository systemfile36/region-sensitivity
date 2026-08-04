"""Resolve user configuration into a deterministic, manifest-ready contract."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeAlias

from ssat.core.adapter.base import SupportsDescribe
from ssat.core.adapter.types import AdapterSpec
from ssat.core.config.schema import (
    AuditConfig,
    DatasetStats,
    PerturbationConfig,
    RegionConfig,
    ResolvedConfig,
    ResolvedRegionConfig,
)
from ssat.core.config.stats import DatasetStatsError, compute_dataset_stats
from ssat.core.source.base import SampleSource
from ssat.core.types import PerturbationOp, RegionKind
from ssat.utils.io import load_yaml, sha256_file
from ssat.utils.logger_factory import get_logger

ConfigInput: TypeAlias = AuditConfig | Mapping[str, Any] | str | Path


class ConfigResolutionError(ValueError):
    """Indicate that pre-run configuration resolution could not complete."""


class ConfigResolver:
    """Resolve all external references and runtime-dependent configuration.

    Args:
        logger: Optional package logger used for resolution audit events.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or get_logger(__name__)

    def resolve(
        self,
        config: ConfigInput,
        adapter: SupportsDescribe,
        sample_source: SampleSource,
        *,
        base_dir: str | Path | None = None,
    ) -> ResolvedConfig:
        """Resolve configuration before work enumeration starts.

        Args:
            config: YAML path, validated model, or mapping to resolve.
            adapter: Adapter metadata provider checked for determinism.
            sample_source: Dataset source used when statistics must be computed.
            base_dir: Path base for in-memory configurations.

        Returns:
            A frozen, JSON-serializable resolved configuration.

        Raises:
            ConfigResolutionError: If any input, reference, or validation step
                cannot be resolved deterministically.
        """

        self._logger.info(
            "config.resolve_started input_type=%s",
            type(config).__name__,
        )
        try:
            audit_config, config_source, config_base_dir = self._load_config(
                config,
                base_dir=base_dir,
            )
            adapter_spec = self._describe_adapter(adapter)
            self._validate_adapter_determinism(audit_config, adapter_spec)

            regions = tuple(
                self._resolve_region(region, config_base_dir)
                for region in audit_config.regions
            )
            for perturbation in audit_config.perturbations:
                self._validate_perturbation(perturbation)

            needs_stats = any(
                perturbation.op is PerturbationOp.MEAN_FILL
                for perturbation in audit_config.perturbations
            )
            dataset_stats = audit_config.dataset_stats
            if dataset_stats is not None:
                self._logger.info("dataset_stats.reused source=config")
            elif needs_stats:
                dataset_stats = compute_dataset_stats(
                    sample_source,
                    logger=self._logger,
                )

            perturbations = tuple(
                self._resolve_perturbation(perturbation, dataset_stats)
                for perturbation in audit_config.perturbations
            )
            resolved = ResolvedConfig(
                schema_version=audit_config.schema_version,
                config_source=config_source,
                config_base_dir=config_base_dir,
                regions=regions,
                perturbations=perturbations,
                controls=audit_config.controls,
                runtime=audit_config.runtime,
                dump=audit_config.dump,
                dataset_stats=dataset_stats,
                adapter_spec=adapter_spec,
            )
        except ConfigResolutionError:
            raise
        except DatasetStatsError as error:
            raise ConfigResolutionError(f"dataset statistics resolution failed: {error}") from error
        except Exception as error:
            raise ConfigResolutionError(f"configuration resolution failed: {error}") from error

        self._logger.info(
            "config.resolve_completed model_id=%s regions=%d perturbations=%d stats=%s",
            resolved.adapter_spec.model_id,
            len(resolved.regions),
            len(resolved.perturbations),
            "available" if resolved.dataset_stats is not None else "not_required",
        )
        return resolved

    def _load_config(
        self,
        config: ConfigInput,
        *,
        base_dir: str | Path | None,
    ) -> tuple[AuditConfig, Path | None, Path]:
        """Load and validate input while establishing its path base."""

        config_source: Path | None = None
        if isinstance(config, (str, Path)):
            config_source = Path(config).expanduser().resolve(strict=True)
            if not config_source.is_file():
                raise ConfigResolutionError(
                    f"configuration source is not a file: {config_source}"
                )
            raw_config = load_yaml(config_source)
            audit_config = AuditConfig.model_validate(raw_config)
            resolved_base_dir = config_source.parent
        elif isinstance(config, AuditConfig):
            audit_config = config
            resolved_base_dir = self._resolve_memory_base(audit_config, base_dir)
        elif isinstance(config, Mapping):
            audit_config = AuditConfig.model_validate(dict(config))
            resolved_base_dir = self._resolve_memory_base(audit_config, base_dir)
        else:
            raise ConfigResolutionError(
                "config must be a YAML path, mapping, or AuditConfig"
            )

        self._logger.debug(
            "config.input_resolved source=%s base_dir=%s",
            config_source,
            resolved_base_dir,
        )
        return audit_config, config_source, resolved_base_dir

    @staticmethod
    def _resolve_memory_base(
        config: AuditConfig,
        base_dir: str | Path | None,
    ) -> Path:
        """Resolve the base directory for mapping or model input."""

        has_relative_ref = any(
            region.kind is RegionKind.EXPLICIT
            and region.ref is not None
            and not region.ref.expanduser().is_absolute()
            for region in config.regions
        )
        if base_dir is None:
            if has_relative_ref:
                raise ConfigResolutionError(
                    "base_dir is required for in-memory config with relative refs"
                )
            return Path.cwd().resolve()

        resolved = Path(base_dir).expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise ConfigResolutionError(f"base_dir is not a directory: {resolved}")
        return resolved

    @staticmethod
    def _describe_adapter(adapter: SupportsDescribe) -> AdapterSpec:
        """Call the adapter metadata boundary and validate its return type."""

        try:
            adapter_spec = adapter.describe()
        except Exception as error:
            raise ConfigResolutionError("adapter.describe() failed") from error
        if not isinstance(adapter_spec, AdapterSpec):
            raise ConfigResolutionError("adapter.describe() must return AdapterSpec")
        return adapter_spec

    def _validate_adapter_determinism(
        self,
        config: AuditConfig,
        adapter_spec: AdapterSpec,
    ) -> None:
        """Reject or explicitly warn about nondeterministic adapters."""

        if adapter_spec.deterministic:
            self._logger.info(
                "adapter.validated model_id=%s deterministic=true",
                adapter_spec.model_id,
            )
            return
        if not config.runtime.allow_nondeterministic:
            raise ConfigResolutionError(
                f"adapter model_id={adapter_spec.model_id!r} is nondeterministic; "
                "set allow_nondeterministic=true to proceed"
            )
        self._logger.warning(
            "adapter.nondeterministic_allowed model_id=%s",
            adapter_spec.model_id,
        )

    def _resolve_region(
        self,
        region: RegionConfig,
        base_dir: Path,
    ) -> ResolvedRegionConfig:
        """Resolve and verify one explicit mask reference when present."""

        if region.kind is not RegionKind.EXPLICIT:
            return ResolvedRegionConfig(
                region_id=region.region_id,
                kind=region.kind,
                params=region.params,
            )

        if region.ref is None:  # RegionConfig validates this; retain a clear boundary.
            raise ConfigResolutionError(
                f"explicit region_id={region.region_id!r} has no ref"
            )
        candidate = region.ref.expanduser()
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        try:
            resolved_ref = candidate.resolve(strict=True)
        except OSError as error:
            raise ConfigResolutionError(
                f"explicit region_id={region.region_id!r} ref does not exist: {candidate}"
            ) from error
        if not resolved_ref.is_file():
            raise ConfigResolutionError(
                f"explicit region_id={region.region_id!r} ref is not a file: {resolved_ref}"
            )

        actual_hash = sha256_file(resolved_ref)
        if region.ref_hash is not None and region.ref_hash.lower() != actual_hash:
            raise ConfigResolutionError(
                f"explicit region_id={region.region_id!r} ref_hash mismatch"
            )
        self._logger.debug(
            "region.reference_resolved region_id=%s ref=%s ref_hash=%s",
            region.region_id,
            resolved_ref,
            actual_hash,
        )
        return ResolvedRegionConfig(
            region_id=region.region_id,
            kind=region.kind,
            params=region.params,
            ref=resolved_ref,
            ref_hash=actual_hash,
        )

    @staticmethod
    def _validate_perturbation(perturbation: PerturbationConfig) -> None:
        """Validate the strict v1 parameter contract for one operation."""

        params = perturbation.params
        op = perturbation.op
        if op is PerturbationOp.CONSTANT_FILL:
            ConfigResolver._require_keys(op, params, {"value"})
            ConfigResolver._validate_fill_value(params["value"])
        elif op is PerturbationOp.MEAN_FILL:
            ConfigResolver._require_keys(op, params, set())
        elif op in (PerturbationOp.BLUR, PerturbationOp.GAUSSIAN_NOISE):
            ConfigResolver._require_keys(op, params, {"sigma"})
            ConfigResolver._validate_positive_number(op, "sigma", params["sigma"])
        elif op is PerturbationOp.PATCH_SHUFFLE:
            ConfigResolver._require_keys(op, params, {"patch_size"})
            patch_size = params["patch_size"]
            if isinstance(patch_size, bool) or not isinstance(patch_size, int):
                raise ConfigResolutionError("patch_shuffle.patch_size must be an integer")
            if patch_size <= 0:
                raise ConfigResolutionError("patch_shuffle.patch_size must be positive")

    @staticmethod
    def _require_keys(
        op: PerturbationOp,
        params: Mapping[str, Any],
        expected: set[str],
    ) -> None:
        """Require an exact parameter-key set for a v1 operation."""

        actual = set(params)
        if actual != expected:
            raise ConfigResolutionError(
                f"{op.value} params must contain exactly {sorted(expected)}; "
                f"received {sorted(actual)}"
            )

    @staticmethod
    def _validate_fill_value(value: Any) -> None:
        """Validate scalar or per-channel constant-fill values."""

        values = value if isinstance(value, list) else [value]
        if not values:
            raise ConfigResolutionError("constant_fill.value must not be empty")
        for item in values:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ConfigResolutionError(
                    "constant_fill.value must contain only numeric values"
                )
            if not math.isfinite(float(item)) or not 0.0 <= float(item) <= 255.0:
                raise ConfigResolutionError(
                    "constant_fill.value values must be finite and within [0, 255]"
                )

    @staticmethod
    def _validate_positive_number(
        op: PerturbationOp,
        name: str,
        value: Any,
    ) -> None:
        """Validate a finite, positive scalar parameter."""

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigResolutionError(f"{op.value}.{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ConfigResolutionError(f"{op.value}.{name} must be finite and positive")

    @staticmethod
    def _resolve_perturbation(
        perturbation: PerturbationConfig,
        dataset_stats: DatasetStats | None,
    ) -> PerturbationConfig:
        """Fill runtime-derived perturbation parameters."""

        if perturbation.op is not PerturbationOp.MEAN_FILL:
            return perturbation
        if dataset_stats is None:
            raise ConfigResolutionError("mean_fill requires resolved dataset statistics")
        return PerturbationConfig(
            op=perturbation.op,
            params={"value": list(dataset_stats.channel_mean)},
            invert_mask=perturbation.invert_mask,
            seed_salts=perturbation.seed_salts,
        )

