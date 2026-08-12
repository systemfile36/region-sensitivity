"""Resolve user configuration into a deterministic, manifest-ready contract."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
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
    ResolvedSkeletonSourceConfig,
    SkeletonSourceConfig,
    SourceProvenance,
)
from ssat.core.config.stats import DatasetStatsError, compute_dataset_stats
from ssat.core.perturb.base import PerturbationError, PerturbationOperator
from ssat.core.perturb.dispatch import find_operator
from ssat.core.perturb.factory import build_operators
from ssat.core.plan.expansion_base import (
    RegionExpansionContext,
    RegionExpansionError,
    RegionFamilyExpander,
)
from ssat.core.plan.region_expander_dispatch import find_family_expander
from ssat.core.plan.region_expander_factory import build_family_expanders
from ssat.core.source.base import SampleSource
from ssat.core.types import RegionKind
from ssat.utils.io import load_yaml, sha256_file
from ssat.utils.logger_factory import get_logger

ConfigInput: TypeAlias = AuditConfig | Mapping[str, Any] | str | Path


class ConfigResolutionError(ValueError):
    """Indicate that pre-run configuration resolution could not complete."""


class ConfigResolver:
    """Resolve all external references and runtime-dependent configuration.

    Args:
        logger: Optional package logger used for resolution audit events.
        perturbation_operators: Optional operators in config dispatch order.
        region_family_expanders: Optional expanders in config dispatch order.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        perturbation_operators: Sequence[PerturbationOperator] | None = None,
        region_family_expanders: Sequence[RegionFamilyExpander] | None = None,
    ) -> None:
        self._logger = logger or get_logger(__name__)
        resolved_operators = tuple(
            build_operators()
            if perturbation_operators is None
            else perturbation_operators
        )
        if not resolved_operators:
            raise ValueError("perturbation_operators must not be empty")
        if any(
            not isinstance(operator, PerturbationOperator)
            for operator in resolved_operators
        ):
            raise TypeError(
                "perturbation_operators must contain PerturbationOperator values"
            )
        self._perturbation_operators = resolved_operators

        resolved_expanders = tuple(
            build_family_expanders(RegionExpansionContext())
            if region_family_expanders is None
            else region_family_expanders
        )
        if not resolved_expanders:
            raise ValueError("region_family_expanders must not be empty")
        if any(
            not isinstance(expander, RegionFamilyExpander)
            for expander in resolved_expanders
        ):
            raise TypeError(
                "region_family_expanders must contain RegionFamilyExpander values"
            )
        self._region_family_expanders = resolved_expanders

    def resolve(
        self,
        config: ConfigInput,
        adapter: SupportsDescribe,
        sample_source: SampleSource,
        *,
        base_dir: str | Path | None = None,
        config_source: str | Path | None = None,
        source_provenance: SourceProvenance | None = None,
    ) -> ResolvedConfig:
        """Resolve configuration before work enumeration starts.

        Args:
            config: YAML path, validated model, or mapping to resolve.
            adapter: Adapter metadata provider checked for determinism.
            sample_source: Dataset source used when statistics must be computed.
            base_dir: Path base for in-memory configurations.
            config_source: Optional original path for an already parsed config.
            source_provenance: Optional resolved application source identity.

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
            requested_config_source = config_source
            audit_config, loaded_config_source, config_base_dir = self._load_config(
                config,
                base_dir=base_dir,
            )
            resolved_config_source = loaded_config_source
            if requested_config_source is not None:
                supplied_source = (
                    Path(requested_config_source).expanduser().resolve(strict=True)
                )
                if not supplied_source.is_file():
                    raise ConfigResolutionError(
                        f"configuration source is not a file: {supplied_source}"
                    )
                if isinstance(config, (str, Path)):
                    if supplied_source != Path(config).expanduser().resolve(strict=True):
                        raise ConfigResolutionError(
                            "config_source disagrees with the configuration path"
                        )
                resolved_config_source = supplied_source
            adapter_spec = self._describe_adapter(adapter)
            self._validate_adapter_determinism(audit_config, adapter_spec)

            regions = tuple(
                self._resolve_region(region, config_base_dir)
                for region in audit_config.regions
            )
            skeleton_source = self._resolve_skeleton_source(
                audit_config.skeleton_source, config_base_dir
            )
            perturbation_handlers = tuple(
                (
                    perturbation,
                    self._validate_perturbation(perturbation),
                )
                for perturbation in audit_config.perturbations
            )

            needs_stats = any(
                self._requires_dataset_stats(perturbation, operator)
                for perturbation, operator in perturbation_handlers
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
                self._resolve_perturbation(
                    perturbation,
                    operator,
                    dataset_stats,
                )
                for perturbation, operator in perturbation_handlers
            )
            resolved = ResolvedConfig(
                schema_version=audit_config.schema_version,
                config_source=resolved_config_source,
                config_base_dir=config_base_dir,
                regions=regions,
                perturbations=perturbations,
                controls=audit_config.controls,
                runtime=audit_config.runtime,
                dump=audit_config.dump,
                dataset_stats=dataset_stats,
                adapter_spec=adapter_spec,
                source_provenance=source_provenance,
                skeleton_source=skeleton_source,
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
        ) or (
            config.skeleton_source is not None
            and not config.skeleton_source.bbox_data.expanduser().is_absolute()
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
        """Validate a family recipe and resolve its external references."""

        self._validate_region_family(region)

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

    def _resolve_skeleton_source(
        self,
        skeleton_source: SkeletonSourceConfig | None,
        base_dir: Path,
    ) -> ResolvedSkeletonSourceConfig | None:
        """Resolve and verify the optional skeleton bbox data reference.

        Mirrors explicit-mask reference resolution: only the path and
        content hash are verified here. The bbox data itself is loaded
        lazily by the caller (e.g. the Application layer), not eagerly
        embedded in this frozen, hashable configuration.

        Args:
            skeleton_source: User-configured reference, or ``None``.
            base_dir: Base directory for relative paths.

        Returns:
            The resolved reference, or ``None`` if none was configured.

        Raises:
            ConfigResolutionError: If the file is missing or its hash
                does not match the configured expectation.
        """

        if skeleton_source is None:
            return None
        candidate = skeleton_source.bbox_data.expanduser()
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        try:
            resolved_path = candidate.resolve(strict=True)
        except OSError as error:
            raise ConfigResolutionError(
                f"skeleton_source.bbox_data does not exist: {candidate}"
            ) from error
        if not resolved_path.is_file():
            raise ConfigResolutionError(
                f"skeleton_source.bbox_data is not a file: {resolved_path}"
            )
        actual_hash = sha256_file(resolved_path)
        if (
            skeleton_source.bbox_data_hash is not None
            and skeleton_source.bbox_data_hash.lower() != actual_hash
        ):
            raise ConfigResolutionError("skeleton_source.bbox_data_hash mismatch")
        self._logger.debug(
            "skeleton_source.reference_resolved bbox_data=%s bbox_data_hash=%s",
            resolved_path,
            actual_hash,
        )
        return ResolvedSkeletonSourceConfig(
            bbox_data=resolved_path,
            bbox_data_hash=actual_hash,
        )

    def _validate_region_family(self, region: RegionConfig) -> None:
        """Select an expander and validate one region-family config.

        Args:
            region: User-configured region family to validate.

        Raises:
            ConfigResolutionError: If dispatch or config validation fails.
        """

        try:
            expander = find_family_expander(
                self._region_family_expanders,
                region,
            )
            expander.validate_config(region)
        except RegionExpansionError as error:
            raise ConfigResolutionError(
                f"invalid {region.kind.value} region configuration: {error}"
            ) from error
        except Exception as error:
            raise ConfigResolutionError(
                f"{region.kind.value} region config validation failed"
            ) from error

    def _validate_perturbation(
        self,
        perturbation: PerturbationConfig,
    ) -> PerturbationOperator:
        """Select an operator and validate one user perturbation config.

        Args:
            perturbation: User operation and unresolved parameters.

        Returns:
            The first registered operator supporting the operation.

        Raises:
            ConfigResolutionError: If dispatch or config validation fails.
        """

        try:
            operator = find_operator(
                self._perturbation_operators,
                perturbation.op,
            )
            operator.validate_config(perturbation.params)
            return operator
        except PerturbationError as error:
            raise ConfigResolutionError(
                f"invalid {perturbation.op.value} configuration: {error}"
            ) from error
        except Exception as error:
            raise ConfigResolutionError(
                f"{perturbation.op.value} config validation failed"
            ) from error

    @staticmethod
    def _requires_dataset_stats(
        perturbation: PerturbationConfig,
        operator: PerturbationOperator,
    ) -> bool:
        """Query an operator's dataset-statistics requirement.

        Args:
            perturbation: User perturbation included for error context.
            operator: Selected supporting operator.

        Returns:
            Whether dataset channel statistics must be available.

        Raises:
            ConfigResolutionError: If the operator hook fails.
        """

        try:
            required = operator.requires_dataset_stats()
        except Exception as error:
            raise ConfigResolutionError(
                f"{perturbation.op.value} stats requirement check failed"
            ) from error
        if not isinstance(required, bool):
            raise ConfigResolutionError(
                f"{perturbation.op.value} requires_dataset_stats() must return bool"
            )
        return required

    @staticmethod
    def _resolve_perturbation(
        perturbation: PerturbationConfig,
        operator: PerturbationOperator,
        dataset_stats: DatasetStats | None,
    ) -> PerturbationConfig:
        """Resolve runtime params through the selected operator hook.

        Args:
            perturbation: Validated user perturbation config.
            operator: First registered supporting operator.
            dataset_stats: Optional computed or preconfigured statistics.

        Returns:
            Perturbation config containing complete runtime parameters.

        Raises:
            ConfigResolutionError: If parameter resolution fails.
        """

        channel_mean = (
            dataset_stats.channel_mean if dataset_stats is not None else None
        )
        try:
            params = operator.resolve_config_params(
                perturbation.params,
                channel_mean,
            )
            return PerturbationConfig(
                op=perturbation.op,
                params=params,
                invert_mask=perturbation.invert_mask,
                seed_salts=perturbation.seed_salts,
            )
        except PerturbationError as error:
            raise ConfigResolutionError(str(error)) from error
        except Exception as error:
            raise ConfigResolutionError(
                f"{perturbation.op.value} parameter resolution failed"
            ) from error
