"""N4 MetricsStore: persist and reload every metrics-engine output.

Writes ``<metrics_dir>/item_metrics.parquet``, ``sample_metrics.parquet``,
``region_metrics.parquet``, ``class_metrics.parquet``, ``spatial_profile.parquet``,
and ``metrics_manifest.json``. Follows the same provenance principle as the
raw dump: metrics are only as trustworthy
as the dump they were computed from, tracked here via
``source_run_manifest_hash`` — a SHA-256 of the source run's actual
``run_manifest.json`` bytes, not a re-serialization of its parsed contents.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pyarrow as pa

from ssat.core.types import RegionKind
from ssat.metrics._storage import atomic_write_parquet, read_validated_table
from ssat.metrics.aggregate import AggregationResult
from ssat.metrics.errors import MetricsCorruptionError, MetricsRegistryError, MetricsSchemaError
from ssat.metrics.registry import MetricRegistry
from ssat.metrics.schema import (
    CLASS_METRICS_SCHEMA,
    ITEM_METRICS_SCHEMA,
    METRICS_SCHEMA_VERSION,
    REGION_METRICS_SCHEMA,
    SAMPLE_METRICS_SCHEMA,
    SPATIAL_PROFILE_SCHEMA,
)
from ssat.metrics.types import (
    ClassMetrics,
    ExclusionReason,
    ItemMetrics,
    RegionGeometryRef,
    RegionMetrics,
    SampleMetrics,
    SpatialProfile,
)
from ssat.utils.io import load_json, sha256_file, write_json_atomic

_TOPK_METRIC_NAME = "topk_exit"


@dataclass(frozen=True, slots=True)
class MetricConfig:
    """Carry the configuration choices that shaped one aggregation run.

    Attributes:
        primary_metric: Registered metric name whose degradation defined
            ``vulnerability_score``.
        topk: The ``topk_exit`` metric's rank cutoff, when that metric was
            registered; ``None`` otherwise.
    """

    primary_metric: str
    topk: int | None

    def __post_init__(self) -> None:
        if not self.primary_metric:
            raise ValueError("primary_metric must not be empty")
        if self.topk is not None and (isinstance(self.topk, bool) or self.topk < 1):
            raise ValueError("topk must be a positive int when provided")


@dataclass(frozen=True, slots=True)
class RegisteredMetricInfo:
    """Describe one metric that took part in a stored aggregation run.

    No ``version`` field is carried here: the ``Metric`` protocol has no
    per-metric version concept, and ``metrics_schema_version`` already tracks
    "a metric's definition changed" at the file-format level.
    """

    name: str
    kind: Literal["binary", "continuous"]
    higher_is_better: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if self.kind not in ("binary", "continuous"):
            raise ValueError("kind must be 'binary' or 'continuous'")


@dataclass(frozen=True, slots=True)
class MetricsManifest:
    """Describe one stored aggregation run and the dump it was computed from.

    Attributes:
        metrics_schema_version: Version of the on-disk metrics format itself.
        source_run_manifest_hash: SHA-256 of the source dump's
            ``run_manifest.json`` bytes at computation time.
        metric_config: Configuration choices that shaped this run.
        registered_metrics: Every metric that took part, sorted by name.
        computed_at: When this aggregation run was persisted.
        exclusion_summary: Status-based exclusion counts, typically
            ``DumpHandle.summary()``'s own return value.
    """

    metrics_schema_version: str
    source_run_manifest_hash: str
    metric_config: MetricConfig
    registered_metrics: tuple[RegisteredMetricInfo, ...]
    computed_at: datetime
    exclusion_summary: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.metrics_schema_version:
            raise ValueError("metrics_schema_version must not be empty")
        _validate_sha256_hex(self.source_run_manifest_hash)
        if self.computed_at.tzinfo is None or self.computed_at.utcoffset() is None:
            raise ValueError("computed_at must be timezone-aware")
        names = [metric.name for metric in self.registered_metrics]
        if len(set(names)) != len(names):
            raise ValueError("registered_metrics must not contain duplicate names")


def save_metrics(
    metrics_dir: Path,
    item_metrics: Sequence[ItemMetrics],
    result: AggregationResult,
    *,
    registry: MetricRegistry,
    primary_metric: str,
    source_run_manifest_path: Path,
    exclusion_summary: Mapping[str, object],
    computed_at: datetime | None = None,
) -> MetricsManifest:
    """Persist every metrics output plus ``metrics_manifest.json``.

    Args:
        metrics_dir: Destination directory (created if missing), typically
            ``<run_dir>/metrics``.
        item_metrics: Item-grain rows from ``registry.compute_item_metrics``.
        result: The four N3 aggregate outputs from ``aggregate_item_metrics``.
        registry: The registry that computed ``item_metrics`` — consulted to
            describe every registered metric and to validate ``primary_metric``.
        primary_metric: Registered metric name recorded as this run's
            vulnerability-score source.
        source_run_manifest_path: Path to the source dump's
            ``run_manifest.json`` (``DumpHandle.manifest_path``), hashed to
            produce ``source_run_manifest_hash``.
        exclusion_summary: Status-based exclusion counts to record verbatim.
        computed_at: Timestamp to record; defaults to now (UTC).

    Returns:
        The ``MetricsManifest`` that was written.

    Raises:
        MetricsRegistryError: If ``primary_metric`` is not registered.
    """

    if primary_metric not in registry.names:
        raise MetricsRegistryError(f"primary_metric not registered: {primary_metric}")

    metrics_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(metrics_dir / "item_metrics.parquet", _item_metrics_to_table(item_metrics))
    atomic_write_parquet(
        metrics_dir / "sample_metrics.parquet", _sample_metrics_to_table(result.sample_metrics)
    )
    atomic_write_parquet(
        metrics_dir / "region_metrics.parquet", _region_metrics_to_table(result.region_metrics)
    )
    atomic_write_parquet(
        metrics_dir / "class_metrics.parquet", _class_metrics_to_table(result.class_metrics)
    )
    atomic_write_parquet(
        metrics_dir / "spatial_profile.parquet", _spatial_profile_to_table(result.spatial_profile)
    )

    topk_metric = registry.get(_TOPK_METRIC_NAME) if _TOPK_METRIC_NAME in registry.names else None
    manifest = MetricsManifest(
        metrics_schema_version=METRICS_SCHEMA_VERSION,
        source_run_manifest_hash=sha256_file(source_run_manifest_path),
        metric_config=MetricConfig(
            primary_metric=primary_metric,
            topk=getattr(topk_metric, "k", None),
        ),
        registered_metrics=tuple(
            RegisteredMetricInfo(
                name=name,
                kind=registry.get(name).kind,
                higher_is_better=registry.get(name).higher_is_better,
            )
            for name in sorted(registry.names)
        ),
        computed_at=computed_at or datetime.now(timezone.utc),
        exclusion_summary=exclusion_summary,
    )
    write_json_atomic(metrics_dir / "metrics_manifest.json", _manifest_to_json(manifest))
    return manifest


def load_metrics(
    metrics_dir: Path,
) -> tuple[list[ItemMetrics], AggregationResult, MetricsManifest]:
    """Reload every metrics output and its manifest.

    Raises:
        MetricsCorruptionError: If a parquet fragment or the manifest is
            missing or unreadable.
        MetricsSchemaError: If ``metrics_schema_version`` or any parquet
            fragment's version/schema does not match this module's version.
    """

    manifest = _load_manifest(metrics_dir / "metrics_manifest.json")

    item_metrics = _table_to_item_metrics(
        read_validated_table(
            metrics_dir / "item_metrics.parquet",
            ITEM_METRICS_SCHEMA,
            expected_version=METRICS_SCHEMA_VERSION,
        )
    )
    sample_metrics = _table_to_sample_metrics(
        read_validated_table(
            metrics_dir / "sample_metrics.parquet",
            SAMPLE_METRICS_SCHEMA,
            expected_version=METRICS_SCHEMA_VERSION,
        )
    )
    region_metrics = _table_to_region_metrics(
        read_validated_table(
            metrics_dir / "region_metrics.parquet",
            REGION_METRICS_SCHEMA,
            expected_version=METRICS_SCHEMA_VERSION,
        )
    )
    class_metrics = _table_to_class_metrics(
        read_validated_table(
            metrics_dir / "class_metrics.parquet",
            CLASS_METRICS_SCHEMA,
            expected_version=METRICS_SCHEMA_VERSION,
        )
    )
    spatial_profile = _table_to_spatial_profile(
        read_validated_table(
            metrics_dir / "spatial_profile.parquet",
            SPATIAL_PROFILE_SCHEMA,
            expected_version=METRICS_SCHEMA_VERSION,
        )
    )

    result = AggregationResult(
        sample_metrics=sample_metrics,
        region_metrics=region_metrics,
        class_metrics=class_metrics,
        spatial_profile=spatial_profile,
    )
    return item_metrics, result, manifest


def verify_source_dump(manifest: MetricsManifest, source_run_manifest_path: Path) -> None:
    """Raise if the source dump's run_manifest.json no longer matches.

    Raises:
        MetricsCorruptionError: If the source dump was re-run or otherwise
            changed since this metrics store was computed — metrics become
            invalid whenever the dump they were computed from changes.
    """

    actual_hash = sha256_file(source_run_manifest_path)
    if actual_hash != manifest.source_run_manifest_hash:
        raise MetricsCorruptionError(
            "source dump has changed since these metrics were computed: "
            f"expected run_manifest hash {manifest.source_run_manifest_hash}, "
            f"found {actual_hash}"
        )


def _load_manifest(path: Path) -> MetricsManifest:
    if not path.is_file():
        raise MetricsCorruptionError(f"metrics manifest does not exist: {path}")
    try:
        payload = load_json(path)
    except Exception as error:
        raise MetricsCorruptionError(f"cannot read metrics manifest: {path}") from error
    if not isinstance(payload, dict):
        raise MetricsCorruptionError("metrics manifest must contain a JSON object")

    version = payload.get("metrics_schema_version")
    if version != METRICS_SCHEMA_VERSION:
        raise MetricsSchemaError(
            f"metrics schema version mismatch: expected {METRICS_SCHEMA_VERSION}, found {version}"
        )

    try:
        return _manifest_from_json(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise MetricsCorruptionError("metrics manifest does not match its schema") from error


def _manifest_to_json(manifest: MetricsManifest) -> dict[str, object]:
    return {
        "metrics_schema_version": manifest.metrics_schema_version,
        "source_run_manifest_hash": manifest.source_run_manifest_hash,
        "metric_config": {
            "primary_metric": manifest.metric_config.primary_metric,
            "topk": manifest.metric_config.topk,
        },
        "registered_metrics": [
            {"name": metric.name, "kind": metric.kind, "higher_is_better": metric.higher_is_better}
            for metric in manifest.registered_metrics
        ],
        "computed_at": manifest.computed_at.isoformat(),
        "exclusion_summary": dict(manifest.exclusion_summary),
    }


def _manifest_from_json(payload: dict[str, object]) -> MetricsManifest:
    metric_config_payload = payload["metric_config"]
    registered_metrics_payload = payload["registered_metrics"]
    return MetricsManifest(
        metrics_schema_version=payload["metrics_schema_version"],
        source_run_manifest_hash=payload["source_run_manifest_hash"],
        metric_config=MetricConfig(
            primary_metric=metric_config_payload["primary_metric"],
            topk=metric_config_payload["topk"],
        ),
        registered_metrics=tuple(
            RegisteredMetricInfo(
                name=entry["name"], kind=entry["kind"], higher_is_better=entry["higher_is_better"]
            )
            for entry in registered_metrics_payload
        ),
        computed_at=datetime.fromisoformat(payload["computed_at"]),
        exclusion_summary=payload["exclusion_summary"],
    )


def _validate_sha256_hex(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("source_run_manifest_hash must be a lowercase SHA-256 hex digest")


def _item_metrics_to_table(rows: Sequence[ItemMetrics]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "item_id": row.item_id,
                "sample_id": row.sample_id,
                "metric_name": row.metric_name,
                "clean_correct": row.clean_correct,
                "value_clean": row.value_clean,
                "value_perturbed": row.value_perturbed,
                "degradation": row.degradation,
                "available": row.available,
                "excluded_reason": row.excluded_reason.value if row.excluded_reason else None,
            }
            for row in rows
        ],
        schema=ITEM_METRICS_SCHEMA,
    )


def _table_to_item_metrics(table: pa.Table) -> list[ItemMetrics]:
    return [
        ItemMetrics(
            item_id=row["item_id"],
            sample_id=row["sample_id"],
            metric_name=row["metric_name"],
            clean_correct=row["clean_correct"],
            value_clean=row["value_clean"],
            value_perturbed=row["value_perturbed"],
            degradation=row["degradation"],
            available=row["available"],
            excluded_reason=(
                ExclusionReason(row["excluded_reason"]) if row["excluded_reason"] is not None else None
            ),
        )
        for row in table.to_pylist()
    ]


def _sample_metrics_to_table(rows: Sequence[SampleMetrics]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "sample_id": row.sample_id,
                "metric_name": row.metric_name,
                "gt_label": row.gt_label,
                "clean_correct": row.clean_correct,
                "n_items": row.n_items,
                "n_valid": row.n_valid,
                "flip_rate": row.flip_rate,
                "vulnerability_score": row.vulnerability_score,
                "metric_mean": row.metric_mean,
                "metric_max": row.metric_max,
                "metric_std": row.metric_std,
            }
            for row in rows
        ],
        schema=SAMPLE_METRICS_SCHEMA,
    )


def _table_to_sample_metrics(table: pa.Table) -> list[SampleMetrics]:
    return [
        SampleMetrics(
            sample_id=row["sample_id"],
            metric_name=row["metric_name"],
            gt_label=row["gt_label"],
            clean_correct=row["clean_correct"],
            n_items=row["n_items"],
            n_valid=row["n_valid"],
            flip_rate=row["flip_rate"],
            vulnerability_score=row["vulnerability_score"],
            metric_mean=row["metric_mean"],
            metric_max=row["metric_max"],
            metric_std=row["metric_std"],
        )
        for row in table.to_pylist()
    ]


def _region_metrics_to_table(rows: Sequence[RegionMetrics]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "region_key": row.region_key,
                "metric_name": row.metric_name,
                "region_kind": row.region_kind.value,
                "intended_area_px": row.intended_area_px,
                "effective_area_px": row.effective_area_px,
                "n_samples": row.n_samples,
                "n_valid": row.n_valid,
                "flip_rate": row.flip_rate,
                "metric_mean": row.metric_mean,
            }
            for row in rows
        ],
        schema=REGION_METRICS_SCHEMA,
    )


def _table_to_region_metrics(table: pa.Table) -> list[RegionMetrics]:
    return [
        RegionMetrics(
            region_key=row["region_key"],
            metric_name=row["metric_name"],
            region_kind=RegionKind(row["region_kind"]),
            intended_area_px=row["intended_area_px"],
            effective_area_px=row["effective_area_px"],
            n_samples=row["n_samples"],
            n_valid=row["n_valid"],
            flip_rate=row["flip_rate"],
            metric_mean=row["metric_mean"],
        )
        for row in table.to_pylist()
    ]


def _class_metrics_to_table(rows: Sequence[ClassMetrics]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "gt_label": row.gt_label,
                "metric_name": row.metric_name,
                "n_samples": row.n_samples,
                "flip_rate": row.flip_rate,
                "metric_mean": row.metric_mean,
            }
            for row in rows
        ],
        schema=CLASS_METRICS_SCHEMA,
    )


def _table_to_class_metrics(table: pa.Table) -> list[ClassMetrics]:
    return [
        ClassMetrics(
            gt_label=row["gt_label"],
            metric_name=row["metric_name"],
            n_samples=row["n_samples"],
            flip_rate=row["flip_rate"],
            metric_mean=row["metric_mean"],
        )
        for row in table.to_pylist()
    ]


def _spatial_profile_to_table(rows: Sequence[SpatialProfile]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "sample_id": row.sample_id,
                "region_key": row.region_key,
                "metric_name": row.metric_name,
                "region_kind": row.region_geometry_ref.region_kind.value,
                "region_params_json": row.region_geometry_ref.region_params_json,
                "ref": row.region_geometry_ref.ref,
                "ref_hash": row.region_geometry_ref.ref_hash,
                "degradation": row.degradation,
            }
            for row in rows
        ],
        schema=SPATIAL_PROFILE_SCHEMA,
    )


def _table_to_spatial_profile(table: pa.Table) -> list[SpatialProfile]:
    return [
        SpatialProfile(
            sample_id=row["sample_id"],
            region_key=row["region_key"],
            metric_name=row["metric_name"],
            region_geometry_ref=RegionGeometryRef(
                region_kind=RegionKind(row["region_kind"]),
                region_params_json=row["region_params_json"],
                ref=row["ref"],
                ref_hash=row["ref_hash"],
            ),
            degradation=row["degradation"],
        )
        for row in table.to_pylist()
    ]
