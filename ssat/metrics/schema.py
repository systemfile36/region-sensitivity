"""Versioned PyArrow contracts for the five N3 aggregate outputs.

Mirrors ``ssat.core.dump.schema``'s shape (one ``pa.Schema`` constant per
dataset, a shared version tag in Arrow metadata, and a ``schema_for`` lookup)
but is independent of it — ``METRICS_SCHEMA_VERSION`` tracks metric
*definition* changes, not the raw dump's own ``SCHEMA_VERSION`` (design
METRIC_ENGINE_DESIGN_v1.md §N4: "지표 정의가 바뀌면 metrics_schema_version이
올라간다").
"""

from __future__ import annotations

import pyarrow as pa

METRICS_SCHEMA_VERSION = "1.0.0"

METRICS_SCHEMA_METADATA = {b"ssat.metrics_schema_version": METRICS_SCHEMA_VERSION.encode("ascii")}

# One row per (item_id, metric_name).
ITEM_METRICS_SCHEMA = pa.schema(
    [
        pa.field("item_id", pa.string(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("metric_name", pa.string(), nullable=False),
        pa.field("clean_correct", pa.bool_(), nullable=False),
        pa.field("value_clean", pa.float64()),
        pa.field("value_perturbed", pa.float64()),
        pa.field("degradation", pa.float64()),
        pa.field("available", pa.bool_(), nullable=False),
        pa.field("excluded_reason", pa.string()),
    ],
    metadata=METRICS_SCHEMA_METADATA,
)

# One row per (sample_id, metric_name).
SAMPLE_METRICS_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("metric_name", pa.string(), nullable=False),
        pa.field("gt_label", pa.int64()),
        pa.field("clean_correct", pa.bool_()),
        pa.field("n_items", pa.int64(), nullable=False),
        pa.field("n_valid", pa.int64(), nullable=False),
        pa.field("flip_rate", pa.float64()),
        pa.field("vulnerability_score", pa.float64()),
        pa.field("metric_mean", pa.float64()),
        pa.field("metric_max", pa.float64()),
        pa.field("metric_std", pa.float64()),
    ],
    metadata=METRICS_SCHEMA_METADATA,
)

# One row per (region_key, metric_name).
REGION_METRICS_SCHEMA = pa.schema(
    [
        pa.field("region_key", pa.string(), nullable=False),
        pa.field("metric_name", pa.string(), nullable=False),
        pa.field("region_kind", pa.string(), nullable=False),
        pa.field("intended_area_px", pa.int64()),
        pa.field("effective_area_px", pa.int64()),
        pa.field("n_samples", pa.int64(), nullable=False),
        pa.field("n_valid", pa.int64(), nullable=False),
        pa.field("flip_rate", pa.float64()),
        pa.field("metric_mean", pa.float64()),
    ],
    metadata=METRICS_SCHEMA_METADATA,
)

# One row per (gt_label, metric_name).
CLASS_METRICS_SCHEMA = pa.schema(
    [
        pa.field("gt_label", pa.int64(), nullable=False),
        pa.field("metric_name", pa.string(), nullable=False),
        pa.field("n_samples", pa.int64(), nullable=False),
        pa.field("flip_rate", pa.float64()),
        pa.field("metric_mean", pa.float64()),
    ],
    metadata=METRICS_SCHEMA_METADATA,
)

# One row per (sample_id, region_key, metric_name). region_geometry_ref is
# flattened into its four scalar fields (design §N5 "DebugViz의 직접 입력").
SPATIAL_PROFILE_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("region_key", pa.string(), nullable=False),
        pa.field("metric_name", pa.string(), nullable=False),
        pa.field("region_kind", pa.string(), nullable=False),
        pa.field("region_params_json", pa.string(), nullable=False),
        pa.field("ref", pa.string()),
        pa.field("ref_hash", pa.string()),
        pa.field("degradation", pa.float64()),
    ],
    metadata=METRICS_SCHEMA_METADATA,
)


def schema_for(name: str) -> pa.Schema:
    """Return a named N3 aggregate output schema.

    Args:
        name: One of ``item_metrics``, ``sample_metrics``, ``region_metrics``,
            ``class_metrics``, or ``spatial_profile``.

    Returns:
        The versioned PyArrow schema for the requested dataset.

    Raises:
        ValueError: If the schema name is unknown.
    """

    schemas = {
        "item_metrics": ITEM_METRICS_SCHEMA,
        "sample_metrics": SAMPLE_METRICS_SCHEMA,
        "region_metrics": REGION_METRICS_SCHEMA,
        "class_metrics": CLASS_METRICS_SCHEMA,
        "spatial_profile": SPATIAL_PROFILE_SCHEMA,
    }
    try:
        return schemas[name]
    except KeyError as error:
        raise ValueError(f"unknown metrics schema: {name}") from error
