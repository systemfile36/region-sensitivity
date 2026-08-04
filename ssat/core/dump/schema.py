"""Versioned PyArrow contracts for clean, perturbed, and resume-index data."""

from __future__ import annotations

import pyarrow as pa

from ssat.core.types import SCHEMA_VERSION

SCHEMA_METADATA = {b"ssat.schema_version": SCHEMA_VERSION.encode("ascii")}
UTC_TIMESTAMP = pa.timestamp("us", tz="UTC")
LOGITS = pa.list_(pa.float32())
SHAPE_4D = pa.list_(pa.int32(), 4)

# One row per source sample, independent of perturbation enumeration.
CLEAN_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("content_hash", pa.string()),
        pa.field("gt_label", pa.int64()),
        pa.field("status", pa.string(), nullable=False),
        pa.field("logits", LOGITS),
        pa.field("original_shape", SHAPE_4D),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("written_at", UTC_TIMESTAMP, nullable=False),
    ],
    metadata=SCHEMA_METADATA,
)

# One row per WorkItem, including intended and adapter-effective region areas.
PERTURBED_SCHEMA = pa.schema(
    [
        pa.field("item_id", pa.string(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("region_id", pa.string(), nullable=False),
        pa.field("region_kind", pa.string(), nullable=False),
        pa.field("region_params_json", pa.string(), nullable=False),
        pa.field("intended_area_px", pa.int64()),
        pa.field("intended_area_ratio", pa.float64()),
        pa.field("generator_kind", pa.string()),
        pa.field("generator_version", pa.string()),
        pa.field("generator_confidence", pa.float64()),
        pa.field("effective_area_px", pa.int64()),
        pa.field("effective_area_available", pa.bool_(), nullable=False),
        pa.field("perturb_op", pa.string(), nullable=False),
        pa.field("perturb_params_json", pa.string(), nullable=False),
        pa.field("invert_mask", pa.bool_(), nullable=False),
        pa.field("is_control", pa.bool_(), nullable=False),
        pa.field("seed_used", pa.uint64(), nullable=False),
        pa.field("logits", LOGITS),
        pa.field("written_at", UTC_TIMESTAMP, nullable=False),
    ],
    metadata=SCHEMA_METADATA,
)

# Lightweight completion records consumed by ResumeIndex.
INDEX_SCHEMA = pa.schema(
    [
        pa.field("item_id", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("chunk_file", pa.string(), nullable=False),
        pa.field("written_at", UTC_TIMESTAMP, nullable=False),
    ],
    metadata=SCHEMA_METADATA,
)


def schema_for(name: str) -> pa.Schema:
    """Return a named raw-dump schema.

    Args:
        name: One of ``clean``, ``perturbed``, or ``index``.

    Returns:
        The versioned PyArrow schema for the requested dataset.

    Raises:
        ValueError: If the schema name is unknown.
    """

    schemas = {
        "clean": CLEAN_SCHEMA,
        "perturbed": PERTURBED_SCHEMA,
        "index": INDEX_SCHEMA,
    }
    try:
        return schemas[name]
    except KeyError as error:
        raise ValueError(f"unknown dump schema: {name}") from error
