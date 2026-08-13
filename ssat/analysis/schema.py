"""Versioned PyArrow contracts for the seven A2-A6 control/stability outputs.

Mirrors ``ssat.metrics.schema``'s shape (one ``pa.Schema`` constant per
dataset, a shared version tag in Arrow metadata) — independent of it:
``ANALYSIS_SCHEMA_VERSION`` tracks this module's on-disk format, not
``ssat.metrics.schema.METRICS_SCHEMA_VERSION``.

Each row type gets its own file/schema, one per (grain) — no two row types
sharing a schema, even when both belong to the same design-doc-level "axis"
(e.g. ``SeedStabilityRow`` and ``StrategyStabilityRow`` are different grains
and get separate schemas), following ``ssat.metrics.schema``'s own
established convention over the design document's higher-level file list
(IMPLE_PLAN_CONTROL_STABILITY_v1.md §5 단계8 결정 — user-confirmed "타입별
분리").

``AnchorKey``/``ConditionKey`` are flattened into their scalar components
(``sample_id``/``region_key``/``invert_mask``,
``perturb_op``/``perturb_params_hash``) rather than nested structs, matching
``ssat.metrics.schema``'s ``SPATIAL_PROFILE_SCHEMA`` flattening
``RegionGeometryRef``.
"""

from __future__ import annotations

import pyarrow as pa

ANALYSIS_SCHEMA_VERSION = "1.0.0"

ANALYSIS_SCHEMA_METADATA = {b"ssat.analysis_schema_version": ANALYSIS_SCHEMA_VERSION.encode("ascii")}

# One row per (target AnchorKey, ConditionKey, metric_name).
CONTROL_COMPARISON_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("region_key", pa.string(), nullable=False),
        pa.field("invert_mask", pa.bool_(), nullable=False),
        pa.field("perturb_op", pa.string(), nullable=False),
        pa.field("perturb_params_hash", pa.string(), nullable=False),
        pa.field("metric_name", pa.string(), nullable=False),
        pa.field("control_available", pa.string(), nullable=False),
        pa.field("area_matched", pa.string(), nullable=False),
        pa.field("control_mean", pa.float64()),
        pa.field("control_std", pa.float64()),
        pa.field("n_controls", pa.int64(), nullable=False),
        pa.field("excess", pa.float64()),
        pa.field("ratio", pa.float64()),
        pa.field("z_vs_control", pa.float64()),
    ],
    metadata=ANALYSIS_SCHEMA_METADATA,
)

# One row per (AnchorKey, ConditionKey, metric_name).
SEED_STABILITY_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("region_key", pa.string(), nullable=False),
        pa.field("invert_mask", pa.bool_(), nullable=False),
        pa.field("perturb_op", pa.string(), nullable=False),
        pa.field("perturb_params_hash", pa.string(), nullable=False),
        pa.field("metric_name", pa.string(), nullable=False),
        pa.field("seed_mean", pa.float64()),
        pa.field("seed_std", pa.float64()),
        pa.field("seed_cv", pa.float64()),
        pa.field("n_seeds", pa.int64(), nullable=False),
    ],
    metadata=ANALYSIS_SCHEMA_METADATA,
)

# One row per (AnchorKey, metric_name) — target anchors only.
STRATEGY_STABILITY_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("region_key", pa.string(), nullable=False),
        pa.field("invert_mask", pa.bool_(), nullable=False),
        pa.field("metric_name", pa.string(), nullable=False),
        pa.field("strategy_signs_json", pa.string(), nullable=False),
        pa.field("strategy_values_json", pa.string(), nullable=False),
        pa.field("sign_agreement_ratio", pa.float64(), nullable=False),
        pa.field("n_strategies", pa.int64(), nullable=False),
    ],
    metadata=ANALYSIS_SCHEMA_METADATA,
)

# One row per (op_a, op_b).
RANK_CORRELATION_SCHEMA = pa.schema(
    [
        pa.field("op_a", pa.string(), nullable=False),
        pa.field("op_b", pa.string(), nullable=False),
        pa.field("spearman", pa.float64()),
        pa.field("n_regions", pa.int64(), nullable=False),
        pa.field("spearman_excl_top1", pa.float64()),
        pa.field("scope", pa.string(), nullable=False),
    ],
    metadata=ANALYSIS_SCHEMA_METADATA,
)

# One row per perturb_op.
STRATEGY_PROFILE_SCHEMA = pa.schema(
    [
        pa.field("perturb_op", pa.string(), nullable=False),
        pa.field("preserves_statistics", pa.bool_(), nullable=False),
        pa.field("preserves_local_texture", pa.bool_(), nullable=False),
        pa.field("is_global_operation", pa.bool_(), nullable=False),
        pa.field("cluster_id", pa.int64()),
        pa.field("mean_corr_within", pa.float64()),
        pa.field("mean_corr_across", pa.float64()),
        pa.field("alignment", pa.string(), nullable=False),
        pa.field("mean_degradation_excl_top", pa.float64()),
        pa.field("sign_ratio_positive", pa.float64()),
        pa.field("n_anchors", pa.int64(), nullable=False),
    ],
    metadata=ANALYSIS_SCHEMA_METADATA,
)

# One row per (region_key, metric).
INTERVALS_SCHEMA = pa.schema(
    [
        pa.field("region_key", pa.string(), nullable=False),
        pa.field("metric", pa.string(), nullable=False),
        pa.field("point_estimate", pa.float64(), nullable=False),
        pa.field("ci_low", pa.float64(), nullable=False),
        pa.field("ci_high", pa.float64(), nullable=False),
        pa.field("ci_method", pa.string(), nullable=False),
        pa.field("n_bootstrap", pa.int64(), nullable=False),
        pa.field("excludes_zero", pa.bool_(), nullable=False),
    ],
    metadata=ANALYSIS_SCHEMA_METADATA,
)

# One row per (AnchorKey, metric_name).
RELIABILITY_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("region_key", pa.string(), nullable=False),
        pa.field("invert_mask", pa.bool_(), nullable=False),
        pa.field("metric_name", pa.string(), nullable=False),
        pa.field("sign_consistent", pa.string(), nullable=False),
        pa.field("exceeds_control", pa.string(), nullable=False),
        pa.field("seed_stable", pa.string(), nullable=False),
        pa.field("jitter_stable", pa.string(), nullable=False),
        pa.field("multi_strategy", pa.string(), nullable=False),
        pa.field("ci_excludes_zero", pa.string(), nullable=False),
        pa.field("area_matched", pa.string(), nullable=False),
        pa.field("reliability_grade", pa.string(), nullable=False),
        # ``pa.list_(pa.string())`` names its child field "item" by default,
        # but the Parquet list logical type round-trips it back as
        # "element" -- name it explicitly so a fresh in-memory schema
        # matches one just read from disk (both compared via
        # ``read_validated_table``'s ``equals(..., check_metadata=True)``).
        pa.field(
            "reliability_reasons",
            pa.list_(pa.field("element", pa.string(), nullable=False)),
            nullable=False,
        ),
    ],
    metadata=ANALYSIS_SCHEMA_METADATA,
)
