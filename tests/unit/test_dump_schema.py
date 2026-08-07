from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ssat.core.dump.schema import (
    CLEAN_SCHEMA,
    INDEX_SCHEMA,
    PERTURBED_SCHEMA,
    schema_for,
)


def records_for(name: str) -> list[dict]:
    now = datetime(2026, 8, 4, tzinfo=timezone.utc)
    if name == "clean":
        return [
            {
                "sample_id": "sample-1",
                "content_hash": "a" * 64,
                "gt_label": 2,
                "status": "ok",
                "logits": [0.1, -0.2, 1.5],
                "original_shape": [1, 64, 64, 3],
                "model_id": "dummy",
                "written_at": now,
            }
        ]
    if name == "perturbed":
        return [
            {
                "item_id": "b" * 64,
                "sample_id": "sample-1",
                "status": "ok",
                "region_id": "grid",
                "region_instance_id": "grid/r0/c0",
                "region_kind": "grid",
                "region_params_json": '{"cols":2,"rows":2}',
                "intended_area_px": 1024,
                "intended_area_ratio": 0.25,
                "generator_kind": "grid",
                "generator_version": "1",
                "generator_confidence": None,
                "effective_area_px": None,
                "effective_area_available": False,
                "perturb_op": "blur",
                "perturb_params_json": '{"sigma":"1.000000000000"}',
                "invert_mask": False,
                "is_control": False,
                "seed_used": "00000000000000000000000000000007",
                "logits": [0.0, 1.0],
                "written_at": now,
            }
        ]
    return [
        {
            "item_id": "b" * 64,
            "status": "ok",
            "chunk_file": "perturbed/chunk_00001.parquet",
            "written_at": now,
        }
    ]


@pytest.mark.parametrize(
    ("name", "schema"),
    [("clean", CLEAN_SCHEMA), ("perturbed", PERTURBED_SCHEMA), ("index", INDEX_SCHEMA)],
)
def test_parquet_schema_round_trip(tmp_path, name: str, schema: pa.Schema) -> None:
    path = tmp_path / f"{name}.parquet"
    table = pa.Table.from_pylist(records_for(name), schema=schema)
    pq.write_table(table, path)
    restored = pq.read_table(path)

    assert restored.schema == schema
    assert restored.to_pylist() == table.to_pylist()
    assert restored.schema.metadata[b"ssat.schema_version"] == b"1.0.0"


def test_perturbed_schema_contains_effective_area_contract() -> None:
    assert "region_instance_id" in PERTURBED_SCHEMA.names
    assert "intended_area_ratio" in PERTURBED_SCHEMA.names
    assert "generator_confidence" in PERTURBED_SCHEMA.names
    assert "effective_area_available" in PERTURBED_SCHEMA.names


def test_schema_lookup_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown dump schema"):
        schema_for("missing")
