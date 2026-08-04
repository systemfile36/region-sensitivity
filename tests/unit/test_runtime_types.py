from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from ssat.core.adapter.types import AdapterSpec, RawOutput
from ssat.core.plan.hashing import compute_item_id
from ssat.core.plan.types import WorkChunk, WorkChunkMeta, WorkItem
from ssat.core.region.types import RegionSpec
from ssat.core.runtime.types import ItemMeta, PreparedChunk
from ssat.core.source.types import LoadedSample
from ssat.core.types import ItemStatus, PerturbationOp, RegionKind


def test_loaded_sample_enforces_thwc_uint8() -> None:
    array = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    sample = LoadedSample(array, "sample", array.shape, "a" * 64, gt_label=1)
    assert sample.original_shape == (1, 8, 8, 3)

    with pytest.raises(TypeError, match="uint8"):
        LoadedSample(array.astype(np.float32), "sample", array.shape, "a" * 64)


def test_resolved_explicit_region_requires_hash() -> None:
    with pytest.raises(ValueError, match="require ref and ref_hash"):
        RegionSpec("mask", RegionKind.EXPLICIT, ref="mask.png")


def test_work_item_params_are_immutable_and_hashable() -> None:
    region = RegionSpec("grid", RegionKind.GRID, params={"rows": 2})
    payload = {
        "sample_id": "sample",
        "region_spec": region,
        "perturb_op": PerturbationOp.BLUR,
        "perturb_params": {"sigma": 1.0},
        "invert_mask": False,
        "seed_salt": 0,
        "is_control": False,
    }
    item = WorkItem(item_id=compute_item_id(payload), **payload)
    assert compute_item_id(item) == item.item_id

    with pytest.raises(TypeError):
        item.perturb_params["sigma"] = 2.0
    with pytest.raises(FrozenInstanceError):
        item.sample_id = "changed"

    chunk = WorkChunk("c" * 64, "sample", (item,))
    assert chunk.items == (item,)

    with pytest.raises(ValueError, match="chunk_id"):
        WorkChunk("chunk-1", "sample", (item,))
    with pytest.raises(ValueError, match="item_id"):
        WorkChunkMeta("c" * 64, "sample", ("item-1",))


def test_prepared_chunk_aligns_arrays_and_metadata() -> None:
    arrays = np.zeros((2, 1, 8, 8, 3), dtype=np.uint8)
    metas = (ItemMeta("a" * 64), ItemMeta("b" * 64))
    prepared = PreparedChunk(arrays, metas)
    assert len(prepared.item_metas) == 2

    with pytest.raises(ValueError, match="aligned"):
        PreparedChunk(arrays, metas[:1])


def test_adapter_contract_is_logits_only() -> None:
    spec = AdapterSpec(model_id="dummy", deterministic=True)
    output = RawOutput(np.array([0.2, -0.1], dtype=np.float32))
    assert spec.output_kind == "logits"
    assert output.logits.shape == (2,)

    with pytest.raises(TypeError):
        RawOutput(np.array([1, 2], dtype=np.int64))


def test_failed_item_cannot_be_in_success_metadata() -> None:
    arrays = np.zeros((1, 1, 2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="successful items only"):
        PreparedChunk(arrays, (ItemMeta("a" * 64, status=ItemStatus.LOAD_FAILED),))
