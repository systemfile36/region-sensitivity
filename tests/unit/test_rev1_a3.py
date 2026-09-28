"""Unit tests for experiments/revision_1/a3_multi_arch (run registry, configs, model inspection helpers)."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest
import yaml

from experiments.revision_1.a3_multi_arch.inspect_models import cell_type, grid_masks
from experiments.revision_1.common.runs import A3_DIR, A3_RUNS, BASELINE_RUNS, get_run
from ssat.core.region.mask_generators import GridMaskGenerator
from ssat.core.region.types import RegionSpec
from ssat.core.types import RegionKind


def test_registry_matches_matrix_and_protocol() -> None:
    matrix = json.loads((A3_DIR / "matrix.json").read_text(encoding="utf-8"))
    protocol = json.loads((A3_DIR / "protocol.json").read_text(encoding="utf-8"))
    assert [run["name"] for run in matrix["runs"]] == list(protocol["runs"])
    assert set(A3_RUNS) == set(protocol["runs"])
    for entry in matrix["runs"]:
        run = get_run(entry["name"])
        assert run.config == (A3_DIR / entry["config"]).resolve()
        assert protocol["runs"][run.name]["minimum_accuracy"] == entry["minimum_accuracy"]
        assert (run.protocol == "exact") == (entry["minimum_accuracy"] is not None)


@pytest.mark.parametrize("name", sorted(A3_RUNS))
def test_config_differs_from_mnv2_050_only_in_model_name(name: str) -> None:
    run = A3_RUNS[name]
    baseline = BASELINE_RUNS[f"imagenet_mnv2_050_{run.protocol}_k3"]
    config = yaml.safe_load(run.config.read_text(encoding="utf-8"))
    reference = yaml.safe_load(baseline.config.read_text(encoding="utf-8"))
    for section in (config, reference):
        section["adapter"].pop("model_name")
    for key in ("root", "annotation_file"):
        assert (run.config.parent / config["source"].pop(key)).resolve() == (baseline.config.parent / reference["source"].pop(key)).resolve()
    assert config == reference
    assert yaml.safe_load(run.config.read_text(encoding="utf-8"))["adapter"]["model_name"].startswith(run.model.split("_")[0])


@pytest.mark.parametrize(("height", "width"), [(375, 500), (500, 333), (224, 224), (17, 30)])
def test_grid_masks_match_ssat_grid_generator(height: int, width: int) -> None:
    generator = GridMaskGenerator(None)  # type: ignore[arg-type]
    masks = grid_masks(height, width)
    for row in range(4):
        for col in range(4):
            spec = RegionSpec(region_id="grid_4x4", region_instance_id=f"grid_4x4/r{row}/c{col}", kind=RegionKind.GRID,
                              params={"rows": 4, "cols": 4, "row_index": row, "col_index": col})
            np.testing.assert_array_equal(masks[row * 4 + col], generator.get_mask(height, width, spec))
    assert masks.sum(0).max() == 1 and masks.any(0).all()


def test_cell_types() -> None:
    types = Counter(cell_type(row, col) for row in range(4) for col in range(4))
    assert types == {"corner": 4, "edge": 8, "center": 4}
    assert cell_type(1, 2) == "center" and cell_type(0, 3) == "corner" and cell_type(2, 0) == "edge"
