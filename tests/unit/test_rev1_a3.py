"""Unit tests for experiments/revision_1/a3_multi_arch (run registry, configs, model inspection helpers)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.revision_1.a3_multi_arch import summarize
from experiments.revision_1.a3_multi_arch.compare_models import (
    KEYS,
    MODELS,
    RunData,
    analyze_protocol,
    common_pattern_rows,
    pair_row,
    populations,
)
from experiments.revision_1.a3_multi_arch.inspect_models import cell_type, grid_masks
from experiments.revision_1.common.agreement import GRADE_ORDER
from experiments.revision_1.common.runs import A3_DIR, A3_RUNS, BASELINE_RUNS, RunRef, get_run
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


def _run_data(model: str, protocol: str, seed: int, *, n_samples: int = 30, center_boost: float = 1.0) -> RunData:
    """Random profile with larger values in the four center cells, plus matching reliability rows."""

    rng = np.random.default_rng(seed)
    keys = sorted(f"grid_4x4::grid_4x4/r{r}/c{c}" for r in range(4) for c in range(4))
    samples = pd.Index([f"s{i:03d}" for i in range(n_samples)], name="sample_id")
    values = rng.normal(size=(n_samples, 16))
    for column, key in enumerate(keys):
        if cell_type(int(key[-4]), int(key[-1])) == "center":
            values[:, column] += center_boost
    profile = pd.DataFrame(values, index=samples, columns=pd.Index(keys, name="region_key"))
    index = pd.MultiIndex.from_product([samples, keys], names=KEYS)
    reliability = pd.DataFrame({
        "reliability_grade": rng.choice(["high", "low", "unreliable"], size=len(index)),
        "sign_consistent": rng.choice(["true", "false"], size=len(index)),
        "exceeds_control": rng.choice(["true", "false"], size=len(index)),
        "seed_stable": "true", "multi_strategy": "true", "ci_excludes_zero": "true", "area_matched": "true",
    }, index=index)
    run = RunRef.co_located(f"{model}_{protocol}", Path("/nonexistent"), dataset="imagenet", model=model, protocol=protocol, n_controls=3)
    return RunData(run, profile, reliability, pd.Series(rng.random(n_samples) < 0.8, index=samples))


@pytest.mark.parametrize(("protocol", "n_pairs"), [("crop_free", 6), ("exact", 3)])
def test_analyze_protocol_follows_the_comparison_rule(protocol: str, n_pairs: int) -> None:
    data = {model: _run_data(model, protocol, seed) for seed, model in enumerate(MODELS)}
    out = analyze_protocol(protocol, data)
    cross = pd.DataFrame(out["cross"])
    pairs = set(zip(cross["model_a"], cross["model_b"]))
    assert len(pairs) == n_pairs
    assert ("deit_small" in set(cross["model_a"]) | set(cross["model_b"])) == (protocol == "crop_free")
    assert set(cross.loc[cross["pair_type"] == "yardstick", "model_b"]) == {"mobilenetv2_100"}
    assert set(cross["population"]) == {"all", "common_correct", "pair_correct"}

    profile = pd.DataFrame(out["profile"])
    for _, group in profile.groupby(["model", "population"]):
        assert sorted(group["rank_in_model"]) == list(range(1, 17))
        assert group["top_share"].sum() == pytest.approx(1.0)
        assert group["z_in_model"].mean() == pytest.approx(0.0)
        assert group["z_in_model"].std(ddof=0) == pytest.approx(1.0)
    grades = pd.DataFrame(out["grades"])
    assert grades[list(GRADE_ORDER)].sum(axis=1).to_numpy() == pytest.approx(1.0)
    common = grades[(grades["population"] == "common_correct") & (grades["cell_type"] == "all")]
    assert common["n_samples"].nunique() == 1


def test_pair_row_of_a_run_with_itself_is_perfect_agreement() -> None:
    data = _run_data("convnext_tiny", "crop_free", 0)
    row = pair_row(data, data, data.profile.index, {})
    assert row["profile_spearman"] == pytest.approx(1.0)
    assert row["sample_spearman_p50"] == pytest.approx(1.0)
    assert row["top1_agreement"] == 1.0 and row["top3_overlap_mean"] == 1.0
    assert row["top1_kappa"] == pytest.approx(1.0)
    assert row["agreement"] == 1.0


def test_common_pattern_detects_center_leading_profile() -> None:
    strong = _run_data("mobilenetv2_050", "exact", 1, n_samples=200, center_boost=3.0)
    flat = _run_data("mobilenetv2_050", "exact", 1, n_samples=200, center_boost=-3.0)
    assert common_pattern_rows(strong, strong.profile.index, {})["center_cells_top4"]
    assert not common_pattern_rows(flat, flat.profile.index, {})["center_cells_top4"]
    assert common_pattern_rows(strong, strong.profile.index, {})["top_share_center"] > 0.9


def test_populations_reject_misaligned_runs() -> None:
    data = {"a": _run_data("a", "exact", 0), "b": _run_data("b", "exact", 1, n_samples=20)}
    with pytest.raises(ValueError, match="different samples"):
        populations(data)


def test_summarize_writes_figures(tmp_path: Path) -> None:
    for protocol in ("crop_free", "exact"):
        out = analyze_protocol(protocol, {model: _run_data(model, protocol, seed) for seed, model in enumerate(MODELS)})
        for key, name in (("profile", "region_profile"), ("grades", "grade_distribution"), ("cross", "cross_model")):
            path = tmp_path / f"{name}.csv"
            pd.DataFrame(out[key]).to_csv(path, index=False, mode="a", header=not path.exists())
    assert summarize.main(["--summary-dir", str(tmp_path)]) == 0
    for name in ("fig_a3_profiles.pdf", "fig_a3_grades.pdf", "fig_a3_similarity.pdf"):
        assert (tmp_path / name).stat().st_size > 0


def test_cell_types() -> None:
    types = Counter(cell_type(row, col) for row in range(4) for col in range(4))
    assert types == {"corner": 4, "edge": 8, "center": 4}
    assert cell_type(1, 2) == "center" and cell_type(0, 3) == "corner" and cell_type(2, 0) == "edge"
