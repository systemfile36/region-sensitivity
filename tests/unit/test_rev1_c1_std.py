"""Unit tests for C1-std: the Captum ImageNet workflow's SSAT-equivalent pieces and the summary helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("captum")

from experiments.revision_1.c1_captum import summarize_std
from experiments.revision_1.c1_captum.imagenet_workflow import analysis, workflow
from experiments.revision_1.c1_captum.measure_std import workflow_steps
from ssat.core.adapter.timm_adapter import TimmSquashPreprocessor
from ssat.core.perturb.perturbator import Perturbator
from ssat.core.region.mask_generators import GridMaskGenerator
from ssat.core.region.types import RegionSpec
from ssat.core.types import PerturbationOp, RegionKind

SHAPES = ((375, 500), (333, 457), (224, 224), (2112, 2816))
CHANNEL_MEAN = [123.675, 116.28, 103.53]


def _grid_mask(height: int, width: int, row: int, col: int) -> np.ndarray:
    spec = RegionSpec(region_id="grid_4x4", region_instance_id=f"grid_4x4/r{row}/c{col}", kind=RegionKind.GRID,
                      params={"rows": 4, "cols": 4, "row_index": row, "col_index": col}, ref=None, ref_hash=None)
    return GridMaskGenerator.__new__(GridMaskGenerator).get_mask(height, width, spec)


@pytest.mark.parametrize("shape", SHAPES)
def test_cell_bounds_match_ssat_grid_masks(shape: tuple[int, int]) -> None:
    height, width = shape
    for index, (r0, r1, c0, c1) in enumerate(workflow.cell_bounds(height, width, 4, 4)):
        expected = np.zeros((height, width), dtype=bool)
        expected[r0:r1, c0:c1] = True
        np.testing.assert_array_equal(_grid_mask(height, width, *divmod(index, 4)), expected)


@pytest.mark.parametrize("shape", SHAPES[:3])
def test_model_area_matches_ssat_squash_mask_transform(shape: tuple[int, int]) -> None:
    height, width = shape
    preprocessor = TimmSquashPreprocessor({"input_size": (3, 224, 224), "mean": (0.485, 0.456, 0.406),
                                           "std": (0.229, 0.224, 0.225), "interpolation": "bicubic"})
    rects = [*workflow.cell_bounds(height, width, 4, 4), (7, 7 + height // 4, 11, 11 + width // 4)]
    for r0, r1, c0, c1 in rects:
        mask = np.zeros((height, width), dtype=bool)
        mask[r0:r1, c0:c1] = True
        expected = int(preprocessor.transform_mask(mask).sum())
        assert workflow.model_space_extent(r0, r1, height, 224) * workflow.model_space_extent(c0, c1, width, 224) == expected


@pytest.mark.parametrize(("operator", "params", "ssat_params"), [
    ("mean_fill", {"value": CHANNEL_MEAN}, {"value": CHANNEL_MEAN}),
    ("blur", {"sigma": 3.0}, {"sigma": 3.0}),
])
def test_deterministic_candidates_composite_to_ssat_perturbations(operator: str, params: dict, ssat_params: dict) -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(61, 83, 3), dtype=np.uint8)
    candidate = workflow.perturbation_baseline(image, operator, params, global_seed=1, sample_id="s", seed_salt=0)
    r0, r1, c0, c1 = workflow.cell_bounds(61, 83, 4, 4)[5]
    mask = np.zeros((61, 83), dtype=bool)
    mask[r0:r1, c0:c1] = True
    composited = np.where(mask[..., None], candidate, image)
    expected = Perturbator().apply(image[None], mask, PerturbationOp(operator), ssat_params)[0]
    np.testing.assert_array_equal(composited, expected)


def test_noise_candidate_is_seeded_per_sample_and_salt() -> None:
    image = np.full((8, 8, 3), 100, dtype=np.uint8)
    draw = lambda sample, salt: workflow.perturbation_baseline(  # noqa: E731
        image, "gaussian_noise", {"sigma": 12.5}, global_seed=1, sample_id=sample, seed_salt=salt)
    np.testing.assert_array_equal(draw("a", 0), draw("a", 0))
    assert not np.array_equal(draw("a", 0), draw("a", 1))
    assert not np.array_equal(draw("a", 0), draw("b", 0))


def test_matched_controls_translate_each_cell_in_bounds() -> None:
    bounds = workflow.cell_bounds(333, 457, 4, 4)
    kwargs = dict(sample_id="x.JPEG", bounds=bounds, cols=4, controls_per_region=3, height=333, width=457, global_seed=7)
    controls = workflow.matched_controls(**kwargs)
    assert controls == workflow.matched_controls(**kwargs)
    assert len(controls) == 48
    for (target, index, row, col, cell_h, cell_w), (r0, r1, c0, c1) in zip(controls, np.repeat(bounds, 3, axis=0)):
        assert (cell_h, cell_w) == (r1 - r0, c1 - c0)
        assert 0 <= row <= 333 - cell_h and 0 <= col <= 457 - cell_w
    assert [control[:2] for control in controls[:4]] == [("grid::grid/r0/c0", 0), ("grid::grid/r0/c0", 1),
                                                          ("grid::grid/r0/c0", 2), ("grid::grid/r0/c1", 0)]


def test_margins() -> None:
    import torch

    logits = torch.tensor([[1.0, 3.0, 2.0], [5.0, 1.0, 0.0]])
    assert workflow.margins(logits, torch.tensor([1, 2])).tolist() == [1.0, -5.0]


def _raw(samples: int = 3) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(1)
    for sample in range(samples):
        for cell in range(4):
            target = f"grid::grid/r{cell // 2}/c{cell % 2}"
            for operator, salt in (("mean_fill", 0), ("gaussian_noise", 0), ("gaussian_noise", 1)):
                value = float(cell) + rng.normal(0, 0.1)
                rows.append({"sample_id": f"s{sample}", "gt_label": sample, "region_key": target, "target_region_key": target,
                             "is_control": False, "control_index": None, "perturbation": operator, "seed_salt": salt,
                             "degradation": value, "source_area": 100, "model_area": 3136})
                for index in range(3):
                    rows.append({"sample_id": f"s{sample}", "gt_label": sample, "region_key": f"control:{target}:{index}@0,0",
                                 "target_region_key": target, "is_control": True, "control_index": index,
                                 "perturbation": operator, "seed_salt": salt, "degradation": float(index) * 0.1,
                                 "source_area": 100, "model_area": 3080 + index})
    return pd.DataFrame(rows)


def test_region_profile_and_top_region_share() -> None:
    target = _raw().query("~is_control")
    profile = analysis._region_profile(target)
    pooled = profile[profile["perturbation"] == "all"].sort_values("rank")
    assert pooled["region_key"].tolist() == ["grid::grid/r1/c1", "grid::grid/r1/c0", "grid::grid/r0/c1", "grid::grid/r0/c0"]
    assert sorted(profile.loc[profile["perturbation"] == "mean_fill", "rank"]) == [1, 2, 3, 4]
    share = analysis._top_region_share(target)
    assert share.set_index("region_key")["share"].to_dict() == {"grid::grid/r1/c1": 1.0}


def test_control_comparison_and_area_sanity() -> None:
    raw = _raw()
    controls = analysis._control_comparison(raw, 1e-6)
    assert len(controls) == len(raw.query("~is_control"))
    first = controls.iloc[0]
    expected_std = float(np.std([0.0, 0.1, 0.2]))
    assert first["control_mean"] == pytest.approx(0.1) and first["control_std"] == pytest.approx(expected_std)
    assert first["z_vs_control"] == pytest.approx((first["degradation"] - 0.1) / expected_std)
    summary = analysis._control_summary(controls, 2.0)
    assert summary["share_z_defined"].eq(1.0).all()
    area = analysis._area_sanity(raw)
    assert area["pass"] and area["control_model_area"] == {"min": 3080, "median": 3081.0, "max": 3082}
    assert not analysis._area_sanity(raw.assign(source_area=np.where(raw["is_control"], 99, 100)))["pass"]


def test_measure_std_steps(tmp_path: Path) -> None:
    captum = workflow_steps("captum", tmp_path)
    assert [argv[2] for _, _, argv in captum] == ["audit", "analyze", "report"]
    ssat = workflow_steps("ssat", tmp_path)
    assert [argv[3] for _, _, argv in ssat] == ["run", "metrics", "analyze", "report"]
    assert [stage for _, stage, _ in ssat] == ["audit"] + ["analysis_report"] * 3
    assert ssat[0][2][-3:] == ["-o", str(tmp_path), "--yes"]
    with pytest.raises(ValueError):
        workflow_steps("other", tmp_path)


def _targets(noise: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows = [{"sample_id": f"s{sample}", "row": row, "col": col, "perturbation": operator,
             "degradation": row * 4 + col + sample * 0.01 + noise * rng.normal()}
            for sample in range(5) for row in range(4) for col in range(4) for operator in ("blur", "mean_fill")]
    return pd.DataFrame(rows)


def test_parity_of_identical_and_perturbed_targets() -> None:
    items = summarize_std.item_parity(_targets(), _targets())
    assert items["pearson"].tolist() == pytest.approx([1.0, 1.0]) and items["max_abs_diff"].eq(0.0).all()
    profile = summarize_std.profile_parity(_targets(), _targets())
    assert set(profile["perturbation"]) == {"all", "blur", "mean_fill"}
    assert profile["rank_order_identical"].all() and profile["top_cell_agreement_per_sample"].eq(1.0).all()
    noisy = summarize_std.profile_parity(_targets(), _targets(noise=5.0))
    assert noisy["profile_spearman"].lt(1.0).any()


def test_line_changes_and_meaningful_lines(tmp_path: Path) -> None:
    assert summarize_std.line_changes(["a", "b", "c"], ["a", "x", "c", "d"]) == {
        "old": 3, "new": 4, "unchanged": 2, "added_or_changed": 2, "removed": 1}
    source = tmp_path / "m.py"
    source.write_text('"""Doc."""\n\n# comment\nx = 1  # trailing\n\ndef f():\n    return x\n', encoding="utf-8")
    assert summarize_std.meaningful_lines(source) == ['"""Doc."""', "x = 1  # trailing", "def f():", "return x"]


def test_engineering_indicators() -> None:
    result = summarize_std.engineering()
    assert result["commands"] == {"captum": 3, "ssat": 4}
    assert result["ssat_config_changes_from_case_study"]["added_or_changed"] == 2
    assert result["captum_imagenet_workflow_sloc"] == result["captum_changes_from_synthetic_reference"]["total"]["new"]
    assert {row["captum_workflow"] for row in result["capabilities"]} >= {"custom", "not implemented"}
    assert "setup_time" in result and not any("effort" in key for key in result)


def test_registered_protocol_counts_match_the_setting() -> None:
    protocol = json.loads(summarize_std.PROTOCOL.read_text(encoding="utf-8"))
    samples, cells, controls, variants = 1000, 16, 3, 5
    assert protocol["setting"]["captum_raw_rows"] == samples * cells * (1 + controls) * variants
    assert protocol["setting"]["ssat_perturbed_items"] == samples * cells * (1 + controls) * variants
    assert protocol["setting"]["ssat_items_with_clean"] == protocol["setting"]["ssat_perturbed_items"] + samples
