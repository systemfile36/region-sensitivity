"""Unit coverage for the independent Captum reference workflow."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from captum.attr import FeatureAblation
from torch import nn

from experiments.reference_comparison.captum_baseline.workflow import (
    RAW_COLUMNS,
    RawMarginModel,
    RawStore,
    _attribution_values,
    _expected_paired_keys,
    grid_feature_mask,
    matched_control_masks,
    perturbation_baseline,
)

REFERENCE_DIR = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "reference_comparison"
    / "captum_baseline"
)


def test_reference_python_has_no_primary_implementation_imports() -> None:
    """Keep the reference implementation independent at the import boundary."""

    forbidden: list[tuple[Path, str]] = []
    for path in REFERENCE_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name == "ssat" or name.startswith("ssat.") or name.startswith(
                    "experiments.synthetic_shortcut"
                ):
                    forbidden.append((path, name))
    assert forbidden == []


def test_grid_and_matched_controls_are_area_matched_and_deterministic() -> None:
    grid = grid_feature_mask(32, 32, 4, 4)
    assert grid.shape == (1, 32, 32, 1)
    assert sorted(torch.unique(grid).tolist()) == list(range(16))
    assert all(int((grid == feature).sum()) == 64 for feature in range(16))

    first, first_keys = matched_control_masks(
        sample_ids=["a", "b"],
        target_row=0,
        target_col=0,
        control_index=1,
        height=32,
        width=32,
        rows=4,
        cols=4,
        global_seed=123,
    )
    second, second_keys = matched_control_masks(
        sample_ids=["a", "b"],
        target_row=0,
        target_col=0,
        control_index=1,
        height=32,
        width=32,
        rows=4,
        cols=4,
        global_seed=123,
    )
    assert torch.equal(first, second)
    assert first_keys == second_keys
    assert [int((row == 0).sum()) for row in first] == [64, 64]

    expected = _expected_paired_keys(
        "shortcut",
        "A",
        ["a", "b"],
        [
            (first_keys[0], "grid::grid/r0/c0", True, 1),
            (first_keys[1], "grid::grid/r0/c0", True, 1),
        ],
        "constant_fill",
        0,
    )
    assert len(expected) == 2


@pytest.mark.parametrize(
    ("operator", "params"),
    [
        ("constant_fill", {"value": 0.0}),
        ("mean_fill", {}),
        ("blur", {"sigma": 4.0}),
        ("gaussian_noise", {"sigma": 50.0}),
        ("patch_shuffle", {"patch_size": 4}),
    ],
)
def test_all_baselines_are_deterministic(operator: str, params: dict) -> None:
    images = torch.arange(2 * 32 * 32 * 3, dtype=torch.int64).remainder(256).to(torch.uint8)
    images = images.reshape(2, 32, 32, 3)
    kwargs = {
        "images": images,
        "operator": operator,
        "params": params,
        "channel_mean": [125.0, 122.0, 113.0],
        "global_seed": 7,
        "sample_ids": ["a", "b"],
        "seed_salt": 1,
    }
    first = perturbation_baseline(**kwargs)
    second = perturbation_baseline(**kwargs)
    assert first.shape == images.shape
    assert torch.equal(first, second)


class _ToyClassifier(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        signal = images.mean(dim=(1, 2, 3))
        return torch.stack((signal, -signal), dim=1)


def test_captum_feature_ablation_returns_margin_drop_per_region() -> None:
    model = RawMarginModel(_ToyClassifier(), (4, 4), [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    images = torch.full((1, 4, 4, 3), 255.0)
    labels = torch.tensor([0])
    baseline = torch.zeros_like(images)
    mask = grid_feature_mask(4, 4, 2, 2)
    clean, values = _attribution_values(
        FeatureAblation(model), images, labels, baseline, mask, (0, 1, 2, 3), 4
    )
    assert clean.shape == (1,)
    assert values.shape == (1, 4)
    assert np.all(values > 0)
    assert np.allclose(values, values[:, :1])


def test_raw_store_deduplicates_and_rejects_identity_mismatch(tmp_path: Path) -> None:
    store = RawStore(tmp_path / "run", {"config": "a"})
    row = dict.fromkeys(RAW_COLUMNS)
    row.update(
        {
            "item_key": "one",
            "is_control": False,
            "seed_salt": 0,
            "clean_margin": 1.0,
            "perturbed_margin": 0.0,
            "degradation": 1.0,
            "source_area": 64,
            "model_area": 3136,
            "status": "complete",
        }
    )
    assert store.append([row]) == 1
    assert store.append([row]) == 0
    store.flush()
    resumed = RawStore(tmp_path / "run", {"config": "a"})
    assert len(resumed.frame) == 1
    with pytest.raises(RuntimeError, match="different config"):
        RawStore(tmp_path / "run", {"config": "b"})
