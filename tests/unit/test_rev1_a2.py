"""Parity and unit tests for the A2 control tensor and ablation helpers."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synthetic_dump_builder import (
    build_resolved_config,
    clean_record,
    compute_and_save_analysis,
    compute_and_save_metrics,
    perturbed_record,
    write_dump,
)

from experiments.revision_1.a1_threshold.build_features import build_features
from experiments.revision_1.a2_control_count.build_control_tensor import (
    ControlTensor,
    build_tensor,
    stored_parity,
    subset_stats,
)
from experiments.revision_1.a2_control_count import summarize
from experiments.revision_1.a2_control_count.run_ablation import (
    draw_subsets,
    official_grades,
    replicate_disagreement,
    top1_cells,
)
from experiments.revision_1.b1_preprocessing.eff_area_controls import ea_control_stats, eligible_items
from experiments.revision_1.common.grade_engine import (
    GradeParams,
    compute_scales,
    grade_anchors,
    grade_labels,
    with_control_stats,
)
from experiments.revision_1.common.loading import load_item_values
from experiments.revision_1.common.runs import RunRef
from ssat.core.config.schema import ResolvedRegionConfig
from ssat.core.types import PerturbationOp, RegionKind
from ssat.metrics.builtin_metrics.continuous import MarginDrop
from ssat.metrics.registry import MetricRegistry

OPS = (PerturbationOp.CONSTANT_FILL, PerturbationOp.MEAN_FILL, PerturbationOp.BLUR)
N_CONTROLS = 12
N_SEEDS = 2


def _cell_params(row: int, col: int) -> dict[str, int]:
    return {"col_index": col, "cols": 2, "row_index": row, "rows": 2}


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> RunRef:
    """A 4-sample, 2x2-grid, 3-operator, 2-seed dump with 12 controls per target (per op and seed)."""

    tmp_path = tmp_path_factory.mktemp("a2")
    rng = np.random.default_rng(11)
    config = build_resolved_config(
        tmp_path,
        regions=(ResolvedRegionConfig(region_id="grid", kind=RegionKind.GRID, params={"rows": 2, "cols": 2}),),
    )
    clean_logits = {f"s{index}": np.array([3.0, 1.0, 0.5]) + rng.normal(0, 0.2, 3) for index in range(4)}
    records = []
    counter = itertools.count()
    for sample_id, (row, col), op in itertools.product(clean_logits, itertools.product(range(2), range(2)), OPS):
        instance = f"grid/r{row}/c{col}"
        effect = rng.choice([0.05, 0.4, 1.5])
        for seed in range(N_SEEDS):
            records.append(perturbed_record(
                next(counter), sample_id=sample_id, region_id="grid", region_instance_id=instance,
                region_params=_cell_params(row, col),
                logits=clean_logits[sample_id] - np.array([effect, 0.0, 0.0]) + rng.normal(0, 0.1, 3),
                perturb_op=op, seed_used=seed + 1,
            ))
            for control_index in range(N_CONTROLS):
                records.append(perturbed_record(
                    next(counter), sample_id=sample_id, region_id="control:grid:0",
                    region_instance_id=f"control:{instance}:0:{control_index}",
                    region_kind=RegionKind.RANDOM_AREA_MATCH,
                    region_params={"control_index": control_index, "control_request_index": 0,
                                   "target_region": {"kind": "grid", "params": _cell_params(row, col),
                                                     "region_id": "grid", "region_instance_id": instance}},
                    logits=clean_logits[sample_id] - rng.normal(0.3, 0.4, 3),
                    perturb_op=op, is_control=True, seed_used=100 * (control_index + 1) + seed,
                ))
    dump, metrics, analysis = tmp_path / "dump", tmp_path / "metrics", tmp_path / "analysis"
    write_dump(
        dump, config,
        clean_records=tuple(clean_record(sample_id, logits=logits) for sample_id, logits in clean_logits.items()),
        perturbed_records=tuple(records),
    )
    registry = MetricRegistry()
    registry.register(MarginDrop())
    compute_and_save_metrics(dump, config, metrics, registry=registry, primary_metric="margin_drop")
    compute_and_save_analysis(dump, metrics, analysis, primary_metric="margin_drop")
    return RunRef(name="fixture", dump=dump, metrics=metrics, analysis=analysis, dataset="synthetic",
                  model="fixture", protocol="crop_free", n_controls=N_CONTROLS)


@pytest.fixture(scope="module")
def tensor(run: RunRef) -> ControlTensor:
    return build_tensor(load_item_values(run, ("margin_drop",)), "margin_drop")


def test_tensor_reproduces_stored_control_comparison(run: RunRef, tensor: ControlTensor) -> None:
    assert tensor.controls.shape == (16, 3, N_CONTROLS)
    assert tensor.stat_order.tolist() == sorted(range(N_CONTROLS), key=str)
    parity = stored_parity(tensor, run.analysis, "margin_drop")
    assert parity["passed"], parity


def test_tensor_round_trips_through_npz(tensor: ControlTensor, tmp_path: Path) -> None:
    tensor.save(tmp_path / "t.npz")
    loaded = ControlTensor.load(tmp_path / "t.npz")
    np.testing.assert_array_equal(loaded.controls, tensor.controls)
    np.testing.assert_array_equal(loaded.sample_id, tensor.sample_id)
    assert loaded.conditions == tensor.conditions


def _engine_grades(features: pd.DataFrame, tensor: ControlTensor, indices: range) -> np.ndarray:
    mean, std0, n = subset_stats(tensor, indices)
    graded = grade_anchors(with_control_stats(features, tensor.target, mean, std0, n), GradeParams(),
                           scales=compute_scales(features))
    return grade_labels(graded["grade"])


def test_full_prefix_grades_match_stored_reliability(run: RunRef, tensor: ControlTensor) -> None:
    features, _ = build_features(run, "margin_drop")
    np.testing.assert_array_equal(_engine_grades(features, tensor, range(N_CONTROLS)),
                                  features["stored__reliability_grade"].to_numpy())


@pytest.mark.parametrize("k", [2, 3, 5])
def test_prefix_grades_match_official_path(run: RunRef, tensor: ControlTensor, k: int) -> None:
    features, _ = build_features(run, "margin_drop")
    expected = official_grades(load_item_values(run, ("margin_drop",)), run.analysis, "margin_drop", k)
    ours = pd.Series(_engine_grades(features, tensor, range(k)),
                     index=pd.MultiIndex.from_frame(features[["sample_id", "region_key", "invert_mask"]]))
    assert expected.sort_index().equals(ours.sort_index())


def test_small_k_changes_some_grades(run: RunRef, tensor: ControlTensor) -> None:
    features, _ = build_features(run, "margin_drop")
    assert (_engine_grades(features, tensor, range(2)) != _engine_grades(features, tensor, range(N_CONTROLS))).any()


def test_draw_subsets() -> None:
    assert draw_subsets("prefix", 3, 20, 5, 7) == [(0, 1, 2)]
    assert draw_subsets("global_random", 20, 20, 5, 7) == [tuple(range(20))]
    random = draw_subsets("global_random", 5, 20, 4, 20260923)
    assert len(random) == 4 and all(len(set(s)) == 5 and list(s) == sorted(s) for s in random)
    assert random == draw_subsets("global_random", 5, 20, 4, 20260923)
    assert random[0] == tuple(sorted(np.random.default_rng(20260923).choice(20, 5, replace=False).tolist()))
    with pytest.raises(ValueError):
        draw_subsets("other", 3, 20, 1, 0)


def test_replicate_disagreement() -> None:
    grades = np.array([[3, 0, 1], [3, 1, 1], [3, 0, 0], [3, 1, 0]])
    pairwise, vs_mode = replicate_disagreement(grades)
    np.testing.assert_allclose(pairwise, [0.0, 0.5, 0.5])
    np.testing.assert_allclose(vs_mode, [0.0, 0.5, 0.5])


def test_top1_cells_break_ties_by_cell_order() -> None:
    sample_ids = pd.Series(["a", "a", "a", "b", "b", "b"])
    cells = pd.Series(["r0/c1", "r0/c0", "r1/c0", "r0/c0", "r0/c1", "r1/c0"])
    top1 = top1_cells(np.array([0.5, 0.5, 0.1, 0.0, 0.2, 0.9]), sample_ids, cells)
    assert top1.to_dict() == {"a": "r0/c0", "b": "r1/c0"}


def test_ea_stats_match_full_set_when_every_item_is_eligible(run: RunRef, tensor: ControlTensor) -> None:
    items = eligible_items(load_item_values(run, ("margin_drop",)))
    assert items["eligible"].all()
    keys = pd.MultiIndex.from_arrays([tensor.sample_id, tensor.region_key, tensor.invert_mask])
    anchor_pos = pd.Series(np.arange(len(keys)), index=keys).reindex(
        pd.MultiIndex.from_frame(items[["sample_id", "region_key", "invert_mask"]])).to_numpy()
    condition_pos = {condition: index for index, condition in enumerate(tensor.conditions)}
    c_pos = np.array([condition_pos[key] for key in zip(items["perturb_op"], items["perturb_params_hash"])])
    mean, std0, n = ea_control_stats(anchor_pos, c_pos, items["control_index"].to_numpy(),
                                     items["degradation"].to_numpy(float), items["eligible"].to_numpy(),
                                     tensor.controls.shape)
    full_mean, full_std0, _ = subset_stats(tensor, range(N_CONTROLS))
    assert (n == N_CONTROLS).all()
    np.testing.assert_allclose(mean, full_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(std0, full_std0, rtol=0, atol=1e-12)


def test_ea_stats_require_three_eligible_controls() -> None:
    # One anchor, one condition, four controls with two items each.
    anchor = np.zeros(8, dtype=np.int64)
    condition = np.zeros(8, dtype=np.int64)
    control = np.repeat(np.arange(4), 2)
    values = np.array([1.0, 3.0, 2.0, 2.0, 5.0, 7.0, 9.0, 9.0])
    eligible = np.array([True, False, True, True, True, True, False, False])
    mean, std0, n = ea_control_stats(anchor, condition, control, values, eligible, (1, 1, 4))
    assert n[0, 0] == 3
    np.testing.assert_allclose(mean[0, 0], np.mean([1.0, 2.0, 6.0]))
    np.testing.assert_allclose(std0[0, 0], np.std([1.0, 2.0, 6.0]))
    eligible[4:6] = False
    mean, _, n = ea_control_stats(anchor, condition, control, values, eligible, (1, 1, 4))
    assert n[0, 0] == 2 and np.isnan(mean[0, 0])


def test_ablate_run_end_to_end(run: RunRef, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import experiments.revision_1.a2_control_count.build_control_tensor as tensor_module
    import experiments.revision_1.a2_control_count.run_ablation as ablation_module

    monkeypatch.setattr(ablation_module, "FEATURE_DIR", tmp_path / "features")
    monkeypatch.setattr(ablation_module, "TENSOR_DIR", tmp_path / "tensors")
    monkeypatch.setattr(ablation_module, "LOG_PATH", tmp_path / "missing.jsonl")
    monkeypatch.setattr(ablation_module, "N_REPLICATES", 5)
    tensor_module.build(run, "margin_drop", tmp_path / "tensors")
    protocol = json.loads(ablation_module.PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["ablation"]["k_values"] = [2, 3, 5, N_CONTROLS]
    protocol["ablation"]["subset_schemes"]["global_random"] = protocol["ablation"]["subset_schemes"][
        "global_random"].replace("R = 200;", "R = 5;")

    tables, parity = ablation_module.ablate_run(run, protocol, "margin_drop", check_official=True)

    assert parity["k20_prefix_vs_reliability_parquet"]["n_grade_mismatch"] == 0
    assert parity["prefix_vs_official_path"]["k3"]["n_grade_mismatch"] == 0
    grade_rows = pd.DataFrame(tables["grade_vs_k"])
    reference = grade_rows[(grade_rows["k"] == N_CONTROLS) & (grade_rows["cell_type"] == "all")]
    assert (reference["agreement_mean"] == 1.0).all()
    assert set(grade_rows["n_replicates"]) == {1, 5}
    for table in ("control_mean_convergence", "effect_convergence", "ranking_vs_k", "transitions_vs_k",
                  "replicate_variability", "marginal_gain", "ddof_effect", "synthetic_gt_vs_k"):
        assert tables[table], table
    transitions = pd.DataFrame(tables["transitions_vs_k"])
    assert np.allclose(transitions.groupby(["scheme", "k", "ddof"])["share"].sum(), 1.0)

    summary = tmp_path / "summary"
    summary.mkdir()
    for table in ("control_mean_convergence", "grade_vs_k", "replicate_variability"):
        pd.DataFrame(tables[table]).to_csv(summary / f"{table}.csv", index=False)
    assert summarize.main(["--summary-dir", str(summary)]) == 0
    assert (summary / "fig_a2_flip.pdf").stat().st_size > 0


def test_with_control_stats_validates_shape(run: RunRef, tensor: ControlTensor) -> None:
    features, _ = build_features(run, "margin_drop")
    mean, std0, n = subset_stats(tensor, range(3))
    replaced = with_control_stats(features, tensor.target, mean, std0, n)
    assert (replaced["cond0__n"] == 3).all()
    with pytest.raises(ValueError):
        with_control_stats(features, tensor.target[:, :2], mean[:, :2], std0[:, :2], n)
