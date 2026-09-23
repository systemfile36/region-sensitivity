"""Unit tests for experiments/revision_1/common (agreement, subsets, loaders)."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synthetic_dump_builder import (
    ENVIRONMENT,
    build_resolved_config,
    clean_record,
    compute_and_save_metrics,
    perturbed_record,
)

from experiments.revision_1.common.agreement import (
    GRADE_ORDER,
    auroc,
    cohen_kappa,
    grade_agreement,
    spearman,
    transition_matrix,
    weighted_kappa,
)
from experiments.revision_1.common.loading import load_clean, load_item_values
from experiments.revision_1.common.runs import BASELINE_RUNS, K1_PARITY_RUNS, RunRef, k1_counterpart
from experiments.revision_1.common.subsets import (
    class_balanced_subset,
    hash_rank,
    top_n_subset,
    write_annotation,
)
from ssat.analysis.reader import AnalysisReader
from ssat.core.config.schema import ResolvedRegionConfig
from ssat.core.dump import DumpWriter
from ssat.core.types import PerturbationOp, RegionKind
from ssat.metrics.builtin_metrics.continuous import GtLogitDrop, MarginDrop
from ssat.metrics.registry import MetricRegistry
from ssat.utils.io import sha256_file

# --- agreement -------------------------------------------------------------


def _grades(values: list[str], start: int = 0) -> pd.Series:
    return pd.Series(values, index=[f"a{index}" for index in range(start, start + len(values))])


def test_grade_agreement_counts_ordinal_steps_and_direction() -> None:
    before = _grades(["high", "high", "low", "unreliable", "moderate"])
    after = _grades(["high", "low", "moderate", "high", "moderate"])
    summary = grade_agreement(before, after)
    assert summary["n"] == 5
    assert summary["agreement"] == pytest.approx(2 / 5)
    assert summary["unchanged"] == pytest.approx(2 / 5)
    assert summary["delta_1"] == pytest.approx(1 / 5)  # low -> moderate
    assert summary["delta_ge2"] == pytest.approx(2 / 5)  # high -> low, unreliable -> high
    assert summary["up"] == pytest.approx(2 / 5)
    assert summary["down"] == pytest.approx(1 / 5)


def test_grade_agreement_aligns_on_index_and_rejects_unknown_grades() -> None:
    before = _grades(["high", "low"])
    # Same labels in reversed row order: identical once aligned on the anchor index.
    after = pd.Series(["low", "high"], index=["a1", "a0"])
    assert grade_agreement(before, after)["agreement"] == 1.0
    with pytest.raises(ValueError, match="unknown grade"):
        grade_agreement(before, _grades(["HIGH", "low"]))


def test_transition_matrix_keeps_every_grade_on_both_axes() -> None:
    before = _grades(["high", "high", "low"])
    after = _grades(["high", "low", "low"])
    matrix = transition_matrix(before, after)
    assert list(matrix.index) == list(GRADE_ORDER)
    assert list(matrix.columns) == list(GRADE_ORDER)
    assert matrix.loc["high", "high"] == 1
    assert matrix.loc["high", "low"] == 1
    assert matrix.loc["low", "low"] == 1
    assert int(matrix.to_numpy().sum()) == 3


def test_weighted_kappa_matches_hand_computation() -> None:
    a = [0, 0, 1, 2, 3, 3]
    b = [0, 1, 1, 3, 3, 2]
    # Direct evaluation of 1 - sum(w_ij * O_ij) / sum(w_ij * E_ij) with
    # disagreement weights |i - j| / (k - 1).
    k = 4
    observed = np.zeros((k, k))
    for i, j in zip(a, b):
        observed[i, j] += 1 / len(a)
    expected = np.outer(observed.sum(1), observed.sum(0))
    disagreement = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    reference = 1 - (disagreement * observed).sum() / (disagreement * expected).sum()
    assert weighted_kappa(a, b, k) == pytest.approx(reference)
    assert weighted_kappa([1, 2, 3], [1, 2, 3], 4) == pytest.approx(1.0)
    assert weighted_kappa([2, 2], [2, 2], 4) is None


def test_cohen_kappa_known_value_and_degenerate_case() -> None:
    # p_o = 0.5; p_e = 0.5 * 0.5 + 0.5 * 0.5 = 0.5 -> kappa = 0.
    assert cohen_kappa(["x", "x", "y", "y"], ["x", "y", "x", "y"]) == pytest.approx(0.0)
    assert cohen_kappa(["x", "y"], ["x", "y"]) == pytest.approx(1.0)
    assert cohen_kappa(["x", "x"], ["x", "x"]) is None


def test_spearman_matches_closed_form_without_ties() -> None:
    x = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
    y = np.array([2.0, 4.0, 5.0, 1.0, 3.0])
    rank_x = x.argsort().argsort() + 1
    rank_y = y.argsort().argsort() + 1
    n = len(x)
    reference = 1 - 6 * ((rank_x - rank_y) ** 2).sum() / (n * (n**2 - 1))
    assert spearman(x, y) == pytest.approx(reference)


def test_spearman_uses_average_ranks_drops_nonfinite_and_rejects_constant() -> None:
    assert spearman([1, 1, 2], [1, 1, 2]) == pytest.approx(1.0)
    assert spearman([1, 2, np.nan, 3], [3, 2, 100, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert spearman([1.0], [2.0]) is None


def test_auroc_matches_pairwise_definition() -> None:
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 5, size=40).astype(float)
    labels = rng.random(40) < 0.4
    positives = scores[labels]
    negatives = scores[~labels]
    pairwise = np.mean(
        [1.0 if p > q else 0.5 if p == q else 0.0 for p, q in itertools.product(positives, negatives)]
    )
    assert auroc(scores, labels) == pytest.approx(pairwise)
    assert auroc([3, 2, 1], [True, False, False]) == pytest.approx(1.0)
    assert auroc([1, 1], [True, False]) == pytest.approx(0.5)
    assert auroc([1, 2], [True, True]) is None


# --- subsets ---------------------------------------------------------------


def _annotation(tmp_path: Path, n_classes: int = 3, per_class: int = 4) -> Path:
    path = tmp_path / "annotation.txt"
    lines = [(f"img_{label}_{index}.JPEG", label) for label in range(n_classes) for index in range(per_class)]
    write_annotation(lines, path)
    return path


def test_hash_rank_is_input_order_independent_and_salt_dependent() -> None:
    ids = [f"id{index}" for index in range(20)]
    assert hash_rank(ids, "s") == hash_rank(list(reversed(ids)), "s")
    assert hash_rank(ids, "s") != hash_rank(ids, "t")
    with pytest.raises(ValueError, match="unique"):
        hash_rank(["a", "a"], "s")


def test_class_balanced_subset_is_nested_across_sizes(tmp_path: Path) -> None:
    annotation = _annotation(tmp_path)
    one = class_balanced_subset(annotation, 1, "rev1-test")
    two = class_balanced_subset(annotation, 2, "rev1-test")
    assert [label for _, label in two] == [0, 0, 1, 1, 2, 2]
    assert set(one) <= set(two)
    with pytest.raises(ValueError, match="has only"):
        class_balanced_subset(annotation, 5, "rev1-test")


def test_top_n_subset_is_a_prefix_and_write_annotation_returns_file_hash(tmp_path: Path) -> None:
    annotation = _annotation(tmp_path)
    small = top_n_subset(annotation, 3, "rev1-test")
    large = top_n_subset(annotation, 7, "rev1-test")
    assert large[:3] == small
    digest = write_annotation(small, tmp_path / "out" / "subset.txt")
    assert digest == sha256_file(tmp_path / "out" / "subset.txt")
    assert (tmp_path / "out" / "subset.txt").read_text().splitlines()[0] == f"{small[0][0]} {small[0][1]}"


# --- runs ------------------------------------------------------------------


def test_run_registry_covers_the_eight_baselines_and_four_k1_runs() -> None:
    assert len(BASELINE_RUNS) == 8
    assert {run.dataset for run in BASELINE_RUNS.values()} == {"synthetic", "imagenet", "ntu60"}
    assert all(run.n_controls == 1 for run in K1_PARITY_RUNS.values())
    k3 = BASELINE_RUNS["imagenet_mnv2_050_exact_k3"]
    assert k1_counterpart(k3).name == "imagenet_mnv2_050_exact_k1"
    assert k1_counterpart(k3).protocol == k3.protocol


# --- loading ---------------------------------------------------------------


def _retried_dump(tmp_path: Path) -> RunRef:
    """Two samples with a target and a control item each; one item is rewritten in a later chunk."""

    config = build_resolved_config(
        tmp_path,
        regions=(
            ResolvedRegionConfig(region_id="grid", kind=RegionKind.GRID, params={"rows": 1, "cols": 1}),
            ResolvedRegionConfig(region_id="control", kind=RegionKind.RANDOM_AREA_MATCH, params={}),
        ),
    )
    records = []
    for sample_index, sample_id in enumerate(("s0", "s1")):
        for offset, (region_id, kind, is_control) in enumerate(
            (("grid", RegionKind.GRID, False), ("control", RegionKind.RANDOM_AREA_MATCH, True))
        ):
            records.append(
                perturbed_record(
                    sample_index * 2 + offset,
                    sample_id=sample_id,
                    region_id=region_id,
                    region_instance_id=f"{region_id}/{sample_id}",
                    region_kind=kind,
                    is_control=is_control,
                    logits=np.array([1.0 + sample_index + offset, 0.5, 0.0]),
                    perturb_op=PerturbationOp.CONSTANT_FILL,
                )
            )
    retried = perturbed_record(
        0,
        sample_id="s0",
        region_id="grid",
        region_instance_id="grid/s0",
        logits=np.array([-2.0, 0.5, 0.0]),
    )
    dump = tmp_path / "dump"
    with DumpWriter(dump, config, code_version="test-code", mode="create", environment=ENVIRONMENT) as writer:
        writer.write_clean_many(
            [clean_record(sample_id, logits=np.array([3.0, 0.0, 0.0])) for sample_id in ("s0", "s1")]
        )
        writer.write_perturbed_many(records)
        writer.flush()
        writer.write_perturbed(retried)
    registry = MetricRegistry()
    registry.register(GtLogitDrop())
    registry.register(MarginDrop())
    compute_and_save_metrics(dump, config, dump / "metrics", registry=registry, primary_metric="margin_drop")
    return RunRef.co_located(
        "fixture", dump, dataset="synthetic", model="tiny", protocol="crop_free", n_controls=1
    )


def test_load_item_values_matches_analysis_reader_including_retried_items(tmp_path: Path) -> None:
    run = _retried_dump(tmp_path)
    reference = AnalysisReader(run.dump, run.metrics).item_values()
    for metrics in (("margin_drop",), ("margin_drop", "gt_logit_drop")):
        ours = load_item_values(run, metrics)
        expected = reference[reference["metric_name"].isin(metrics)]
        keys = ["item_id", "metric_name"]
        pd.testing.assert_frame_equal(
            ours.sort_values(keys).reset_index(drop=True),
            expected.sort_values(keys).reset_index(drop=True),
        )
    retried = load_item_values(run, ("gt_logit_drop",)).set_index("item_id")
    # The rewritten row (gt logit -2.0) wins over the first write (1.0).
    assert retried.loc[f"{0:064x}", "degradation"] == pytest.approx(3.0 - (-2.0))


def test_load_item_values_rejects_metrics_from_another_dump(tmp_path: Path) -> None:
    run = _retried_dump(tmp_path)
    manifest = run.dump / "run_manifest.json"
    manifest.write_text(manifest.read_text().replace('"test-code"', '"other-code"'))
    with pytest.raises(ValueError, match="does not match"):
        load_item_values(run)


def test_load_clean_returns_one_row_per_sample(tmp_path: Path) -> None:
    run = _retried_dump(tmp_path)
    clean = load_clean(run, with_logits=True)
    assert sorted(clean["sample_id"]) == ["s0", "s1"]
    assert list(clean.loc[0, "logits"]) == pytest.approx([3.0, 0.0, 0.0])
