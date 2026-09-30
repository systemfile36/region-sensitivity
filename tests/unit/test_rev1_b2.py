"""Unit tests for experiments/revision_1/b2_ntu_semantic (annotation files, scores, alignment, figures)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.revision_1.b2_ntu_semantic import annotations, evaluate_alignment, group_breakdown, plot_figures, ssat_part_scores
from experiments.revision_1.b2_ntu_semantic.annotations import GROUPS
from ssat.report.assembler import _build_class_semantic_matrix, _sample_semantic_group_degradation

REGION_GROUPS = {"head": "head", "torso": "torso", "left_arm": "arms", "right_arm": "arms", "left_hand": "hands",
                 "right_hand": "hands", "left_leg": "legs", "right_leg": "legs", "upper_body": "upper_body",
                 "lower_body": "lower_body"}


def _fill(path: Path, ratings: np.ndarray) -> None:
    frame = annotations.read_sheet(path)
    for column, group in enumerate(GROUPS):
        frame[group] = [str(value) for value in ratings[:, column]]
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def test_class_rows_follow_the_official_numbering() -> None:
    rows = annotations.class_rows()
    assert len(rows) == 60 and rows[0]["action_id"] == "A001" and rows[0]["action_name"] == "drink water"
    assert rows[59] == {"label_id": 59, "action_id": "A060", "action_name": "walking apart from each other", "two_person": 1}
    assert sum(row["two_person"] for row in rows) == 11 and rows[48]["two_person"] == 0


def test_sheets_are_written_validated_and_never_overwritten(tmp_path: Path) -> None:
    annotations.write_files(["A"], tmp_path)
    path = annotations.sheet_path("A", tmp_path)
    errors, complete = annotations.validate_sheet(annotations.read_sheet(path))
    assert errors == [] and complete == 0
    ratings = np.random.default_rng(0).integers(0, 3, size=(60, 5))
    _fill(path, ratings)
    annotations.write_files(["A"], tmp_path)  # must keep the filled sheet
    loaded = annotations.load_ratings(path)
    assert (loaded.to_numpy() == ratings).all() and list(loaded.columns) == list(GROUPS)
    frame = annotations.read_sheet(path)
    frame.loc[3, "hands"] = "3"
    frame.loc[5, "action_name"] = "edited"
    errors, complete = annotations.validate_sheet(frame)
    assert complete == 59 and any("label 3" in error for error in errors) and any("classes.csv" in error for error in errors)
    with pytest.raises(ValueError):
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        annotations.load_ratings(path)


def test_scores_are_blocked_until_sheets_are_committed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    annotations.write_files(["A"], tmp_path)
    _fill(annotations.sheet_path("A", tmp_path), np.ones((60, 5), dtype=int))
    assert not annotations.committed_unchanged(annotations.sheet_path("A", tmp_path))
    with pytest.raises(PermissionError):
        annotations.require_blind_annotations(["A"], tmp_path)
    monkeypatch.setattr(annotations, "committed_unchanged", lambda path: True)
    assert annotations.require_blind_annotations(["A"], tmp_path)["A"].shape == (60, 5)


def _region_values(seed: int = 0, n_per_class: int = 3, n_classes: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label in range(n_classes):
        for index in range(n_per_class):
            sample = f"S{label:03d}_{index}"
            for region in REGION_GROUPS:
                rows.append({"sample_id": sample, "region_key": f"{region}::{region}/{sample}", "region_id": region,
                             "degradation": float(rng.normal()), "effective_area_px": float(rng.integers(100, 5000)),
                             "gt_label": label, "clean_correct": bool(index)})
    return pd.DataFrame(rows)


def test_class_matrix_matches_the_report_assembler() -> None:
    frame = _region_values()
    ours = ssat_part_scores.class_group_matrix(frame, REGION_GROUPS).set_index(["label_id", "group"])
    spatial = [SimpleNamespace(sample_id=row.sample_id, region_key=row.region_key, degradation=row.degradation)
               for row in frame.itertuples()]
    per_sample = _sample_semantic_group_degradation(spatial, REGION_GROUPS)
    report, excluded = _build_class_semantic_matrix(per_sample, dict(zip(frame["sample_id"], frame["gt_label"])))
    assert excluded == 0 and len(report) == len(ours)
    for row in report:
        assert ours.loc[(row.gt_label, row.semantic_group), "value"] == pytest.approx(row.mean_degradation)
        assert ours.loc[(row.gt_label, row.semantic_group), "n_samples"] == row.n_samples


def test_lift_area_residuals_and_score_rows() -> None:
    frame = _region_values()
    matrix = ssat_part_scores.class_group_matrix(frame, REGION_GROUPS)
    lifted = matrix.assign(value=ssat_part_scores.lift(matrix)).groupby("group")["value"]
    assert np.allclose(lifted.mean(), 0) and np.allclose(lifted.std(ddof=0), 1)
    frame.loc[0, "effective_area_px"] = 0.0
    residuals = ssat_part_scores.area_residuals(frame, REGION_GROUPS)
    assert np.isnan(residuals[0]) and residuals[frame["region_id"].isin(["upper_body", "lower_body"])].isna().all()
    used = residuals.notna()
    assert abs(np.corrcoef(np.log(frame.loc[used, "effective_area_px"]), residuals[used])[0, 1]) < 1e-9
    rows = ssat_part_scores.score_rows(frame, REGION_GROUPS)
    assert set(rows["score"]) == {"raw", "lift", "area_adjusted_lift"} and set(rows["population"]) == {"all", "clean_correct"}
    assert rows[(rows["population"] == "clean_correct") & (rows["score"] == "raw")]["n_samples"].max() == 2
    assert ssat_part_scores.region_id("left_hand::left_hand/S001") == "left_hand"


def test_weighted_kappa_and_auroc() -> None:
    assert evaluate_alignment.weighted_kappa([0, 1, 2, 2], [0, 2, 2, 1]) == pytest.approx(3 / 7)
    assert evaluate_alignment.weighted_kappa([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)
    assert np.isnan(evaluate_alignment.weighted_kappa([1, 1], [1, 1]))
    assert evaluate_alignment.auroc(np.array([3.0, 1.0]), np.array([1.0, 0.0])) == pytest.approx(0.875)
    assert np.isnan(evaluate_alignment.auroc(np.array([1.0]), np.array([])))
    masks = evaluate_alignment.PATTERNS
    assert (evaluate_alignment.pattern_index(masks) == np.arange(len(masks))).all()


def _aligned_case(seed: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ratings = rng.integers(0, 3, size=(60, 5))
    scores = ratings + rng.normal(0, 0.1, size=ratings.shape)
    return ratings >= 1, ratings == 2, scores


def test_statistics_detect_alignment_and_null() -> None:
    relevant, primary, scores = _aligned_case()
    stats = evaluate_alignment.statistics(relevant, primary, scores, n_permutations=500, n_bootstrap=500)
    eligible = relevant.any(axis=1) & ~relevant.all(axis=1)
    assert stats["n_classes_eligible"] == eligible.sum() and stats["mean_auroc"] == pytest.approx(1.0)
    assert stats["auroc_permutation_p"] < 0.01 and stats["top1_hit_rate"] == pytest.approx(1.0)
    assert stats["top1_chance"] == pytest.approx(primary[primary.any(axis=1)].mean())
    assert stats["n_excluded_all_relevant"] + stats["n_excluded_none_relevant"] + stats["n_classes_eligible"] == 60
    shuffled = scores[np.random.default_rng(2).permutation(60)]
    null = evaluate_alignment.statistics(relevant, primary, shuffled, n_permutations=500, n_bootstrap=500)
    assert null["auroc_permutation_p"] > 0.01 and 0.35 < null["mean_auroc"] < 0.65


def test_pipeline_end_to_end_on_synthetic_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    relevant, primary, scores = _aligned_case()
    ratings = relevant.astype(int) + primary.astype(int)
    sheets = {"A": pd.DataFrame(ratings, columns=GROUPS, index=pd.Index(range(60), name="label_id")),
              "B": pd.DataFrame(np.clip(ratings - (np.arange(300).reshape(60, 5) % 7 == 0), 0, 2), columns=GROUPS,
                                index=pd.Index(range(60), name="label_id"))}
    monkeypatch.setattr(evaluate_alignment, "require_blind_annotations", lambda names: {name: sheets[name] for name in names})
    monkeypatch.setattr(evaluate_alignment, "N_PERMUTATIONS", 200)
    monkeypatch.setattr(evaluate_alignment, "N_BOOTSTRAP", 200)
    rows = [{"run": run, "population": population, "score": score, "label_id": label, "group": group,
             "value": scores[label, column] + offset, "n_samples": 20}
            for run, offset in (("exact", 0.0), ("crop_free", 0.05)) for population in ssat_part_scores.POPULATIONS
            for score in ("raw", "lift", "area_adjusted_lift") for label in range(60) for column, group in enumerate(GROUPS)]
    pd.DataFrame(rows).to_csv(tmp_path / "class_group_scores.csv", index=False)
    assert evaluate_alignment.main(["--annotators", "A", "B", "--summary-dir", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "alignment_summary.json").read_text())
    assert summary["mean_auroc"] > 0.95 and len(summary["representative_classes"]["top"]) == 3
    robustness = pd.read_csv(tmp_path / "robustness.csv")
    assert set(robustness["variant"]) >= {"primary", "crop_free", "single_person_classes", "annotator_B_only"}
    assert robustness.set_index("variant").loc["single_person_classes", "n_classes_eligible"] <= 49
    agreement = pd.read_csv(tmp_path / "annotator_agreement.csv")
    assert len(agreement) == 6 and (agreement["weighted_kappa"] <= 1).all()
    assert plot_figures.main(["--summary-dir", str(tmp_path)]) == 0
    for name in ("fig_b2_alignment.pdf", "fig_b2_examples.pdf"):
        assert (tmp_path / name).stat().st_size > 0


def test_committed_check_ignores_line_endings(tmp_path: Path) -> None:
    assert annotations._normalized(b"a,b\r\n1,2\r\n") == annotations._normalized(b"a,b\n1,2\n")
    outside = tmp_path / "annotator_A.csv"
    outside.write_text("x\n")
    assert not annotations.committed_unchanged(outside)


def test_group_breakdown_rows() -> None:
    rating = pd.DataFrame([[2, 0, 0, 0, 0], [0, 0, 0, 0, 2], [0, 0, 1, 1, 0]], columns=GROUPS)
    matrix = pd.DataFrame([[3.0, 0, 0, 0, 0], [0, 0, 0, 0, 3.0], [0, 0, 1.0, 2.0, 0]], columns=GROUPS)
    rows = group_breakdown.group_rows(rating, matrix).set_index("group")
    assert rows.loc["head", "auroc_across_classes"] == 1.0 and rows.loc["head", "n_primary_classes"] == 1
    assert rows.loc["hands", "n_classes_top_group"] == 1 and np.isnan(rows.loc["torso", "auroc_across_classes"])
