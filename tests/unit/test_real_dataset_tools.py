from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.real_dataset_case_study.parity.compare_logits import compare
from scripts.dataset_prep.imagenet_val import prepare_subset
from scripts.dataset_prep.ntu_rgb_d import select_per_class


def test_ntu_class_selection_is_balanced_and_deterministic() -> None:
    frame = pd.DataFrame(
        [
            {"label": label, "video_key": f"c{label}-{index}", "path": f"{index}.avi"}
            for label in range(3)
            for index in range(5)
        ]
    )
    first = select_per_class(frame, count=2, seed=7)
    second = select_per_class(frame.sample(frac=1, random_state=9), count=2, seed=7)
    assert first["video_key"].tolist() == second["video_key"].tolist()
    assert first.groupby("label").size().tolist() == [2, 2, 2]


def test_imagenet_prep_writes_one_sample_for_each_of_1000_classes(tmp_path: Path) -> None:
    val_root = tmp_path / "val"
    val_root.mkdir()
    mapping = tmp_path / "LOC_synset_mapping.txt"
    mapping.write_text(
        "".join(f"n{index:08d} class {index}\n" for index in range(1000)),
        encoding="utf-8",
    )
    solution = tmp_path / "LOC_val_solution.csv"
    with solution.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["ImageId", "PredictionString"])
        writer.writeheader()
        for index in range(1000):
            image_id = f"ILSVRC2012_val_{index:08d}"
            (val_root / f"{image_id}.JPEG").touch()
            writer.writerow(
                {"ImageId": image_id, "PredictionString": f"n{index:08d} 1 2 3 4"}
            )
    annotation = tmp_path / "subset.txt"
    metadata = prepare_subset(
        val_root, solution, mapping, annotation, samples_per_class=1, seed=11
    )
    assert metadata["num_samples"] == 1000
    assert len(annotation.read_text(encoding="utf-8").splitlines()) == 1000


def test_logit_parity_comparison_checks_ids_top1_and_tolerances(tmp_path: Path) -> None:
    reference = tmp_path / "reference.npz"
    native = tmp_path / "native.npz"
    ids = np.asarray(["a", "b"], dtype=np.str_)
    logits = np.asarray([[1.0, 2.0], [3.0, 1.0]], dtype=np.float32)
    np.savez(reference, sample_ids=ids, logits=logits)
    np.savez(native, sample_ids=ids, logits=logits + 1e-6)
    assert compare(reference, native, atol=1e-4, rtol=1e-4)["passed"]
    np.savez(native, sample_ids=ids[::-1], logits=logits)
    with pytest.raises(ValueError, match="sample_ids"):
        compare(reference, native, atol=1e-4, rtol=1e-4)
