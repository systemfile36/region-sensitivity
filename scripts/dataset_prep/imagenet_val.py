#!/usr/bin/env python3
"""Prepare a deterministic class-balanced ImageNet-1k validation subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from ssat.utils.io import sha256_file, write_json_atomic


def _rank(seed: int, relative_path: str) -> str:
    return hashlib.sha256(f"{seed}:{relative_path}".encode("utf-8")).hexdigest()


def load_synset_indices(path: Path) -> dict[str, int]:
    synsets = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            synsets.append(line.split(maxsplit=1)[0])
    if len(synsets) != 1000 or len(set(synsets)) != 1000:
        raise ValueError(f"expected 1,000 unique synsets in {path}, found {len(synsets)}")
    return {synset: index for index, synset in enumerate(synsets)}


def prepare_subset(
    val_root: Path,
    solution_csv: Path,
    synset_mapping: Path,
    output_annotation: Path,
    *,
    samples_per_class: int = 10,
    seed: int = 20260820,
) -> dict:
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    synset_indices = load_synset_indices(synset_mapping)
    by_class: dict[int, list[str]] = {index: [] for index in range(1000)}
    with solution_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"ImageId", "PredictionString"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"{solution_csv} must contain {sorted(required)}")
        for row in reader:
            image_id = row["ImageId"].strip()
            tokens = row["PredictionString"].split()
            if not image_id or not tokens:
                raise ValueError("solution CSV contains an empty image ID or prediction")
            synset = tokens[0]
            if synset not in synset_indices:
                raise ValueError(f"unknown validation synset {synset!r}")
            candidates = [
                val_root / image_id,
                val_root / f"{image_id}.JPEG",
                val_root / f"{image_id}.jpg",
                val_root / f"{image_id}.jpeg",
            ]
            image_path = next((path for path in candidates if path.is_file()), None)
            if image_path is None:
                raise FileNotFoundError(f"validation image not found for {image_id!r}")
            relative_path = image_path.relative_to(val_root).as_posix()
            by_class[synset_indices[synset]].append(relative_path)

    selected: list[tuple[str, int]] = []
    class_counts = {}
    for label in range(1000):
        candidates = by_class[label]
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"ImageNet class {label} has {len(candidates)} validation images; "
                f"need {samples_per_class}"
            )
        chosen = sorted(candidates, key=lambda path: (_rank(seed, path), path))[
            :samples_per_class
        ]
        selected.extend((path, label) for path in chosen)
        class_counts[str(label)] = len(chosen)

    output_annotation.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{path} {label}\n" for path, label in selected)
    output_annotation.write_text(text, encoding="utf-8")
    metadata = {
        "dataset": "ImageNet-1k validation",
        "selection": "sha256(seed:relative_path)",
        "seed": seed,
        "samples_per_class": samples_per_class,
        "num_classes": 1000,
        "num_samples": len(selected),
        "class_counts": class_counts,
        "source_solution_sha256": sha256_file(solution_csv),
        "source_synset_mapping_sha256": sha256_file(synset_mapping),
        "annotation_sha256": sha256_file(output_annotation),
    }
    metadata_path = output_annotation.with_suffix(".metadata.json")
    write_json_atomic(metadata_path, metadata)
    output_annotation.chmod(0o644)
    metadata_path.chmod(0o644)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--solution-csv", type=Path, required=True)
    parser.add_argument("--synset-mapping", type=Path, required=True)
    parser.add_argument("--output-annotation", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_subset(
        args.val_root.expanduser().resolve(strict=True),
        args.solution_csv.expanduser().resolve(strict=True),
        args.synset_mapping.expanduser().resolve(strict=True),
        args.output_annotation.expanduser().resolve(),
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    print(metadata)


if __name__ == "__main__":
    main()
