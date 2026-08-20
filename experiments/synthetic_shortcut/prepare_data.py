#!/usr/bin/env python3
"""Build the A/B/C image sets and manifests for the L3 synthetic-shortcut experiment.

Downloads CIFAR-10, class-balance-samples a train and a test pool with a
fixed seed, then renders three physically distinct PNG sets per pool:

    A (contaminated): every image gets its ground-truth class's solid-color
        patch painted into the fixed top-left grid cell.
    B (irrelevant, test pool only): the same class-color patches, but each
        placed at a *uniformly random* grid cell per image, so the patch no
        longer correlates with any single region. B is only used as an
        auxiliary audit control, never for training.
    C (clean): the original CIFAR-10 pixels, unmodified.

A/B/C for one pool are rendered from the *same* underlying photographs (only
the pixels inside one grid cell differ), so any difference measured between
them is attributable to the patch alone, not to different image content.

Why the patch fills a whole grid cell instead of a small fixed-size dot in
the corner: ssat's TorchvisionAdapter always resizes to the model's default
weights preset (Resize+CenterCrop) before inference, and CenterCrop trims a
symmetric border off every image (see run_audit.py's module docstring for
the exact numbers). A patch confined to a few corner pixels could be
partially or fully cropped away; filling an entire, generously-sized grid
cell keeps most of the patch inside the retained center region.

Run as: python3 experiments/synthetic_shortcut/prepare_data.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from common import (
    GRID_COLS,
    GRID_ROWS,
    NUM_CLASSES,
    PATCH_COL,
    PATCH_ROW,
    ManifestSample,
    draw_patch,
    write_manifest,
)
from PIL import Image

DEFAULT_SEED = 42
DEFAULT_TRAIN_PER_CLASS = 3000
DEFAULT_TEST_PER_CLASS = 200
DEFAULT_AUDIT_PER_CLASS = 20


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one data-preparation run."""

    parser = argparse.ArgumentParser(
        description="Build CIFAR-10-based A/B/C datasets for the synthetic-shortcut experiment."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="destination for the CIFAR-10 cache, rendered PNGs, and manifests",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="fixed sampling seed")
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=DEFAULT_TRAIN_PER_CLASS,
        help="training images sampled per class (default: %(default)s)",
    )
    parser.add_argument(
        "--test-per-class",
        type=int,
        default=DEFAULT_TEST_PER_CLASS,
        help="held-out (accuracy-eval) images sampled per class (default: %(default)s)",
    )
    parser.add_argument(
        "--audit-per-class",
        type=int,
        default=DEFAULT_AUDIT_PER_CLASS,
        help="region-audit subset size per class, taken from the test pool (default: %(default)s)",
    )
    return parser.parse_args()


def _sample_indices_per_class(
    targets: np.ndarray, per_class: int, rng: np.random.Generator
) -> list[int]:
    """Draw a class-balanced, reproducible subset of dataset indices.

    Indices are grouped by class and sorted within each class, so that later
    "first N per class" slicing (used to build the smaller audit subset from
    the test pool) is a deterministic, order-stable operation rather than a
    second independent sample.
    """

    indices: list[int] = []
    for class_index in range(NUM_CLASSES):
        class_indices = np.flatnonzero(targets == class_index)
        if len(class_indices) < per_class:
            raise ValueError(
                f"class {class_index} has only {len(class_indices)} images, "
                f"but {per_class} were requested"
            )
        chosen = rng.choice(class_indices, size=per_class, replace=False)
        chosen.sort()
        indices.extend(int(value) for value in chosen)
    return indices


def _save_png(path: Path, array: np.ndarray) -> None:
    """Write one RGB image array to disk, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def _build_pool(
    *,
    split: str,
    dataset,
    indices: list[int],
    targets: np.ndarray,
    images_dir: Path,
    rng: np.random.Generator | None,
) -> tuple[list[ManifestSample], list[ManifestSample], list[ManifestSample] | None, np.ndarray]:
    """Render A/C (and, for the test pool, B) PNGs for one sampled index pool.

    Args:
        split: "train" or "test", used for sample_id prefixes and paths.
        dataset: A torchvision CIFAR10 dataset (used for its .data array).
        indices: Sampled dataset indices, grouped by class (see
            _sample_indices_per_class).
        targets: Full per-image label array for ``dataset``.
        images_dir: Root directory PNGs are written under.
        rng: Random generator for B's per-image cell placement. Train pools
            never build B, so this may be None there.

    Returns:
        (a_samples, c_samples, b_samples_or_none, pixel_sum) where pixel_sum
        is the running per-channel pixel sum of the *A* images (used by the
        caller to compute mean_fill's dataset_stats -- see the module-level
        note on why A's mean, not C's, is what mean_fill should use).
    """

    a_samples: list[ManifestSample] = []
    c_samples: list[ManifestSample] = []
    b_samples: list[ManifestSample] | None = [] if rng is not None else None
    channel_pixel_sum = np.zeros(3, dtype=np.float64)
    all_cells = [(row, col) for row in range(GRID_ROWS) for col in range(GRID_COLS)]

    for position, index in enumerate(indices):
        image = dataset.data[index]
        label = int(targets[index])
        sample_id = f"{split}-{label:02d}-{position:05d}"

        c_path = images_dir / split / "C" / f"{sample_id}.png"
        _save_png(c_path, image)
        c_samples.append(ManifestSample(sample_id, c_path, label))

        a_image = draw_patch(image, label, row=PATCH_ROW, col=PATCH_COL)
        a_path = images_dir / split / "A" / f"{sample_id}.png"
        _save_png(a_path, a_image)
        a_samples.append(ManifestSample(sample_id, a_path, label))
        channel_pixel_sum += a_image.reshape(-1, 3).sum(axis=0)

        if rng is not None and b_samples is not None:
            random_row, random_col = all_cells[int(rng.integers(len(all_cells)))]
            b_image = draw_patch(image, label, row=random_row, col=random_col)
            b_path = images_dir / split / "B" / f"{sample_id}.png"
            _save_png(b_path, b_image)
            b_samples.append(ManifestSample(sample_id, b_path, label))

    return a_samples, c_samples, b_samples, channel_pixel_sum


def main() -> int:
    """Download CIFAR-10, render A/B/C, and write manifests + dataset_stats."""

    args = parse_args()
    # Local import: torchvision's dataset module is heavy and only this
    # script needs it (train.py/evaluate_accuracy.py read pre-rendered PNGs
    # via ssat's own image_manifest source instead).
    from torchvision.datasets import CIFAR10

    rng = np.random.default_rng(args.seed)
    cifar_root = args.data_dir / "cifar10"
    train_dataset = CIFAR10(root=str(cifar_root), train=True, download=True)
    test_dataset = CIFAR10(root=str(cifar_root), train=False, download=True)
    train_targets = np.asarray(train_dataset.targets)
    test_targets = np.asarray(test_dataset.targets)

    train_indices = _sample_indices_per_class(train_targets, args.train_per_class, rng)
    test_indices = _sample_indices_per_class(test_targets, args.test_per_class, rng)

    images_dir = args.data_dir / "images"
    manifests_dir = args.data_dir / "manifests"

    a_train, c_train, _, train_pixel_sum = _build_pool(
        split="train",
        dataset=train_dataset,
        indices=train_indices,
        targets=train_targets,
        images_dir=images_dir,
        rng=None,  # B is a test-only, audit-only control.
    )
    write_manifest(manifests_dir / "A_train.json", a_train)
    write_manifest(manifests_dir / "C_train.json", c_train)

    # mean_fill (core PerturbationOp.MEAN_FILL) needs the *contaminated*
    # dataset's channel mean: that is the distribution M_shortcut was
    # actually trained and audited on, so it is the mean that makes
    # mean_fill a meaningful "replace with an uninformative average" probe
    # for that model. Using C's mean here would introduce a small, avoidable
    # mismatch between the perturbation and the audited model's input space.
    channel_mean = (train_pixel_sum / (len(train_indices) * 32 * 32)).tolist()
    stats_path = args.data_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps({"channel_mean": channel_mean}, indent=2), encoding="utf-8")

    a_test, c_test, b_test, _ = _build_pool(
        split="test",
        dataset=test_dataset,
        indices=test_indices,
        targets=test_targets,
        images_dir=images_dir,
        rng=rng,
    )
    write_manifest(manifests_dir / "A_test.json", a_test)
    write_manifest(manifests_dir / "C_test.json", c_test)
    assert b_test is not None
    write_manifest(manifests_dir / "B_test.json", b_test)

    # The region-level audits (Q1-Q4, section 3.5) run one forward pass per
    # image *per grid cell* -- 17x the cost of a plain accuracy check, per
    # fill strategy. Slicing the front of each class's already-sorted block
    # keeps the audit set a strict, reproducible subset of the larger
    # accuracy-eval set instead of drawing an independent sample that could
    # happen to look different from it.
    audit_ids = {
        sample_id
        for label in range(NUM_CLASSES)
        for sample_id in sorted(s.sample_id for s in a_test if s.gt_label == label)[
            : args.audit_per_class
        ]
    }
    write_manifest(manifests_dir / "A_audit.json", [s for s in a_test if s.sample_id in audit_ids])
    write_manifest(manifests_dir / "B_audit.json", [s for s in b_test if s.sample_id in audit_ids])

    print(f"wrote manifests to {manifests_dir}")
    print(f"train: A/C = {len(a_train)} images each; test: A/B/C = {len(a_test)} images each")
    print(f"audit subset: {len(audit_ids)} images; channel_mean = {channel_mean}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
