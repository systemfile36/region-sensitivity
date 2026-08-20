#!/usr/bin/env python3
"""Compute Q5's plain top-1 accuracy numbers for the L3 synthetic-shortcut experiment.

Implements Q5: M_shortcut's accuracy drop when evaluated on C (unpatched)
should be larger than M_normal's. Q5 is a plain generalization check -- it
asks how much each model's accuracy changes when the patch it may or may
not depend on is removed -- and has no region or perturbation dimension.
It is therefore computed here as a direct forward pass over each
checkpoint, entirely independent of run_audit.py's ssat pipeline (which
exists to answer Q1-Q4 and the fill-strategy sensitivity check, not Q5).

Both models are evaluated on both A_test (patched) and C_test (clean) so
evaluate.py can compare each model's own (A accuracy - C accuracy) drop
against the other's, as the pre-registered Q5 criterion requires
(thresholds.json's q5_min_margin_points).

Run as: python3 experiments/synthetic_shortcut/evaluate_accuracy.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from train import ManifestImageDataset  # reuse the manifest-backed Dataset, not duplicate it
from torch.utils.data import DataLoader

NUM_CLASSES = 10


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one accuracy-evaluation run."""

    parser = argparse.ArgumentParser(description="Evaluate M_shortcut/M_normal top-1 accuracy.")
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoints"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--preprocessing",
        choices=("preset", "crop_free"),
        default="preset",
        help="Must match the --preprocessing the checkpoints were trained with (see train.py).",
    )
    return parser.parse_args()


def _load_model(checkpoint_path: Path, device: torch.device):
    """Reconstruct the trained squeezenet1_0 architecture and load its weights.

    weights_only=True (torch.load's restricted unpickler) matches
    ssat.core.adapter.checkpoint.load_state_dict_checkpoint's own loading
    discipline for locally-produced, trusted checkpoints.
    """

    from torchvision.models import squeezenet1_0

    model = squeezenet1_0(weights=None, num_classes=NUM_CLASSES)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    return model.eval().to(device)


@torch.no_grad()
def _top1_accuracy(model, manifest_path: Path, transform, device: torch.device, batch_size: int) -> float:
    """Return the fraction of correctly classified images in one manifest."""

    dataset = ManifestImageDataset(manifest_path, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += images.size(0)
    return correct / total


def main() -> int:
    """Evaluate both models on both A_test and C_test, and save the results."""

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Same preprocessing the checkpoint was trained under, and the same one
    # ssat's TorchvisionAdapter applies at audit time (see train.py's module
    # docstring for why this must not be re-derived by hand) -- otherwise Q5
    # would silently evaluate under a third, mismatched input distribution.
    if args.preprocessing == "crop_free":
        from common import build_crop_free_transform

        transform = build_crop_free_transform()
    else:
        from torchvision.models import SqueezeNet1_0_Weights

        transform = SqueezeNet1_0_Weights.DEFAULT.transforms()

    manifests_dir = args.data_dir / "manifests"
    accuracy: dict[str, dict[str, float]] = {}
    for model_name in ("shortcut", "normal"):
        model = _load_model(args.checkpoint_dir / f"m_{model_name}.pt", device)
        accuracy[model_name] = {
            dataset_name: _top1_accuracy(
                model,
                manifests_dir / f"{dataset_name}_test.json",
                transform,
                device,
                args.batch_size,
            )
            for dataset_name in ("A", "C")
        }
        print(f"[{model_name}] A={accuracy[model_name]['A']:.4f} C={accuracy[model_name]['C']:.4f}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.results_dir / "accuracy.json"
    output_path.write_text(json.dumps(accuracy, indent=2), encoding="utf-8")
    print(f"saved accuracy results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
