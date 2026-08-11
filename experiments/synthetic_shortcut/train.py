#!/usr/bin/env python3
"""Train M_shortcut (on A) or M_normal (on C) for the L3 synthetic-shortcut experiment.

Implements docs/STAGE9_SYNTHETIC_SHORTCUT_DESIGN_v1.md section 2: a
from-scratch (no pretrained weights) torchvision squeezenet1_0, trained with
a standard CIFAR-style recipe (SGD + cosine annealing).

Two things here are non-negotiable given how the tool being validated works,
not just "typical" choices:

1. No RandomCrop/RandomHorizontalFlip (or any other spatially-random
   augmentation). The entire experiment's premise is that the synthetic
   patch sits at one *fixed* pixel location the model can learn to key off
   of. Cropping or flipping during training would move or mirror that
   location per-sample, destroying the fixed-location shortcut before the
   model ever gets a chance to learn it. Only deterministic normalization is
   applied.

2. The preprocessing transform is pulled directly from
   ``torchvision.models.SqueezeNet1_0_Weights.DEFAULT.transforms()`` instead
   of being hand-written. ssat's TorchvisionAdapter uses this exact object
   internally for inference, *even when weights=None*
   (ssat/core/adapter/torchvision_adapter.py: "preprocessing_weights =
   resolved_weights or weights_enum.DEFAULT"), and that choice is not
   configurable from the audit YAML. If training used a different resize or
   normalization, the audited model would see two different input
   distributions -- one at train time, one at audit time -- as a confound.
   Calling the same torchvision API here removes that confound by
   construction: any spatial trimming (e.g. CenterCrop's fixed-size border
   trim) automatically happens identically on both sides.

A third point, discovered only after a full 40-epoch run (the earlier
smoke test's 2-3 epochs were too short to expose it): squeezenet1_0's
Fire modules are plain Conv2d+ReLU with no batch normalization, and
training at a high lr can push several layers' pre-activations
permanently negative within a few steps -- every ReLU in that path then
outputs exactly 0 for every input, and since ReLU'(x<0)=0, no gradient
ever reaches those weights again to revive them. Once enough of the
network is dead this way, the model degenerates into a constant
function (equal logits for every input), which is why a collapsed run
shows loss pinned at exactly ln(10)=2.3026 and acc=0.1000 forever, not
a NaN. Gradient clipping (see _train_one_epoch below) only bounds how
far a single step can move weights; it does not stop the network from
drifting into this dead region.

A linear LR warmup (see main()) was tried first, on the theory that
only the initial jump to a high lr was unsafe. It was not sufficient by
itself: a 6-epoch verification run with a 3-epoch warmup to lr=0.1
trained normally through the warmup (acc climbing past 17%), then
collapsed to the exact ln(10)/0.10 dead state in the very first epoch
after warmup handed off to the full lr=0.1 cosine schedule -- proof
that lr=0.1 is unstable for this architecture on a sustained basis, not
just as a first step. DEFAULT_LR was therefore lowered from design
section 2's original 0.1 to 0.01 (a value more typical for a from-
scratch, batch-norm-free small conv net); warmup is kept alongside it
as cheap extra insurance since it did help during the epochs it was
active.

Run as: python3 experiments/synthetic_shortcut/train.py --dataset shortcut
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 128
# Lowered from design section 2's original 0.1: at 0.1, squeezenet1_0 (no
# batch norm) reliably collapses into a dead-ReLU constant-output state --
# confirmed to recur even with LR warmup, so the peak value itself, not
# just how it's reached, was the problem. See the module docstring.
DEFAULT_LR = 0.01
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 5e-4
# Epochs spent linearly ramping lr from a small fraction of DEFAULT_LR up to
# DEFAULT_LR before cosine annealing takes over. See the module docstring's
# third point: without this, lr=0.1 from step 1 reliably kills squeezenet1_0
# (no batch norm) via a dead-ReLU collapse to the ln(10) constant-loss state.
DEFAULT_WARMUP_EPOCHS = 10
NUM_CLASSES = 10


class ManifestImageDataset(Dataset):
    """Load (image, label) pairs from an ssat image_manifest JSON document.

    Reading through the same manifest format ssat's own source loader
    consumes (rather than, say, a torchvision ImageFolder layout) keeps
    prepare_data.py's output as the single source of truth for which images
    belong to which dataset variant.
    """

    def __init__(self, manifest_path: Path, transform: Callable[[Image.Image], torch.Tensor]) -> None:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._samples = document["samples"]
        self._transform = transform

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        entry = self._samples[index]
        image = Image.open(entry["path"]).convert("RGB")
        return self._transform(image), int(entry["gt_label"])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one training run."""

    parser = argparse.ArgumentParser(description="Train M_shortcut or M_normal from scratch.")
    parser.add_argument(
        "--dataset",
        choices=("shortcut", "normal"),
        required=True,
        help="'shortcut' trains on A_train (patched); 'normal' trains on C_train (clean)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoints"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--momentum", type=float, default=DEFAULT_MOMENTUM)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=DEFAULT_WARMUP_EPOCHS,
        help="epochs spent linearly ramping lr up to --lr before cosine annealing starts",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch and return (mean loss, top-1 accuracy)."""

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        # squeezenet1_0 has no batch normalization (its Fire modules are
        # plain Conv2d+ReLU), which makes a from-scratch model considerably
        # more prone to an early loss spike/NaN at lr=0.1 than a
        # BatchNorm-equipped architecture would be. Clipping the gradient
        # norm is a standard, minimally-invasive stabilizer that lets the
        # design doc's chosen learning rate stand rather than lowering it.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def main() -> int:
    """Train one model end to end and save its checkpoint."""

    args = parse_args()
    # Local import: keep torchvision's model zoo out of prepare_data.py's
    # import graph, and out of any lightweight consumer of common.py.
    from torchvision.models import SqueezeNet1_0_Weights, squeezenet1_0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # See the module docstring for why this transform is fetched from
    # torchvision instead of hand-written: it must be bit-identical to what
    # ssat's TorchvisionAdapter applies at audit time.
    transform = SqueezeNet1_0_Weights.DEFAULT.transforms()

    manifest_name = "A_train.json" if args.dataset == "shortcut" else "C_train.json"
    manifest_path = args.data_dir / "manifests" / manifest_name
    dataset = ManifestImageDataset(manifest_path, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    # weights=None: train entirely from scratch (design section 2), not
    # fine-tuned from ImageNet weights.
    model = squeezenet1_0(weights=None, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Linear warmup -> cosine annealing, chained with SequentialLR. The
    # warmup phase starts at 1% of --lr (not 0%, since LinearLR's start
    # factor multiplies the base lr and a literal 0 would take a full
    # optimizer step with zero effective lr) and ramps linearly to 100% of
    # --lr over --warmup-epochs epochs; see the module docstring for why
    # this is needed for a batch-norm-free network like squeezenet1_0.
    # After the handoff at epoch warmup_epochs, cosine annealing runs over
    # the remaining epochs, ending at 0 exactly at the final epoch, same as
    # design section 2 originally specified.
    warmup_epochs = min(args.warmup_epochs, args.epochs)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - warmup_epochs, 1)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        loss, accuracy = _train_one_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
        print(f"[{args.dataset}] epoch {epoch}/{args.epochs}: loss={loss:.4f} acc={accuracy:.4f}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / f"m_{args.dataset}.pt"
    # {"model": state_dict} matches the shape
    # ssat.core.adapter.checkpoint.load_state_dict_checkpoint expects when
    # given state_dict_key="model" (see run_audit.py's config builder).
    torch.save({"model": model.state_dict()}, checkpoint_path)
    print(f"saved checkpoint to {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
