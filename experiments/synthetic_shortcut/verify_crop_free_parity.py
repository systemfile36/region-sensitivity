#!/usr/bin/env python3
"""Verify common.build_crop_free_transform() and the audit-time adapter agree.

train.py/evaluate_accuracy.py's crop-free path (common.build_crop_free_transform)
and run_audit.py's crop-free path (TorchvisionAdapter(preprocessing_ops=...) via
common.CROP_FREE_PREPROCESSING_OPS) both drive
ssat.core.adapter.preprocessing.DeclarativePreprocessor, so they are the same
code path by construction, not two independent implementations that merely
happen to agree numerically (see common.build_crop_free_transform's
docstring). This script exists as a cheap, explicit sanity check of that
claim before spending GPU time on a real training run -- if this ever fails,
train/audit preprocessing has silently diverged.

Run as: python3 experiments/synthetic_shortcut/verify_crop_free_parity.py
"""

from __future__ import annotations

import numpy as np
from common import CROP_FREE_PREPROCESSING_OPS, build_crop_free_transform
from PIL import Image

from ssat.core.adapter.torchvision_adapter import TorchvisionAdapter


def main() -> int:
    """Compare build_crop_free_transform()'s output to the adapter's own path."""

    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    image = Image.fromarray(array, mode="RGB")

    train_side = build_crop_free_transform()(image).numpy()

    adapter = TorchvisionAdapter(
        model_name="squeezenet1_0",
        weights=None,
        model_kwargs={"num_classes": 10},
        device="cpu",
        preprocessing_ops=CROP_FREE_PREPROCESSING_OPS,
    )
    batch = array[None, None, :, :, :]
    audit_side = adapter._preprocessor.transform_batch(batch)[0]

    if train_side.shape != audit_side.shape:
        print(f"SHAPE MISMATCH: train={train_side.shape} audit={audit_side.shape}")
        return 1
    max_abs_diff = float(np.max(np.abs(train_side - audit_side)))
    print(f"shape={train_side.shape} max_abs_diff={max_abs_diff:.10f}")
    if max_abs_diff != 0.0:
        print("FAIL: train-time and audit-time crop-free preprocessing diverge")
        return 1
    print("PASS: train-time and audit-time crop-free preprocessing are bit-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
