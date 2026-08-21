#!/usr/bin/env python3
"""Enforce the raw-logit parity gate against an MMAction2 reference NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare(reference_path: Path, native_path: Path, *, atol: float, rtol: float) -> dict:
    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        native_path, allow_pickle=False
    ) as native:
        reference_ids = reference["sample_ids"]
        native_ids = native["sample_ids"]
        reference_logits = reference["logits"].astype(np.float64)
        native_logits = native["logits"].astype(np.float64)
    if not np.array_equal(reference_ids, native_ids):
        raise ValueError("reference and native sample_ids differ")
    if reference_logits.shape != native_logits.shape:
        raise ValueError("reference and native logits shapes differ")
    difference = np.abs(reference_logits - native_logits)
    relative = difference / np.maximum(np.abs(reference_logits), 1e-12)
    top1_match = np.argmax(reference_logits, axis=1) == np.argmax(native_logits, axis=1)
    result = {
        "sample_ids": reference_ids.tolist(),
        "shape": list(reference_logits.shape),
        "max_absolute_error": float(difference.max(initial=0.0)),
        "max_relative_error": float(relative.max(initial=0.0)),
        "top1_matches": int(top1_match.sum()),
        "top1_total": int(len(top1_match)),
        "atol": atol,
        "rtol": rtol,
        "passed": bool(
            top1_match.all()
            and np.allclose(reference_logits, native_logits, atol=atol, rtol=rtol)
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.reference, args.native, atol=args.atol, rtol=args.rtol)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
