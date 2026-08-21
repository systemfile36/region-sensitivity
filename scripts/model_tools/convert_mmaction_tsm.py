#!/usr/bin/env python3
"""Convert an MMAction2 ResNetTSM checkpoint into tensor-only SSAT form."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ssat.core.adapter.mmaction_checkpoint import (
    convert_mmaction_tsm_state_dict,
    load_mmaction_checkpoint_restricted,
)
from ssat.core.adapter.torchvision_tsm_adapter import TSMResNet50Model
from ssat.utils.io import sha256_file, write_json_atomic


def _safe_meta(meta: Any) -> dict[str, Any]:
    if not isinstance(meta, Mapping):
        return {}
    result = {}
    for key in ("epoch", "iter", "experiment_name", "mmengine_version"):
        value = meta.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    return result


def convert_checkpoint(
    source: Path,
    destination: Path,
    *,
    mmaction_commit: str,
    training_config: Path | None,
    num_segments: int = 8,
    num_classes: int = 60,
    shift_div: int = 8,
) -> dict[str, Any]:
    import torch

    payload = load_mmaction_checkpoint_restricted(source)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint has no mapping-valued 'state_dict'")
    converted = convert_mmaction_tsm_state_dict(state_dict)
    if len(converted) != 320:
        raise ValueError(f"expected 320 TSM tensors, found {len(converted)}")
    model = TSMResNet50Model.create(
        num_segments=num_segments,
        num_classes=num_classes,
        shift_div=shift_div,
    )
    model.load_state_dict(converted, strict=True)
    head = converted.get("backbone.fc.weight")
    if head is None or tuple(head.shape) != (num_classes, 2048):
        raise ValueError("converted classification head has an unexpected shape")

    source_hash = sha256_file(source)
    provenance = {
        "format": "ssat.torchvision_tsm.state_dict.v1",
        "source_checkpoint": source.name,
        "source_sha256": source_hash,
        "mmaction2_commit": mmaction_commit,
        "training_config": None if training_config is None else training_config.name,
        "training_config_sha256": (
            None if training_config is None else sha256_file(training_config)
        ),
        "num_tensors": len(converted),
        "num_segments": num_segments,
        "num_classes": num_classes,
        "shift_div": shift_div,
        "source_meta": _safe_meta(payload.get("meta")),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": converted, "provenance": provenance}, destination)
    provenance["converted_checkpoint"] = destination.name
    provenance["converted_sha256"] = sha256_file(destination)
    sidecar = destination.with_suffix(destination.suffix + ".json")
    write_json_atomic(sidecar, provenance)
    destination.chmod(0o644)
    sidecar.chmod(0o644)
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--mmaction-commit", default="a5a167df")
    parser.add_argument("--training-config", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provenance = convert_checkpoint(
        args.source.expanduser().resolve(strict=True),
        args.destination.expanduser().resolve(),
        mmaction_commit=args.mmaction_commit,
        training_config=(
            None
            if args.training_config is None
            else args.training_config.expanduser().resolve(strict=True)
        ),
    )
    print(provenance)


if __name__ == "__main__":
    main()
