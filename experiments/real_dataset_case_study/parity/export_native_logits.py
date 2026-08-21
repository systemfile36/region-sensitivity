#!/usr/bin/env python3
"""Export clean raw logits from a configured SSAT adapter to a compact NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ssat.application.config import load_application_config
from ssat.core.adapter import default_adapter_provider_registry
from ssat.core.source import LoadError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    registry = default_adapter_provider_registry()
    loaded = load_application_config(args.config, registry)
    adapter = registry.build(loaded.adapter_config, base_dir=loaded.base_dir)
    sample_ids = []
    logits = []
    for sample in sorted(loaded.sample_source.list_samples(), key=lambda item: item.sample_id):
        decoded = loaded.sample_source.load(sample.sample_id)
        if isinstance(decoded, LoadError):
            raise RuntimeError(f"failed to load {sample.sample_id}: {decoded.message}")
        sample_ids.append(sample.sample_id)
        logits.append(adapter.predict(decoded.array[None])[0].logits)
        if len(sample_ids) == args.count:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        logits=np.stack(logits).astype(np.float32),
    )


if __name__ == "__main__":
    main()
