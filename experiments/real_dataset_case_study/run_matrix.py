#!/usr/bin/env python3
"""Run the six Phase-3 audits through metrics, analysis, and report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def commands_for(config: Path, output: Path, minimum_accuracy: float | None) -> list[list[str]]:
    run = [sys.executable, "-m", "ssat", "run", str(config), "-o", str(output), "--yes"]
    if minimum_accuracy is not None:
        run.extend(["--minimum-accuracy", str(minimum_accuracy)])
    return [
        run,
        [sys.executable, "-m", "ssat", "metrics", str(output)],
        [sys.executable, "-m", "ssat", "analyze", str(output)],
        [sys.executable, "-m", "ssat", "report", str(output)],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/real_dataset_case_study"))
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    matrix = json.loads((root / "matrix.json").read_text(encoding="utf-8"))
    known = {run["name"] for run in matrix["runs"]}
    unknown = sorted(set(args.only) - known)
    if unknown:
        raise SystemExit(f"unknown run name(s): {', '.join(unknown)}")
    for run in matrix["runs"]:
        if args.only and run["name"] not in args.only:
            continue
        config = root / run["config"]
        output = args.output_root / run["name"]
        for command in commands_for(config, output, run["minimum_accuracy"]):
            print("+", " ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
