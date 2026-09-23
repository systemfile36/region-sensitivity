#!/usr/bin/env python3
"""Run a matrix of SSAT audits through run -> metrics -> analyze -> report.

Behaves like ``experiments/real_dataset_case_study/run_matrix.py`` but takes
the matrix file and output root as arguments, and appends one JSON line per
executed step (command, wall time, return code) to
``<output-root>/run_matrix_log.jsonl`` so later cost analyses can reuse the
timings.

Matrix format::

    {"runs": [{"name": "...", "config": "configs/x.yaml", "minimum_accuracy": 0.5}]}

``config`` is resolved relative to the matrix file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STEPS = ("run", "metrics", "analyze", "report")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", type=Path, required=True, help="Matrix JSON file.")
    parser.add_argument("--output-root", type=Path, required=True, help="Parent of the run output directories.")
    parser.add_argument("--only", action="append", default=[], help="Run only this name (repeatable).")
    parser.add_argument("--skip-report", action="store_true", help="Stop after analyze.")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEPS,
        default=None,
        help="Run only these steps (default: all, minus report with --skip-report).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args(argv)


def commands_for(
    config: Path, output: Path, minimum_accuracy: float | None, steps: tuple[str, ...]
) -> list[tuple[str, list[str]]]:
    """Return the ``(step, argv)`` pairs for one run."""

    run = [sys.executable, "-m", "ssat", "run", str(config), "-o", str(output), "--yes"]
    if minimum_accuracy is not None:
        run.extend(["--minimum-accuracy", str(minimum_accuracy)])
    all_commands = {
        "run": run,
        "metrics": [sys.executable, "-m", "ssat", "metrics", str(output)],
        "analyze": [sys.executable, "-m", "ssat", "analyze", str(output)],
        "report": [sys.executable, "-m", "ssat", "report", str(output)],
    }
    return [(step, all_commands[step]) for step in steps]


def _log(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    matrix_path = args.matrix.resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    known = {run["name"] for run in matrix["runs"]}
    unknown = sorted(set(args.only) - known)
    if unknown:
        raise SystemExit(f"unknown run name(s): {', '.join(unknown)}")
    if args.steps is not None:
        steps = tuple(step for step in STEPS if step in args.steps)
    else:
        steps = tuple(step for step in STEPS if not (args.skip_report and step == "report"))
    log_path = args.output_root / "run_matrix_log.jsonl"
    for run in matrix["runs"]:
        if args.only and run["name"] not in args.only:
            continue
        config = (matrix_path.parent / run["config"]).resolve()
        output = args.output_root / run["name"]
        for step, command in commands_for(config, output, run.get("minimum_accuracy"), steps):
            print("+", " ".join(command), flush=True)
            if args.dry_run:
                continue
            started_at = datetime.now(timezone.utc).isoformat()
            start = time.perf_counter()
            completed = subprocess.run(command, check=False)
            _log(
                log_path,
                {
                    "run": run["name"],
                    "step": step,
                    "command": command,
                    "started_at": started_at,
                    "elapsed_s": time.perf_counter() - start,
                    "returncode": completed.returncode,
                },
            )
            if completed.returncode != 0:
                print(f"step {step!r} of {run['name']!r} failed ({completed.returncode})", file=sys.stderr)
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
