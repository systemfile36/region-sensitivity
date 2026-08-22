#!/usr/bin/env python3
"""Assert every pre-registered Q1-Q5 verdict in a results dir's verdicts.json
is PASS. Exit-code-driven, dependency-free (stdlib only), for scripted/CI
verification of `reproduce_demo.sh` (and equally usable against the full
retrain path's `results_crop_free/verdicts.json`).

Run as: python3 experiments/synthetic_shortcut/check_demo_verdicts.py --results-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

QUESTIONS = (
    "Q1_identifies_patch_region",
    "Q2_separated_from_baseline",
    "Q3_distinguishes_normal_model",
    "Q4_robust_to_fill_strategy",
    "Q5_predicts_generalization_gap",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Assert all pre-registered Q1-Q5 verdicts in verdicts.json are PASS."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Load verdicts.json and check every pre-registered question passed."""

    args = parse_args()
    verdicts_path = args.results_dir / "verdicts.json"
    if not verdicts_path.is_file():
        print(f"error: {verdicts_path} does not exist (run evaluate.py first)")
        return 1

    verdicts = json.loads(verdicts_path.read_text(encoding="utf-8"))
    failed = [q for q in QUESTIONS if not verdicts.get(q, {}).get("pass")]

    if failed:
        print(f"FAIL: {len(failed)}/{len(QUESTIONS)} question(s) did not pass:")
        for q in failed:
            print(f"  - {q}: {verdicts.get(q)}")
        return 1

    print(f"PASS: all {len(QUESTIONS)} pre-registered Q1-Q5 questions passed ({verdicts_path}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
