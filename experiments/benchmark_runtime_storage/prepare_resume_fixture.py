#!/usr/bin/env python3
"""Generate the larger synthetic fixture used by the cache/resume benchmark.

Wraps ``scripts/generate_synthetic_classification_fixture.py`` to produce
enough samples (default 300) that an interrupted ``ssat run`` has a wide,
reliable window in which to be stopped mid-flight. The committed 20-image
test fixture at ``tests/fixtures/synthetic_classification/`` completes far
too fast for a reliable interruption and must not be regenerated at a
different size -- other tests depend on its exact committed content -- so
this script writes into its own gitignored directory under this experiment
instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
GENERATOR = REPO_ROOT / "scripts" / "generate_synthetic_classification_fixture.py"

DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "data" / "resume_fixture"
DEFAULT_VALID_COUNT = 300
DEFAULT_SEED = 20260824


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments without mutating the filesystem."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"fixture destination (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--valid-count",
        type=int,
        default=DEFAULT_VALID_COUNT,
        help=f"number of valid RGB PNG files (default: {DEFAULT_VALID_COUNT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"non-negative root seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an already-generated fixture at --output-dir",
    )
    return parser.parse_args()


def main() -> int:
    """Delegate fixture generation to the shared generator at a larger scale."""

    args = parse_args()
    command = [
        sys.executable,
        str(GENERATOR),
        "--output-dir",
        str(args.output_dir),
        "--valid-count",
        str(args.valid_count),
        "--seed",
        str(args.seed),
    ]
    if args.force:
        command.append("--force")
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
