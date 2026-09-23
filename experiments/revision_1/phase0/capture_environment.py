#!/usr/bin/env python3
"""P0-5: snapshot the execution environment used for revision-1 runs.

Run inside the ``region-sensitivity-workspace`` container. Export
``SSAT_CONTAINER_IMAGE_ID`` from the host (``docker inspect --format
'{{.Id}}' local/region-sensitivity-workspace:latest``) to record the image,
which is not visible from inside the container. ``--note`` appends
free-text facts (e.g. packages installed by hand after the container was
created).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiments.revision_1.common.provenance import capture_environment, git_state, write_json_shared

SUMMARY_DIR = Path(__file__).resolve().parent / "summary"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=SUMMARY_DIR / "environment.json")
    parser.add_argument("--note", action="append", default=[], help="Free-text note (repeatable).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "environment": capture_environment(),
        "notes": args.note,
    }
    write_json_shared(args.output, payload)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
