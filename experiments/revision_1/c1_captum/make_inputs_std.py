#!/usr/bin/env python3
"""Write the C1-std sample subset (implementation plan section 9.3).

``data/revision_1/imagenet/val_1_per_class_c1.txt``: within every class of
``data/phase3/imagenet/val_10_per_class.txt``, the file with the lowest
``sha256("rev1-c1:" + filename)`` (1,000 samples). Both workflows read this
file: the SSAT config as ``source.annotation_file``, the Captum workflow as
``data.annotation``.

With ``--check``, nothing is written; the script fails unless the file on
disk matches what it would write and the SHA-256 registered in
``protocol_std.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from experiments.revision_1.common.runs import REPO_ROOT
from experiments.revision_1.common.subsets import class_balanced_subset

C1_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = C1_DIR / "protocol_std.json"
SOURCE_ANNOTATION = REPO_ROOT / "data" / "phase3" / "imagenet" / "val_10_per_class.txt"
SUBSET_PATH = REPO_ROOT / "data" / "revision_1" / "imagenet" / "val_1_per_class_c1.txt"
SUBSET_SALT = "rev1-c1"
PER_CLASS = 1


def subset_text() -> str:
    """Return the annotation file content of the C1-std subset."""

    lines = class_balanced_subset(SOURCE_ANNOTATION, PER_CLASS, SUBSET_SALT)
    return "".join(f"{filename} {label}\n" for filename, label in lines)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Verify instead of writing.")
    args = parser.parse_args(argv)

    text = subset_text()
    key = SUBSET_PATH.relative_to(REPO_ROOT).as_posix()
    if args.check:
        registered = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))["inputs_sha256"]
        failures = []
        if not SUBSET_PATH.is_file() or SUBSET_PATH.read_text(encoding="utf-8") != text:
            failures.append(f"{key}: differs from the generated content")
        if registered.get(key) != _sha256(text):
            failures.append(f"{key}: SHA-256 differs from protocol_std.json")
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1 if failures else 0
    SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUBSET_PATH.write_text(text, encoding="utf-8")
    SUBSET_PATH.chmod(0o644)
    print(f"{key} sha256={_sha256(text)} lines={text.count(chr(10))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
