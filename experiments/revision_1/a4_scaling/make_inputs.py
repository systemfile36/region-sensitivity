#!/usr/bin/env python3
"""Write the A4 sample subsets ``data/revision_1/imagenet/a4_N{N}.txt``.

Each subset is the ``N`` files of ``data/phase3/imagenet/val_10_per_class.txt``
with the lowest ``sha256("rev1-a4:" + filename)``, so every smaller subset
is a prefix of every larger one (``sweep.SUBSET_SIZES``, including the N=50
warm-up and the N=200 component profile).

With ``--check``, nothing is written; the script fails unless the files on
disk match what it would write and the SHA-256 values registered in
``protocol.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

from experiments.revision_1.a4_scaling.sweep import A4_DIR, SOURCE_ANNOTATION, SUBSET_SALT, SUBSET_SIZES, subset_path
from experiments.revision_1.common.runs import REPO_ROOT
from experiments.revision_1.common.subsets import top_n_subset

PROTOCOL_PATH = A4_DIR / "protocol.json"


def subset_text(n: int) -> str:
    """Return the annotation file content of the ``n``-sample subset."""

    return "".join(f"{filename} {label}\n" for filename, label in top_n_subset(SOURCE_ANNOTATION, n, SUBSET_SALT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Verify instead of writing.")
    args = parser.parse_args(argv)

    outputs = {subset_path(n): subset_text(n) for n in SUBSET_SIZES}
    if args.check:
        registered = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))["inputs_sha256"]
        failures = []
        for path, text in outputs.items():
            key = path.relative_to(REPO_ROOT).as_posix()
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                failures.append(f"{key}: differs from the generated content")
            if registered.get(key) != hashlib.sha256(text.encode("utf-8")).hexdigest():
                failures.append(f"{key}: SHA-256 differs from protocol.json")
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1 if failures else 0
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        path.chmod(0o644)
        print(f"{path.relative_to(REPO_ROOT)} sha256={hashlib.sha256(text.encode('utf-8')).hexdigest()} lines={text.count(chr(10))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
