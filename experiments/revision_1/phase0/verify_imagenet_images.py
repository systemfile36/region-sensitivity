#!/usr/bin/env python3
"""Check the restored ImageNet val JPEGs against the images the baselines audited.

Every ImageNet baseline dump records a ``content_hash`` (SHA-256 of the file
bytes) per sample in its clean rows. This script hashes each file under
``--image-root`` once and requires it to equal the recorded hash in all four
``K=3`` baseline runs, so new runs (A2, A3) read exactly the baseline images.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.revision_1.common.provenance import write_json_shared, write_provenance
from experiments.revision_1.common.runs import BASELINE_RUNS, REPO_ROOT
from ssat.core.dump._storage import fragment_files
from ssat.utils.io import sha256_file

SUMMARY_DIR = Path(__file__).resolve().parent / "summary"
DEFAULT_IMAGE_ROOT = REPO_ROOT / "data" / "imagenet" / "ILSVRC" / "Data" / "CLS-LOC" / "val"


def recorded_hashes(dump: Path) -> dict[str, str]:
    """Return ``{sample_id: content_hash}`` from a dump's clean fragments (last write wins)."""

    tables = [pq.read_table(path, columns=["sample_id", "content_hash"]) for _, path in fragment_files(dump / "clean", "part")]
    frame = pa.concat_tables(tables).to_pandas().drop_duplicates("sample_id", keep="last")
    return dict(zip(frame["sample_id"], frame["content_hash"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=SUMMARY_DIR / "imagenet_images.json")
    args = parser.parse_args(argv)

    runs = [run for run in BASELINE_RUNS.values() if run.dataset == "imagenet"]
    recorded = {run.name: recorded_hashes(run.dump) for run in runs}
    sample_ids = sorted(set().union(*recorded.values()))
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = dict(zip(sample_ids, pool.map(lambda sid: sha256_file(args.image_root / sid) if (args.image_root / sid).is_file() else None, sample_ids)))

    per_run = {}
    for name, hashes in recorded.items():
        missing = sorted(sid for sid in hashes if actual[sid] is None)
        mismatched = sorted(sid for sid, digest in hashes.items() if actual[sid] is not None and actual[sid] != digest)
        per_run[name] = {"n_samples": len(hashes), "n_missing": len(missing), "n_mismatched": len(mismatched),
                         "missing_head": missing[:20], "mismatched_head": mismatched[:20]}
    ok = all(entry["n_missing"] == 0 and entry["n_mismatched"] == 0 for entry in per_run.values())
    n_files = sum(1 for _ in args.image_root.glob("*.JPEG")) if args.image_root.is_dir() else 0
    result = {"image_root": str(args.image_root.relative_to(REPO_ROOT)) if args.image_root.is_relative_to(REPO_ROOT) else str(args.image_root),
              "n_files_in_root": n_files, "n_audited_samples": len(sample_ids), "all_match": ok, "runs": per_run}
    write_json_shared(args.output, result)
    write_provenance(args.output.parent, inputs={run.name: run.dump for run in runs},
                     extra={"image_root": result["image_root"]}, filename=f"{args.output.stem}.provenance.json")
    print(f"{len(sample_ids)} audited samples, {n_files} files in root, all_match={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
