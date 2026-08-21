#!/usr/bin/env python3
"""NTU-RGB+D dataset-prep recipe: raw files -> SSAT ``video_manifest``/``skeleton_bbox`` JSON.

This script is a NTU-RGB+D-specific reference implementation, NOT a stable
SSAT API. If you need another dataset, copy this file and rewrite only its
raw-file parsing section (``collect_ntu_rgb_data``/``parse_ntu_rgb_name``/
``parse_ntu_skeleton_file``) -- the SSAT-facing half (manifest/bbox writing)
is dataset-agnostic and can stay as-is.

It bridges two things that already exist and are independently tested:

- Raw-file parsing adapted from an existing NTU-RGB+D preprocessing
  reference implementation (a prior project's script, not itself part of
  SSAT): file-name parsing, xsub/xview splitting, and ``.skeleton`` file
  parsing into per-frame joint arrays.
- ``ssat.core.region.skeleton_bbox_builder``, which turns those joint
  arrays into the exact bounding-box JSON ``ssat.core.region.skeleton_store``
  reads at audit time.

Usage:
    python scripts/dataset_prep/ntu_rgb_d.py \\
        --rgb-root /path/to/nturgb+d_rgb \\
        --skeleton-root /path/to/nturgb+d_skeletons \\
        --annotation-file /path/to/ntu60_xsub_test.txt \\
        --samples-per-class 20 --num-frames 8 --sampling segment_center \\
        --out /path/to/output_dir

Writes ``video_manifest.json``, ``skeleton_bbox.json``, and an example
``config.yaml`` into ``--out``, restricted to samples whose skeleton file
parsed successfully (a sample present in the manifest but absent from the
bbox store would fail every ``skeleton_parts`` region at audit time -- see
``ssat.core.region.skeleton_provider.SkeletonRegionProvider``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import sys

import decord
import numpy as np
import pandas as pd

from ssat.core.region.skeleton_bbox_builder import (
    joints_to_part_bboxes,
    write_skeleton_bbox_json,
)
from ssat.core.source.video_folder import (
    segment_center_frame_indices,
    uniform_frame_indices,
)
from ssat.utils.io import sha256_file, write_json_atomic

# ---------------------------------------------------------------------------
# Adapted from an existing NTU-RGB+D preprocessing reference
# implementation. Only file_io.* calls were replaced with plain pathlib, and
# collect_ntu_rgb_data now also records normalize_video_key(...) as
# video_key -- everything else is unchanged.
# ---------------------------------------------------------------------------

# For cross-subject evaluation.
TRAIN_PERFORMER_IDS: set[int] = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
}

# For cross-view evaluation.
TRAIN_CAMERA_IDS: set[int] = {2, 3}

SPLIT_STRATEGIES: set[str] = {"xsub", "xview"}

STRATEGY_COLUMN_MAP: dict[str, str] = {"xsub": "performer", "xview": "camera"}

BODY_PARTS: dict[str, list[int]] = {
    # NTU RGB+D 25-joint skeleton, 0-based indices.
    "head": [2, 3, 20],  # Neck, Head, SpineShoulder
    "torso": [0, 1, 2, 20, 4, 8, 12, 16],
    "left_arm": [4, 5, 6, 7, 21, 22],
    "right_arm": [8, 9, 10, 11, 23, 24],
    "left_hand": [6, 7, 21, 22],
    "right_hand": [10, 11, 23, 24],
    "left_leg": [12, 13, 14, 15],
    "right_leg": [16, 17, 18, 19],
    "upper_body": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23, 24],
    "lower_body": [0, 12, 13, 14, 15, 16, 17, 18, 19],
}


@dataclass
class ParsedSkeleton:
    num_frames: int
    frames: list[list[dict]]


def normalize_video_key(video_file_name_or_path: str | Path) -> str:
    """Convert a video file name to its skeleton key.

    Examples:
        S001C001P001R001A001_rgb.avi -> S001C001P001R001A001
        S001C001P001R001A001.avi     -> S001C001P001R001A001
    """

    stem = Path(video_file_name_or_path).stem
    if stem.endswith("_rgb"):
        stem = stem[: -len("_rgb")]
    return stem


def parse_ntu_skeleton_file(path: Path) -> ParsedSkeleton:
    """Parse one NTU RGB+D ``.skeleton`` file.

    Joint line format is assumed as:
        x y z depthX depthY colorX colorY orientationW orientationX orientationY orientationZ trackingState
    """

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise ValueError(f"empty skeleton file: {path}")

    idx = 0
    num_frames = int(lines[idx])
    idx += 1

    frames: list[list[dict]] = []
    for _frame_idx in range(num_frames):
        if idx >= len(lines):
            raise ValueError(f"unexpected EOF while reading frame body count: {path}")
        num_bodies = int(lines[idx])
        idx += 1

        bodies: list[dict] = []
        for _body_idx in range(num_bodies):
            if idx >= len(lines):
                raise ValueError(f"unexpected EOF while reading body info: {path}")
            body_info = lines[idx].split()
            idx += 1

            if idx >= len(lines):
                raise ValueError(f"unexpected EOF while reading num_joints: {path}")
            num_joints = int(lines[idx])
            idx += 1

            joints = []
            for _joint_idx in range(num_joints):
                if idx >= len(lines):
                    raise ValueError(f"unexpected EOF while reading joint info: {path}")
                vals = list(map(float, lines[idx].split()))
                idx += 1
                joints.append(vals)

            bodies.append({"body_info": body_info, "joints": np.asarray(joints, dtype=np.float32)})
        frames.append(bodies)

    return ParsedSkeleton(num_frames=num_frames, frames=frames)


def choose_primary_body_index(parsed: ParsedSkeleton) -> int:
    """Choose the primary body index.

    Simple robust rule: select the body index with the largest accumulated
    number of valid joints across frames. This assumes body ordering is
    reasonably stable, which is usually acceptable for single-person NTU
    actions. For two-person actions (A050-A060) this heuristic can pick the
    wrong body -- a known limitation inherited as-is from the reference
    project.
    """

    max_bodies = max((len(bodies) for bodies in parsed.frames), default=0)
    if max_bodies == 0:
        return -1

    scores = np.zeros((max_bodies,), dtype=np.float64)
    for bodies in parsed.frames:
        for body_idx, body in enumerate(bodies):
            joints = body["joints"]
            if joints.ndim != 2 or joints.shape[0] == 0:
                continue
            tracking_state = joints[:, -1]
            scores[body_idx] += float(np.sum(tracking_state > 0))

    return int(np.argmax(scores))


def skeleton_to_primary_arrays(parsed: ParsedSkeleton) -> dict:
    """Convert a parsed skeleton into fixed arrays for the primary body.

    Returns:
        A dict with ``joints3d`` (T,25,3), ``joints2d_color`` (T,25,2),
        ``joint_valid`` (T,25), and ``primary_body_idx``.
    """

    num_frames = parsed.num_frames
    primary_body_idx = choose_primary_body_index(parsed)

    joints3d = np.full((num_frames, 25, 3), np.nan, dtype=np.float32)
    joints2d_color = np.full((num_frames, 25, 2), np.nan, dtype=np.float32)
    joint_valid = np.zeros((num_frames, 25), dtype=bool)

    if primary_body_idx < 0:
        return {
            "joints3d": joints3d,
            "joints2d_color": joints2d_color,
            "joint_valid": joint_valid,
            "primary_body_idx": primary_body_idx,
        }

    for frame_idx, bodies in enumerate(parsed.frames):
        if len(bodies) <= primary_body_idx:
            continue
        joints = bodies[primary_body_idx]["joints"]
        if joints.shape[0] < 25 or joints.shape[1] < 12:
            continue
        joints = joints[:25]
        joints3d[frame_idx] = joints[:, 0:3]
        joints2d_color[frame_idx] = joints[:, 5:7]
        # trackingState: 0 = not tracked, 1 = inferred, 2 = tracked.
        joint_valid[frame_idx] = joints[:, -1] > 0

    return {
        "joints3d": joints3d,
        "joints2d_color": joints2d_color,
        "joint_valid": joint_valid,
        "primary_body_idx": primary_body_idx,
    }


def parse_ntu_rgb_name(name: str) -> dict[str, str]:
    """Parse a NTU-RGB+D file name of the form ``S{s}C{c}P{p}R{r}A{a}_rgb.avi``."""

    pattern = r"S(\d+)C(\d+)P(\d+)R(\d+)A(\d+)_rgb\.avi"
    match = re.match(pattern, name)
    if not match:
        raise ValueError(f"invalid NTU-RGB+D file name: {name}")
    return {
        "setup": match.group(1),
        "camera": match.group(2),
        "performer": match.group(3),
        "replication": match.group(4),
        "action": match.group(5),
    }


def collect_ntu_rgb_data(root: str | Path) -> pd.DataFrame:
    """Collect every ``*.avi`` under ``root`` into one row-per-sample DataFrame."""

    root = Path(root).expanduser().resolve(strict=True)
    data = []
    for file in root.glob("*.avi"):
        try:
            info = parse_ntu_rgb_name(file.name)
        except ValueError:
            continue
        info["path"] = str(file)
        info["video_key"] = normalize_video_key(file.name)
        info["label"] = int(info["action"]) - 1  # 0-based gt_label
        data.append(info)
    return pd.DataFrame(data)


def split_ntu_rgb_data(df: pd.DataFrame, strategy: str = "xsub") -> pd.DataFrame:
    """Label every row with a ``split`` column per the ``xsub``/``xview`` strategy."""

    column = STRATEGY_COLUMN_MAP.get(strategy)
    if column is None:
        raise ValueError(f"invalid strategy: {strategy}")
    train_value_set = TRAIN_PERFORMER_IDS if strategy == "xsub" else TRAIN_CAMERA_IDS
    df = df.copy()
    df["split"] = df[column].apply(lambda x: "train" if int(x) in train_value_set else "test")
    return df


def load_ntu_annotation(annotation_file: Path, rgb_root: Path) -> pd.DataFrame:
    """Load the exact MMAction2 ``<video> <0-based-label>`` annotation."""

    rows = []
    for line_number, raw_line in enumerate(
        annotation_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.rsplit(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"{annotation_file}:{line_number}: malformed annotation")
        relative_path, label_text = parts
        path = (rgb_root / relative_path).resolve()
        rows.append(
            {
                "path": str(path),
                "video_key": normalize_video_key(relative_path),
                "label": int(label_text),
                "action": int(label_text) + 1,
            }
        )
    if not rows:
        raise ValueError(f"annotation contains no samples: {annotation_file}")
    return pd.DataFrame(rows)


def select_per_class(df: pd.DataFrame, *, count: int, seed: int) -> pd.DataFrame:
    """Select a class-balanced subset by stable SHA-256 ranking."""

    if count <= 0:
        raise ValueError("per-class count must be positive")
    selected = []
    for label, group in df.groupby("label", sort=True):
        if len(group) < count:
            raise ValueError(f"class {label} has {len(group)} samples; need {count}")
        ranked = group.assign(
            _rank=group["video_key"].map(
                lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
            )
        ).sort_values(["_rank", "video_key"])
        selected.append(ranked.head(count).drop(columns=["_rank"]))
    return pd.concat(selected, ignore_index=True).sort_values(
        ["label", "video_key"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# New: bridge collected samples into SSAT's video_manifest/skeleton_bbox JSON.
# ---------------------------------------------------------------------------


def probe_rgb_video(path: Path) -> tuple[int, tuple[int, int]]:
    """Return the ``(frame_count, (width, height))`` SSAT's source actually decodes.

    ``skeleton_bbox.json``'s ``frame_size`` must match the decoded
    resolution, not the ``.skeleton`` file's depth-camera resolution -- a
    mismatch silently mispositions every skeleton_parts mask without
    failing any schema check (docs/CONFIG_REFERENCE.md:85-87).
    """

    reader = decord.VideoReader(str(path))
    frame_count = len(reader)
    frame = reader[0].asnumpy()
    height, width = frame.shape[:2]
    return frame_count, (width, height)


def build_outputs(
    df: pd.DataFrame,
    *,
    skeleton_root: Path,
    out_dir: Path,
    num_frames: int,
    sampling: str = "uniform",
    selection_metadata: dict | None = None,
) -> tuple[int, list[tuple[str, str]]]:
    """Parse each row's skeleton, write video_manifest.json/skeleton_bbox.json.

    Samples whose skeleton file is missing or fails to parse are skipped
    (and reported) rather than aborting the whole batch -- real NTU-RGB+D
    data has known-corrupt/missing skeleton entries.

    The skeleton file records one joint set per *native* clip frame (e.g. a
    98-frame clip has 98 skeleton frames), but ``VideoFolderSource``
    deterministically subsamples every clip down to ``num_frames`` before a
    region is ever resolved (see
    ``ssat.core.region.resolver.RegionResolver.resolve``,
    which requires a ``(T, H, W)`` mask's ``T`` to equal the *loaded* sample's
    frame count, not the source video's native length). So the configured
    frame-index selection SSAT's own video loader uses is applied here to the
    skeleton arrays before computing bounding boxes --
    otherwise every skeleton_parts region fails at audit time with "mask
    frame count does not match the source sample".

    Returns:
        The number of samples written and a list of ``(video_key, reason)``
        skip records.
    """

    if sampling not in {"uniform", "segment_center"}:
        raise ValueError("sampling must be 'uniform' or 'segment_center'")
    sample_bboxes: dict[str, dict[str, list]] = {}
    frame_sizes: dict[str, tuple[int, int]] = {}
    manifest_samples: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for row in df.itertuples(index=False):
        skeleton_path = skeleton_root / f"{row.video_key}.skeleton"
        try:
            if not skeleton_path.is_file():
                raise FileNotFoundError(str(skeleton_path))
            parsed = parse_ntu_skeleton_file(skeleton_path)
            arrays = skeleton_to_primary_arrays(parsed)
            video_frame_count, frame_size = probe_rgb_video(Path(row.path))
            # Skeleton and RGB frames are captured 1:1 by the same Kinect
            # sensor; clamp defensively in case a clip's counts drift by a
            # frame or two.
            sampler = (
                segment_center_frame_indices
                if sampling == "segment_center"
                else uniform_frame_indices
            )
            indices = np.minimum(
                sampler(video_frame_count, num_frames),
                parsed.num_frames - 1,
            )
            part_bboxes = joints_to_part_bboxes(
                arrays["joints2d_color"][indices],
                arrays["joint_valid"][indices],
                BODY_PARTS,
                frame_size=frame_size,
            )
        except Exception as error:  # noqa: BLE001 -- batch script, one bad file must not abort the run
            skipped.append((row.video_key, f"{error.__class__.__name__}: {error}"))
            continue

        sample_bboxes[row.video_key] = part_bboxes
        frame_sizes[row.video_key] = frame_size
        manifest_samples.append(
            {
                "sample_id": row.video_key,
                "path": Path(os.path.relpath(row.path, out_dir)).as_posix(),
                "gt_label": int(row.label),
            }
        )

    if not manifest_samples:
        raise RuntimeError("no sample produced valid skeleton bbox data; nothing to write")
    requested_per_class = (selection_metadata or {}).get("samples_per_class")
    if requested_per_class is not None:
        output_counts: dict[int, int] = {}
        for sample in manifest_samples:
            label = int(sample["gt_label"])
            output_counts[label] = output_counts.get(label, 0) + 1
        short = {
            label: count
            for label, count in output_counts.items()
            if count != requested_per_class
        }
        expected_labels = set(int(label) for label in df["label"].unique())
        if set(output_counts) != expected_labels or short:
            raise RuntimeError(
                "skeleton filtering broke the requested per-class balance; "
                f"counts={output_counts}"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / "video_manifest.json", {"samples": manifest_samples})
    write_skeleton_bbox_json(sample_bboxes, frame_sizes, out_dir / "skeleton_bbox.json")
    metadata = {
        "dataset": "NTU-RGB+D 60",
        "sampling": sampling,
        "num_frames": num_frames,
        "num_samples": len(manifest_samples),
        "body_parts": sorted(BODY_PARTS),
        **(selection_metadata or {}),
    }
    metadata["manifest_sha256"] = sha256_file(out_dir / "video_manifest.json")
    metadata["skeleton_bbox_sha256"] = sha256_file(out_dir / "skeleton_bbox.json")
    write_json_atomic(out_dir / "selection_metadata.json", metadata)
    _write_quickstart_config(out_dir, num_frames=num_frames, sampling=sampling)
    for path in (
        out_dir / "video_manifest.json",
        out_dir / "skeleton_bbox.json",
        out_dir / "selection_metadata.json",
        out_dir / "config.yaml",
    ):
        path.chmod(0o644)
    return len(manifest_samples), skipped


def _write_quickstart_config(out_dir: Path, *, num_frames: int, sampling: str) -> None:
    """Write a ready-to-run SSAT config pointing at this run's own outputs."""

    body_part = next(iter(BODY_PARTS))
    config = f"""schema_version: 1.0.0
source:
  kind: video_manifest
  manifest: video_manifest.json
  num_frames: {num_frames}
  sampling: {sampling}
adapter:
  provider: torchvision_video
  model_name: r3d_18
  weights: null
  device: cpu
  max_batch_size: 2
skeleton_source:
  bbox_data: skeleton_bbox.json
regions:
  - region_id: occlude_{body_part}
    kind: skeleton_parts
    params:
      body_part: {body_part}
      bbox_scale: 1.15
perturbations:
  - op: constant_fill
    params:
      value: [0, 0, 0]
runtime:
  variants_per_chunk: 2
  target_batch_size: 2
  num_workers: 0
dump:
  flush_every: 8
"""
    (out_dir / "config.yaml").write_text(config, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a video_manifest.json/skeleton_bbox.json/config.yaml trio "
            "for one NTU-RGB+D subset, ready for `ssat estimate`/`ssat run`."
        )
    )
    parser.add_argument("--rgb-root", type=Path, required=True, help="directory of *_rgb.avi files")
    parser.add_argument(
        "--skeleton-root", type=Path, required=True, help="directory of *.skeleton files"
    )
    parser.add_argument(
        "--split", choices=sorted(SPLIT_STRATEGIES), default="xsub", help="xsub or xview"
    )
    parser.add_argument("--num-frames", type=int, default=16, help="frames sampled per clip")
    parser.add_argument(
        "--sampling", choices=("uniform", "segment_center"), default="uniform"
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--actions",
        type=str,
        default=None,
        help=(
            "comma-separated 1-based NTU action IDs to include, e.g. '8,9,23' "
            "(default: no filter, every collected action)"
        ),
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        help="MMAction2 annotation file; when set, use its exact sample set and labels",
    )
    parser.add_argument(
        "--partition", choices=("train", "test"), default="test",
        help="partition selected when deriving a split without --annotation-file",
    )
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the number of collected samples (useful for a smoke-test subset)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rgb_root = Path(args.rgb_root).expanduser().resolve(strict=True)
    annotation_file = None
    if args.annotation_file is not None:
        annotation_file = args.annotation_file.expanduser().resolve(strict=True)
        df = load_ntu_annotation(annotation_file, rgb_root)
    else:
        df = collect_ntu_rgb_data(rgb_root)
    if df.empty:
        raise SystemExit(f"no *_rgb.avi files matched under {args.rgb_root}")
    # Deterministic order independent of the filesystem's glob() ordering, so
    # --actions/--limit selects the same samples on every run.
    df = df.sort_values("video_key").reset_index(drop=True)
    if annotation_file is None:
        df = split_ntu_rgb_data(df, strategy=args.split)
        df = df[df["split"] == args.partition].reset_index(drop=True)
    if args.actions is not None:
        wanted = {int(action_id) for action_id in args.actions.split(",")}
        df = df[df["action"].astype(int).isin(wanted)].reset_index(drop=True)
        if df.empty:
            raise SystemExit(f"no samples matched --actions {args.actions!r}")
    if args.limit is not None:
        df = df.head(args.limit)
    if args.samples_per_class is not None:
        df = select_per_class(df, count=args.samples_per_class, seed=args.seed)
    class_counts = {
        str(int(label)): int(count)
        for label, count in df.groupby("label").size().items()
    }

    skeleton_root = Path(args.skeleton_root).expanduser().resolve(strict=True)
    written, skipped = build_outputs(
        df,
        skeleton_root=skeleton_root,
        out_dir=args.out.expanduser().resolve(),
        num_frames=args.num_frames,
        sampling=args.sampling,
        selection_metadata={
            "selection": (
                "sha256(seed:video_key)" if args.samples_per_class is not None else "all"
            ),
            "seed": args.seed,
            "samples_per_class": args.samples_per_class,
            "class_counts_before_skeleton_filter": class_counts,
            "annotation_file": None if annotation_file is None else annotation_file.name,
            "annotation_sha256": (
                None if annotation_file is None else sha256_file(annotation_file)
            ),
            "split_strategy": None if annotation_file is not None else args.split,
            "partition": None if annotation_file is not None else args.partition,
        },
    )

    print(f"wrote {written} sample(s) to {args.out}")
    if skipped:
        print(f"skipped {len(skipped)} sample(s):", file=sys.stderr)
        for video_key, reason in skipped:
            print(f"  {video_key}: {reason}", file=sys.stderr)


if __name__ == "__main__":
    main()
