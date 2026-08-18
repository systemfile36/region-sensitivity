from dataclasses import dataclass
import json
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import os
from pathlib import Path
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any
import src.file_io as file_io

# Make NTU-RGB+D dataset to training compatible dataframe
# We will use VideoDataset in MMAction2 to load the RGB video data as the input of the model, and the label will be the action label extracted from the file name of the RGB video file.
# See https://mmaction2.readthedocs.io/en/latest/user_guides/prepare_dataset.html#action-recognition

# further information about evaluation split strategy, See https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf

# For cross-subject evaluation
TRAIN_PERFORMER_IDS: set[int] = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
}

# For cross-view evaluation
TRAIN_CAMERA_IDS: set[int] = {2, 3}

SPLIT_STRATEGIES: set[str] = {"xsub", "xview"}

STRATEGY_COLUMN_MAP: Dict[str, str] = {
    "xsub": "performer",
    "xview": "camera"
}

BODY_PARTS: Dict[str, List[int]] = {
    # NTU RGB+D 25-joint skeleton, 0-based indices.
    "head": [2, 3, 20],  # Neck, Head, SpineShoulder
    "torso": [0, 1, 2, 20, 4, 8, 12, 16],
    "left_arm": [4, 5, 6, 7, 21, 22],
    "right_arm": [8, 9, 10, 11, 23, 24],
    "left_hand": [6, 7, 21, 22],
    "right_hand": [10, 11, 23, 24],
    "left_leg": [12, 13, 14, 15],
    "right_leg": [16, 17, 18, 19],
    "both_hands": [6, 7, 10, 11, 21, 22, 23, 24],
    "upper_body": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23, 24],
    "lower_body": [0, 12, 13, 14, 15, 16, 17, 18, 19],
}

@dataclass
class ParsedSkeleton:
    num_frames: int
    frames: list[list[dict]]

# ------
# Skeleton parsing
# ------
def normalize_video_key(video_file_name_or_path: str | Path) -> str:
    """
    Convert video file name to skeleton key.

    Examples:
        S001C001P001R001A001_rgb.avi -> S001C001P001R001A001
        S001C001P001R001A001.avi     -> S001C001P001R001A001
    """
    stem = file_io.ensure_path(video_file_name_or_path).stem
    if stem.endswith("_rgb"):
        stem = stem[: -len("_rgb")]
    return stem

def read_video_keys_from_label_file(label_file: Path) -> list[str]:
    keys: list[str] = []

    with label_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Expected: "{video_file_name} {gt_label}"
            parts = line.split()
            if len(parts) < 1:
                continue

            video_file_name = parts[0]
            keys.append(normalize_video_key(video_file_name))

    # Preserve order while removing duplicates.
    seen = set()
    unique_keys = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique_keys.append(key)

    return unique_keys

def parse_ntu_skeleton_file(path: Path) -> ParsedSkeleton:
    """
    Parse NTU RGB+D .skeleton file.

    Joint line format is assumed as:
        x y z depthX depthY colorX colorY orientationW orientationX orientationY orientationZ trackingState
    """
    lines = path.read_text(encoding="utf-8").strip().splitlines()

    if not lines:
        raise ValueError(f"Empty skeleton file: {path}")

    idx = 0
    num_frames = int(lines[idx])
    idx += 1

    frames: list[list[dict]] = []

    for _frame_idx in range(num_frames):
        if idx >= len(lines):
            raise ValueError(f"Unexpected EOF while reading frame body count: {path}")

        num_bodies = int(lines[idx])
        idx += 1

        bodies: list[dict] = []

        for _body_idx in range(num_bodies):
            if idx >= len(lines):
                raise ValueError(f"Unexpected EOF while reading body info: {path}")

            body_info = lines[idx].split()
            idx += 1

            if idx >= len(lines):
                raise ValueError(f"Unexpected EOF while reading num_joints: {path}")

            num_joints = int(lines[idx])
            idx += 1

            joints = []
            for _joint_idx in range(num_joints):
                if idx >= len(lines):
                    raise ValueError(f"Unexpected EOF while reading joint info: {path}")

                vals = list(map(float, lines[idx].split()))
                idx += 1
                joints.append(vals)

            bodies.append(
                {
                    "body_info": body_info,
                    "joints": np.asarray(joints, dtype=np.float32),
                }
            )

        frames.append(bodies)

    return ParsedSkeleton(num_frames=num_frames, frames=frames)

def choose_primary_body_index(parsed: ParsedSkeleton) -> int:
    """
    Choose primary body index.

    Simple robust rule:
        Select the body index with the largest accumulated number of valid joints
        across frames.

    This assumes body ordering is reasonably stable, which is usually acceptable
    for single-person NTU actions. For two-person actions, this is still a
    practical first-pass heuristic.
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
            valid_count = float(np.sum(tracking_state > 0))
            scores[body_idx] += valid_count

    return int(np.argmax(scores))

def skeleton_to_primary_arrays(parsed: ParsedSkeleton) -> dict:
    """
    Convert parsed skeleton into fixed arrays for the primary body.

    Returns:
        joints3d:        [T, 25, 3]
        joints2d_color:  [T, 25, 2]
        joint_valid:     [T, 25]
        primary_body_idx
    """
    T = parsed.num_frames
    primary_body_idx = choose_primary_body_index(parsed)

    joints3d = np.full((T, 25, 3), np.nan, dtype=np.float32)
    joints2d_color = np.full((T, 25, 2), np.nan, dtype=np.float32)
    joint_valid = np.zeros((T, 25), dtype=bool)

    if primary_body_idx < 0:
        return {
            "joints3d": joints3d,
            "joints2d_color": joints2d_color,
            "joint_valid": joint_valid,
            "primary_body_idx": primary_body_idx,
        }

    for t, bodies in enumerate(parsed.frames):
        if len(bodies) <= primary_body_idx:
            continue

        joints = bodies[primary_body_idx]["joints"]

        if joints.shape[0] < 25 or joints.shape[1] < 12:
            continue

        joints = joints[:25]

        joints3d[t] = joints[:, 0:3]
        joints2d_color[t] = joints[:, 5:7]

        # trackingState:
        #   0 = not tracked
        #   1 = inferred
        #   2 = tracked
        joint_valid[t] = joints[:, -1] > 0

    return {
        "joints3d": joints3d,
        "joints2d_color": joints2d_color,
        "joint_valid": joint_valid,
        "primary_body_idx": primary_body_idx,
    }

def parse_ntu_rgb_name(name: str) -> Dict[str, str]:
    """
    Parse the name of NTU-RGB+D dataset to extract the information of the sample.

    Args:
        name (str): The file name of the RGB video file, which is in the format of "S{setup}C{camera}P{performer}R{replication}A{action}_rgb.avi"

    Returns:
        Dict[str, str]: A dictionary containing the information of the sample, including "setup", "camera", "performer", "replication", "action"
    """
    pattern = r"S(\d+)C(\d+)P(\d+)R(\d+)A(\d+)_rgb\.avi"
    match = re.match(pattern, name)
    if match:
        return {
            "setup": match.group(1),
            "camera": match.group(2),
            "performer": match.group(3),
            "replication": match.group(4),
            "action": match.group(5)
        }
    else:
        raise ValueError(f"Invalid file name: {name}")
    
def collect_ntu_rgb_data(root: str | Path) -> pd.DataFrame:
    """
    Collect the data of NTU-RGB+D dataset and return a dataframe containing the information of each sample.

    Args:
        root (str | Path): The root directory of the NTU-RGB+D dataset, which contains the RGB video files.

    Returns:
        pd.DataFrame: A dataframe containing the information of each sample.
    """
    root = file_io.ensure_path(file_io.check_path_raise(root))
    data = []
    for file in root.glob("*.avi"):

        # Parse the file name to extract the information of the sample
        info = parse_ntu_rgb_name(file.name)
        info["path"] = str(file)

        # Convert the action label to 0-based index
        info["label"] = int(info["action"]) - 1

        data.append(info)
    return pd.DataFrame(data)

def split_ntu_rgb_data(df: pd.DataFrame, strategy: str = "xsub") -> pd.DataFrame:
    """
    Split the data of NTU-RGB+D dataset into training and testing sets according to the given strategy.

    Args:
        df (pd.DataFrame): The dataframe containing the information of each sample.
        strategy (str): The strategy for splitting the data, which can be "xsub" or "xview". Default is "xsub".

    Returns:
        pd.DataFrame: A dataframe containing the information of each sample, with an additional column "split" indicating whether the sample is in the training set or the testing set.
    """

    column = STRATEGY_COLUMN_MAP.get(strategy, None)

    if column is None:
        raise ValueError(f"Invalid strategy: {strategy}")
    
    train_value_set = TRAIN_PERFORMER_IDS if strategy == "xsub" else TRAIN_CAMERA_IDS

    df["split"] = df[column].apply(lambda x: "train" if int(x) in train_value_set else "test")

    return df

def _build_skeleton_info_chunk(
    summary_df: pd.DataFrame,
    seg_bbox_df: pd.DataFrame,
    seg_motion_df: pd.DataFrame,
    frame_bbox_df: pd.DataFrame | None,
    frame_motion_df: pd.DataFrame | None,
    *,
    load_frame_level: bool,
) -> dict[str, dict[str, Any]]:
    """
    Build video_key -> skeleton_info dict for a subset of summary_df.

    This function is multiprocessing-friendly.
    Each worker receives only its chunk DataFrames.
    """
    seg_bbox_groups = (
        dict(tuple(seg_bbox_df.groupby("video_key", sort=False)))
        if seg_bbox_df is not None and len(seg_bbox_df)
        else {}
    )
    seg_motion_groups = (
        dict(tuple(seg_motion_df.groupby("video_key", sort=False)))
        if seg_motion_df is not None and len(seg_motion_df)
        else {}
    )

    frame_bbox_groups = {}
    frame_motion_groups = {}

    if load_frame_level and frame_bbox_df is not None and len(frame_bbox_df):
        frame_bbox_groups = dict(tuple(frame_bbox_df.groupby("video_key", sort=False)))

    if load_frame_level and frame_motion_df is not None and len(frame_motion_df):
        frame_motion_groups = dict(tuple(frame_motion_df.groupby("video_key", sort=False)))

    info_by_key: dict[str, dict[str, Any]] = {}

    for row in summary_df.itertuples(index=False):
        video_key = row.video_key

        num_segments = int(row.num_segments)
        skeleton_num_frames = int(row.skeleton_num_frames)

        info: dict[str, Any] = {
            "video_key": video_key,
            "skeleton_path": row.skeleton_path,
            "skeleton_num_frames": skeleton_num_frames,
            "video_num_frames": (
                None if pd.isna(row.video_num_frames) else int(row.video_num_frames)
            ),
            "image_width": int(row.image_width),
            "image_height": int(row.image_height),
            "num_segments": num_segments,
            "skeleton_valid": bool(row.skeleton_valid),
            "skeleton_valid_ratio": float(row.skeleton_valid_ratio),
            "primary_body_idx": int(row.primary_body_idx),

            "segment_boundaries": np.asarray(
                json.loads(row.segment_boundaries_json),
                dtype=np.int64,
            ),
            "segment_motion": np.asarray(
                json.loads(row.segment_motion_json),
                dtype=np.float32,
            ),
            "segment_motion_rank": np.asarray(
                json.loads(row.segment_motion_rank_json),
                dtype=np.int64,
            ),

            "high_motion_segment_idx": int(row.high_motion_segment_idx),
            "low_motion_segment_idx": int(row.low_motion_segment_idx),
            "transition_segment_idx": int(row.transition_segment_idx),

            "segment_bodypart_bbox": {},
            "segment_bodypart_valid_ratio": {},
            "segment_motion_by_part": {},
        }

        # segment-level body-part bbox
        if video_key in seg_bbox_groups:
            g = seg_bbox_groups[video_key]

            for part_name, pg in g.groupby("body_part", sort=False):
                pg = pg.sort_values("segment_idx")

                boxes = np.full((num_segments, 4), np.nan, dtype=np.float32)
                valid_ratio = np.zeros((num_segments,), dtype=np.float32)

                idx = pg["segment_idx"].to_numpy(dtype=np.int64)
                boxes[idx] = pg[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
                valid_ratio[idx] = pg["valid_ratio"].to_numpy(dtype=np.float32)

                info["segment_bodypart_bbox"][part_name] = boxes
                info["segment_bodypart_valid_ratio"][part_name] = valid_ratio

        # segment-level motion
        if video_key in seg_motion_groups:
            g = seg_motion_groups[video_key].sort_values("segment_idx")

            for col in g.columns:
                if not col.startswith("motion_"):
                    continue

                name = col[len("motion_"):]
                arr = np.zeros((num_segments,), dtype=np.float32)

                idx = g["segment_idx"].to_numpy(dtype=np.int64)
                arr[idx] = g[col].to_numpy(dtype=np.float32)

                info["segment_motion_by_part"][name] = arr

        # optional frame-level info
        if load_frame_level:
            info["frame_bodypart_bbox"] = {}
            info["frame_bodypart_valid"] = {}
            info["frame_motion_by_part"] = {}

            if video_key in frame_bbox_groups:
                g = frame_bbox_groups[video_key]

                for part_name, pg in g.groupby("body_part", sort=False):
                    pg = pg.sort_values("frame_idx")

                    boxes = np.full((skeleton_num_frames, 4), np.nan, dtype=np.float32)
                    valid = np.zeros((skeleton_num_frames,), dtype=bool)

                    idx = pg["frame_idx"].to_numpy(dtype=np.int64)
                    boxes[idx] = pg[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
                    valid[idx] = pg["valid"].to_numpy(dtype=np.int8).astype(bool)

                    info["frame_bodypart_bbox"][part_name] = boxes
                    info["frame_bodypart_valid"][part_name] = valid

            if video_key in frame_motion_groups:
                g = frame_motion_groups[video_key].sort_values("frame_idx")

                for col in g.columns:
                    if not col.startswith("motion_"):
                        continue

                    name = col[len("motion_"):]
                    arr = np.zeros((skeleton_num_frames,), dtype=np.float32)

                    idx = g["frame_idx"].to_numpy(dtype=np.int64)
                    arr[idx] = g[col].to_numpy(dtype=np.float32)

                    info["frame_motion_by_part"][name] = arr

        info_by_key[video_key] = info

    return info_by_key

def load_skeleton_metadata_to_memory(
    metadata_dir: str | Path,
    *,
    load_frame_level: bool = True,
    workers: int = 6,
) -> dict[str, dict[str, Any]]:
    """
    Load parquet skeleton metadata and return a video_key-based in-memory dict.

    If workers > 0:
        Build video_key chunks in parallel.

    Expected files:
        skeleton_summary.parquet
        segment_bodypart_bbox.parquet
        segment_bodypart_motion.parquet
        frame_bodypart_bbox.parquet      optional
        frame_motion.parquet             optional
    """
    metadata_dir = Path(metadata_dir)

    summary_df = pd.read_parquet(metadata_dir / "skeleton_summary.parquet")
    seg_bbox_df = pd.read_parquet(metadata_dir / "segment_bodypart_bbox.parquet")
    seg_motion_df = pd.read_parquet(metadata_dir / "segment_bodypart_motion.parquet")

    frame_bbox_df = None
    frame_motion_df = None

    if load_frame_level:
        frame_bbox_path = metadata_dir / "frame_bodypart_bbox.parquet"
        frame_motion_path = metadata_dir / "frame_motion.parquet"

        if frame_bbox_path.exists():
            frame_bbox_df = pd.read_parquet(frame_bbox_path)

        if frame_motion_path.exists():
            frame_motion_df = pd.read_parquet(frame_motion_path)

    # Sequential path: exactly one chunk.
    if workers is None or workers <= 0:
        return _build_skeleton_info_chunk(
            summary_df,
            seg_bbox_df,
            seg_motion_df,
            frame_bbox_df,
            frame_motion_df,
            load_frame_level=load_frame_level,
        )

    video_keys = summary_df["video_key"].drop_duplicates().to_numpy()
    workers = min(int(workers), len(video_keys))

    if workers <= 1:
        return _build_skeleton_info_chunk(
            summary_df,
            seg_bbox_df,
            seg_motion_df,
            frame_bbox_df,
            frame_motion_df,
            load_frame_level=load_frame_level,
        )

    key_chunks = np.array_split(video_keys, workers)

    jobs = []

    for key_chunk in key_chunks:
        key_set = set(key_chunk.tolist())

        summary_part = summary_df[summary_df["video_key"].isin(key_set)].copy()
        seg_bbox_part = seg_bbox_df[seg_bbox_df["video_key"].isin(key_set)].copy()
        seg_motion_part = seg_motion_df[seg_motion_df["video_key"].isin(key_set)].copy()

        frame_bbox_part = None
        frame_motion_part = None

        if load_frame_level and frame_bbox_df is not None:
            frame_bbox_part = frame_bbox_df[
                frame_bbox_df["video_key"].isin(key_set)
            ].copy()

        if load_frame_level and frame_motion_df is not None:
            frame_motion_part = frame_motion_df[
                frame_motion_df["video_key"].isin(key_set)
            ].copy()

        jobs.append(
            (
                summary_part,
                seg_bbox_part,
                seg_motion_part,
                frame_bbox_part,
                frame_motion_part,
            )
        )

    info_by_key: dict[str, dict[str, Any]] = {}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _build_skeleton_info_chunk,
                summary_part,
                seg_bbox_part,
                seg_motion_part,
                frame_bbox_part,
                frame_motion_part,
                load_frame_level=load_frame_level,
            )
            for (
                summary_part,
                seg_bbox_part,
                seg_motion_part,
                frame_bbox_part,
                frame_motion_part,
            ) in jobs
        ]

        for future in as_completed(futures):
            partial = future.result()
            info_by_key.update(partial)

    return info_by_key

def filter_invalid_skeleton_path_from_df(
    df: pd.DataFrame,
    skeleton_info_by_key: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """
    Filter out rows with invalid skeleton paths.

    A skeleton path is considered valid if it exists in the skeleton_info_by_key and is marked as valid.
    """
    def is_valid(row):
        path = row['path']
        video_key = normalize_video_key(path)
        if video_key not in skeleton_info_by_key:
            return False
        info = skeleton_info_by_key[video_key]
        return info.get("skeleton_valid", False)

    valid_mask = df.apply(is_valid, axis=1)
    return df[valid_mask].reset_index(drop=True)