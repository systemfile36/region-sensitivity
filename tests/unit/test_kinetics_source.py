"""Tests for the Kinetics-style annotation-CSV video source provider.

No real Kinetics data is used or required. Each test builds a tiny CSV and
clip directory that mirror the documented DeepMind Kinetics annotation/
filename convention (see ``ssat/core/source/kinetics.py``), then exercises
the provider against it -- this dataset's own real-data validation is
intentionally deferred.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from ssat.core.source import (
    KineticsSourceConfig,
    LoadedSample,
    SourceProviderError,
    VideoFolderSource,
    default_source_provider_registry,
)

_SIZE = 40  # matches test_video_folder_source.py's decord-safe clip size


def _write_video(path: Path, frame_count: int = 6) -> None:
    """Write a small mp4v clip whose frames each hold a distinct flat color."""

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (_SIZE, _SIZE))
    for index in range(frame_count):
        frame = np.full((_SIZE, _SIZE, 3), index * 20 % 256, dtype=np.uint8)
        writer.write(frame[:, :, ::-1])
    writer.release()


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["label", "youtube_id", "time_start", "time_end", "split", "is_cc"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


_ROWS = [
    {
        "label": "abseiling",
        "youtube_id": "aaaaaaaaaaa",
        "time_start": "0",
        "time_end": "10",
        "split": "train",
        "is_cc": "0",
    },
    {
        "label": "zumba",
        "youtube_id": "bbbbbbbbbbb",
        "time_start": "5",
        "time_end": "15",
        "split": "train",
        "is_cc": "0",
    },
    {
        "label": "abseiling",
        "youtube_id": "ccccccccccc",
        "time_start": "0",
        "time_end": "10",
        "split": "val",
        "is_cc": "0",
    },
]


def _build_clips(video_root: Path, rows: list[dict[str, str]]) -> None:
    for row in rows:
        start, end = int(row["time_start"]), int(row["time_end"])
        _write_video(video_root / f"{row['youtube_id']}_{start:06d}_{end:06d}.mp4")


def test_builds_source_from_annotation_csv_and_clip_root(tmp_path: Path) -> None:
    """Class indices default to the CSV's distinct labels, sorted."""

    csv_path = tmp_path / "kinetics400_train.csv"
    _write_csv(csv_path, _ROWS)
    video_root = tmp_path / "clips"
    _build_clips(video_root, _ROWS)

    registry = default_source_provider_registry()
    config = registry.parse(
        {
            "kind": "kinetics400",
            "csv_path": str(csv_path),
            "video_root": str(video_root),
            "num_frames": 4,
        }
    )
    assert isinstance(config, KineticsSourceConfig)
    source, provenance = registry.build(config, base_dir=tmp_path)

    assert isinstance(source, VideoFolderSource)
    assert provenance.kind == "kinetics400"
    assert provenance.manifest == csv_path.resolve()

    samples = {sample.sample_id: sample for sample in source.list_samples()}
    assert set(samples) == {
        "aaaaaaaaaaa_000000_000010",
        "bbbbbbbbbbb_000005_000015",
        "ccccccccccc_000000_000010",
    }
    # sorted(["abseiling", "zumba"]) -> abseiling=0, zumba=1
    assert samples["aaaaaaaaaaa_000000_000010"].gt_label == 0
    assert samples["bbbbbbbbbbb_000005_000015"].gt_label == 1
    assert samples["ccccccccccc_000000_000010"].gt_label == 0

    loaded = source.load("aaaaaaaaaaa_000000_000010")
    assert isinstance(loaded, LoadedSample)
    assert loaded.array.shape == (4, _SIZE, _SIZE, 3)
    assert loaded.gt_label == 0


def test_split_filters_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "kinetics400.csv"
    _write_csv(csv_path, _ROWS)
    video_root = tmp_path / "clips"
    _build_clips(video_root, _ROWS)

    registry = default_source_provider_registry()
    config = registry.parse(
        {
            "kind": "kinetics400",
            "csv_path": str(csv_path),
            "video_root": str(video_root),
            "split": "val",
        }
    )
    source, _ = registry.build(config, base_dir=tmp_path)
    assert [sample.sample_id for sample in source.list_samples()] == [
        "ccccccccccc_000000_000010"
    ]


def test_explicit_classes_pins_label_order(tmp_path: Path) -> None:
    csv_path = tmp_path / "kinetics400.csv"
    _write_csv(csv_path, _ROWS)
    video_root = tmp_path / "clips"
    _build_clips(video_root, _ROWS)

    registry = default_source_provider_registry()
    config = registry.parse(
        {
            "kind": "kinetics400",
            "csv_path": str(csv_path),
            "video_root": str(video_root),
            "classes": ["zumba", "abseiling"],
        }
    )
    source, _ = registry.build(config, base_dir=tmp_path)
    samples = {sample.sample_id: sample for sample in source.list_samples()}
    assert samples["aaaaaaaaaaa_000000_000010"].gt_label == 1  # abseiling -> index 1
    assert samples["bbbbbbbbbbb_000005_000015"].gt_label == 0  # zumba -> index 0


def test_explicit_classes_missing_a_label_is_rejected(tmp_path: Path) -> None:
    csv_path = tmp_path / "kinetics400.csv"
    _write_csv(csv_path, _ROWS)
    video_root = tmp_path / "clips"
    _build_clips(video_root, _ROWS)

    registry = default_source_provider_registry()
    config = registry.parse(
        {
            "kind": "kinetics400",
            "csv_path": str(csv_path),
            "video_root": str(video_root),
            "classes": ["zumba"],
        }
    )
    with pytest.raises(SourceProviderError, match="not in the configured classes"):
        registry.build(config, base_dir=tmp_path)


def test_rejects_missing_required_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["label", "youtube_id"])
        writer.writeheader()
        writer.writerow({"label": "abseiling", "youtube_id": "aaaaaaaaaaa"})
    video_root = tmp_path / "clips"
    video_root.mkdir()

    registry = default_source_provider_registry()
    config = registry.parse(
        {"kind": "kinetics400", "csv_path": str(csv_path), "video_root": str(video_root)}
    )
    with pytest.raises(SourceProviderError, match="missing required column"):
        registry.build(config, base_dir=tmp_path)


def test_rejects_duplicate_clip_id(tmp_path: Path) -> None:
    csv_path = tmp_path / "dup.csv"
    _write_csv(csv_path, [_ROWS[0], _ROWS[0]])
    video_root = tmp_path / "clips"
    _build_clips(video_root, [_ROWS[0]])

    registry = default_source_provider_registry()
    config = registry.parse(
        {"kind": "kinetics400", "csv_path": str(csv_path), "video_root": str(video_root)}
    )
    with pytest.raises(SourceProviderError, match="duplicate clip id"):
        registry.build(config, base_dir=tmp_path)


def test_rejects_split_with_no_matching_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "kinetics400.csv"
    _write_csv(csv_path, _ROWS)
    video_root = tmp_path / "clips"
    _build_clips(video_root, _ROWS)

    registry = default_source_provider_registry()
    config = registry.parse(
        {
            "kind": "kinetics400",
            "csv_path": str(csv_path),
            "video_root": str(video_root),
            "split": "test",
        }
    )
    with pytest.raises(SourceProviderError, match="matched no rows"):
        registry.build(config, base_dir=tmp_path)
