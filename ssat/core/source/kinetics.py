"""Kinetics-style video source provider.

Builds a :class:`VideoFolderSource` from DeepMind's Kinetics annotation CSV
format (``label,youtube_id,time_start,time_end,split[,is_cc]``, one header
row) -- the format Kinetics-400/600/700 ship, and that most third-party
download mirrors (e.g. https://github.com/cvdfoundation/kinetics-dataset)
preserve for the local clip filenames they produce:
``<youtube_id>_<time_start:06d>_<time_end:06d>.<ext>`` inside a flat
directory per split.

The CSV does not publish class indices, so this provider derives them by
sorting the CSV's distinct ``label`` strings alphabetically -- the same
convention most public Kinetics ``label_map.txt`` files and training recipes
(e.g. mmaction2) use. Pass an explicit ``classes`` list in the config to pin
a different order (e.g. to match a pretrained checkpoint's head).

This provider has been validated only against small hand-built fixtures that
mirror the documented CSV/filename convention (see
``tests/unit/test_kinetics_source.py``) -- it has not been exercised against
a real, full-scale Kinetics download.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

from pydantic import Field

from ssat.core.config.schema import SourceProvenance
from ssat.core.source.provider import SourceProvider, SourceProviderConfig, SourceProviderError
from ssat.core.source.types import SampleMeta
from ssat.core.source.video_folder import VideoFolderSource
from ssat.utils.io import sha256_file

_REQUIRED_COLUMNS = frozenset({"label", "youtube_id", "time_start", "time_end"})


class KineticsSourceConfig(SourceProviderConfig):
    """Configure a Kinetics-style source from an annotation CSV and clip root.

    Attributes:
        csv_path: Kinetics-format annotation CSV (``label,youtube_id,
            time_start,time_end,split`` header, extra columns ignored).
        video_root: Directory containing ``<youtube_id>_<start>_<end>.<ext>``
            clip files.
        num_frames: Positive number of frames sampled per clip.
        split: Optional ``split`` column value to filter rows to (e.g.
            ``"train"``). ``None`` keeps every row.
        extension: Clip filename extension, without a leading dot.
        classes: Optional explicit, ordered class list. When omitted, classes
            are the CSV's distinct ``label`` values sorted alphabetically.
    """

    kind: Literal["kinetics400"] = "kinetics400"
    csv_path: Path
    video_root: Path
    num_frames: int = Field(default=16, gt=0)
    split: str | None = None
    extension: str = "mp4"
    classes: tuple[str, ...] | None = None


class KineticsSourceProvider(SourceProvider):
    """Build a :class:`VideoFolderSource` from a Kinetics-style annotation CSV."""

    name = "kinetics400"
    config_model = KineticsSourceConfig

    def build(
        self, config: SourceProviderConfig, *, base_dir: Path
    ) -> tuple[VideoFolderSource, SourceProvenance]:
        if not isinstance(config, KineticsSourceConfig):
            raise TypeError("config must be KineticsSourceConfig")

        csv_path = _resolve_existing(config.csv_path, base_dir)
        video_root = _resolve_existing(config.video_root, base_dir)
        if not video_root.is_dir():
            raise SourceProviderError(f"kinetics video_root is not a directory: {video_root}")

        rows = _read_annotation_rows(csv_path, split=config.split)
        classes = config.classes or tuple(sorted({row["label"] for row in rows}))
        class_to_index = {name: index for index, name in enumerate(classes)}

        samples: list[SampleMeta] = []
        seen_ids: set[str] = set()
        for row in rows:
            label = row["label"]
            if label not in class_to_index:
                raise SourceProviderError(
                    f"{csv_path}: label {label!r} is not in the configured classes list"
                )
            try:
                start, end = int(row["time_start"]), int(row["time_end"])
            except ValueError as error:
                raise SourceProviderError(
                    f"{csv_path}: time_start/time_end must be integers "
                    f"(youtube_id={row['youtube_id']!r})"
                ) from error
            sample_id = f"{row['youtube_id']}_{start:06d}_{end:06d}"
            if sample_id in seen_ids:
                raise SourceProviderError(f"{csv_path}: duplicate clip id {sample_id!r}")
            seen_ids.add(sample_id)
            video_path = video_root / f"{sample_id}.{config.extension}"
            samples.append(SampleMeta(sample_id, video_path, class_to_index[label]))

        if not samples:
            raise SourceProviderError(
                f"kinetics csv matched no rows: {csv_path} (split={config.split!r})"
            )

        provenance = SourceProvenance(
            kind=config.kind,
            manifest=csv_path,
            manifest_hash=sha256_file(csv_path),
        )
        return VideoFolderSource(samples, num_frames=config.num_frames), provenance


def _resolve_existing(path: Path, base_dir: Path) -> Path:
    """Resolve a config-relative path, requiring it to already exist."""

    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved.resolve(strict=True)


def _read_annotation_rows(csv_path: Path, *, split: str | None) -> list[dict[str, str]]:
    """Read and optionally split-filter a Kinetics annotation CSV's rows.

    Args:
        csv_path: Existing CSV file with a header row.
        split: Optional ``split`` column value to keep; ``None`` keeps all.

    Returns:
        Row dictionaries in file order.

    Raises:
        SourceProviderError: If the file is missing or lacks a required
            column.
    """

    if not csv_path.is_file():
        raise SourceProviderError(f"kinetics csv_path is not a file: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = _REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise SourceProviderError(f"{csv_path}: missing required column(s): {sorted(missing)}")
        rows = list(reader)
    if split is not None:
        rows = [row for row in rows if row.get("split") == split]
    return rows
