#!/usr/bin/env python3
"""Generate the deterministic video fixture used by SSAT video source tests.

The command performs no network access. It creates small mp4v-encoded MP4
clips (a moving bright square over a per-class background, so frame content
is distinguishable across time) plus two intentionally corrupt files, and a
JSON manifest mirroring ``generate_synthetic_classification_fixture.py``.
Existing generated targets are preserved unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray


FIXTURE_VERSION = "1.0.0"
DEFAULT_SEED = 20260812
DEFAULT_VALID_COUNT = 12
DEFAULT_FRAME_SIZE = 64  # some small sizes trip a decord/mp4v get_batch size bug
DEFAULT_FPS = 8
CORRUPT_FILES = (
    ("synthetic-corrupt-truncated", "corrupt/truncated.mp4", "truncated_mp4"),
    ("synthetic-corrupt-invalid", "corrupt/invalid.mp4", "invalid_video_bytes"),
)
CLASS_NAMES = ("sweep", "pulse", "bounce")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "tests" / "fixtures" / "synthetic_video"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments without mutating the filesystem."""

    parser = argparse.ArgumentParser(
        description="Generate deterministic, repository-safe video fixtures."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--valid-count", type=int, default=DEFAULT_VALID_COUNT)
    parser.add_argument("--size", type=int, default=DEFAULT_FRAME_SIZE)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only manifest.json, videos/, and corrupt/ in output-dir",
    )
    return parser.parse_args()


def main() -> int:
    """Generate a staged fixture and publish it after all files validate."""

    args = parse_args()
    _validate_options(args.seed, args.valid_count, args.size)
    output_dir = args.output_dir.expanduser().resolve()
    _validate_output_dir(output_dir)
    _ensure_publishable(output_dir, force=args.force)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.generate-",
    ) as temporary:
        staging_dir = Path(temporary)
        manifest = generate_fixture(
            staging_dir,
            seed=args.seed,
            valid_count=args.valid_count,
            size=args.size,
        )
        _validate_staged_fixture(staging_dir, manifest)
        _publish(staging_dir, output_dir, force=args.force)

    print(
        f"Generated {args.valid_count} valid and {len(CORRUPT_FILES)} corrupt "
        f"video samples in {output_dir}"
    )
    return 0


def generate_fixture(
    output_dir: Path,
    *,
    seed: int,
    valid_count: int,
    size: int,
) -> dict[str, object]:
    """Generate all artifacts under an empty staging directory."""

    videos_dir = output_dir / "videos"
    corrupt_dir = output_dir / "corrupt"
    videos_dir.mkdir(parents=True)
    corrupt_dir.mkdir(parents=True)
    samples: list[dict[str, object]] = []

    generators: tuple[Callable[..., NDArray[np.uint8]], ...] = (
        _sweep_clip,
        _pulse_clip,
        _bounce_clip,
    )
    for index in range(valid_count):
        label = index % len(CLASS_NAMES)
        variation = index // len(CLASS_NAMES)
        rng = _sample_rng(seed, index)
        num_frames = 10 + int(rng.integers(0, 8))  # varies clip length on purpose
        frames = generators[label](
            size=size, num_frames=num_frames, variation=variation, rng=rng
        )
        _validate_clip_array(frames, size=size, num_frames=num_frames)
        relative_path = Path("videos") / f"sample_{index:03d}.mp4"
        destination = output_dir / relative_path
        _write_clip(destination, frames, fps=DEFAULT_FPS)
        samples.append(
            {
                "sample_id": f"synthetic-video-{index:03d}",
                "path": relative_path.as_posix(),
                "gt_label": label,
                "class_name": CLASS_NAMES[label],
                "expected_status": "ok",
                "width": size,
                "height": size,
                "num_source_frames": num_frames,
                "content_sha256": _sha256_file(destination),
            }
        )

    corrupt_payloads = (
        b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00",
        b"This file is intentionally not a decodable video.\n",
    )
    for (sample_id, relative, error_kind), payload in zip(
        CORRUPT_FILES,
        corrupt_payloads,
        strict=True,
    ):
        destination = output_dir / relative
        destination.write_bytes(payload)
        samples.append(
            {
                "sample_id": sample_id,
                "path": relative,
                "gt_label": None,
                "class_name": None,
                "expected_status": "load_failed",
                "expected_error_kind": error_kind,
                "width": None,
                "height": None,
                "num_source_frames": None,
                "content_sha256": _sha256_file(destination),
            }
        )

    manifest: dict[str, object] = {
        "fixture_version": FIXTURE_VERSION,
        "description": (
            "Deterministic procedural video clips for SSAT video source tests."
        ),
        "generator": Path(__file__).name,
        "seed": seed,
        "frame_size": [size, size],
        "fps": DEFAULT_FPS,
        "classes": [
            {"id": class_id, "name": name}
            for class_id, name in enumerate(CLASS_NAMES)
        ],
        "valid_count": valid_count,
        "corrupt_count": len(CORRUPT_FILES),
        "samples": samples,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_clip(path: Path, frames: NDArray[np.uint8], *, fps: int) -> None:
    """Encode RGB frames as an mp4v MP4 clip readable by decord/ffmpeg."""

    height, width = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    try:
        for frame in frames:
            writer.write(frame[:, :, ::-1])  # RGB -> BGR for cv2's writer
    finally:
        writer.release()


def _sweep_clip(
    *, size: int, num_frames: int, variation: int, rng: np.random.Generator
) -> NDArray[np.uint8]:
    """A bright square sweeping left-to-right over a vertical gradient."""

    frames = np.empty((num_frames, size, size, 3), dtype=np.uint8)
    gradient = (np.arange(size, dtype=np.uint16) * 255 // max(size - 1, 1)).astype(np.uint8)
    base_color = rng.integers(24, 200, size=3, dtype=np.uint8)
    square = max(size // 6, 4)
    for t in range(num_frames):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:, :, 0] = base_color[0]
        frame[:, :, 1] = gradient[:, None]
        frame[:, :, 2] = base_color[2]
        x = int((t / max(num_frames - 1, 1)) * (size - square))
        y = (size - square) // 2 + (variation * 3) % max(size - square, 1)
        y = min(y, size - square)
        frame[y : y + square, x : x + square] = (250, 250, 20)
        frames[t] = frame
    return frames


def _pulse_clip(
    *, size: int, num_frames: int, variation: int, rng: np.random.Generator
) -> NDArray[np.uint8]:
    """A centered square that pulses in size across the clip."""

    frames = np.empty((num_frames, size, size, 3), dtype=np.uint8)
    background = rng.integers(16, 96, size=3, dtype=np.uint8)
    max_radius = size // 2 - 2
    for t in range(num_frames):
        frame = np.broadcast_to(background, (size, size, 3)).copy()
        phase = (t + variation) / max(num_frames, 1)
        radius = max(2, int(max_radius * abs(np.sin(phase * np.pi))))
        yy, xx = np.ogrid[:size, :size]
        center = size // 2
        disk = (xx - center) ** 2 + (yy - center) ** 2 <= radius**2
        frame[disk] = (30, 220, 220)
        frames[t] = frame
    return frames


def _bounce_clip(
    *, size: int, num_frames: int, variation: int, rng: np.random.Generator
) -> NDArray[np.uint8]:
    """A small square bouncing vertically over a checkerboard background."""

    frames = np.empty((num_frames, size, size, 3), dtype=np.uint8)
    yy, xx = np.indices((size, size), dtype=np.int32)
    cell = 2 + variation % 5
    checker = ((xx // cell + yy // cell) % 2) * 160 + 32
    square = max(size // 7, 4)
    period = max(num_frames - 1, 1)
    for t in range(num_frames):
        frame = np.stack([checker] * 3, axis=-1).astype(np.uint8)
        triangle = abs((t % (2 * period)) - period) / period
        y = int(triangle * (size - square))
        x = (size - square) // 2 + (variation * 5) % max(size - square, 1)
        x = min(x, size - square)
        frame[y : y + square, x : x + square] = (250, 60, 60)
        frames[t] = frame
    return frames


def _sample_rng(seed: int, index: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([seed, index])
    return np.random.Generator(np.random.PCG64(sequence))


def _validate_options(seed: int, valid_count: int, size: int) -> None:
    if seed < 0:
        raise SystemExit("--seed must be non-negative")
    if valid_count <= 0:
        raise SystemExit("--valid-count must be positive")
    if size < 16:
        raise SystemExit("--size must be at least 16")


def _validate_output_dir(output_dir: Path) -> None:
    if output_dir == output_dir.parent:
        raise SystemExit("refusing to use a filesystem root as --output-dir")
    if output_dir == REPOSITORY_ROOT:
        raise SystemExit("refusing to use the repository root as --output-dir")


def _ensure_publishable(output_dir: Path, *, force: bool) -> None:
    generated_targets = (
        output_dir / "manifest.json",
        output_dir / "videos",
        output_dir / "corrupt",
    )
    existing = [path for path in generated_targets if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise SystemExit(f"generated targets already exist ({names}); use --force")


def _publish(staging_dir: Path, output_dir: Path, *, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = ("videos", "corrupt", "manifest.json")
    if force:
        for name in targets:
            destination = output_dir / name
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink(missing_ok=True)
    for name in targets:
        os.replace(staging_dir / name, output_dir / name)


def _validate_staged_fixture(staging_dir: Path, manifest: dict[str, object]) -> None:
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise RuntimeError("generated manifest has no sample list")
    sample_ids: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise RuntimeError("generated manifest contains an invalid sample")
        sample_id = sample.get("sample_id")
        relative = sample.get("path")
        if not isinstance(sample_id, str) or sample_id in sample_ids:
            raise RuntimeError("generated manifest contains duplicate sample IDs")
        if not isinstance(relative, str):
            raise RuntimeError("generated manifest contains an invalid path")
        path = staging_dir / relative
        if not path.is_file() or _sha256_file(path) != sample.get("content_sha256"):
            raise RuntimeError(f"generated fixture hash mismatch: {relative}")
        sample_ids.add(sample_id)


def _validate_clip_array(frames: NDArray[np.uint8], *, size: int, num_frames: int) -> None:
    if frames.dtype != np.uint8 or frames.shape != (num_frames, size, size, 3):
        raise RuntimeError("clip generator returned an invalid RGB array")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
