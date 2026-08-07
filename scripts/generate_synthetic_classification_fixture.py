#!/usr/bin/env python3
"""Generate the deterministic image fixture used by SSAT E2E tests.

The command performs no network access. It creates 18 valid RGB PNG files and
two intentionally corrupt files by default, together with a JSON manifest.
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

import numpy as np
from numpy.typing import NDArray
from PIL import Image


FIXTURE_VERSION = "1.0.0"
DEFAULT_SEED = 20260807
DEFAULT_VALID_COUNT = 18
DEFAULT_IMAGE_SIZE = 64
CORRUPT_FILES = (
    ("synthetic-corrupt-truncated", "corrupt/truncated.png", "truncated_png"),
    ("synthetic-corrupt-invalid", "corrupt/invalid.png", "invalid_image_bytes"),
)
CLASS_NAMES = ("gradient", "geometry", "texture")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "synthetic_classification"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments without mutating the filesystem."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic, repository-safe classification fixtures "
            "without downloading external images."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"fixture destination (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"non-negative root seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--valid-count",
        type=int,
        default=DEFAULT_VALID_COUNT,
        help=f"number of valid RGB PNG files (default: {DEFAULT_VALID_COUNT})",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"square image size in pixels (default: {DEFAULT_IMAGE_SIZE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only manifest.json, images/, and corrupt/ in output-dir",
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
        f"samples in {output_dir}"
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

    images_dir = output_dir / "images"
    corrupt_dir = output_dir / "corrupt"
    images_dir.mkdir(parents=True)
    corrupt_dir.mkdir(parents=True)
    samples: list[dict[str, object]] = []

    generators: tuple[Callable[..., NDArray[np.uint8]], ...] = (
        _gradient_image,
        _geometry_image,
        _texture_image,
    )
    for index in range(valid_count):
        label = index % len(CLASS_NAMES)
        variation = index // len(CLASS_NAMES)
        rng = _sample_rng(seed, index)
        array = generators[label](size=size, variation=variation, rng=rng)
        _validate_image_array(array, size=size)
        relative_path = Path("images") / f"sample_{index:03d}.png"
        destination = output_dir / relative_path
        Image.fromarray(array, mode="RGB").save(
            destination,
            format="PNG",
            optimize=False,
            compress_level=9,
        )
        samples.append(
            {
                "sample_id": f"synthetic-{index:03d}",
                "path": relative_path.as_posix(),
                "gt_label": label,
                "class_name": CLASS_NAMES[label],
                "expected_status": "ok",
                "width": size,
                "height": size,
                "mode": "RGB",
                "content_sha256": _sha256_file(destination),
                "pixel_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        )

    corrupt_payloads = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
        b"This file is intentionally not a decodable image.\n",
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
                "mode": None,
                "content_sha256": _sha256_file(destination),
                "pixel_sha256": None,
            }
        )

    manifest: dict[str, object] = {
        "fixture_version": FIXTURE_VERSION,
        "description": (
            "Deterministic procedural images for SSAT execution-layer E2E tests."
        ),
        "generator": Path(__file__).name,
        "seed": seed,
        "image_size": [size, size],
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


def _gradient_image(
    *,
    size: int,
    variation: int,
    rng: np.random.Generator,
) -> NDArray[np.uint8]:
    coordinates = np.arange(size, dtype=np.uint32)
    denominator = max(size - 1, 1)
    horizontal = (coordinates * 255 // denominator).astype(np.uint8)
    vertical = horizontal[:, None]
    horizontal = horizontal[None, :]
    offset = int(rng.integers(0, 256))
    red = np.broadcast_to((horizontal.astype(np.uint16) + offset) % 256, (size, size))
    green = np.broadcast_to(
        (vertical.astype(np.uint16) + variation * 29) % 256,
        (size, size),
    )
    blue = (
        horizontal.astype(np.uint16)
        + vertical.astype(np.uint16)
        + variation * 17
    ) % 256
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


def _geometry_image(
    *,
    size: int,
    variation: int,
    rng: np.random.Generator,
) -> NDArray[np.uint8]:
    array = np.empty((size, size, 3), dtype=np.uint8)
    palette = rng.integers(24, 232, size=(4, 3), dtype=np.uint8)
    midpoint = size // 2
    array[:midpoint, :midpoint] = palette[0]
    array[:midpoint, midpoint:] = palette[1]
    array[midpoint:, :midpoint] = palette[2]
    array[midpoint:, midpoint:] = palette[3]

    radius = max(size // 8, size // 5 + variation % max(size // 16, 1))
    center_x = size // 3 + (variation * 7) % max(size // 3, 1)
    center_y = size // 3 + (variation * 5) % max(size // 3, 1)
    yy, xx = np.ogrid[:size, :size]
    circle = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    array[circle] = np.array([245, 245 - variation % 80, 32], dtype=np.uint8)

    rectangle_width = max(size // 6, 1)
    start = (variation * 11) % max(size - rectangle_width, 1)
    array[size // 8 : size // 4, start : start + rectangle_width] = (16, 240, 208)
    return array


def _texture_image(
    *,
    size: int,
    variation: int,
    rng: np.random.Generator,
) -> NDArray[np.uint8]:
    yy, xx = np.indices((size, size), dtype=np.int32)
    cell_size = 2 + variation % 7
    checker = ((xx // cell_size + yy // cell_size) % 2) * 176 + 40
    stripe_width = 1 + variation % 5
    stripes = ((xx + variation * yy) // stripe_width % 2) * 160 + 48
    diagonal = (xx * 5 + yy * 3 + variation * 19) % 256
    array = np.stack((checker, stripes, diagonal), axis=-1).astype(np.int16)
    noise = rng.integers(-18, 19, size=array.shape, dtype=np.int16)
    return np.clip(array + noise, 0, 255).astype(np.uint8)


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
        output_dir / "images",
        output_dir / "corrupt",
    )
    existing = [path for path in generated_targets if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise SystemExit(f"generated targets already exist ({names}); use --force")


def _publish(staging_dir: Path, output_dir: Path, *, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = ("images", "corrupt", "manifest.json")
    if force:
        for name in targets:
            destination = output_dir / name
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink(missing_ok=True)
    for name in targets:
        os.replace(staging_dir / name, output_dir / name)


def _validate_staged_fixture(
    staging_dir: Path,
    manifest: dict[str, object],
) -> None:
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


def _validate_image_array(array: NDArray[np.uint8], *, size: int) -> None:
    if array.dtype != np.uint8 or array.shape != (size, size, 3):
        raise RuntimeError("image generator returned an invalid RGB array")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
