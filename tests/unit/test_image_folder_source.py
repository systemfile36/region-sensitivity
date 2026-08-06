"""Tests for the explicit image catalog source."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ssat.core.source import ImageFolderSource, LoadError, LoadedSample, SampleMeta
from ssat.utils.io import sha256_file


def _write_image(path: Path, mode: str, size: tuple[int, int], value: object) -> None:
    """Write a small Pillow image used by source tests."""

    Image.new(mode, size, color=value).save(path)


def test_lists_metadata_without_loading_and_rejects_duplicates(tmp_path: Path) -> None:
    """The source preserves its explicit catalog and enforces unique IDs."""

    sample = SampleMeta("sample", tmp_path / "sample.png", gt_label=4)
    source = ImageFolderSource((sample,))

    first = source.list_samples()
    second = source.list_samples()

    assert first == [sample]
    assert second == [sample]
    assert first is not second
    first.clear()
    assert source.list_samples() == [sample]

    with pytest.raises(ValueError, match="duplicate"):
        ImageFolderSource((sample, sample))
    with pytest.raises(TypeError, match="SampleMeta"):
        ImageFolderSource((sample, object()))  # type: ignore[arg-type]


@pytest.mark.parametrize("size", [(3, 2), (7, 5)])
def test_loads_images_as_thwc_rgb_with_provenance(
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    """Decoded images use RGB uint8 layout and retain labels and hashes."""

    path = tmp_path / f"image-{size[0]}x{size[1]}.png"
    _write_image(path, "L", size, 37)
    source = ImageFolderSource((SampleMeta("image", path, gt_label=2),))

    loaded = source.load("image")

    assert isinstance(loaded, LoadedSample)
    assert loaded.array.shape == (1, size[1], size[0], 3)
    assert loaded.original_shape == loaded.array.shape
    assert loaded.array.dtype == np.uint8
    assert np.all(loaded.array == 37)
    assert loaded.content_hash == sha256_file(path)
    assert loaded.gt_label == 2


def test_load_returns_recoverable_errors(tmp_path: Path) -> None:
    """Unknown, missing, and corrupt samples are represented as values."""

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    source = ImageFolderSource(
        (
            SampleMeta("missing", tmp_path / "missing.png"),
            SampleMeta("corrupt", corrupt),
        )
    )

    unknown = source.load("unknown")
    missing = source.load("missing")
    broken = source.load("corrupt")

    assert isinstance(unknown, LoadError)
    assert unknown.error_type == "sample_not_found"
    assert isinstance(missing, LoadError)
    assert missing.error_type == "file_not_found"
    assert isinstance(broken, LoadError)
    assert broken.error_type == "decode_error"
    assert "UnidentifiedImageError" in broken.message
