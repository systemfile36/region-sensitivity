"""Tests for the ImageNet-style file-list source provider.

No real ImageNet data is used or required. Each test builds a tiny directory
tree and annotation file that mirror the documented file-list convention
(see ``ssat/core/source/imagenet.py``), then exercises the provider against
it -- this dataset's own real-data validation is intentionally deferred.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ssat.core.source import (
    ImageFolderSource,
    ImageNetSourceConfig,
    LoadedSample,
    SourceProviderError,
    default_source_provider_registry,
)


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), color=value).save(path)


def _write_annotation_file(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_builds_source_from_synset_style_tree_and_file_list(tmp_path: Path) -> None:
    """The classic ``<relative_path> <label>`` file list resolves against root."""

    root = tmp_path / "images"
    _write_image(root / "n01440764" / "a.png", 10)
    _write_image(root / "n02391049" / "b.png", 20)
    annotation_file = tmp_path / "train.txt"
    _write_annotation_file(
        annotation_file,
        ["n01440764/a.png 0", "n02391049/b.png 1"],
    )

    registry = default_source_provider_registry()
    config = registry.parse(
        {"kind": "imagenet", "root": str(root), "annotation_file": str(annotation_file)}
    )
    assert isinstance(config, ImageNetSourceConfig)
    source, provenance = registry.build(config, base_dir=tmp_path)

    assert isinstance(source, ImageFolderSource)
    assert provenance.kind == "imagenet"
    assert provenance.manifest == annotation_file.resolve()
    assert len(provenance.manifest_hash) == 64
    assert provenance.loader_parameters == {"root": str(root.resolve())}

    samples = {sample.sample_id: sample for sample in source.list_samples()}
    assert set(samples) == {"n01440764/a.png", "n02391049/b.png"}
    assert samples["n01440764/a.png"].gt_label == 0
    assert samples["n02391049/b.png"].gt_label == 1

    loaded = source.load("n01440764/a.png")
    assert isinstance(loaded, LoadedSample)
    assert loaded.array.shape == (1, 3, 4, 3)
    assert loaded.gt_label == 0


def test_paths_resolve_relative_to_base_dir(tmp_path: Path) -> None:
    """root/annotation_file given as relative paths resolve against base_dir."""

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    root = tmp_path / "images"
    _write_image(root / "n01440764" / "a.png", 5)
    annotation_file = tmp_path / "train.txt"
    _write_annotation_file(annotation_file, ["n01440764/a.png 0"])

    registry = default_source_provider_registry()
    config = registry.parse(
        {"kind": "imagenet", "root": "../images", "annotation_file": "../train.txt"}
    )
    source, _ = registry.build(config, base_dir=config_dir)
    assert len(source.list_samples()) == 1


@pytest.mark.parametrize(
    "lines,match",
    [
        (["n01440764/a.png"], "expected"),
        (["n01440764/a.png notanumber"], "must be an integer"),
        (["n01440764/a.png -1"], "non-negative"),
        (["n01440764/a.png 0", "n01440764/a.png 1"], "duplicate"),
    ],
)
def test_rejects_malformed_annotation_lines(
    tmp_path: Path, lines: list[str], match: str
) -> None:
    root = tmp_path / "images"
    _write_image(root / "n01440764" / "a.png", 5)
    annotation_file = tmp_path / "train.txt"
    _write_annotation_file(annotation_file, lines)

    registry = default_source_provider_registry()
    config = registry.parse(
        {"kind": "imagenet", "root": str(root), "annotation_file": str(annotation_file)}
    )
    with pytest.raises(SourceProviderError, match=match):
        registry.build(config, base_dir=tmp_path)


def test_rejects_empty_annotation_file(tmp_path: Path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    annotation_file = tmp_path / "train.txt"
    annotation_file.write_text("\n\n", encoding="utf-8")

    registry = default_source_provider_registry()
    config = registry.parse(
        {"kind": "imagenet", "root": str(root), "annotation_file": str(annotation_file)}
    )
    with pytest.raises(SourceProviderError, match="no samples"):
        registry.build(config, base_dir=tmp_path)


def test_rejects_missing_root_or_annotation_file(tmp_path: Path) -> None:
    registry = default_source_provider_registry()

    with pytest.raises(SourceProviderError):
        config = registry.parse(
            {
                "kind": "imagenet",
                "root": str(tmp_path / "missing-root"),
                "annotation_file": str(tmp_path / "missing.txt"),
            }
        )
        registry.build(config, base_dir=tmp_path)
