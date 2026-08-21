"""ImageNet-style source provider.

Builds an :class:`ImageFolderSource` from a flat plain-text annotation file
listing ``<relative_path> <label_index>`` pairs, one sample per line,
resolved against an image root directory. This "file list" format predates
ImageNet itself (Caffe's ``ImageDataLayer``) but remains the most common way
ImageNet-derived classification datasets are consumed outside torchvision's
directory-per-class ``ImageFolder`` layout, so this provider also covers any
other dataset that reuses the same annotation convention -- nothing here is
tied to ImageNet's specific 1000 classes.

Real ImageNet distributions do not ship this file directly; it is the
normalized form most training pipelines reduce ``LOC_val_solution.csv``,
``valprep.sh``-reorganized folders, or the devkit's ground-truth files into
before training. Producing it from a raw ImageNet download is left to the
caller, mirroring how ``scripts/dataset_prep/ntu_rgb_d.py`` is a separate,
dataset-specific offline step for NTU-RGB+D.

This provider has been validated only against small hand-built fixtures that
mirror the documented annotation format (see
``tests/unit/test_imagenet_source.py``) -- it has not been exercised against
a real, full-scale ImageNet download.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ssat.core.config.schema import SourceProvenance
from ssat.core.source.image_folder import ImageFolderSource
from ssat.core.source.provider import SourceProvider, SourceProviderConfig, SourceProviderError
from ssat.core.source.types import SampleMeta
from ssat.utils.io import sha256_file


class ImageNetSourceConfig(SourceProviderConfig):
    """Configure an ImageNet-style source from an image root and file list.

    Attributes:
        root: Directory that annotation file paths are resolved against.
        annotation_file: Text file with one ``<relative_path> <label>`` pair
            per line (whitespace-separated, 0-based non-negative integer
            label).
    """

    kind: Literal["imagenet"] = "imagenet"
    root: Path
    annotation_file: Path


class ImageNetSourceProvider(SourceProvider):
    """Build an :class:`ImageFolderSource` from an ImageNet-style file list."""

    name = "imagenet"
    config_model = ImageNetSourceConfig

    def build(
        self, config: SourceProviderConfig, *, base_dir: Path
    ) -> tuple[ImageFolderSource, SourceProvenance]:
        if not isinstance(config, ImageNetSourceConfig):
            raise TypeError("config must be ImageNetSourceConfig")

        annotation_path = _resolve_existing(config.annotation_file, base_dir)
        root = _resolve_existing(config.root, base_dir)
        if not root.is_dir():
            raise SourceProviderError(f"imagenet root is not a directory: {root}")

        samples = _parse_annotation_file(annotation_path, root)
        provenance = SourceProvenance(
            kind=config.kind,
            manifest=annotation_path,
            manifest_hash=sha256_file(annotation_path),
            loader_parameters={"root": str(root)},
        )
        return ImageFolderSource(samples), provenance


def _resolve_existing(path: Path, base_dir: Path) -> Path:
    """Resolve a config-relative path, requiring it to already exist."""

    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved.resolve(strict=True)


def _parse_annotation_file(annotation_path: Path, root: Path) -> list[SampleMeta]:
    """Parse a ``<relative_path> <label>`` file list into ``SampleMeta`` values.

    Args:
        annotation_path: File containing one sample per non-blank line.
        root: Directory relative paths are resolved against.

    Returns:
        One ``SampleMeta`` per line, in file order.

    Raises:
        SourceProviderError: If the file is missing, empty, malformed, or
            contains a duplicate relative path.
    """

    if not annotation_path.is_file():
        raise SourceProviderError(f"imagenet annotation_file is not a file: {annotation_path}")

    samples: list[SampleMeta] = []
    seen_paths: set[str] = set()
    with annotation_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.rsplit(maxsplit=1)
            if len(parts) != 2:
                raise SourceProviderError(
                    f"{annotation_path}:{line_number}: expected '<relative_path> <label>'"
                )
            relative_path, label_text = parts
            try:
                label = int(label_text)
            except ValueError as error:
                raise SourceProviderError(
                    f"{annotation_path}:{line_number}: label must be an integer, "
                    f"got {label_text!r}"
                ) from error
            if label < 0:
                raise SourceProviderError(
                    f"{annotation_path}:{line_number}: label must be non-negative, got {label}"
                )
            if relative_path in seen_paths:
                raise SourceProviderError(
                    f"{annotation_path}:{line_number}: duplicate sample path {relative_path!r}"
                )
            seen_paths.add(relative_path)
            samples.append(SampleMeta(relative_path, (root / relative_path).resolve(), label))

    if not samples:
        raise SourceProviderError(f"imagenet annotation_file has no samples: {annotation_path}")
    return samples
