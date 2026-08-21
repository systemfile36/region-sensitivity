"""Private helpers shared by every DebugViz view.

Mirrors the precedent set by ``ssat.metrics._storage`` (a private module
``store.py`` alone consumes): each public view module (``mask_check.py``,
``heatmap.py``, ``ranking.py``) imports from here instead of from one
another, so a bug in one view's own logic can never break another view's
import — even if one view breaks, the rest must stay usable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ssat.core.source import (
    ImageFolderSource,
    ImageNetSourceConfig,
    ImageNetSourceProvider,
    SourceProviderError,
)
from ssat.core.source.types import SampleMeta
from ssat.metrics.dump_reader import DumpHandle
from ssat.metrics.errors import DebugVizError
from ssat.utils.io import load_json, load_yaml

__all__ = ["decanonicalize", "open_image_source"]

_CANONICAL_FLOAT_RE = re.compile(r"^-?\d+\.\d{12}$")


def decanonicalize(value: object) -> object:
    """Reverse the dump's canonical-JSON float encoding for one parsed value.

    ``ssat.core.plan.hashing.canonical_json`` (used to write
    ``region_params_json``/``perturb_params_json`` for stable hashing)
    formats every float as a fixed 12-decimal-place string — e.g.
    ``constant_fill``'s ``value: 200.0`` becomes the *string*
    ``"200.000000000000"``. ``RegionResolver``/``Perturbator`` require real
    numeric params, so this walks a ``json.loads()`` result and converts any
    string matching that exact canonical float shape back to a ``float``.
    Ints are unaffected — ``canonical_json`` never stringifies them.

    Args:
        value: A value (possibly nested) produced by ``json.loads()``.

    Returns:
        The same structure with canonical-float strings converted to float.
    """

    if isinstance(value, str):
        return float(value) if _CANONICAL_FLOAT_RE.fullmatch(value) else value
    if isinstance(value, dict):
        return {key: decanonicalize(child) for key, child in value.items()}
    if isinstance(value, list):
        return [decanonicalize(child) for child in value]
    return value


def open_image_source(dump_root: Path | str) -> ImageFolderSource:
    """Open a dump and build the ``ImageFolderSource`` it points at.

    Args:
        dump_root: Root directory of a raw audit dump.

    Returns:
        A source ready to load the run's original images by sample_id.

    Raises:
        DebugVizError: If the dump has no ``source_provenance``, or its
            manifest cannot be loaded.
    """

    handle = DumpHandle(dump_root)
    resolved_config = handle.manifest.resolved_config
    source_provenance = resolved_config.source_provenance
    if source_provenance is None:
        raise DebugVizError(
            f"dump at {handle.root} has no source_provenance; "
            "DebugViz cannot load its original images"
        )

    if source_provenance.kind == "imagenet":
        root = _resolve_imagenet_root(
            source_provenance.loader_parameters,
            manifest_path=source_provenance.manifest,
            config_source=resolved_config.config_source,
        )
        try:
            source, _ = ImageNetSourceProvider().build(
                ImageNetSourceConfig(
                    root=root,
                    annotation_file=source_provenance.manifest,
                ),
                base_dir=Path("/"),
            )
        except (OSError, SourceProviderError, ValueError) as error:
            raise DebugVizError(
                f"cannot rebuild ImageNet source from {source_provenance.manifest}: {error}"
            ) from error
        return source

    manifest_path = source_provenance.manifest
    try:
        document = load_json(manifest_path)
    except (OSError, ValueError) as error:
        raise DebugVizError(f"cannot read source manifest: {manifest_path}") from error

    manifest_dir = manifest_path.parent
    samples = []
    for entry in document["samples"]:
        image_path = Path(entry["path"])
        if not image_path.is_absolute():
            image_path = manifest_dir / image_path
        samples.append(
            SampleMeta(
                sample_id=entry["sample_id"],
                path=image_path,
                gt_label=entry.get("gt_label"),
            )
        )
    return ImageFolderSource(samples)


def _resolve_imagenet_root(
    loader_parameters: dict[str, Any],
    *,
    manifest_path: Path,
    config_source: Path | None,
) -> Path:
    """Resolve the image root recorded by new or legacy ImageNet dumps.

    Current dumps record the resolved root in ``loader_parameters``. Dumps
    written before that provenance field was added retain only the annotation
    path, so their recorded ``config_source`` is used as a compatibility
    source. The annotation path is cross-checked before accepting that root.
    """

    recorded_root = loader_parameters.get("root")
    if recorded_root is not None:
        if not isinstance(recorded_root, str) or not recorded_root:
            raise DebugVizError(
                "ImageNet source loader_parameters.root must be a non-empty string"
            )
        try:
            root = Path(recorded_root).expanduser().resolve(strict=True)
        except OSError as error:
            raise DebugVizError(f"cannot resolve ImageNet image root: {recorded_root}") from error
        if not root.is_dir():
            raise DebugVizError(f"ImageNet image root is not a directory: {root}")
        return root

    if config_source is None:
        raise DebugVizError(
            "legacy ImageNet source provenance has no loader root and no config_source"
        )

    try:
        document = load_yaml(config_source)
    except (OSError, ValueError) as error:
        raise DebugVizError(f"cannot read source configuration: {config_source}") from error
    if not isinstance(document, dict) or not isinstance(document.get("source"), dict):
        raise DebugVizError(f"source configuration has no source mapping: {config_source}")

    source = document["source"]
    if source.get("kind") != "imagenet":
        raise DebugVizError(f"source configuration is not ImageNet: {config_source}")
    root_value = source.get("root")
    annotation_value = source.get("annotation_file")
    if not isinstance(root_value, (str, Path)) or not isinstance(
        annotation_value, (str, Path)
    ):
        raise DebugVizError(
            f"ImageNet source configuration requires root and annotation_file: {config_source}"
        )

    try:
        configured_annotation = _resolve_config_path(annotation_value, config_source)
        expected_annotation = manifest_path.expanduser().resolve(strict=True)
    except OSError as error:
        raise DebugVizError(
            f"cannot resolve ImageNet annotation from source configuration: {config_source}"
        ) from error
    if configured_annotation != expected_annotation:
        raise DebugVizError(
            "ImageNet annotation in config_source does not match source provenance: "
            f"{configured_annotation} != {expected_annotation}"
        )

    try:
        root = _resolve_config_path(root_value, config_source)
    except OSError as error:
        raise DebugVizError(
            f"cannot resolve ImageNet image root from source configuration: {config_source}"
        ) from error
    if not root.is_dir():
        raise DebugVizError(f"ImageNet image root is not a directory: {root}")
    return root


def _resolve_config_path(value: str | Path, config_source: Path) -> Path:
    """Resolve one source-config path relative to its YAML file."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_source.parent / path
    return path.resolve(strict=True)
