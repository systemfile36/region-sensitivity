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

from ssat.core.region.skeleton_store import SkeletonBBoxStore, SkeletonDataError, load_skeleton_bbox_store
from ssat.core.source import (
    ImageFolderSource,
    ImageNetSourceConfig,
    ImageNetSourceProvider,
    SampleSource,
    SourceProviderError,
    VideoFolderSource,
)
from ssat.core.source.types import SampleMeta
from ssat.metrics.dump_reader import DumpHandle
from ssat.metrics.errors import DebugVizError
from ssat.utils.io import load_json, load_yaml

__all__ = ["decanonicalize", "open_image_source", "open_skeleton_store"]

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


def open_image_source(dump_root: Path | str) -> SampleSource:
    """Open a dump and build the sample source it points at.

    Despite the name (kept for call-site/test stability — every ``kind`` this
    function has ever supported loads an image or a video, never anything
    else), this returns an :class:`ImageFolderSource` *or* a
    :class:`VideoFolderSource`, depending on the dump's recorded
    ``source_provenance.kind``.

    Args:
        dump_root: Root directory of a raw audit dump.

    Returns:
        A source ready to load the run's original samples by sample_id.

    Raises:
        DebugVizError: If the dump has no ``source_provenance``, its
            ``kind`` cannot be reproduced from provenance alone (currently
            ``"kinetics400"`` — see below), or its manifest cannot be loaded.
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

    if source_provenance.kind == "kinetics400":
        # Unlike "video_manifest" (whose num_frames/sampling are recorded in
        # loader_parameters by VideoManifestProvider.build), KineticsSource
        # Provider.build records no loader_parameters at all (ssat/core/
        # source/kinetics.py) -- there is nothing here to rebuild a
        # VideoFolderSource from. Raise clearly rather than silently falling
        # through to the manifest-JSON branch below, which would misread the
        # Kinetics annotation CSV as a sample manifest.
        raise DebugVizError(
            f"dump at {handle.root} uses a kinetics400 source, which records no "
            "loader_parameters (num_frames/sampling) in its provenance; DebugViz "
            "cannot rebuild a VideoFolderSource from a kinetics400 dump alone"
        )

    manifest_path = source_provenance.manifest
    try:
        document = load_json(manifest_path)
    except (OSError, ValueError) as error:
        raise DebugVizError(f"cannot read source manifest: {manifest_path}") from error

    samples = _load_manifest_samples(document, manifest_path.parent)

    if source_provenance.kind == "video_manifest":
        num_frames, sampling = _video_loader_parameters(source_provenance.loader_parameters)
        return VideoFolderSource(samples, num_frames=num_frames, sampling=sampling)

    return ImageFolderSource(samples)


def open_skeleton_store(dump_root: Path | str) -> SkeletonBBoxStore | None:
    """Open a dump and, if configured, build the ``SkeletonBBoxStore`` it points at.

    Near-duplicates ``ssat.application._session_service.load_skeleton_store``
    rather than importing it: ``ssat.metrics.viz`` cannot depend on
    ``ssat.application`` (wrong dependency direction), the same constraint
    that already keeps ``assets.py::_save_heatmap_png`` a duplicate of
    ``heatmap.py::_save_view_png`` rather than a shared import.

    Args:
        dump_root: Root directory of a raw audit dump.

    Returns:
        A store ready to look up per-frame body-part bounding boxes, or
        ``None`` if the dump's run was never configured with a
        ``skeleton_source`` (e.g. every non-``skeleton_parts`` run).

    Raises:
        DebugVizError: If ``skeleton_source`` is configured but its bbox
            data file cannot be read or fails hash verification.
    """

    handle = DumpHandle(dump_root)
    skeleton_source = handle.manifest.resolved_config.skeleton_source
    if skeleton_source is None:
        return None
    try:
        return load_skeleton_bbox_store(
            skeleton_source.bbox_data, expected_hash=skeleton_source.bbox_data_hash
        )
    except SkeletonDataError as error:
        raise DebugVizError(
            f"cannot load skeleton_source data for dump at {handle.root}: {error}"
        ) from error


def _load_manifest_samples(document: dict, manifest_dir: Path) -> list[SampleMeta]:
    """Parse a manifest JSON document's ``samples`` list into ``SampleMeta`` values.

    Shared by the image-manifest and video-manifest branches of
    :func:`open_image_source` -- both manifest formats share this exact
    shape (``ssat.core.source.provider._SampleManifest``).

    Args:
        document: Parsed manifest JSON (``{"samples": [...]}``).
        manifest_dir: Directory relative paths in the document resolve against.

    Returns:
        One ``SampleMeta`` per manifest entry, in document order.
    """

    samples = []
    for entry in document["samples"]:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = manifest_dir / path
        samples.append(
            SampleMeta(
                sample_id=entry["sample_id"],
                path=path,
                gt_label=entry.get("gt_label"),
            )
        )
    return samples


def _video_loader_parameters(loader_parameters: dict[str, Any]) -> tuple[int, str]:
    """Validate and extract ``num_frames``/``sampling`` from video provenance.

    Args:
        loader_parameters: ``SourceProvenance.loader_parameters`` recorded by
            ``VideoManifestProvider.build``.

    Returns:
        ``(num_frames, sampling)``.

    Raises:
        DebugVizError: If either value is missing or has an invalid type.
    """

    num_frames = loader_parameters.get("num_frames")
    sampling = loader_parameters.get("sampling")
    if (
        isinstance(num_frames, bool)
        or not isinstance(num_frames, int)
        or num_frames <= 0
    ):
        raise DebugVizError(
            "video_manifest source loader_parameters.num_frames must be a positive integer"
        )
    if not isinstance(sampling, str) or not sampling:
        raise DebugVizError(
            "video_manifest source loader_parameters.sampling must be a non-empty string"
        )
    return num_frames, sampling


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
