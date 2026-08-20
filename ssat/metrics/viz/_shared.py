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

from ssat.core.source import ImageFolderSource
from ssat.core.source.types import SampleMeta
from ssat.metrics.dump_reader import DumpHandle
from ssat.metrics.errors import DebugVizError
from ssat.utils.io import load_json

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
    source_provenance = handle.manifest.resolved_config.source_provenance
    if source_provenance is None:
        raise DebugVizError(
            f"dump at {handle.root} has no source_provenance; "
            "DebugViz cannot load its original images"
        )

    manifest_path = source_provenance.manifest
    try:
        document = load_json(manifest_path)
    except OSError as error:
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
