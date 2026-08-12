"""Expand a configured skeleton body part into a concrete tracked region."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from ssat.core.config.schema import ResolvedRegionConfig
from ssat.core.plan.expansion_base import RegionExpansionError
from ssat.core.region.skeleton_store import SkeletonBBoxStore
from ssat.core.region.types import RegionSpec
from ssat.core.source.types import SampleMeta

_ALLOWED_PARAMS = {"body_part", "bbox_scale"}
_DEFAULT_BBOX_SCALE = 1.0


def validate_skeleton_family_params(params: Mapping[str, Any]) -> tuple[str, float]:
    """Validate a ``skeleton_parts`` family's ``body_part``/``bbox_scale``.

    Shared by config-time validation (``SampleDependentRegionExpander``) and
    per-sample expansion (``SkeletonRegionProvider``), so both layers enforce
    the identical structural recipe.

    Args:
        params: Family or resolved-family parameter mapping.

    Returns:
        The validated ``(body_part, bbox_scale)`` pair, with ``bbox_scale``
        defaulted to ``1.0`` when omitted.

    Raises:
        RegionExpansionError: If the params contain unknown keys, omit
            ``body_part``, or ``body_part``/``bbox_scale`` are invalid.
    """

    if not set(params) <= _ALLOWED_PARAMS or "body_part" not in params:
        raise RegionExpansionError(
            "skeleton_parts params must contain body_part and may "
            "contain bbox_scale"
        )
    body_part = params["body_part"]
    if not isinstance(body_part, str) or not body_part:
        raise RegionExpansionError(
            "skeleton_parts.body_part must be a non-empty string"
        )
    bbox_scale = params.get("bbox_scale", _DEFAULT_BBOX_SCALE)
    if (
        isinstance(bbox_scale, bool)
        or not isinstance(bbox_scale, (int, float))
        or not isfinite(bbox_scale)
        or bbox_scale <= 0
    ):
        raise RegionExpansionError(
            "skeleton_parts.bbox_scale must be a positive number"
        )
    return body_part, float(bbox_scale)


class SkeletonRegionProvider:
    """Expand one ``skeleton_parts`` family into its per-sample tracked region.

    Implements the ``SampleRegionProvider`` protocol
    (``ssat.core.plan.expansion_base``): one instance covers every family
    configured with ``kind: skeleton_parts``, reading the requested body
    part from ``family.params`` per call rather than at construction.

    Args:
        store: Pre-computed per-sample, per-body-part frame bounding boxes.
    """

    def __init__(self, store: SkeletonBBoxStore) -> None:
        self._store = store

    def expand(
        self,
        sample: SampleMeta,
        family: ResolvedRegionConfig,
    ) -> Sequence[RegionSpec]:
        """Create the single concrete region tracking one sample's body part.

        Args:
            sample: Lightweight metadata for the source sample.
            family: Resolved skeleton-parts family recipe.

        Returns:
            A one-item sequence containing the concrete tracked region.

        Raises:
            RegionExpansionError: If the family params are invalid, or the
                store has no data for this sample and body part.
        """

        body_part, bbox_scale = validate_skeleton_family_params(family.params)

        if self._store.get(sample.sample_id, body_part) is None:
            raise RegionExpansionError(
                f"no skeleton bbox data for sample_id={sample.sample_id!r} "
                f"part={body_part!r}"
            )

        return (
            RegionSpec(
                region_id=family.region_id,
                region_instance_id=f"{family.region_id}/{sample.sample_id}",
                kind=family.kind,
                params={
                    "sample_id": sample.sample_id,
                    "body_part": body_part,
                    "bbox_scale": bbox_scale,
                },
            ),
        )
