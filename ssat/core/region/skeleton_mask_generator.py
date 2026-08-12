"""Rasterize tracked skeleton body-part bounding boxes into per-frame masks."""

from __future__ import annotations

from math import isfinite

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ssat.core.region.mask_base import RegionMaskGenerator, RegionResolutionError
from ssat.core.region.types import RegionSpec
from ssat.core.types import RegionKind

_EXPECTED_PARAMS = {"sample_id", "body_part", "bbox_scale"}


class SkeletonPartsMaskGenerator(RegionMaskGenerator):
    """Materialize a ``(T, H, W)`` mask from pre-computed body-part bboxes.

    Requires ``MaskResolutionContext.skeleton_store`` to be configured; masks
    are read from that shared store rather than recomputed from raw joints.
    """

    def supports(self, spec: RegionSpec) -> bool:
        """Return whether the recipe is a concrete skeleton body-part region.

        Args:
            spec: Concrete region recipe.

        Returns:
            ``True`` only for the skeleton-parts kind.
        """

        return spec.kind is RegionKind.SKELETON_PARTS

    def get_mask(
        self,
        height: int,
        width: int,
        spec: RegionSpec,
        rng: Generator | None = None,
    ) -> NDArray[np.bool_]:
        """Rasterize one tracked body part into a per-frame boolean mask.

        Args:
            height: Source frame height.
            width: Source frame width.
            spec: Concrete skeleton-parts recipe with ``sample_id``,
                ``body_part``, and ``bbox_scale`` params.
            rng: Unused item-local generator.

        Returns:
            A ``(T, H, W)`` boolean mask, one frame per stored bounding box,
            with each box scaled about its center by ``bbox_scale`` and
            clipped to the source frame.

        Raises:
            RegionResolutionError: If no store is configured, the recipe is
                invalid, or the sample/part/frame size is unavailable or
                inconsistent.
        """

        store = self._context.skeleton_store
        if store is None:
            raise RegionResolutionError(
                "skeleton_parts requires a SkeletonBBoxStore; configure "
                "MaskResolutionContext.skeleton_store"
            )
        if set(spec.params) != _EXPECTED_PARAMS:
            raise RegionResolutionError(
                "skeleton_parts params must contain sample_id, body_part, "
                "and bbox_scale"
            )
        sample_id = spec.params["sample_id"]
        body_part = spec.params["body_part"]
        bbox_scale = spec.params["bbox_scale"]
        if not isinstance(sample_id, str) or not sample_id:
            raise RegionResolutionError(
                "skeleton_parts.sample_id must be a non-empty string"
            )
        if not isinstance(body_part, str) or not body_part:
            raise RegionResolutionError(
                "skeleton_parts.body_part must be a non-empty string"
            )
        if (
            isinstance(bbox_scale, bool)
            or not isinstance(bbox_scale, (int, float))
            or not isfinite(bbox_scale)
            or bbox_scale <= 0
        ):
            raise RegionResolutionError(
                "skeleton_parts.bbox_scale must be a positive number"
            )

        frame_size = store.frame_size(sample_id)
        if frame_size is None:
            raise RegionResolutionError(
                f"no skeleton bbox data for sample_id={sample_id!r}"
            )
        if frame_size != (width, height):
            raise RegionResolutionError(
                f"skeleton frame_size {frame_size} does not match source "
                f"({width}, {height})"
            )
        boxes = store.get(sample_id, body_part)
        if boxes is None:
            raise RegionResolutionError(
                f"no skeleton bbox data for sample_id={sample_id!r} "
                f"part={body_part!r}"
            )

        mask = np.zeros((len(boxes), height, width), dtype=np.bool_)
        for frame_index, box in enumerate(boxes):
            if box is None:
                continue
            x1, y1, x2, y2 = box
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            half_width = (x2 - x1) / 2 * bbox_scale
            half_height = (y2 - y1) / 2 * bbox_scale
            x_start = max(0, int(np.floor(center_x - half_width)))
            x_end = min(width, int(np.ceil(center_x + half_width)))
            y_start = max(0, int(np.floor(center_y - half_height)))
            y_end = min(height, int(np.ceil(center_y + half_height)))
            if x_start < x_end and y_start < y_end:
                mask[frame_index, y_start:y_end, x_start:x_end] = True
        return mask
