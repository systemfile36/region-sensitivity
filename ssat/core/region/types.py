"""Resolved region specifications and materialization metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from ssat.core.types import JsonValue, RegionKind, freeze_json_mapping


@dataclass(frozen=True, slots=True)
class RegionSpec:
    """Describe one concrete, fully resolved region stored in a WorkItem.

    A configured region family may expand into several instances of this type.
    Explicit instances always contain a resolved reference and content hash.

    Attributes:
        region_id: Configured family ID used for grouping and controls.
        region_instance_id: Stable identity of this concrete spatial unit.
        kind: Mask materialization strategy.
        params: Immutable concrete mask recipe.
        ref: Resolved explicit-mask path, when applicable.
        ref_hash: Content hash for the explicit mask, when applicable.
    """

    region_id: str
    region_instance_id: str
    kind: RegionKind
    params: Mapping[str, JsonValue] = field(default_factory=dict)
    ref: str | None = None
    ref_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.region_id or not self.region_instance_id:
            raise ValueError("region_id and region_instance_id must not be empty")
        frozen = freeze_json_mapping(dict(self.params))
        object.__setattr__(self, "params", frozen)
        if self.kind is RegionKind.EXPLICIT:
            if self.ref is None or self.ref_hash is None:
                raise ValueError("resolved explicit regions require ref and ref_hash")
        elif self.ref is not None or self.ref_hash is not None:
            raise ValueError("procedural regions cannot define ref or ref_hash")


@dataclass(frozen=True, slots=True)
class RegionMeta:
    """Record mask properties measured during region materialization.

    Attributes:
        intended_area_px: Number of pixels selected before model preprocessing.
        intended_area_ratio: Selected fraction of the original image area.
        generator_kind: Generator implementation that created the mask.
        generator_version: Version of the generator contract.
        confidence: Optional confidence supplied by automatic generators.
    """

    intended_area_px: int
    intended_area_ratio: float
    generator_kind: str
    generator_version: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.intended_area_px < 0:
            raise ValueError("intended_area_px must be non-negative")
        if not 0.0 <= self.intended_area_ratio <= 1.0:
            raise ValueError("intended_area_ratio must be between 0 and 1")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
