"""Types shared by work enumeration and worker materialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ssat.core.region.types import RegionSpec
from ssat.core.types import JsonValue, PerturbationOp, freeze_json_mapping


@dataclass(frozen=True, slots=True)
class WorkItem:
    """Define one deterministic sample-region-perturbation evaluation.

    PlanBuilder creates these records during worker-side materialization. All
    fields except ``item_id`` form the stable identity payload.

    Attributes:
        item_id: Schema-versioned SHA-256 identity of this work item.
        sample_id: Source sample to load.
        region_spec: Resolved mask recipe applied to the sample.
        perturb_op: Perturbation operation to run.
        perturb_params: Immutable operation-specific parameters.
        invert_mask: Whether to perturb outside the resolved mask.
        seed_salt: Salt combined with the global seed and item ID.
        is_control: Whether this item is an area-matched random control.
    """

    item_id: str
    sample_id: str
    region_spec: RegionSpec
    perturb_op: PerturbationOp
    perturb_params: Mapping[str, JsonValue] = field(default_factory=dict)
    invert_mask: bool = False
    seed_salt: int = 0
    is_control: bool = False

    def __post_init__(self) -> None:
        if len(self.item_id) != 64 or any(c not in "0123456789abcdef" for c in self.item_id):
            raise ValueError("item_id must be a 64-character lowercase SHA-256 hex digest")
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if self.seed_salt < 0:
            raise ValueError("seed_salt must be non-negative")
        object.__setattr__(
            self,
            "perturb_params",
            freeze_json_mapping(dict(self.perturb_params)),
        )

    def identity_payload(self) -> dict[str, object]:
        """Return the fields covered by the item ID hash.

        Returns:
            A mapping ready for canonical serialization.
        """

        return {
            "sample_id": self.sample_id,
            "region_spec": self.region_spec,
            "perturb_op": self.perturb_op,
            "perturb_params": self.perturb_params,
            "invert_mask": self.invert_mask,
            "seed_salt": self.seed_salt,
            "is_control": self.is_control,
        }


@dataclass(frozen=True, slots=True)
class WorkChunkMeta:
    """Carry lightweight chunk identifiers from the main process to workers.

    Attributes:
        chunk_id: Deterministic identifier used to rematerialize the chunk.
        sample_id: Single source sample shared by every item in the chunk.
        item_ids: Ordered identifiers expected from materialization.
    """

    chunk_id: str
    sample_id: str
    item_ids: tuple[str, ...]

    @property
    def n_items(self) -> int:
        """Return the number of work items represented by the chunk."""

        return len(self.item_ids)


@dataclass(frozen=True, slots=True)
class WorkChunk:
    """Group fully materialized work items for one source sample.

    Attributes:
        chunk_id: Identifier corresponding to the lightweight chunk metadata.
        sample_id: Source sample loaded once for this chunk.
        items: Ordered work items prepared from the loaded sample.
    """

    chunk_id: str
    sample_id: str
    items: tuple[WorkItem, ...]

    def __post_init__(self) -> None:
        if not self.chunk_id or not self.sample_id:
            raise ValueError("chunk_id and sample_id must not be empty")
        if not self.items:
            raise ValueError("items must not be empty")
        if any(item.sample_id != self.sample_id for item in self.items):
            raise ValueError("every work item must belong to the chunk sample_id")
