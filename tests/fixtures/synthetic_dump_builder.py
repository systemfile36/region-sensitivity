"""Build synthetic core dumps directly via DumpWriter for L2 metrics tests.

This module never imports ``PlanBuilder``, ``ssat.core.runtime``, or any
adapter — every record is constructed by hand and written straight to
``DumpWriter`` (design IMPLE_PLAN_METRIC_DESIGN_v1.md §3.3 "코어를 실행하지
않고 의도적으로 구성한 dump", §5 단계 5). This keeps L2 failures attributable
to the metrics engine rather than to the core execution path. It generalizes
the ad hoc helpers already proven in tests/unit/test_metrics_dump_reader.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ssat.core.adapter.types import AdapterSpec, RawOutput
from ssat.core.config.schema import (
    DumpConfig,
    PerturbationConfig,
    ResolvedConfig,
    ResolvedRegionConfig,
    RuntimeConfig,
)
from ssat.core.dump import CleanDumpRecord, DumpWriter, EnvironmentSpec, PerturbedDumpRecord
from ssat.core.plan.types import WorkItem
from ssat.core.region.types import RegionMeta, RegionSpec
from ssat.core.types import ItemStatus, JsonValue, PerturbationOp, RegionKind

ENVIRONMENT = EnvironmentSpec(python_version="3.11.0", platform="test-platform")


def build_resolved_config(
    tmp_path: Path,
    *,
    regions: tuple[ResolvedRegionConfig, ...],
    perturbations: tuple[PerturbationConfig, ...] = (
        PerturbationConfig(op=PerturbationOp.CONSTANT_FILL, params={"value": 0.0}),
    ),
    mask_transform_available: bool = True,
) -> ResolvedConfig:
    """Build a minimal, valid ResolvedConfig for a synthetic dump."""

    return ResolvedConfig(
        config_base_dir=tmp_path.resolve(),
        regions=regions,
        perturbations=perturbations,
        runtime=RuntimeConfig(),
        dump=DumpConfig(),
        adapter_spec=AdapterSpec(
            model_id="model",
            deterministic=True,
            adapter_kind="callable",
            model_name="tiny",
            weights_hash="a" * 64,
            preprocessing_fingerprint="b" * 64,
            mask_transform_available=mask_transform_available,
        ),
    )


def raw_output(values: NDArray[np.floating]) -> RawOutput:
    """Wrap a logits vector as the RawOutput DumpWriter expects."""

    return RawOutput(np.asarray(values, dtype=np.float64))


def clean_record(
    sample_id: str,
    *,
    logits: NDArray[np.floating] | None,
    gt_label: int = 0,
    status: ItemStatus = ItemStatus.OK,
) -> CleanDumpRecord:
    """Build one clean-sample record; ``logits`` is ignored for failure statuses."""

    successful = status is ItemStatus.OK
    return CleanDumpRecord(
        sample_id=sample_id,
        status=status,
        model_id="model",
        logits=raw_output(logits) if successful else None,
        content_hash="c" * 64 if successful else None,
        gt_label=gt_label,
        original_shape=(1, 4, 4, 3) if successful else None,
    )


def perturbed_record(
    index: int,
    *,
    sample_id: str,
    region_id: str,
    region_instance_id: str,
    logits: NDArray[np.floating] | None,
    region_kind: RegionKind = RegionKind.GRID,
    region_params: Mapping[str, JsonValue] | None = None,
    perturb_op: PerturbationOp = PerturbationOp.CONSTANT_FILL,
    perturb_params: Mapping[str, JsonValue] | None = None,
    status: ItemStatus = ItemStatus.OK,
    is_control: bool = False,
    seed_used: int = 7,
    intended_area_px: int = 4,
    effective_area_px: int | None = 4,
) -> PerturbedDumpRecord:
    """Build one perturbed-item record with a deterministic, unique item_id.

    ``index`` is formatted as a 64-char hex string — DumpWriter/WorkItem only
    validate item_id's SHA-256-hex *shape*, not its provenance from
    ``compute_item_id`` (confirmed against tests/unit/test_dump_io.py's
    ``character * 64`` placeholders), so a plain formatted index is enough to
    keep item_ids unique across one synthetic dump.
    """

    successful = status is ItemStatus.OK
    work_item = WorkItem(
        item_id=f"{index:064x}",
        sample_id=sample_id,
        region_spec=RegionSpec(
            region_id=region_id,
            region_instance_id=region_instance_id,
            kind=region_kind,
            params=dict(region_params) if region_params is not None else {},
        ),
        perturb_op=perturb_op,
        perturb_params=dict(perturb_params) if perturb_params is not None else {"value": 0.0},
        is_control=is_control,
    )
    return PerturbedDumpRecord(
        work_item=work_item,
        status=status,
        seed_used=seed_used,
        logits=raw_output(logits) if successful else None,
        region_meta=(
            RegionMeta(intended_area_px, 0.25, "synthetic", "1") if successful else None
        ),
        effective_area_px=effective_area_px if successful else None,
    )


def write_dump(
    root: Path,
    config: ResolvedConfig,
    *,
    clean_records: tuple[CleanDumpRecord, ...],
    perturbed_records: tuple[PerturbedDumpRecord, ...],
) -> None:
    """Write one complete synthetic dump under ``root``."""

    with DumpWriter(
        root, config, code_version="test-code", mode="create", environment=ENVIRONMENT
    ) as writer:
        writer.write_clean_many(clean_records)
        writer.write_perturbed_many(perturbed_records)
