"""Build synthetic core dumps directly via DumpWriter for L2 metrics tests.

This module never imports ``PlanBuilder``, ``ssat.core.runtime``, or any
adapter — every record is constructed by hand and written straight to
``DumpWriter`` (design IMPLE_PLAN_METRIC_DESIGN_v1.md §3.3 "코어를 실행하지
않고 의도적으로 구성한 dump", §5 단계 5). This keeps L2 failures attributable
to the metrics engine rather than to the core execution path. It generalizes
the ad hoc helpers already proven in tests/unit/test_metrics_dump_reader.py.

``compute_and_save_metrics`` additionally wires the downstream N2-N4 metrics
pipeline (``ssat.metrics.*``) over one already-written dump — this is the
metrics engine consuming its own contract, not the core execution path, so
it does not conflict with the constraint above; it exists so DebugViz V2/V3
tests can get a ready-to-load ``metrics_dir`` without repeating that wiring
in every test module.

``compute_and_save_analysis`` extends this the same way one layer up: it
wires the A0-A6 control/stability analysis pipeline over an already-computed
metrics store. Rather than re-implementing that sequence by hand (the way
``compute_and_save_metrics`` above wires N2-N4 directly), it delegates to
``AuditApplication.analyze()`` — that method already is this exact sequence
(IMPLE_PLAN_REPORTING_v1.md §5 단계0), so re-deriving it here would just be a
second copy to keep in sync. This exists so report-layer tests (and any
future consumer needing dump+metrics+analysis all three) can get a
ready-to-load ``analysis_dir`` without repeating that wiring themselves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ssat.analysis.store import AnalysisManifest, load_analysis
from ssat.application import AnalyzeRequest, AuditApplication
from ssat.core.adapter.types import AdapterSpec, RawOutput
from ssat.core.config.schema import (
    DumpConfig,
    PerturbationConfig,
    ResolvedConfig,
    ResolvedRegionConfig,
    RuntimeConfig,
    SourceProvenance,
)
from ssat.core.dump import CleanDumpRecord, DumpWriter, EnvironmentSpec, PerturbedDumpRecord
from ssat.core.plan.types import WorkItem
from ssat.core.region.types import RegionMeta, RegionSpec
from ssat.core.types import ItemStatus, JsonValue, PerturbationOp, RegionKind
from ssat.metrics.aggregate import aggregate_item_metrics
from ssat.metrics.dump_reader import DumpHandle
from ssat.metrics.registry import MetricRegistry
from ssat.metrics.store import MetricsManifest, save_metrics
from ssat.utils.io import sha256_file

ENVIRONMENT = EnvironmentSpec(python_version="3.11.0", platform="test-platform")


def build_resolved_config(
    tmp_path: Path,
    *,
    regions: tuple[ResolvedRegionConfig, ...],
    perturbations: tuple[PerturbationConfig, ...] = (
        PerturbationConfig(op=PerturbationOp.CONSTANT_FILL, params={"value": 0.0}),
    ),
    mask_transform_available: bool = True,
    source_provenance: SourceProvenance | None = None,
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
        source_provenance=source_provenance,
    )


def image_manifest_source_provenance(manifest_path: Path) -> SourceProvenance:
    """Build a SourceProvenance pointing at a real, committed image manifest.

    Lets L2/DebugViz tests build synthetic dumps that reference actual images
    (e.g. tests/fixtures/synthetic_classification/manifest.json) instead of
    the placeholder logits-only records the rest of this module produces.
    """

    return SourceProvenance(
        kind="image_manifest",
        manifest=manifest_path.resolve(),
        manifest_hash=sha256_file(manifest_path),
    )


def raw_output(values: NDArray[np.floating]) -> RawOutput:
    """Wrap a logits vector as the RawOutput DumpWriter expects."""

    return RawOutput(np.asarray(values, dtype=np.float64))


def clean_record(
    sample_id: str,
    *,
    logits: NDArray[np.floating] | None,
    gt_label: int | None = 0,
    status: ItemStatus = ItemStatus.OK,
) -> CleanDumpRecord:
    """Build one clean-sample record; ``logits`` is ignored for failure statuses.

    ``gt_label=None`` builds a label-free sample (core layer's supported
    inference-only auditing scenario) — every metric requiring a reference
    class becomes unavailable for it (``ExclusionReason.GT_LABEL_UNKNOWN``).
    """

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
    # Default deliberately contains hex letters (unlike a plain digits-only
    # value such as 7) so any code that mistakenly parses the dump's
    # persisted seed_used with int(..., base=10) instead of base=16 fails
    # loudly in tests instead of getting silently masked.
    seed_used: int = 0xDEADBEEF,
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


def compute_and_save_metrics(
    dump_root: Path,
    config: ResolvedConfig,
    metrics_dir: Path,
    *,
    registry: MetricRegistry,
    primary_metric: str,
) -> MetricsManifest:
    """Run the N2-N4 metrics pipeline over one already-written dump and persist it.

    Args:
        dump_root: Root directory of a dump previously written by ``write_dump``.
        config: The same ``ResolvedConfig`` the dump was written with.
        metrics_dir: Destination directory for the stored aggregation run.
        registry: Registered metrics to compute (e.g. from ``_registry()``
            test helpers).
        primary_metric: Registered metric name whose degradation defines
            ``vulnerability_score``.

    Returns:
        The ``MetricsManifest`` that was written to ``metrics_dir``.
    """

    handle = DumpHandle(dump_root)
    joined = handle.joined()
    item_metrics = registry.compute_item_metrics(joined, adapter_spec=config.adapter_spec)
    result = aggregate_item_metrics(
        item_metrics, joined, registry, config, primary_metric=primary_metric
    )
    return save_metrics(
        metrics_dir,
        item_metrics,
        result,
        registry=registry,
        primary_metric=primary_metric,
        source_run_manifest_path=handle.manifest_path,
        exclusion_summary=handle.summary(),
    )


def compute_and_save_analysis(
    dump_root: Path,
    metrics_dir: Path,
    analysis_dir: Path,
    *,
    primary_metric: str,
    **analyze_kwargs: object,
) -> AnalysisManifest:
    """Run the A0-A6 analysis pipeline over one dump+metrics pair and persist it.

    Thin wrapper around ``AuditApplication.analyze()`` (module docstring) —
    every threshold/bootstrap keyword ``AnalyzeRequest`` accepts
    (``n_bootstrap``, ``random_seed``, ``z_vs_control_threshold``,
    ``seed_cv_threshold``, ``area_match_tolerance``) can be passed through
    ``analyze_kwargs`` and otherwise falls back to that request type's own
    defaults, so this helper never re-declares them.

    Args:
        dump_root: Root directory of a dump previously written by ``write_dump``.
        metrics_dir: Metrics store previously written by
            ``compute_and_save_metrics`` for the same dump.
        analysis_dir: Destination directory for the stored analysis run.
        primary_metric: Registered metric name to score stability/rank
            correlation on.
        **analyze_kwargs: Forwarded verbatim to ``AnalyzeRequest``.

    Returns:
        The ``AnalysisManifest`` reloaded from ``analysis_dir`` after the
        run completes — reloaded rather than returned from ``analyze()``
        directly, so a caller always observes exactly what was persisted.
    """

    AuditApplication().analyze(
        AnalyzeRequest(
            dump=dump_root,
            metrics_dir=metrics_dir,
            analysis_dir=analysis_dir,
            primary_metric=primary_metric,
            **analyze_kwargs,  # type: ignore[arg-type]
        )
    )
    *_rest, manifest = load_analysis(analysis_dir)
    return manifest
