"""Tests for A0 AnalysisReader (ssat/analysis/reader.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from synthetic_dump_builder import (
    build_resolved_config,
    clean_record,
    compute_and_save_metrics,
    image_manifest_source_provenance,
    perturbed_record,
    write_dump,
)

from ssat.analysis.reader import AnalysisReader
from ssat.analysis.types import AvailableAnalyses
from ssat.core.config.schema import ResolvedRegionConfig
from ssat.core.types import PerturbationOp, RegionKind
from ssat.metrics.builtin_metrics.continuous import GtLogitDrop
from ssat.metrics.dump_reader import DumpHandle
from ssat.metrics.errors import MetricsCorruptionError
from ssat.metrics.registry import MetricRegistry

_METRIC_NAME = "gt_logit_drop"
_TARGET_REGIONS = (
    ResolvedRegionConfig(region_id="grid", kind=RegionKind.GRID, params={"rows": 1, "cols": 1}),
)
_LOGITS = np.array([1.0, 0.0])
_CONTEXT_COLUMNS = [
    "item_id",
    "sample_id",
    "region_id",
    "region_instance_id",
    "region_kind",
    "region_params_json",
    "intended_area_px",
    "effective_area_px",
    "perturb_op",
    "perturb_params_json",
    "invert_mask",
    "is_control",
    "seed_used",
]
_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_classification"
_FIXTURE_MANIFEST = _FIXTURE_ROOT / "manifest.json"


def _registry() -> MetricRegistry:
    registry = MetricRegistry()
    registry.register(GtLogitDrop())
    return registry


def _build_reader_fixture(
    tmp_path: Path,
    *,
    perturb_ops: tuple[PerturbationOp, ...] = (PerturbationOp.CONSTANT_FILL,),
    n_seed_repeats: int = 1,
    include_control: bool = False,
) -> tuple[Path, Path]:
    """Build a hand-built dump with configurable op/seed-repeat/control variety."""

    config = build_resolved_config(tmp_path, regions=_TARGET_REGIONS)
    sample_id = "s0"
    perturbed_records = []
    index = 0
    for op in perturb_ops:
        for seed_index in range(n_seed_repeats):
            perturbed_records.append(
                perturbed_record(
                    index,
                    sample_id=sample_id,
                    region_id="grid",
                    region_instance_id="grid/r0/c0",
                    logits=_LOGITS,
                    perturb_op=op,
                    seed_used=0xDEADBEEF + seed_index,
                )
            )
            index += 1
    if include_control:
        perturbed_records.append(
            perturbed_record(
                index,
                sample_id=sample_id,
                region_id="control",
                region_instance_id="control/0",
                logits=_LOGITS,
                region_kind=RegionKind.RANDOM_AREA_MATCH,
                is_control=True,
            )
        )

    dump_root = tmp_path / "dump"
    write_dump(
        dump_root,
        config,
        clean_records=(clean_record(sample_id, logits=_LOGITS),),
        perturbed_records=tuple(perturbed_records),
    )
    metrics_dir = tmp_path / "metrics"
    compute_and_save_metrics(
        dump_root, config, metrics_dir, registry=_registry(), primary_metric=_METRIC_NAME
    )
    return dump_root, metrics_dir


def _build_fixture_reader(tmp_path: Path) -> tuple[Path, Path]:
    """Build a dump backed by the committed real-image fixture, with a control item."""

    config = build_resolved_config(
        tmp_path,
        regions=_TARGET_REGIONS,
        source_provenance=image_manifest_source_provenance(_FIXTURE_MANIFEST),
    )
    sample_id = "synthetic-000"
    perturbed_records = (
        perturbed_record(
            0, sample_id=sample_id, region_id="grid", region_instance_id="grid/r0/c0",
            logits=_LOGITS,
        ),
        perturbed_record(
            1, sample_id=sample_id, region_id="control", region_instance_id="control/0",
            logits=_LOGITS, region_kind=RegionKind.RANDOM_AREA_MATCH, is_control=True,
        ),
    )
    dump_root = tmp_path / "dump"
    write_dump(
        dump_root,
        config,
        clean_records=(clean_record(sample_id, logits=_LOGITS),),
        perturbed_records=perturbed_records,
    )
    metrics_dir = tmp_path / "metrics"
    compute_and_save_metrics(
        dump_root, config, metrics_dir, registry=_registry(), primary_metric=_METRIC_NAME
    )
    return dump_root, metrics_dir


# --- item_context() ----------------------------------------------------


def test_item_context_matches_dump_handle_items_rows_and_columns(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_fixture_reader(tmp_path)
    context = AnalysisReader(dump_root, metrics_dir).item_context()
    expected = DumpHandle(dump_root).items()

    assert list(context.columns) == _CONTEXT_COLUMNS
    assert len(context) == len(expected)
    merged = context.merge(expected, on="item_id", suffixes=("", "_dump"))
    assert len(merged) == len(context)
    for column in _CONTEXT_COLUMNS:
        if column == "item_id":
            continue
        assert (merged[column] == merged[f"{column}_dump"]).all()


def test_item_context_returns_a_copy_not_the_cached_frame(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_reader_fixture(tmp_path)
    reader = AnalysisReader(dump_root, metrics_dir)

    first = reader.item_context()
    first.loc[0, "seed_used"] = "tampered"

    assert reader.item_context().loc[0, "seed_used"] != "tampered"


# --- verify_source_dump propagation -------------------------------------


def test_verify_source_dump_failure_propagates_from_init(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_reader_fixture(tmp_path)
    manifest_path = dump_root / "run_manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["started_at"] = "2099-01-01T00:00:00+00:00"
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(MetricsCorruptionError, match="source dump has changed"):
        AnalysisReader(dump_root, metrics_dir)


# --- available_analyses() -----------------------------------------------


def test_available_analyses_all_false_for_minimal_dump(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_reader_fixture(tmp_path)
    result = AnalysisReader(dump_root, metrics_dir).available_analyses()
    assert result == AvailableAnalyses(
        control_comparison=False,
        fill_strategy_stability=False,
        seed_stability=False,
        jitter_stability=False,
    )


def test_available_analyses_control_comparison_true_when_control_present(
    tmp_path: Path,
) -> None:
    dump_root, metrics_dir = _build_reader_fixture(tmp_path, include_control=True)
    result = AnalysisReader(dump_root, metrics_dir).available_analyses()
    assert result.control_comparison is True
    assert result.fill_strategy_stability is False
    assert result.seed_stability is False


def test_available_analyses_fill_strategy_stability_true_with_two_ops(tmp_path: Path) -> None:
    dump_root, metrics_dir = _build_reader_fixture(
        tmp_path, perturb_ops=(PerturbationOp.CONSTANT_FILL, PerturbationOp.BLUR)
    )
    result = AnalysisReader(dump_root, metrics_dir).available_analyses()
    assert result.fill_strategy_stability is True
    assert result.control_comparison is False
    assert result.seed_stability is False


def test_available_analyses_seed_stability_true_with_repeated_condition(
    tmp_path: Path,
) -> None:
    dump_root, metrics_dir = _build_reader_fixture(tmp_path, n_seed_repeats=2)
    result = AnalysisReader(dump_root, metrics_dir).available_analyses()
    assert result.seed_stability is True
    assert result.fill_strategy_stability is False


def test_available_analyses_all_true_except_jitter_when_fully_populated(
    tmp_path: Path,
) -> None:
    dump_root, metrics_dir = _build_reader_fixture(
        tmp_path,
        perturb_ops=(PerturbationOp.CONSTANT_FILL, PerturbationOp.BLUR),
        n_seed_repeats=2,
        include_control=True,
    )
    result = AnalysisReader(dump_root, metrics_dir).available_analyses()
    assert result == AvailableAnalyses(
        control_comparison=True,
        fill_strategy_stability=True,
        seed_stability=True,
        jitter_stability=False,
    )


def test_jitter_stability_is_always_false_regression(tmp_path: Path) -> None:
    # Even at maximal availability (control + >=2 ops + repeated seeds),
    # jitter_stability must stay False: the core has no jitter axis to
    # detect. If this ever flips, the core gained jitter support and both
    # this test and reader.py's hardcoded constant need an intentional
    # update.
    dump_root, metrics_dir = _build_reader_fixture(
        tmp_path,
        perturb_ops=(PerturbationOp.CONSTANT_FILL, PerturbationOp.BLUR),
        n_seed_repeats=2,
        include_control=True,
    )
    result = AnalysisReader(dump_root, metrics_dir).available_analyses()
    assert result.jitter_stability is False
