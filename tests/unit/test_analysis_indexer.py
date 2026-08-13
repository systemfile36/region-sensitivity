"""Tests for A1 ComparisonIndexer (ssat/analysis/indexer.py)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ssat.analysis.errors import AnalysisCorruptionError
from ssat.analysis.indexer import ComparisonIndexer
from ssat.analysis.types import AnchorKey, MatchMethod
from ssat.core.types import RegionKind


def _context_row(
    *,
    sample_id: str,
    region_id: str = "grid",
    region_instance_id: str = "grid/r0/c0",
    region_kind: RegionKind = RegionKind.GRID,
    region_params_json: str = "{}",
    intended_area_px: int | None = 100,
    effective_area_px: int | None = 100,
    perturb_op: str = "constant_fill",
    perturb_params_json: str = "{}",
    invert_mask: bool = False,
    is_control: bool = False,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "region_id": region_id,
        "region_instance_id": region_instance_id,
        "region_kind": region_kind.value,
        "region_params_json": region_params_json,
        "intended_area_px": intended_area_px,
        "effective_area_px": effective_area_px,
        "perturb_op": perturb_op,
        "perturb_params_json": perturb_params_json,
        "invert_mask": invert_mask,
        "is_control": is_control,
    }


def _context(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _control_region_params_json(
    *,
    target_region_id: str = "grid",
    target_region_instance_id: str = "grid/r0/c0",
    target_kind: RegionKind = RegionKind.GRID,
    control_request_index: int = 0,
    control_index: int = 0,
) -> str:
    # Mirrors PlanBuilder._region_recipe's shape (ssat/core/plan/builder.py:327-338)
    # nested under "target_region", as canonical_json actually produces it
    # (None-valued ref/ref_hash are omitted, not the flat top-level shape).
    return json.dumps(
        {
            "target_region": {
                "region_id": target_region_id,
                "region_instance_id": target_region_instance_id,
                "kind": target_kind.value,
                "params": {},
            },
            "control_request_index": control_request_index,
            "control_index": control_index,
        }
    )


# --- exact-reference matching --------------------------------------------


def test_target_and_matched_controls_are_paired_by_exact_reference() -> None:
    context = _context(
        [
            _context_row(sample_id="s1"),
            _context_row(
                sample_id="s1",
                region_id="control:grid:0",
                region_instance_id="control:grid/r0/c0:0:0",
                region_kind=RegionKind.RANDOM_AREA_MATCH,
                region_params_json=_control_region_params_json(control_index=0),
                intended_area_px=98,
                effective_area_px=98,
                is_control=True,
            ),
            _context_row(
                sample_id="s1",
                region_id="control:grid:0",
                region_instance_id="control:grid/r0/c0:0:1",
                region_kind=RegionKind.RANDOM_AREA_MATCH,
                region_params_json=_control_region_params_json(control_index=1),
                intended_area_px=102,
                effective_area_px=102,
                is_control=True,
            ),
        ]
    )

    indexer = ComparisonIndexer(context)

    assert len(indexer.anchor_table) == 3
    assert len(indexer.control_pairs) == 2
    target_anchor_key = AnchorKey(sample_id="s1", region_key="grid::grid/r0/c0", invert_mask=False)
    for pair in indexer.control_pairs:
        assert pair.match_method is MatchMethod.EXACT_REFERENCE
        assert pair.target_anchor_key == target_anchor_key
    assert {pair.control_anchor_key.region_key for pair in indexer.control_pairs} == {
        "control:grid:0::control:grid/r0/c0:0:0",
        "control:grid:0::control:grid/r0/c0:0:1",
    }
    assert indexer.coverage_report.n_controls_unmatched == 0
    assert indexer.coverage_report.n_area_mismatch_warnings == 0


# --- area-tolerance fallback ----------------------------------------------


def test_control_with_malformed_target_region_falls_back_to_area_tolerance() -> None:
    context = _context(
        [
            _context_row(sample_id="s1"),
            _context_row(
                sample_id="s1",
                region_id="control:grid:0",
                region_instance_id="control:grid/r0/c0:0:0",
                region_kind=RegionKind.RANDOM_AREA_MATCH,
                # Missing "target_region" key -- not a resolvable recipe.
                region_params_json=json.dumps({"control_request_index": 0, "control_index": 0}),
                intended_area_px=96,
                effective_area_px=96,
                is_control=True,
            ),
        ]
    )

    indexer = ComparisonIndexer(context)

    assert len(indexer.control_pairs) == 1
    pair = indexer.control_pairs[0]
    assert pair.match_method is MatchMethod.AREA_TOLERANCE
    assert pair.target_anchor_key == AnchorKey(
        sample_id="s1", region_key="grid::grid/r0/c0", invert_mask=False
    )
    assert pair.area_match_ratio == pytest.approx(0.96)
    assert indexer.coverage_report.n_controls_unmatched == 0


def test_control_outside_area_tolerance_is_unmatched() -> None:
    context = _context(
        [
            _context_row(sample_id="s1"),
            _context_row(
                sample_id="s1",
                region_id="control:grid:0",
                region_instance_id="control:grid/r0/c0:0:0",
                region_kind=RegionKind.RANDOM_AREA_MATCH,
                region_params_json=json.dumps({"control_request_index": 0, "control_index": 0}),
                intended_area_px=50,
                effective_area_px=50,
                is_control=True,
            ),
        ]
    )

    indexer = ComparisonIndexer(context)

    assert indexer.control_pairs == ()
    assert indexer.coverage_report.n_controls_unmatched == 1


def test_exact_reference_match_outside_tolerance_still_warns() -> None:
    context = _context(
        [
            _context_row(sample_id="s1", intended_area_px=100, effective_area_px=100),
            _context_row(
                sample_id="s1",
                region_id="control:grid:0",
                region_instance_id="control:grid/r0/c0:0:0",
                region_kind=RegionKind.RANDOM_AREA_MATCH,
                region_params_json=_control_region_params_json(),
                intended_area_px=200,
                effective_area_px=200,
                is_control=True,
            ),
        ]
    )

    indexer = ComparisonIndexer(context)

    assert len(indexer.control_pairs) == 1
    pair = indexer.control_pairs[0]
    assert pair.match_method is MatchMethod.EXACT_REFERENCE
    assert pair.area_match_ratio == pytest.approx(2.0)
    assert indexer.coverage_report.n_area_mismatch_warnings == 1
    assert indexer.coverage_report.n_controls_unmatched == 0


# --- coverage: insufficient conditions -------------------------------------


def test_anchor_with_single_condition_is_flagged_insufficient() -> None:
    context = _context(
        [
            # Two conditions on this anchor -- not insufficient.
            _context_row(sample_id="s1", perturb_op="constant_fill"),
            _context_row(sample_id="s1", perturb_op="blur"),
            # Only one condition on this separate anchor -- insufficient.
            _context_row(
                sample_id="s1",
                region_id="grid",
                region_instance_id="grid/r0/c1",
                perturb_op="constant_fill",
            ),
        ]
    )

    indexer = ComparisonIndexer(context)

    assert indexer.coverage_report.n_anchors == 2
    assert indexer.coverage_report.n_conditions_insufficient == 1
    by_region = {row.region_key: row for row in indexer.anchor_table}
    assert by_region["grid::grid/r0/c0"].n_conditions == 2
    assert by_region["grid::grid/r0/c1"].n_conditions == 1


# --- geometry consistency ---------------------------------------------------


def test_inconsistent_region_kind_raises() -> None:
    context = _context(
        [
            _context_row(sample_id="s1", region_kind=RegionKind.GRID),
            _context_row(sample_id="s1", region_kind=RegionKind.BBOX_PARTITION, perturb_op="blur"),
        ]
    )

    with pytest.raises(AnalysisCorruptionError, match="inconsistent region_kind"):
        ComparisonIndexer(context)


def test_inconsistent_area_raises() -> None:
    context = _context(
        [
            _context_row(sample_id="s1", intended_area_px=100),
            _context_row(sample_id="s1", intended_area_px=200, perturb_op="blur"),
        ]
    )

    with pytest.raises(AnalysisCorruptionError, match="inconsistent intended_area_px"):
        ComparisonIndexer(context)


# --- determinism -------------------------------------------------------------


def test_anchor_table_is_deterministically_ordered() -> None:
    context = _context(
        [
            _context_row(sample_id="s3", region_instance_id="grid/r0/c0"),
            _context_row(sample_id="s1", region_instance_id="grid/r0/c1"),
            _context_row(sample_id="s2", region_instance_id="grid/r0/c0"),
        ]
    )

    indexer = ComparisonIndexer(context)

    keys = [(row.sample_id, row.region_key, row.invert_mask) for row in indexer.anchor_table]
    assert keys == sorted(keys)
