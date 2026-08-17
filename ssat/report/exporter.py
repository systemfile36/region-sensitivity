"""R1 Exporter: serialize an assembled report to JSON + CSV (design §R1).

``report_model.json`` is the whole ``ReportModel``, byte-reproducible via
``ssat.utils.io.write_json_atomic`` (sorted keys, fsynced, atomic replace —
the same convention every other store's manifest already follows). The
CSVs exist because ``ReportModel`` itself only ever carries the top-K/
bottom-K slice of a run (IMPLE_PLAN_REPORTING_v1.md §1 격차#5) — they are
the only place the *full* population (every sample, every region, every
semantic_group, every flagged anchor) is reachable without re-running the
assembler. ``semantic_summary.csv``/``class_semantic_matrix.csv`` are the
exception: ``semantic_summary``/``class_semantic_matrix`` are already
dataset-wide (not top-K-limited) on ``ReportModel`` itself
(IMPLE_PLAN_SEMANTIC_VULNERABILITY_v1.md §3.4), so their CSVs are flat
re-exports rather than a fuller population than the JSON carries — kept as
separate files anyway for the same "primary artifact deserves its own
file, not a JSON blob a reader has to parse" reasoning that motivated
``region_summary.csv``.

Every ``ReportModel``/``SampleCard`` field this module writes to a CSV is
represented, including the ones that are not scalars (``top_regions``,
``task_extra``, ``reliability_reasons``, ``region_ids``): each becomes its
own flat column — ``*_json`` (``json.dumps(..., sort_keys=True)``) for
nested structures, matching the precedent ``ssat.analysis.store`` already
set for ``strategy_signs_json``/``strategy_values_json``, or a delimited
string for a flat tuple of identifiers (``region_ids`` joined with ``;``,
plan §3.4 — a JSON array would be needlessly heavy for a list of bare
strings a reader wants to skim directly in a spreadsheet). Rows are never
silently thinned to fit CSV's flat shape (design §6.2 "결측이... 조용히
생략되지 않음" — the same principle, applied to *columns* here rather than
values).
"""

from __future__ import annotations

import csv
import dataclasses
import json
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Protocol

from ssat.report.types import (
    ClassSemanticRow,
    FlaggedItem,
    RegionRow,
    ReportGrade,
    ReportModel,
    SampleCard,
    SemanticGroupRow,
)
from ssat.utils.io import write_json_atomic


class AssembledReportLike(Protocol):
    """Structural shape this module reads off R0's output.

    Matches ``ssat.report.assembler.AssembledReport`` field-for-field
    without importing it, keeping this module on its §3.3 dependency list
    ("report.exporter → report.types, ssat.utils") — the same duck-typing
    trade ``report.adapters.SampleMetricLike`` already made for the same
    reason (see that module's docstring).

    Attributes:
        model: The JSON-serializable ``ReportModel``.
        full_sample_rankings: Every sample the run produced (not just the
            top-K/bottom-K ``model`` keeps) — the source of
            ``sample_rankings.csv``.
    """

    model: ReportModel
    full_sample_rankings: Sequence[SampleCard]


@dataclasses.dataclass(frozen=True, slots=True)
class ExportedPaths:
    """Where :func:`export` wrote each of its six output files (design §R1).

    Attributes:
        report_model_json: The full ``ReportModel``, serialized as-is.
        sample_rankings_csv: Every sample, vulnerability-ranked (§1 격차#5).
        region_summary_csv: ``region_summary.rows``, one row per region_key.
        semantic_summary_csv: ``semantic_summary``, one row per
            semantic_group (IMPLE_PLAN_SEMANTIC_VULNERABILITY_v1.md §3.4);
            empty (header only) when ``semantic_concentration.
            n_semantic_groups <= 1``.
        class_semantic_matrix_csv: ``class_semantic_matrix``, one row per
            ``(gt_label, semantic_group)`` — the primary artifact the
            semantic vulnerability plan exists to produce (plan §3.4);
            empty (header only) under the same condition as
            ``semantic_summary_csv``.
        flagged_items_csv: ``reliability_spotlight.flagged_examples``,
            unchanged in content and order.
    """

    report_model_json: Path
    sample_rankings_csv: Path
    region_summary_csv: Path
    semantic_summary_csv: Path
    class_semantic_matrix_csv: Path
    flagged_items_csv: Path


_SAMPLE_RANKINGS_FIELDS = (
    "sample_id",
    "gt_label",
    "clean_correct",
    "vulnerability_score",
    "reliability_grade",
    "heatmap_asset_ref",
    "thumbnail_asset_ref",
    "top_regions_json",
    "task_extra_json",
)

_REGION_SUMMARY_FIELDS = (
    "region_key",
    "region_kind",
    "intended_area_px",
    "effective_area_px",
    "mean_degradation",
    "flip_rate",
    "n_valid",
    "reliability_grade",
    "high_count",
    "moderate_count",
    "low_count",
    "unreliable_count",
    "top_region_share",
    "high_rate",
)

_SEMANTIC_SUMMARY_FIELDS = (
    "semantic_group",
    "region_ids",
    "n_samples",
    "mean_degradation",
    "high_rate",
    "flip_rate",
)

_CLASS_SEMANTIC_MATRIX_FIELDS = (
    "gt_label",
    "semantic_group",
    "n_samples",
    "mean_degradation",
    "flip_rate",
)

_FLAGGED_ITEMS_FIELDS = (
    "anchor_key_repr",
    "reason_summary",
    "reliability_reasons_json",
)


def export(assembled: AssembledReportLike, output_dir: Path) -> ExportedPaths:
    """Write ``report_model.json`` plus three flattened CSVs (design §R1).

    Args:
        assembled: R0's output — see :class:`AssembledReportLike` for why
            this is not typed as the concrete ``AssembledReport``.
        output_dir: Directory the four files are written into; created
            (including parents) if it does not already exist.

    Returns:
        The four files' paths, in ``output_dir``, for a caller (typically
        R4) to link to.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = ExportedPaths(
        report_model_json=output_dir / "report_model.json",
        sample_rankings_csv=output_dir / "sample_rankings.csv",
        region_summary_csv=output_dir / "region_summary.csv",
        semantic_summary_csv=output_dir / "semantic_summary.csv",
        class_semantic_matrix_csv=output_dir / "class_semantic_matrix.csv",
        flagged_items_csv=output_dir / "flagged_items.csv",
    )

    write_json_atomic(paths.report_model_json, dataclasses.asdict(assembled.model))
    _write_csv(
        paths.sample_rankings_csv,
        _SAMPLE_RANKINGS_FIELDS,
        (_sample_ranking_row(card) for card in assembled.full_sample_rankings),
    )
    _write_csv(
        paths.region_summary_csv,
        _REGION_SUMMARY_FIELDS,
        (_region_summary_row(row) for row in assembled.model.region_summary.rows),
    )
    _write_csv(
        paths.semantic_summary_csv,
        _SEMANTIC_SUMMARY_FIELDS,
        (_semantic_summary_row(row) for row in assembled.model.semantic_summary),
    )
    _write_csv(
        paths.class_semantic_matrix_csv,
        _CLASS_SEMANTIC_MATRIX_FIELDS,
        (_class_semantic_row(row) for row in assembled.model.class_semantic_matrix),
    )
    _write_csv(
        paths.flagged_items_csv,
        _FLAGGED_ITEMS_FIELDS,
        (
            _flagged_item_row(item)
            for item in assembled.model.reliability_spotlight.flagged_examples
        ),
    )
    return paths


# --- row builders -------------------------------------------------------


def _sample_ranking_row(card: SampleCard) -> dict[str, object]:
    return {
        "sample_id": card.sample_id,
        "gt_label": _cell(card.gt_label),
        "clean_correct": _cell(card.clean_correct),
        "vulnerability_score": _cell(card.vulnerability_score),
        "reliability_grade": _cell(card.reliability_grade),
        "heatmap_asset_ref": _cell(card.heatmap_asset_ref),
        "thumbnail_asset_ref": _cell(card.thumbnail_asset_ref),
        "top_regions_json": json.dumps(
            [dataclasses.asdict(entry) for entry in card.top_regions], sort_keys=True
        ),
        "task_extra_json": json.dumps(dict(card.task_extra), sort_keys=True),
    }


def _region_summary_row(row: RegionRow) -> dict[str, object]:
    distribution = row.reliability_distribution
    return {
        "region_key": row.region_key,
        "region_kind": row.region_kind,
        "intended_area_px": _cell(row.intended_area_px),
        "effective_area_px": _cell(row.effective_area_px),
        "mean_degradation": _cell(row.mean_degradation),
        "flip_rate": _cell(row.flip_rate),
        "n_valid": row.n_valid,
        "reliability_grade": _cell(row.reliability_grade),
        "high_count": distribution.get(ReportGrade.HIGH.value, 0),
        "moderate_count": distribution.get(ReportGrade.MODERATE.value, 0),
        "low_count": distribution.get(ReportGrade.LOW.value, 0),
        "unreliable_count": distribution.get(ReportGrade.UNRELIABLE.value, 0),
        "top_region_share": _cell(row.top_region_share),
        "high_rate": _cell(row.high_rate),
    }


def _semantic_summary_row(row: SemanticGroupRow) -> dict[str, object]:
    return {
        "semantic_group": row.semantic_group,
        "region_ids": ";".join(row.region_ids),
        "n_samples": row.n_samples,
        "mean_degradation": _cell(row.mean_degradation),
        "high_rate": _cell(row.high_rate),
        "flip_rate": _cell(row.flip_rate),
    }


def _class_semantic_row(row: ClassSemanticRow) -> dict[str, object]:
    return {
        "gt_label": row.gt_label,
        "semantic_group": row.semantic_group,
        "n_samples": row.n_samples,
        "mean_degradation": _cell(row.mean_degradation),
        "flip_rate": _cell(row.flip_rate),
    }


def _flagged_item_row(item: FlaggedItem) -> dict[str, object]:
    return {
        "anchor_key_repr": item.anchor_key_repr,
        "reason_summary": item.reason_summary,
        "reliability_reasons_json": json.dumps(list(item.reliability_reasons), sort_keys=True),
    }


def _cell(value: object) -> object:
    """Render one scalar CSV cell: ``None`` becomes ``""``, an ``Enum`` becomes its plain value."""

    if value is None:
        return ""
    if isinstance(value, Enum):
        return value.value
    return value


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
