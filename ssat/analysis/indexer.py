"""A1 ComparisonIndexer.

Decomposes ``AnalysisReader.item_context()``'s joined item context into an
AnchorTable and ControlPairs, keyed by ``AnchorKey``/``ConditionKey``
(``ssat.analysis.types``), and reports incomplete combinations via
``CoverageReport``.

This module intentionally does not import ``ssat.metrics.aggregate`` even
though ``_check_region_geometry`` below reproduces that module's policy for
detecting inconsistent per-region geometry: this package's dependency
direction restricts ``analysis.indexer`` to ``analysis.types`` and
``analysis.reader``'s output (a plain DataFrame), and this package already
accepts the smaller duplication of the ``region_key`` format string for the
same reason. Everything here consumes only ``AnalysisReader.item_context()``
column values.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ssat.analysis.errors import AnalysisCorruptionError
from ssat.analysis.types import (
    AnchorKey,
    AnchorRow,
    ConditionKey,
    ControlPairRow,
    CoverageReport,
    MatchMethod,
)
from ssat.utils.io import sha256_bytes

DEFAULT_AREA_MATCH_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class ComparisonIndex:
    """Bundle A1's three outputs computed from one item-context frame.

    Attributes:
        anchor_table: One ``AnchorRow`` per (sample_id, region_key,
            invert_mask), covering both target and control anchors.
        control_pairs: One ``ControlPairRow`` per successfully matched
            (control anchor, ``ConditionKey``) combination. A control anchor
            that could not be matched under a given condition contributes no
            row here -- see ``coverage_report.n_controls_unmatched``.
        coverage_report: Dataset-wide accounting of skipped/incomplete
            combinations.
    """

    anchor_table: tuple[AnchorRow, ...]
    control_pairs: tuple[ControlPairRow, ...]
    coverage_report: CoverageReport


class ComparisonIndexer:
    """Decompose an item-context frame into AnchorTable/ControlPairs/CoverageReport.

    Construction eagerly computes and caches the full ``ComparisonIndex``, in
    the same style as ``analysis.reader.AnalysisReader`` -- the input is a
    pure value (no I/O), so paying the computation cost once at ``__init__``
    is safe and keeps every property below a cheap attribute lookup.
    """

    def __init__(
        self,
        context: pd.DataFrame,
        *,
        area_match_tolerance: float = DEFAULT_AREA_MATCH_TOLERANCE,
    ) -> None:
        """Build the comparison index from one ``AnalysisReader.item_context()`` frame.

        Args:
            context: The item-context frame produced by
                ``AnalysisReader.item_context()``.
            area_match_tolerance: Fractional tolerance for the area-based
                control match, and for flagging area mismatches on
                exact-reference matches (default 5%, chosen when A1 was
                implemented).

        Raises:
            AnalysisCorruptionError: If the same concrete region
                (``region_key``) reports inconsistent ``region_kind`` or a
                conflicting non-null area across the rows that reference it,
                or if one anchor's rows disagree on ``is_control``.
        """

        anchor_table, anchor_by_key, rows_by_anchor = _build_anchor_table(context)
        control_pairs, n_unmatched, n_area_mismatch = _match_controls(
            rows_by_anchor,
            anchor_by_key,
            anchor_table,
            area_match_tolerance=area_match_tolerance,
        )
        n_insufficient = sum(1 for row in anchor_table if row.n_conditions < 2)

        self._index = ComparisonIndex(
            anchor_table=tuple(anchor_table),
            control_pairs=tuple(control_pairs),
            coverage_report=CoverageReport(
                n_anchors=len(anchor_table),
                n_conditions_insufficient=n_insufficient,
                n_controls_unmatched=n_unmatched,
                n_area_mismatch_warnings=n_area_mismatch,
            ),
        )

    @property
    def anchor_table(self) -> tuple[AnchorRow, ...]:
        """Return one ``AnchorRow`` per (sample_id, region_key, invert_mask)."""

        return self._index.anchor_table

    @property
    def control_pairs(self) -> tuple[ControlPairRow, ...]:
        """Return one ``ControlPairRow`` per matched (control anchor, ConditionKey)."""

        return self._index.control_pairs

    @property
    def coverage_report(self) -> CoverageReport:
        """Return dataset-wide accounting of skipped/incomplete combinations."""

        return self._index.coverage_report

    @property
    def index(self) -> ComparisonIndex:
        """Return the full ``ComparisonIndex`` bundle."""

        return self._index


def _region_key_column(df: pd.DataFrame) -> pd.Series:
    """Vectorized form of ``AnchorKey.region_key``'s ``f"{region_id}::{region_instance_id}"`` formula.

    Duplicated per module rather than extracted into a shared helper, the
    same trade-off ``AnchorKey.region_key`` already documents for the
    scalar formula (``ssat.analysis.types``).
    """

    return df["region_id"].astype(str) + "::" + df["region_instance_id"].astype(str)


def _perturb_params_hash_column(df: pd.DataFrame) -> pd.Series:
    """Vectorized form of ``ConditionKey.perturb_params_hash``.

    ``perturb_params_json`` takes only a handful of distinct values across
    an entire frame (one per configured (op, params) combination actually
    run), so this hashes each distinct value once via ``.map()`` instead of
    calling ``sha256_bytes`` once per row.
    """

    distinct = df["perturb_params_json"].unique()
    hash_by_json = {value: sha256_bytes(value.encode("utf-8")) for value in distinct}
    return df["perturb_params_json"].map(hash_by_json)


def _build_anchor_table(
    context: pd.DataFrame,
) -> tuple[list[AnchorRow], dict[AnchorKey, AnchorRow], dict[AnchorKey, list[Any]]]:
    """Decompose the item-context frame into one AnchorRow per (sample, region, mask).

    Also returns ``rows_by_anchor`` so ``_match_controls`` can reuse the same
    grouping instead of re-scanning ``context``.

    Raises:
        AnalysisCorruptionError: If the same concrete region reports
            inconsistent ``region_kind``, or inconsistent area within one
            sample, or if one anchor's rows disagree on ``is_control``.
    """

    # region_key/perturb_params_hash computed once as vectorized columns
    # (rather than per row below) since perturb_params_json only takes a
    # handful of distinct values across the whole frame -- sha256_bytes is
    # then called once per distinct value, not once per row. The geometry
    # conflict check below still runs row-by-row in frame order (it raises
    # on the first conflicting row it sees, with a message naming that
    # row's own values), so that sequencing is deliberately left alone.
    context = context.assign(
        region_key_col=_region_key_column(context),
        perturb_params_hash_col=_perturb_params_hash_column(context),
    )

    kind_by_region: dict[str, str] = {}
    area_by_region: dict[tuple[str, str], tuple[int | None, int | None]] = {}
    rows_by_anchor: dict[AnchorKey, list[Any]] = defaultdict(list)

    for row in context.itertuples(index=False):
        region_key = row.region_key_col
        _check_region_geometry(
            kind_by_region,
            area_by_region,
            sample_id=row.sample_id,
            region_key=region_key,
            region_kind=row.region_kind,
            intended_area_px=_area_or_none(row.intended_area_px),
            effective_area_px=_area_or_none(row.effective_area_px),
        )
        anchor_key = AnchorKey(
            sample_id=row.sample_id, region_key=region_key, invert_mask=bool(row.invert_mask)
        )
        rows_by_anchor[anchor_key].append(row)

    anchor_table: list[AnchorRow] = []
    anchor_by_key: dict[AnchorKey, AnchorRow] = {}
    for anchor_key, rows in rows_by_anchor.items():
        is_control_values = {bool(row.is_control) for row in rows}
        if len(is_control_values) != 1:
            raise AnalysisCorruptionError(
                f"anchor {anchor_key!r} mixes is_control=True and is_control=False rows"
            )
        is_control = next(iter(is_control_values))

        condition_keys = sorted(
            {
                ConditionKey(
                    perturb_op=row.perturb_op,
                    perturb_params_hash=row.perturb_params_hash_col,
                )
                for row in rows
            },
            key=lambda key: (key.perturb_op, key.perturb_params_hash),
        )
        # Taken from this anchor's own rows, not a region-wide map: an
        # anchor is already per-sample, so a region whose geometry varies
        # per sample must report *this* sample's area. Averaged because a
        # RANDOM_AREA_MATCH control is re-drawn per seed salt, so even one
        # anchor can hold several areas -- and its degradation is likewise
        # averaged over those draws by A2.
        intended_area_px = _mean_area([row.intended_area_px for row in rows])
        effective_area_px = _mean_area([row.effective_area_px for row in rows])

        anchor_row = AnchorRow(
            anchor_key=anchor_key,
            sample_id=anchor_key.sample_id,
            region_key=anchor_key.region_key,
            invert_mask=anchor_key.invert_mask,
            intended_area_px=intended_area_px,
            effective_area_px=effective_area_px,
            is_control=is_control,
            n_conditions=len(condition_keys),
            condition_keys=tuple(condition_keys),
        )
        anchor_table.append(anchor_row)
        anchor_by_key[anchor_key] = anchor_row

    anchor_table.sort(key=lambda row: (row.sample_id, row.region_key, row.invert_mask))
    return anchor_table, anchor_by_key, dict(rows_by_anchor)


def _area_or_none(value: Any) -> int | None:
    """Coerce one context area cell to ``int``, mapping null/NaN to ``None``.

    A missing area arrives from parquet as float NaN rather than ``None``,
    and ``NaN != NaN`` would otherwise read as a genuine geometry conflict.
    """

    if value is None or value != value:  # NaN is the only value unequal to itself.
        return None
    return int(value)


def _mean_area(areas: list[Any]) -> int | None:
    """Average the known areas of one anchor's rows, or ``None`` if all unknown."""

    known = [area for area in map(_area_or_none, areas) if area is not None]
    return int(round(sum(known) / len(known))) if known else None


def _check_region_geometry(
    kind_by_region: dict[str, str],
    area_by_region: dict[tuple[str, str], tuple[int | None, int | None]],
    *,
    sample_id: str,
    region_key: str,
    region_kind: str,
    intended_area_px: int | None,
    effective_area_px: int | None,
) -> None:
    """Track one region's geometry and reject a later, conflicting report.

    Mirrors ``ssat.metrics.aggregate._check_region_geometry``'s policy
    without importing it (module docstring), including its scoping: areas
    are compared only within one sample, because sample-dependent kinds
    (gt_bbox, skeleton_parts) legitimately differ per sample, and
    random_area_match is exempt entirely since its mask is re-drawn per
    item. ``None`` areas never conflict; they simply defer to whatever
    non-null value is already known.

    Raises:
        AnalysisCorruptionError: If ``region_kind`` differs from a
            previously recorded value for the same ``region_key``, or a
            non-null area differs from a previous value for the same
            ``(sample_id, region_key)``.
    """

    existing_kind = kind_by_region.setdefault(region_key, region_kind)
    if existing_kind != region_kind:
        raise AnalysisCorruptionError(f"region {region_key!r} reports inconsistent region_kind")

    # Literal rather than ``ssat.core.types.RegionKind.RANDOM_AREA_MATCH``:
    # this module intentionally stays off ``ssat.core``, the same trade-off
    # already accepted for the ``region_key`` format string.
    if region_kind == "random_area_match":
        return

    new_areas = (intended_area_px, effective_area_px)
    geometry_key = (sample_id, region_key)
    existing_areas = area_by_region.get(geometry_key)
    if existing_areas is None:
        area_by_region[geometry_key] = new_areas
        return

    merged: list[int | None] = []
    for existing, new, field_name in zip(
        existing_areas, new_areas, ("intended_area_px", "effective_area_px")
    ):
        if existing is not None and new is not None and existing != new:
            raise AnalysisCorruptionError(
                f"region {region_key!r} reports inconsistent {field_name} "
                f"within sample {sample_id!r}"
            )
        merged.append(existing if existing is not None else new)
    area_by_region[geometry_key] = (merged[0], merged[1])


def _match_controls(
    rows_by_anchor: dict[AnchorKey, list[Any]],
    anchor_by_key: dict[AnchorKey, AnchorRow],
    anchor_table: list[AnchorRow],
    *,
    area_match_tolerance: float,
) -> tuple[list[ControlPairRow], int, int]:
    """Pair each control anchor's conditions to a target anchor.

    For every control anchor and every distinct ``ConditionKey`` it exercises
    (one ``ControlPairRow`` per combination -- seed repeats collapse into the
    same ``ConditionKey`` and are not separately paired here, consistent with
    every other A1 output being condition-key-granular rather than
    item-granular): first try the embedded ``target_region`` reference
    (exact match); if that reference is absent, malformed, or does not
    resolve to a target anchor exercising this same condition, fall back to
    an area-tolerance search among every non-control anchor sharing this
    control's ``sample_id``/``invert_mask``/``ConditionKey``. A control
    (anchor, condition) combination that resolves neither way is counted in
    ``n_controls_unmatched`` and produces no row.

    Returns:
        A tuple of (control_pairs, n_controls_unmatched,
        n_area_mismatch_warnings).
    """

    control_pairs: list[ControlPairRow] = []
    n_unmatched = 0
    n_area_mismatch = 0

    for anchor_key, rows in rows_by_anchor.items():
        anchor_row = anchor_by_key[anchor_key]
        if not anchor_row.is_control:
            continue

        # All rows of one anchor share the same RegionSpec.params (params
        # are tied to the region, not the perturbation loop), so any one
        # row's region_params_json carries the same target_region recipe.
        target_ref = _parse_target_region_recipe(rows[0].region_params_json)

        rows_by_condition: dict[ConditionKey, list[Any]] = defaultdict(list)
        for row in rows:
            condition_key = ConditionKey(
                perturb_op=row.perturb_op,
                perturb_params_hash=row.perturb_params_hash_col,
            )
            rows_by_condition[condition_key].append(row)

        for condition_key in rows_by_condition:
            match: AnchorRow | None = None
            match_method: MatchMethod | None = None

            if target_ref is not None:
                target_region_key = (
                    f"{target_ref['region_id']}::{target_ref['region_instance_id']}"
                )
                candidate = anchor_by_key.get(
                    AnchorKey(
                        sample_id=anchor_key.sample_id,
                        region_key=target_region_key,
                        invert_mask=anchor_key.invert_mask,
                    )
                )
                if candidate is not None and condition_key in candidate.condition_keys:
                    match = candidate
                    match_method = MatchMethod.EXACT_REFERENCE

            if match is None:
                match = _area_tolerance_fallback(
                    anchor_row,
                    condition_key=condition_key,
                    anchor_table=anchor_table,
                    area_match_tolerance=area_match_tolerance,
                )
                if match is not None:
                    match_method = MatchMethod.AREA_TOLERANCE

            if match is None or match_method is None:
                n_unmatched += 1
                continue

            ratio = _area_match_ratio(anchor_row, match)
            if ratio is not None and not (
                1.0 - area_match_tolerance <= ratio <= 1.0 + area_match_tolerance
            ):
                n_area_mismatch += 1

            control_pairs.append(
                ControlPairRow(
                    target_anchor_key=match.anchor_key,
                    control_anchor_key=anchor_key,
                    condition_key=condition_key,
                    match_method=match_method,
                    area_match_ratio=ratio,
                )
            )

    control_pairs.sort(
        key=lambda pair: (
            pair.target_anchor_key.sample_id,
            pair.target_anchor_key.region_key,
            pair.control_anchor_key.region_key,
            pair.condition_key.perturb_op,
            pair.condition_key.perturb_params_hash,
        )
    )
    return control_pairs, n_unmatched, n_area_mismatch


def _parse_target_region_recipe(region_params_json: str) -> dict[str, Any] | None:
    """Parse a control item's embedded target-region reference, if usable.

    The recipe is produced by ``PlanBuilder._region_recipe``
    (``ssat/core/plan/builder.py``) and embedded under a ``target_region``
    key inside the control's ``region_params_json`` (itself
    ``canonical_json`` of ``{"target_region": {...}, "control_request_index":
    ..., "control_index": ...}`` -- not a flat top-level recipe).

    Returns:
        The ``target_region`` sub-dict when present with non-empty
        ``region_id``/``region_instance_id`` strings; ``None`` for any
        parse failure or malformed/missing recipe -- callers treat that as
        "exact reference unavailable" and fall through to the area-tolerance
        match.
    """

    try:
        parsed = json.loads(region_params_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None

    target_region = parsed.get("target_region")
    if not isinstance(target_region, dict):
        return None

    region_id = target_region.get("region_id")
    region_instance_id = target_region.get("region_instance_id")
    if not isinstance(region_id, str) or not region_id:
        return None
    if not isinstance(region_instance_id, str) or not region_instance_id:
        return None
    return target_region


def _area_tolerance_fallback(
    control_anchor: AnchorRow,
    *,
    condition_key: ConditionKey,
    anchor_table: list[AnchorRow],
    area_match_tolerance: float,
) -> AnchorRow | None:
    """Search every non-control anchor for the closest in-tolerance area match.

    Restricted to anchors sharing this control's ``sample_id``,
    ``invert_mask``, and this ``condition_key`` (the fallback path). Among
    candidates within ``area_match_tolerance``, the one with the smallest
    deviation from a 1.0 ratio wins.
    """

    best: tuple[float, AnchorRow] | None = None
    for candidate in anchor_table:
        if candidate.is_control:
            continue
        if candidate.sample_id != control_anchor.sample_id:
            continue
        if candidate.invert_mask != control_anchor.invert_mask:
            continue
        if condition_key not in candidate.condition_keys:
            continue
        ratio = _area_match_ratio(control_anchor, candidate)
        if ratio is None:
            continue
        deviation = abs(ratio - 1.0)
        if deviation > area_match_tolerance:
            continue
        if best is None or deviation < best[0]:
            best = (deviation, candidate)
    return best[1] if best is not None else None


def _area_match_ratio(control_anchor: AnchorRow, target_anchor: AnchorRow) -> float | None:
    """Compute control-area / target-area, preferring effective over intended area.

    Area basis: ``effective_area_px`` is used when *both* sides have it;
    otherwise both sides fall back to ``intended_area_px`` together (never
    mixed bases).

    Returns:
        The ratio, or ``None`` when the chosen area basis is unavailable on
        either side or the target's area is zero.
    """

    if control_anchor.effective_area_px is not None and target_anchor.effective_area_px is not None:
        control_area = control_anchor.effective_area_px
        target_area = target_anchor.effective_area_px
    else:
        control_area = control_anchor.intended_area_px
        target_area = target_anchor.intended_area_px

    if control_area is None or target_area is None or target_area == 0:
        return None
    return control_area / target_area
