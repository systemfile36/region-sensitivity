"""N0 DumpReader wrapper: joins raw dump chunks into a JoinedFrame.

``ssat.core.dump.DumpReader`` is the schema-validated, chunk-streaming
contract between the core and everything downstream. This module is the
*only* place in the metrics engine that imports ``ssat.core.dump`` — every
other metrics module consumes the ``JoinedFrame``/``summary()`` output of
:class:`DumpHandle` instead (design METRIC_ENGINE_DESIGN_v1.md §N0,
IMPLE_PLAN_METRIC_DESIGN_v1.md §3.1).
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import pandas as pd

from ssat.core.dump import DumpReader
from ssat.core.dump.types import RunManifest
from ssat.core.types import ItemStatus, SCHEMA_VERSION
from ssat.metrics.errors import MetricsCorruptionError

JoinedFrame: TypeAlias = pd.DataFrame
"""Item-grain frame combining each perturbed item with its clean sample.

Columns (design §N0, plus ``region_id``/``region_instance_id`` needed by
later Aggregator/DebugViz stages to identify and re-resolve a concrete
region):
``item_id``, ``sample_id``, ``gt_label``, ``region_id``,
``region_instance_id``, ``region_kind``, ``region_params_json``,
``intended_area_px``, ``effective_area_px``, ``effective_area_available``,
``perturb_op``, ``perturb_params_json``, ``invert_mask``, ``is_control``,
``seed_used``, ``logits_perturbed``, ``logits_clean``, ``status_perturbed``,
``status_clean``.

Only items whose sample's clean status is ``ok`` are present. Samples whose
clean status is not ``ok`` are excluded entirely and reported through
:meth:`DumpHandle.summary` instead of appearing here (design §N0 status
table, row "clean≠ok"). Items whose *perturbed* status is not ``ok`` remain
present with ``logits_perturbed=None`` — later metric computation is
responsible for treating them as unavailable (design §N0 status table, row
"clean=ok, perturbed≠ok").
"""

_JOINED_COLUMNS = [
    "item_id",
    "sample_id",
    "gt_label",
    "region_id",
    "region_instance_id",
    "region_kind",
    "region_params_json",
    "intended_area_px",
    "effective_area_px",
    "effective_area_available",
    "perturb_op",
    "perturb_params_json",
    "invert_mask",
    "is_control",
    "seed_used",
    "logits_perturbed",
    "logits_clean",
    "status_perturbed",
    "status_clean",
]


class DumpHandle:
    """Expose clean/perturbed/joined/summary views over one raw audit dump."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_schema_version: str = SCHEMA_VERSION,
    ) -> None:
        self._root = Path(root)
        self._reader = DumpReader(root, expected_schema_version=expected_schema_version)

    @property
    def root(self) -> Path:
        """Return the dump directory this handle wraps."""

        return self._root

    @property
    def manifest_path(self) -> Path:
        """Return the path to this dump's run_manifest.json file.

        Exists so that MetricsStore (metrics/store.py) can hash the
        manifest's actual on-disk bytes for ``source_run_manifest_hash``,
        without importing ``ssat.core.dump`` itself — this remains the only
        module in the metrics engine that touches it.
        """

        return self._root / "run_manifest.json"

    @property
    def manifest(self) -> RunManifest:
        """Return the validated run manifest of the wrapped dump."""

        return self._reader.read_manifest()

    def clean(self) -> pd.DataFrame:
        """Return every authoritative clean-sample row."""

        return self._reader.read_clean().to_pandas()

    def items(self) -> pd.DataFrame:
        """Return every authoritative perturbed-item row."""

        return self._reader.read_perturbed().to_pandas()

    def joined(self) -> JoinedFrame:
        """Combine perturbed items with their clean sample's status and logits.

        Raises:
            MetricsCorruptionError: If a perturbed item references a
                sample_id absent from the clean data entirely (design §N0
                status table, row "clean 레코드 없음").
        """

        clean_df = self.clean()
        items_df = self.items()
        _reject_orphan_items(clean_df, items_df)

        clean_slim = clean_df[["sample_id", "gt_label", "status", "logits"]].rename(
            columns={"status": "status_clean", "logits": "logits_clean"}
        )
        clean_ok = clean_slim[clean_slim["status_clean"] == ItemStatus.OK.value]

        items_slim = items_df.rename(
            columns={"status": "status_perturbed", "logits": "logits_perturbed"}
        )
        merged = items_slim.merge(clean_ok, on="sample_id", how="inner")
        return merged[_JOINED_COLUMNS].reset_index(drop=True)

    def summary(self) -> dict[str, object]:
        """Report status counts and dropout/exclusion accounting.

        Returns:
            A dict with ``total_clean_samples``, ``total_perturbed_items``,
            ``clean_status_counts`` (``dict[ItemStatus, int]``),
            ``perturbed_status_counts`` (``dict[ItemStatus, int]``),
            ``samples_excluded_clean_failed`` (samples dropped from
            :meth:`joined` entirely because clean status was not ``ok``),
            and ``items_excluded_perturbed_failed`` (items kept in
            :meth:`joined` but unavailable because their perturbed status
            was not ``ok`` while clean succeeded).

        Raises:
            MetricsCorruptionError: If a perturbed item references a
                sample_id absent from the clean data entirely.
        """

        clean_df = self.clean()
        items_df = self.items()
        _reject_orphan_items(clean_df, items_df)

        clean_ok_ids = set(
            clean_df.loc[clean_df["status"] == ItemStatus.OK.value, "sample_id"]
        )
        samples_excluded_clean_failed = clean_df["sample_id"].nunique() - len(
            clean_ok_ids
        )
        items_of_ok_clean = items_df[items_df["sample_id"].isin(clean_ok_ids)]
        items_excluded_perturbed_failed = int(
            (items_of_ok_clean["status"] != ItemStatus.OK.value).sum()
        )

        return {
            "total_clean_samples": len(clean_df),
            "total_perturbed_items": len(items_df),
            "clean_status_counts": _status_counts(clean_df["status"]),
            "perturbed_status_counts": _status_counts(items_df["status"]),
            "samples_excluded_clean_failed": int(samples_excluded_clean_failed),
            "items_excluded_perturbed_failed": items_excluded_perturbed_failed,
        }


def _reject_orphan_items(clean_df: pd.DataFrame, items_df: pd.DataFrame) -> None:
    """Raise if any perturbed item references a sample_id missing from clean.

    Raises:
        MetricsCorruptionError: If one or more perturbed items reference a
            sample_id absent from the clean data entirely.
    """

    known_samples = set(clean_df["sample_id"])
    orphan_mask = ~items_df["sample_id"].isin(known_samples)
    if orphan_mask.any():
        orphan_ids = sorted(items_df.loc[orphan_mask, "sample_id"].unique())
        raise MetricsCorruptionError(
            f"perturbed items reference sample_ids absent from clean data: "
            f"{orphan_ids[:5]}"
        )


def _status_counts(status_column: pd.Series) -> dict[ItemStatus, int]:
    """Count rows per ItemStatus, including statuses with zero occurrences."""

    return {
        status: int((status_column == status.value).sum()) for status in ItemStatus
    }


def coerce_nullable_int(value: object) -> int | None:
    """Coerce one nullable-int64 ``JoinedFrame`` cell to ``int | None``.

    A parquet nullable ``int64`` column with at least one missing value
    arrives through pandas as float ``NaN`` for the missing rows (arrow's
    default ``to_pandas`` conversion never uses bare ``None`` for a numeric
    column), and ``NaN`` famously fails equality with itself, so neither a
    bare ``value is None`` check nor a naive ``int(value)`` is safe. Shared
    here — rather than duplicated per call site — because both N2
    (``MetricRegistry.compute_item_metrics``'s ``gt_label``) and N3
    (``aggregate._build_context``'s ``gt_label``) need the exact same
    coercion for the exact same column (design METRIC_ENGINE_DESIGN_v1.md's
    "결측을 조용히 버리지 않는다" principle applied here: a naive cast would
    crash the entire run instead of letting a caller decide how to record
    the exclusion).

    Args:
        value: One cell read from a ``JoinedFrame`` int64 column, e.g. via
            ``row.gt_label`` from ``itertuples()``.

    Returns:
        The value as a plain ``int``, or ``None`` if it was missing.
    """

    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN is unequal to itself.
        return None
    return int(value)
