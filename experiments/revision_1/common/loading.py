"""Column-selective loaders for dumps, metrics stores, and analysis stores.

``ssat.metrics.dump_reader.DumpHandle`` materializes every perturbed row
(logits included) through Python objects, which does not scale to the
3.2 M-item ImageNet runs. The loaders here read only the requested parquet
columns while applying the same authoritative-row rule as
``ssat.core.dump.DumpReader``: an item's row is the one in the perturbed
chunk named by its latest index entry, and a sample's clean row is its last
occurrence in fragment order.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from experiments.revision_1.common.runs import RunRef
from ssat.analysis.store import AnalysisManifest, load_analysis
from ssat.analysis.types import (
    ControlComparisonRow,
    CoverageReport,
    IntervalRow,
    RankCorrelationRow,
    ReliabilityRow,
    SeedStabilityRow,
    StrategyProfileRow,
    StrategyStabilityRow,
)
from ssat.core.dump._storage import fragment_files
from ssat.core.types import SCHEMA_VERSION
from ssat.utils.io import load_json, sha256_file

CONTEXT_COLUMNS: tuple[str, ...] = (
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
)
"""Same columns as ``ssat.analysis.reader.AnalysisReader.item_context()``."""


class AnalysisRows(NamedTuple):
    """Typed rows of one analysis store, as returned by ``load_analysis``."""

    control: list[ControlComparisonRow]
    seed: list[SeedStabilityRow]
    strategy: list[StrategyStabilityRow]
    rank_correlation: list[RankCorrelationRow]
    strategy_profile: list[StrategyProfileRow]
    intervals: list[IntervalRow]
    reliability: list[ReliabilityRow]
    coverage: CoverageReport
    manifest: AnalysisManifest


def _check_dump_version(path: Path) -> None:
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    version = metadata.get(b"ssat.schema_version", b"").decode("ascii")
    if version != SCHEMA_VERSION:
        raise ValueError(f"{path}: dump schema {version!r}, expected {SCHEMA_VERSION!r}")


def latest_index(dump: Path) -> pd.DataFrame:
    """Return one ``(item_id, ordinal)`` row per item from its latest index entry."""

    frames = []
    for ordinal, path in fragment_files(dump / "index", "part"):
        item_ids = pq.read_table(path, columns=["item_id"]).column("item_id").to_numpy()
        frames.append(pd.DataFrame({"item_id": item_ids, "ordinal": ordinal}))
    if not frames:
        return pd.DataFrame({"item_id": pd.Series(dtype=object), "ordinal": pd.Series(dtype=np.int64)})
    index = pd.concat(frames, ignore_index=True)
    return index.drop_duplicates("item_id", keep="last").reset_index(drop=True)


def iter_latest_perturbed(
    dump: Path, columns: Sequence[str], *, filter_expression: pc.Expression | None = None
) -> Iterator[pa.Table]:
    """Yield authoritative perturbed rows chunk by chunk, restricted to ``columns``.

    Args:
        dump: Dump directory.
        columns: Perturbed-schema columns to read (``item_id`` is always added).
        filter_expression: Optional row filter applied after the latest-row
            selection (e.g. ``pc.field("is_control") == False``).

    Raises:
        ValueError: If a fragment has the wrong schema version, or an index
            entry has no matching row in its referenced chunk.
    """

    columns = list(dict.fromkeys(["item_id", *columns]))
    index = latest_index(dump)
    ids_by_ordinal = {
        ordinal: group["item_id"].to_numpy()
        for ordinal, group in index.groupby("ordinal", sort=False)
    }
    found = 0
    for ordinal, path in fragment_files(dump / "perturbed", "chunk"):
        expected = ids_by_ordinal.get(ordinal)
        if expected is None:
            continue
        _check_dump_version(path)
        table = pq.read_table(path, columns=columns)
        keep = pc.is_in(table.column("item_id"), value_set=pa.array(expected))
        table = table.filter(keep)
        if table.num_rows != len(expected):
            raise ValueError(f"{path}: {len(expected)} indexed items but {table.num_rows} rows")
        found += table.num_rows
        if filter_expression is not None:
            table = table.filter(filter_expression)
        if table.num_rows:
            yield table
    if found != len(index):
        raise ValueError(f"{dump}: {len(index) - found} indexed items have no data fragment")


def load_item_context(run: RunRef, columns: Sequence[str] = CONTEXT_COLUMNS) -> pd.DataFrame:
    """Return the per-item context frame (no logits) for every authoritative item."""

    tables = list(iter_latest_perturbed(run.dump, columns))
    if not tables:
        raise ValueError(f"{run.dump} has no perturbed items")
    return pa.concat_tables(tables).select(list(columns)).to_pandas()


def verify_metrics_source(run: RunRef) -> None:
    """Raise if the metrics store was computed from a different ``run_manifest.json``.

    Raises:
        ValueError: If the recorded source hash does not match the dump.
    """

    manifest = load_json(run.metrics / "metrics_manifest.json")
    actual = sha256_file(run.dump / "run_manifest.json")
    if manifest["source_run_manifest_hash"] != actual:
        raise ValueError(f"{run.name}: metrics store does not match its dump's run_manifest.json")


def read_metric_table(
    path: Path,
    metrics: Sequence[str] | None,
    *,
    columns: Sequence[str] | None = None,
    metric_column: str = "metric_name",
) -> pd.DataFrame:
    """Read a metrics/analysis parquet file, keeping only rows for ``metrics``."""

    filters = None if metrics is None else [(metric_column, "in", list(metrics))]
    return pq.read_table(path, columns=None if columns is None else list(columns), filters=filters).to_pandas()


def load_item_values(
    run: RunRef, metrics: Sequence[str] = ("margin_drop",), *, verify: bool = True
) -> pd.DataFrame:
    """Return ``AnalysisReader.item_values()`` restricted to ``metrics``.

    Columns are ``CONTEXT_COLUMNS`` plus ``metric_name``, ``degradation``,
    and ``available``; rows follow the item context order.
    """

    if verify:
        verify_metrics_source(run)
    context = load_item_context(run)
    values = read_metric_table(
        run.metrics / "item_metrics.parquet",
        metrics,
        columns=("item_id", "metric_name", "degradation", "available"),
    )
    return context.merge(values, on="item_id", how="inner")


def load_clean(run: RunRef, *, with_logits: bool = False) -> pd.DataFrame:
    """Return one authoritative clean row per sample (the last in fragment order)."""

    columns = ["sample_id", "gt_label", "status", *(["logits"] if with_logits else [])]
    tables = []
    for _, path in fragment_files(run.dump / "clean", "part"):
        _check_dump_version(path)
        tables.append(pq.read_table(path, columns=columns))
    frame = pa.concat_tables(tables).to_pandas()
    return frame.drop_duplicates("sample_id", keep="last").reset_index(drop=True)


def load_analysis_rows(run: RunRef) -> AnalysisRows:
    """Load the typed analysis rows via ``ssat.analysis.store.load_analysis``.

    Builds Python objects for every row; use the parquet readers below for
    the full-size ImageNet runs.
    """

    return AnalysisRows(*load_analysis(run.analysis))


def load_reliability(run: RunRef, metric: str = "margin_drop") -> pd.DataFrame:
    """Return ``reliability.parquet`` rows for one metric."""

    return read_metric_table(run.analysis / "reliability.parquet", [metric])


def load_spatial_profile(run: RunRef, metric: str = "margin_drop") -> pd.DataFrame:
    """Return ``spatial_profile.parquet`` rows (sample x region) for one metric."""

    return read_metric_table(run.metrics / "spatial_profile.parquet", [metric])


def load_region_metrics(run: RunRef, metric: str = "margin_drop") -> pd.DataFrame:
    """Return ``region_metrics.parquet`` rows for one metric."""

    return read_metric_table(run.metrics / "region_metrics.parquet", [metric])


def load_sample_meta(run: RunRef, metric: str = "margin_drop") -> pd.DataFrame:
    """Return ``sample_id``, ``gt_label``, ``clean_correct`` for every scored sample."""

    return read_metric_table(
        run.metrics / "sample_metrics.parquet",
        [metric],
        columns=("sample_id", "metric_name", "gt_label", "clean_correct"),
    ).drop(columns="metric_name")
