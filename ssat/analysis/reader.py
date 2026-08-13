"""A0 AnalysisReader: open one dump and its metrics store together.

This is the only module in ``ssat.analysis`` allowed to import
``ssat.metrics.dump_reader``/``ssat.metrics.store`` (design
IMPLE_PLAN_CONTROL_STABILITY_v1.md §3.1, §5 단계1) — every other
``ssat.analysis`` module must consume ``AnalysisReader``'s output instead of
reopening the dump or metrics store directly. This split exists because none
of the metrics engine's persisted parquet tables retain the condition axis
(``region_key``/``perturb_op``/``perturb_params``/``seed``/``invert_mask``/
``is_control``) needed to reconstruct an ``AnchorKey``/``ConditionKey`` — the
only surviving source for that axis is the raw dump
(``ssat.metrics.dump_reader.DumpHandle.items()``), so this reader must join
it against the loaded metrics by ``item_id``.

Error wrapping: this module intentionally does NOT wrap
``ssat.metrics.errors.*``/``ssat.core.dump.errors.*`` failures into
``ssat.analysis.errors.*``. ``ssat.metrics.dump_reader.DumpHandle`` — despite
being documented as "the only module importing ssat.core.dump" — already
propagates ``ssat.core.dump.errors.DumpCorruptionError``/``DumpSchemaError``
unwrapped from its own ``__init__`` (no try/except around ``DumpReader(...)``
there); this module treats the same boundary the same way.
``ssat.analysis.errors.AnalysisError`` and its subclasses remain reserved for
this package's own analysis-logic failures (e.g. a later step's geometry
inconsistency check), not for translating a dependency's own public
failures.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from ssat.analysis.types import AnchorKey, AvailableAnalyses, ConditionKey
from ssat.metrics.aggregate import AggregationResult
from ssat.metrics.dump_reader import DumpHandle
from ssat.metrics.store import MetricsManifest, load_metrics, verify_source_dump
from ssat.metrics.types import ItemMetrics
from ssat.utils.io import sha256_bytes

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


class AnalysisReader:
    """Open one dump + metrics store pair and expose a shared item context.

    Construction eagerly loads and validates both sources, and computes the
    condition-context frame once, so that failures and the join cost are
    paid at ``__init__`` time rather than on every downstream call.
    """

    def __init__(self, dump_dir: str | Path, metrics_dir: str | Path) -> None:
        """Load metrics, open the dump, and verify they still agree.

        Raises:
            MetricsCorruptionError: If a parquet fragment or manifest under
                ``metrics_dir`` is missing/unreadable, or if the dump under
                ``dump_dir`` was re-run since these metrics were computed
                (propagated from ``load_metrics``/``verify_source_dump``,
                unwrapped — see module docstring).
            MetricsSchemaError: If the metrics store's schema version does
                not match this codebase's (propagated, unwrapped).
        """

        self._item_metrics, self._result, self._manifest = load_metrics(
            Path(metrics_dir)
        )
        self._handle = DumpHandle(Path(dump_dir))

        # Raises MetricsCorruptionError if this dump was re-run since the
        # metrics store was computed (design §N4). Propagates unwrapped —
        # see module docstring "Error wrapping".
        verify_source_dump(self._manifest, self._handle.manifest_path)

        self._context = self._handle.items()[_CONTEXT_COLUMNS].reset_index(drop=True)

    @property
    def item_metrics(self) -> list[ItemMetrics]:
        """Return the loaded item-grain metrics rows."""

        return list(self._item_metrics)

    @property
    def result(self) -> AggregationResult:
        """Return the loaded sample/region/class/spatial aggregation outputs."""

        return self._result

    @property
    def manifest(self) -> MetricsManifest:
        """Return the loaded metrics store's manifest."""

        return self._manifest

    def item_context(self) -> pd.DataFrame:
        """Return the per-item condition-context frame, joinable on ``item_id``.

        Built from ``DumpHandle.items()``, not ``.joined()`` — clean-side
        status/logits are not needed by any A0 consumer (design §1 항목3).
        Returns a copy so callers cannot mutate the reader's cached frame.
        """

        return self._context.copy()

    def item_values(self) -> pd.DataFrame:
        """Join ``item_context()`` with ``item_metrics`` into the shared A2-A5 input shape.

        Every A2-A5 consumer (``analysis.control``, ``analysis.stability``,
        ``analysis.interval``) documents the same expected frame: the
        identity columns from ``item_context()`` plus one row per (item,
        metric_name) — ``metric_name``, ``degradation``, ``available``. This
        was originally hand-rolled per call site
        (``experiments/synthetic_shortcut/common.py``'s ``build_item_values``,
        the only production caller before the Application layer became a
        second one); it now lives here so neither has to re-derive the join.
        """

        metrics_frame = pd.DataFrame(
            {
                "item_id": item.item_id,
                "metric_name": item.metric_name,
                "degradation": item.degradation,
                "available": item.available,
            }
            for item in self._item_metrics
        )
        return self._context.merge(metrics_frame, on="item_id", how="inner")

    def available_analyses(self) -> AvailableAnalyses:
        """Report which stability/control analyses this dump+metrics pair supports."""

        context = self._context
        non_control = context.loc[~context["is_control"]]

        return AvailableAnalyses(
            control_comparison=bool(context["is_control"].any()),
            fill_strategy_stability=bool(non_control["perturb_op"].nunique() >= 2),
            seed_stability=_has_repeated_condition_group(context),
            # Always False: the core has no jitter axis to detect (see
            # module docstring / IMPLE_PLAN_CONTROL_STABILITY_v1.md §1
            # 항목2). Stays interface-only until the core gains jitter
            # support; pinned by a regression test.
            jitter_stability=False,
        )


def _has_repeated_condition_group(context: pd.DataFrame) -> bool:
    """Return True if any (AnchorKey, ConditionKey) pair covers >= 2 items.

    Reuses AnchorKey/ConditionKey's own key format (``region_key =
    f"{region_id}::{region_instance_id}"``, ``perturb_params_hash =
    sha256(perturb_params_json)``) rather than reinventing it. Deliberately
    not exposed as a column on ``item_context()`` — its column list is
    pinned by design §5 단계1; ``analysis.indexer`` will compute the same
    values again for its own AnchorTable/ConditionKey construction, matching
    the intentional ``region_key``-format duplication already documented on
    ``AnchorKey``.
    """

    counts: Counter[tuple[AnchorKey, ConditionKey]] = Counter()
    for row in context.itertuples(index=False):
        anchor = AnchorKey(
            sample_id=row.sample_id,
            region_key=f"{row.region_id}::{row.region_instance_id}",
            invert_mask=bool(row.invert_mask),
        )
        condition = ConditionKey(
            perturb_op=row.perturb_op,
            perturb_params_hash=sha256_bytes(row.perturb_params_json.encode("utf-8")),
        )
        counts[(anchor, condition)] += 1
    return any(count >= 2 for count in counts.values())
