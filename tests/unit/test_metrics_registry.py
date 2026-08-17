"""Tests for the metrics engine's N2 MetricRegistry (registration, compute_item_metrics)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pytest

from ssat.core.adapter.types import AdapterSpec
from ssat.metrics.errors import MetricsRegistryError
from ssat.metrics.normalize import NormalizedOutput
from ssat.metrics.registry import Metric, MetricRegistry, MetricResult
from ssat.metrics.types import ExclusionReason


def _adapter_spec() -> AdapterSpec:
    return AdapterSpec(model_id="model", deterministic=True)


class _FakeMetric:
    """A fixed-output metric that counts how often compute() is called."""

    name = "fake_metric"
    requires: tuple[str, ...] = ()
    higher_is_better = True
    kind: Literal["continuous"] = "continuous"

    def __init__(self, *, available: bool = True) -> None:
        self._available = available
        self.compute_calls = 0

    def available_when(self, adapter_spec: AdapterSpec) -> bool:
        return self._available

    def compute(self, clean: NormalizedOutput, perturbed: NormalizedOutput) -> MetricResult:
        self.compute_calls += 1
        return MetricResult(value_clean=1.0, value_perturbed=0.5, degradation=0.5)


def _joined_frame(
    *, include_failed_item: bool = False, include_unlabeled_item: bool = False
) -> pd.DataFrame:
    rows = [
        {
            "item_id": "a" * 64,
            "sample_id": "sample-a",
            "gt_label": 0,
            "logits_clean": np.array([2.0, 0.0], dtype=np.float64),
            "logits_perturbed": np.array([2.0, 0.0], dtype=np.float64),
        },
        {
            "item_id": "b" * 64,
            "sample_id": "sample-b",
            "gt_label": 1,
            "logits_clean": np.array([2.0, 0.0], dtype=np.float64),
            "logits_perturbed": np.array([0.0, 2.0], dtype=np.float64),
        },
    ]
    if include_failed_item:
        rows.append(
            {
                "item_id": "c" * 64,
                "sample_id": "sample-c",
                "gt_label": 0,
                "logits_clean": np.array([2.0, 0.0], dtype=np.float64),
                "logits_perturbed": None,
            }
        )
    if include_unlabeled_item:
        # gt_label=None mixed with int rows above upcasts the whole pandas
        # column to float64 NaN for this row (arrow/pandas nullable-int
        # quirk) — exactly the shape MetricRegistry.compute_item_metrics
        # must handle without crashing.
        rows.append(
            {
                "item_id": "d" * 64,
                "sample_id": "sample-d",
                "gt_label": None,
                "logits_clean": np.array([2.0, 0.0], dtype=np.float64),
                "logits_perturbed": np.array([2.0, 0.0], dtype=np.float64),
            }
        )
    return pd.DataFrame(rows)


def test_register_accepts_a_conforming_metric() -> None:
    registry = MetricRegistry()
    registry.register(_FakeMetric())
    assert registry.names == ("fake_metric",)


def test_register_rejects_duplicate_names() -> None:
    registry = MetricRegistry()
    registry.register(_FakeMetric())
    with pytest.raises(MetricsRegistryError, match="already registered"):
        registry.register(_FakeMetric())


def test_register_rejects_non_conforming_objects() -> None:
    registry = MetricRegistry()
    with pytest.raises(TypeError, match="Metric protocol"):
        registry.register("not a metric")  # type: ignore[arg-type]


def test_compute_item_metrics_skips_metrics_unavailable_for_this_run() -> None:
    registry = MetricRegistry()
    unavailable = _FakeMetric(available=False)
    registry.register(unavailable)

    rows = registry.compute_item_metrics(_joined_frame(), adapter_spec=_adapter_spec())

    assert rows == []
    assert unavailable.compute_calls == 0


def test_compute_item_metrics_produces_one_row_per_item_and_shares_clean_correct() -> None:
    registry = MetricRegistry()
    metric = _FakeMetric()
    registry.register(metric)

    rows = registry.compute_item_metrics(_joined_frame(), adapter_spec=_adapter_spec())

    assert len(rows) == 2
    assert metric.compute_calls == 2
    by_item = {row.item_id: row for row in rows}
    assert by_item["a" * 64].clean_correct is True
    assert by_item["a" * 64].value_clean == pytest.approx(1.0)
    assert by_item["a" * 64].value_perturbed == pytest.approx(0.5)
    assert by_item["a" * 64].degradation == pytest.approx(0.5)
    assert by_item["a" * 64].available is True
    assert by_item["a" * 64].excluded_reason is None
    # sample-b's gt_label=1 has clean logits favoring class 0 -> clean_correct=False.
    assert by_item["b" * 64].clean_correct is False


def test_compute_item_metrics_excludes_items_with_unavailable_perturbed_side() -> None:
    registry = MetricRegistry()
    metric = _FakeMetric()
    registry.register(metric)

    rows = registry.compute_item_metrics(
        _joined_frame(include_failed_item=True), adapter_spec=_adapter_spec()
    )

    by_item = {row.item_id: row for row in rows}
    failed_row = by_item["c" * 64]
    assert failed_row.available is False
    assert failed_row.excluded_reason is ExclusionReason.PERTURBED_STATUS_NOT_OK
    assert failed_row.value_clean is None
    assert failed_row.value_perturbed is None
    assert failed_row.degradation is None
    # compute() is only called for the two available items, not the failed one.
    assert metric.compute_calls == 2


def test_compute_item_metrics_excludes_items_with_unknown_gt_label() -> None:
    """Previously crashed with ``ValueError: cannot convert float NaN to integer``.

    Core/source layer explicitly supports ``gt_label: int | None`` for
    label-free, inference-only auditing (``ssat/core/source/types.py``);
    the registry must exclude such items with a clear reason instead of
    crashing the entire run.
    """

    registry = MetricRegistry()
    metric = _FakeMetric()
    registry.register(metric)

    rows = registry.compute_item_metrics(
        _joined_frame(include_unlabeled_item=True), adapter_spec=_adapter_spec()
    )

    by_item = {row.item_id: row for row in rows}
    unlabeled_row = by_item["d" * 64]
    assert unlabeled_row.available is False
    assert unlabeled_row.excluded_reason is ExclusionReason.GT_LABEL_UNKNOWN
    assert unlabeled_row.clean_correct is None
    assert unlabeled_row.value_clean is None
    assert unlabeled_row.value_perturbed is None
    assert unlabeled_row.degradation is None
    # compute() is only called for the two labeled items.
    assert metric.compute_calls == 2
