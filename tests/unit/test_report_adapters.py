"""Unit tests for ssat.report.adapters (R5 TaskPresentationAdapter)."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from ssat.metrics.types import SampleMetrics
from ssat.report.adapters import ClassificationAdapter, DetectionAdapter
from ssat.report.types import MetricCard

_PRIMARY_METRIC = "gt_logit_drop"


def _sample_row(
    sample_id: str,
    *,
    clean_correct: bool | None,
    metric_mean: float | None,
    flip_rate: float | None,
    n_items: int = 4,
    n_valid: int = 4,
) -> SampleMetrics:
    return SampleMetrics(
        sample_id=sample_id,
        metric_name=_PRIMARY_METRIC,
        gt_label=0,
        clean_correct=clean_correct,
        n_items=n_items,
        n_valid=n_valid,
        flip_rate=flip_rate,
        vulnerability_score=metric_mean,
        metric_mean=metric_mean,
        metric_max=metric_mean,
        metric_std=0.0 if metric_mean is not None else None,
    )


def _card(cards: list[MetricCard], key: str) -> MetricCard:
    for card in cards:
        if card.key == key:
            return card
    raise AssertionError(f"no card with key={key!r} in {[c.key for c in cards]}")


# --- dependency direction ------------------------------------------------------


def test_report_adapters_module_has_no_analysis_or_metrics_or_core_imports() -> None:
    """Statically enforce report.adapters → report.types only."""

    source_path = Path(__file__).resolve().parents[2] / "ssat" / "report" / "adapters.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_prefixes = ("ssat.analysis", "ssat.metrics", "ssat.core", "ssat.application")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(forbidden_prefixes), alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith(forbidden_prefixes), module


# --- ClassificationAdapter.summarize_performance --------------------------------


def test_summarize_performance_matches_hand_calculation_for_binary_metric() -> None:
    rows = [
        _sample_row("s0", clean_correct=True, metric_mean=0.2, flip_rate=0.0),
        _sample_row("s1", clean_correct=False, metric_mean=0.8, flip_rate=1.0),
        _sample_row("s2", clean_correct=True, metric_mean=0.5, flip_rate=0.5),
    ]
    adapter = ClassificationAdapter(primary_metric=_PRIMARY_METRIC)

    cards = adapter.summarize_performance(rows)

    accuracy = _card(cards, "accuracy")
    assert accuracy.value == pytest.approx(2 / 3)
    assert accuracy.unit == "%"
    assert accuracy.higher_is_better is True
    assert accuracy.note is None

    degradation = _card(cards, f"mean_{_PRIMARY_METRIC}")
    assert degradation.value == pytest.approx((0.2 + 0.8 + 0.5) / 3)
    assert degradation.higher_is_better is False
    assert degradation.note is None

    flip_rate = _card(cards, "flip_rate")
    assert flip_rate.value == pytest.approx((0.0 + 1.0 + 0.5) / 3)
    assert flip_rate.higher_is_better is False
    assert flip_rate.note is None


def test_summarize_performance_omits_flip_rate_value_for_continuous_metric() -> None:
    """primary_metric is continuous (kind="continuous") -> flip_rate is None on every row."""

    rows = [
        _sample_row("s0", clean_correct=True, metric_mean=0.1, flip_rate=None),
        _sample_row("s1", clean_correct=True, metric_mean=0.3, flip_rate=None),
    ]
    adapter = ClassificationAdapter(primary_metric=_PRIMARY_METRIC)

    cards = adapter.summarize_performance(rows)

    flip_rate = _card(cards, "flip_rate")
    assert flip_rate.value is None
    assert flip_rate.note is not None and "해당 없음" in flip_rate.note
    # The card is still present — unavailable is expressed via note, not omission.
    assert len(cards) == 3


def test_summarize_performance_marks_accuracy_unavailable_without_gt_labels() -> None:
    rows = [
        _sample_row("s0", clean_correct=None, metric_mean=0.1, flip_rate=None),
        _sample_row("s1", clean_correct=None, metric_mean=0.2, flip_rate=None),
    ]
    adapter = ClassificationAdapter(primary_metric=_PRIMARY_METRIC)

    accuracy = _card(adapter.summarize_performance(rows), "accuracy")

    assert accuracy.value is None
    assert accuracy.note is not None and "해당 없음" in accuracy.note


def test_summarize_performance_handles_empty_sample_metrics() -> None:
    adapter = ClassificationAdapter(primary_metric=_PRIMARY_METRIC)

    cards = adapter.summarize_performance([])

    assert len(cards) == 3
    assert all(card.value is None for card in cards)
    assert all(card.note is not None for card in cards)


def test_summarize_performance_skips_none_rows_when_averaging() -> None:
    """A row with metric_mean=None (e.g. n_valid=0) must not be treated as 0."""

    rows = [
        _sample_row("s0", clean_correct=True, metric_mean=1.0, flip_rate=None),
        _sample_row("s1", clean_correct=True, metric_mean=None, flip_rate=None, n_valid=0),
    ]
    adapter = ClassificationAdapter(primary_metric=_PRIMARY_METRIC)

    degradation = _card(adapter.summarize_performance(rows), f"mean_{_PRIMARY_METRIC}")

    assert degradation.value == pytest.approx(1.0)
    assert not math.isnan(degradation.value)


# --- ClassificationAdapter.sample_extra_fields -----------------------------------


def test_sample_extra_fields_is_empty_for_classification() -> None:
    adapter = ClassificationAdapter(primary_metric=_PRIMARY_METRIC)
    row = _sample_row("s0", clean_correct=True, metric_mean=0.1, flip_rate=0.0)

    assert adapter.sample_extra_fields(row) == {}


# --- ClassificationAdapter.applicable_charts -------------------------------------


def test_applicable_charts_excludes_correlation_heatmap_when_unavailable() -> None:
    adapter = ClassificationAdapter(
        primary_metric=_PRIMARY_METRIC, fill_strategy_stability_available=False
    )

    assert adapter.applicable_charts() == ["vulnerability_histogram", "region_bar"]


def test_applicable_charts_includes_correlation_heatmap_when_available() -> None:
    adapter = ClassificationAdapter(
        primary_metric=_PRIMARY_METRIC, fill_strategy_stability_available=True
    )

    assert adapter.applicable_charts() == [
        "vulnerability_histogram",
        "region_bar",
        "fill_strategy_correlation_heatmap",
    ]


def test_classification_adapter_rejects_empty_primary_metric() -> None:
    with pytest.raises(ValueError, match="primary_metric"):
        ClassificationAdapter(primary_metric="")


# --- DetectionAdapter stub --------------------------------------------------------


def test_detection_adapter_instantiates_without_error() -> None:
    DetectionAdapter(primary_metric=_PRIMARY_METRIC)


def test_detection_adapter_rejects_empty_primary_metric() -> None:
    with pytest.raises(ValueError, match="primary_metric"):
        DetectionAdapter(primary_metric="")


def test_detection_adapter_raises_not_implemented_on_summarize_performance() -> None:
    adapter = DetectionAdapter(primary_metric=_PRIMARY_METRIC)
    with pytest.raises(NotImplementedError):
        adapter.summarize_performance([])


def test_detection_adapter_raises_not_implemented_on_sample_extra_fields() -> None:
    adapter = DetectionAdapter(primary_metric=_PRIMARY_METRIC)
    row = _sample_row("s0", clean_correct=True, metric_mean=0.1, flip_rate=0.0)
    with pytest.raises(NotImplementedError):
        adapter.sample_extra_fields(row)


def test_detection_adapter_raises_not_implemented_on_applicable_charts() -> None:
    adapter = DetectionAdapter(primary_metric=_PRIMARY_METRIC)
    with pytest.raises(NotImplementedError):
        adapter.applicable_charts()
