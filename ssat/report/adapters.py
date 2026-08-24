"""TaskPresentationAdapter: translate task-specific concepts into ReportModel's shared vocabulary.

This is the extension point for supporting new tasks: adding a new task
(e.g. detection) means writing a new adapter here, never touching
``ReportModel`` or the rest of the report pipeline — the goal is that
adding detection support only requires a new detection adapter, with no
changes to the shared report stages.

By design, this module does not import ``ssat.metrics`` or
``ssat.analysis``. Instead of importing ``ssat.metrics.types.SampleMetrics``,
:class:`SampleMetricLike` below declares the same shape structurally
(PEP 544) — the real ``SampleMetrics`` dataclass satisfies it without either
module knowing about the other, the same duck-typing trade already made in
``report.types`` (see that module's docstring).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ssat.report.types import MetricCard, ReportJsonValue


class SampleMetricLike(Protocol):
    """Structural shape this module reads off one sample's aggregated metric row.

    Matches ``ssat.metrics.types.SampleMetrics`` field-for-field, filtered to
    a single ``metric_name`` (the run's primary metric) by the caller before
    it ever reaches here — this module never filters or joins by metric_name
    itself.

    Attributes:
        sample_id: Sample this row summarizes.
        clean_correct: Whether the clean prediction matched ground truth,
            ``None`` when unknown.
        metric_mean: Mean sign-normalized degradation across valid items
            (positive always means worse performance), ``None`` when no
            valid item exists.
        flip_rate: Fraction of valid items that flipped, ``None`` when the
            metric is continuous rather than binary.
    """

    sample_id: str
    clean_correct: bool | None
    metric_mean: float | None
    flip_rate: float | None


class TaskPresentationAdapter(Protocol):
    """Translate task-specific concepts into ReportModel's task-agnostic vocabulary.

    ``ReportModel``, the assembler, and the HTML renderer never branch on
    task kind — they only ever consume what one of these produces.
    """

    def summarize_performance(
        self, sample_metrics: Sequence[SampleMetricLike]
    ) -> list[MetricCard]:
        """Build this task's scorecard from one primary-metric-filtered sample_metrics slice."""

    def sample_extra_fields(
        self, sample_metrics_row: SampleMetricLike
    ) -> dict[str, ReportJsonValue]:
        """Return this task's extra fields for one sample's ``SampleCard.task_extra``."""

    def applicable_charts(self) -> list[str]:
        """List chart identifiers this task/run combination can render."""


class ClassificationAdapter:
    """TaskPresentationAdapter for classification runs — the only task rendered end-to-end in v1."""

    def __init__(
        self, *, primary_metric: str, fill_strategy_stability_available: bool = False
    ) -> None:
        """Configure this adapter with the run's primary metric and available analyses.

        Args:
            primary_metric: Registered metric name this adapter's scorecard
                is built from. Used only for card key/label text — the
                underlying values are already sign-normalized degradation,
                so no metric-specific math happens here.
            fill_strategy_stability_available: Whether AnalysisStore has
                fill-strategy stability rows for this run
                (``AvailableAnalyses.fill_strategy_stability``, passed as a
                plain bool rather than importing that type — see module
                docstring). Controls whether the correlation heatmap chart
                is offered.

        Raises:
            ValueError: If ``primary_metric`` is empty.
        """

        if not primary_metric:
            raise ValueError("primary_metric must not be empty")
        self._primary_metric = primary_metric
        self._fill_strategy_stability_available = fill_strategy_stability_available

    def summarize_performance(
        self, sample_metrics: Sequence[SampleMetricLike]
    ) -> list[MetricCard]:
        """Build ``[accuracy, mean_<primary_metric>, flip_rate]``.

        Each card is always present — an unavailable value is expressed as
        ``value=None`` plus an explanatory ``note``, never by omitting the
        card entirely, so a missing value is never silently hidden.
        """

        rows = list(sample_metrics)
        return [
            self._accuracy_card(rows),
            self._mean_degradation_card(rows),
            self._flip_rate_card(rows),
        ]

    def _accuracy_card(self, rows: Sequence[SampleMetricLike]) -> MetricCard:
        values = [row.clean_correct for row in rows if row.clean_correct is not None]
        if not values:
            return MetricCard(
                key="accuracy",
                label="Clean Accuracy",
                value=None,
                unit="%",
                higher_is_better=True,
                note="N/A: this run has no ground-truth labels.",
            )
        return MetricCard(
            key="accuracy",
            label="Clean Accuracy",
            value=sum(values) / len(values),
            unit="%",
            higher_is_better=True,
        )

    def _mean_degradation_card(self, rows: Sequence[SampleMetricLike]) -> MetricCard:
        key = f"mean_{self._primary_metric}"
        label = f"Mean {self._primary_metric} Degradation"
        values = [row.metric_mean for row in rows if row.metric_mean is not None]
        if not values:
            return MetricCard(
                key=key,
                label=label,
                value=None,
                unit="",
                higher_is_better=False,
                note="N/A: no valid items.",
            )
        # metric_mean is already sign-normalized (positive == worse), so a
        # higher mean degradation is always the undesired direction —
        # independent of the underlying metric's own higher_is_better.
        return MetricCard(
            key=key,
            label=label,
            value=sum(values) / len(values),
            unit="",
            higher_is_better=False,
        )

    def _flip_rate_card(self, rows: Sequence[SampleMetricLike]) -> MetricCard:
        values = [row.flip_rate for row in rows if row.flip_rate is not None]
        if not values:
            return MetricCard(
                key="flip_rate",
                label="Flip Rate",
                value=None,
                unit="%",
                higher_is_better=False,
                note=(
                    f"N/A: {self._primary_metric} is a continuous metric, so "
                    "flip has no meaning for it."
                ),
            )
        return MetricCard(
            key="flip_rate",
            label="Flip Rate",
            value=sum(values) / len(values),
            unit="%",
            higher_is_better=False,
        )

    def sample_extra_fields(
        self, sample_metrics_row: SampleMetricLike
    ) -> dict[str, ReportJsonValue]:
        """Return no extra fields — classification has none."""

        return {}

    def applicable_charts(self) -> list[str]:
        """List ``vulnerability_histogram``/``region_bar``, plus the correlation heatmap when available."""

        charts = ["vulnerability_histogram", "region_bar"]
        if self._fill_strategy_stability_available:
            charts.append("fill_strategy_correlation_heatmap")
        return charts


class DetectionAdapter:
    """Schema-only stub for a future detection TaskPresentationAdapter, out of scope for now.

    Instantiates without error so a future task_kind -> adapter selector can
    hold a reference to this class unconditionally; every method raises only
    when actually called, never at instantiation time.
    """

    def __init__(
        self, *, primary_metric: str, fill_strategy_stability_available: bool = False
    ) -> None:
        """Store the same configuration ``ClassificationAdapter`` accepts.

        Raises:
            ValueError: If ``primary_metric`` is empty.
        """

        if not primary_metric:
            raise ValueError("primary_metric must not be empty")
        self._primary_metric = primary_metric
        self._fill_strategy_stability_available = fill_strategy_stability_available

    def summarize_performance(
        self, sample_metrics: Sequence[SampleMetricLike]
    ) -> list[MetricCard]:
        raise NotImplementedError("Detection scorecards are out of scope for v1.")

    def sample_extra_fields(
        self, sample_metrics_row: SampleMetricLike
    ) -> dict[str, ReportJsonValue]:
        raise NotImplementedError("Detection sample fields are out of scope for v1.")

    def applicable_charts(self) -> list[str]:
        raise NotImplementedError("Detection chart selection is out of scope for v1.")
