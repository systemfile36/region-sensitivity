"""R5 TaskPresentationAdapter: translate task-specific concepts into ReportModel's shared vocabulary.

This is the extension point design §R5 designates as "the key extension
point": adding a new task (e.g. detection) means writing a new adapter here,
never touching ``ReportModel`` or R0-R4 (design §5 "검출 지원을 실제로 붙일
때, R5의 검출 어댑터만 추가하고 R0~R4는 손대지 않는 것이 목표다").

Per the dependency table (IMPLE_PLAN_REPORTING_v1.md §3.3 "report.adapters →
report.types"), this module does not import ``ssat.metrics`` or
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
    """Translate task-specific concepts into ReportModel's task-agnostic vocabulary (design §R5).

    ``ReportModel``, R0 (assembler), and R4 (HTMLRenderer) never branch on
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
        """List chart identifiers (design §R2) this task/run combination can render."""


class ClassificationAdapter:
    """TaskPresentationAdapter for classification runs — the only task rendered end-to-end in v1."""

    def __init__(
        self, *, primary_metric: str, fill_strategy_stability_available: bool = False
    ) -> None:
        """Configure this adapter with the run's primary metric and available analyses.

        Args:
            primary_metric: Registered metric name this adapter's scorecard
                is built from. Used only for card key/label text — the
                underlying values are already sign-normalized degradation
                (§1 격차#4), so no metric-specific math happens here.
            fill_strategy_stability_available: Whether AnalysisStore has
                fill-strategy stability rows for this run
                (``AvailableAnalyses.fill_strategy_stability``, passed as a
                plain bool rather than importing that type — see module
                docstring). Controls whether the correlation heatmap chart
                is offered (design §5 단계 1).

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
        """Build ``[accuracy, mean_<primary_metric>, flip_rate]`` (design §5 단계 1).

        Each card is always present — an unavailable value is expressed as
        ``value=None`` plus an explanatory ``note``, never by omitting the
        card (design §6.2 "결측이... 조용히 생략되지 않음").
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
                note="해당 없음: ground-truth 라벨이 없는 실행입니다.",
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
                note="해당 없음: 유효한 항목이 없습니다.",
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
                    f"해당 없음: {self._primary_metric}는 continuous 지표라 "
                    "flip 개념이 없습니다."
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
        """Return no extra fields — classification has none (design §R5, §3)."""

        return {}

    def applicable_charts(self) -> list[str]:
        """List ``vulnerability_histogram``/``region_bar``, plus the correlation heatmap when available."""

        charts = ["vulnerability_histogram", "region_bar"]
        if self._fill_strategy_stability_available:
            charts.append("fill_strategy_correlation_heatmap")
        return charts


class DetectionAdapter:
    """Schema-only stub for a future detection TaskPresentationAdapter (design §5, out of v1 scope).

    Instantiates without error so a future task_kind -> adapter selector can
    hold a reference to this class unconditionally; every method raises only
    when actually called (IMPLE_PLAN_REPORTING_v1.md §5 단계 1 성공 조건
    "인스턴스화 시점에는 실패하지 않고 호출 시점에만 NotImplementedError").
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
        raise NotImplementedError("Detection scorecards are out of scope for v1 (design §5).")

    def sample_extra_fields(
        self, sample_metrics_row: SampleMetricLike
    ) -> dict[str, ReportJsonValue]:
        raise NotImplementedError("Detection sample fields are out of scope for v1 (design §5).")

    def applicable_charts(self) -> list[str]:
        raise NotImplementedError("Detection chart selection is out of scope for v1 (design §5).")
