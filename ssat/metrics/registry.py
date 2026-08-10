"""N2 MetricRegistry: Metric protocol, registration, and item-level computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ssat.core.adapter.types import AdapterSpec
from ssat.metrics.dump_reader import JoinedFrame
from ssat.metrics.errors import MetricsRegistryError
from ssat.metrics.normalize import NormalizedOutput, normalize_output
from ssat.metrics.types import ExclusionReason, ItemMetrics


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Carry one metric's raw and sign-normalized item-level values.

    Attributes:
        value_clean: Raw clean-side metric value.
        value_perturbed: Raw perturbed-side metric value.
        degradation: Sign-normalized value; positive always means worse
            performance (design METRIC_ENGINE_DESIGN_v1.md §N2).
    """

    value_clean: float
    value_perturbed: float
    degradation: float


@runtime_checkable
class Metric(Protocol):
    """Compute one registered metric's item-level clean/perturbed comparison.

    Concrete metrics own their full :class:`MetricResult` triple —
    :class:`MetricRegistry` performs no generic sign math (design §N2
    "부호 정규화"; IMPLE_PLAN_METRIC_DESIGN_v1.md §5 단계 3 확정 사항). This
    lets binary "event" metrics (e.g. flip_correct_to_wrong) encode
    degradation directly as a 0/1 occurrence, which a generic
    clean-minus-perturbed formula cannot guarantee.

    Attributes:
        name: Unique registered metric name.
        requires: Names of NormalizedOutput derived fields this metric reads.
        higher_is_better: Direction metadata for documentation and
            validation; does not drive any registry computation.
        kind: Whether this metric reports a binary event or a continuous
            change.
    """

    name: str
    requires: tuple[str, ...]
    higher_is_better: bool
    kind: Literal["binary", "continuous"]

    def available_when(self, adapter_spec: AdapterSpec) -> bool:
        """Return whether this metric applies to an entire run."""

    def compute(
        self, clean: NormalizedOutput, perturbed: NormalizedOutput
    ) -> MetricResult:
        """Compute this metric's clean/perturbed comparison for one item."""


class MetricRegistry:
    """Instance-local name registry that computes ItemMetrics over a JoinedFrame."""

    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}

    @property
    def names(self) -> tuple[str, ...]:
        """Return every registered metric name."""

        return tuple(self._metrics)

    def register(self, metric: Metric) -> None:
        """Register one metric under its name.

        Raises:
            TypeError: If ``metric`` does not implement the Metric protocol.
            MetricsRegistryError: If the metric name is empty or already
                registered.
        """

        if not isinstance(metric, Metric):
            raise TypeError("metric must implement the Metric protocol")
        if not metric.name:
            raise MetricsRegistryError("metric name must not be empty")
        if metric.name in self._metrics:
            raise MetricsRegistryError(f"metric already registered: {metric.name}")
        self._metrics[metric.name] = metric

    def get(self, name: str) -> Metric:
        """Return one registered metric by name.

        Raises:
            MetricsRegistryError: If no metric is registered under ``name``.
        """

        try:
            return self._metrics[name]
        except KeyError:
            raise MetricsRegistryError(f"metric not registered: {name}") from None

    def compute_item_metrics(
        self, joined: JoinedFrame, *, adapter_spec: AdapterSpec
    ) -> list[ItemMetrics]:
        """Compute every registered, run-available metric for every item.

        Metrics whose ``available_when(adapter_spec)`` is False are skipped
        entirely for this run — no rows are emitted for that metric_name,
        since this is a run-level omission rather than a per-item
        :class:`ExclusionReason`. Items whose perturbed side is unavailable
        (``logits_perturbed`` is ``None``) still produce one unavailable
        row per applicable metric, without calling that metric's
        ``compute``.
        """

        applicable = [
            metric for metric in self._metrics.values() if metric.available_when(adapter_spec)
        ]
        rows: list[ItemMetrics] = []
        for row in joined.itertuples(index=False):
            clean_derived = normalize_output(
                row.logits_clean, gt_label=int(row.gt_label), adapter_spec=adapter_spec
            )
            clean_correct = clean_derived.gt_rank == 1
            perturbed_derived = (
                normalize_output(
                    row.logits_perturbed,
                    gt_label=int(row.gt_label),
                    adapter_spec=adapter_spec,
                )
                if row.logits_perturbed is not None
                else None
            )
            for metric in applicable:
                if perturbed_derived is None:
                    rows.append(
                        ItemMetrics(
                            item_id=row.item_id,
                            sample_id=row.sample_id,
                            metric_name=metric.name,
                            clean_correct=clean_correct,
                            value_clean=None,
                            value_perturbed=None,
                            degradation=None,
                            available=False,
                            excluded_reason=ExclusionReason.PERTURBED_STATUS_NOT_OK,
                        )
                    )
                    continue
                result = metric.compute(clean_derived, perturbed_derived)
                rows.append(
                    ItemMetrics(
                        item_id=row.item_id,
                        sample_id=row.sample_id,
                        metric_name=metric.name,
                        clean_correct=clean_correct,
                        value_clean=result.value_clean,
                        value_perturbed=result.value_perturbed,
                        degradation=result.degradation,
                        available=True,
                        excluded_reason=None,
                    )
                )
        return rows
