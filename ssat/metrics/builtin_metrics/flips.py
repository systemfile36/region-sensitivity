"""Primary error-transition (binary) metrics: flip_correct_to_wrong, flip_wrong_to_correct, pred_changed, topk_exit."""

from __future__ import annotations

from typing import Literal

from ssat.core.adapter.types import AdapterSpec
from ssat.metrics.normalize import NormalizedOutput
from ssat.metrics.registry import MetricResult

DEFAULT_TOPK = 5


class FlipCorrectToWrong:
    """Detect items that were correct on clean but wrong after perturbation."""

    name = "flip_correct_to_wrong"
    requires: tuple[str, ...] = ("gt_rank",)
    higher_is_better = False
    kind: Literal["binary"] = "binary"

    def available_when(self, adapter_spec: AdapterSpec) -> bool:
        """Always available: gt_rank never depends on output_kind."""

        return True

    def compute(self, clean: NormalizedOutput, perturbed: NormalizedOutput) -> MetricResult:
        clean_correct = clean.gt_rank == 1
        perturbed_correct = perturbed.gt_rank == 1
        return MetricResult(
            value_clean=float(clean_correct),
            value_perturbed=float(perturbed_correct),
            degradation=float(clean_correct and not perturbed_correct),
        )


class FlipWrongToCorrect:
    """Detect items that were wrong on clean but correct after perturbation."""

    name = "flip_wrong_to_correct"
    requires: tuple[str, ...] = ("gt_rank",)
    higher_is_better = False
    kind: Literal["binary"] = "binary"

    def available_when(self, adapter_spec: AdapterSpec) -> bool:
        """Always available: gt_rank never depends on output_kind."""

        return True

    def compute(self, clean: NormalizedOutput, perturbed: NormalizedOutput) -> MetricResult:
        clean_correct = clean.gt_rank == 1
        perturbed_correct = perturbed.gt_rank == 1
        return MetricResult(
            value_clean=float(clean_correct),
            value_perturbed=float(perturbed_correct),
            degradation=float(not clean_correct and perturbed_correct),
        )


class PredChanged:
    """Detect a top-1 class change regardless of correctness."""

    name = "pred_changed"
    requires: tuple[str, ...] = ("top1_index",)
    higher_is_better = False
    kind: Literal["binary"] = "binary"

    def available_when(self, adapter_spec: AdapterSpec) -> bool:
        """Always available: top1_index never depends on output_kind."""

        return True

    def compute(self, clean: NormalizedOutput, perturbed: NormalizedOutput) -> MetricResult:
        changed = clean.top1_index != perturbed.top1_index
        return MetricResult(
            # No genuine single-sided reading exists for a purely comparative
            # event, so value_clean anchors at 0.
            value_clean=0.0,
            value_perturbed=float(changed),
            degradation=float(changed),
        )


class TopkExit:
    """Detect ground truth exiting the top-k ranking after perturbation.

    Args:
        k: Rank cutoff; automatically narrowed to the item's class count
            when smaller.
    """

    name = "topk_exit"
    requires: tuple[str, ...] = ("gt_rank",)
    higher_is_better = False
    kind: Literal["binary"] = "binary"

    def __init__(self, k: int = DEFAULT_TOPK) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive int")
        self.k = k

    def available_when(self, adapter_spec: AdapterSpec) -> bool:
        """Always available: gt_rank never depends on output_kind."""

        return True

    def compute(self, clean: NormalizedOutput, perturbed: NormalizedOutput) -> MetricResult:
        effective_k = min(self.k, clean.prob.size)
        in_topk_clean = clean.gt_rank <= effective_k
        in_topk_perturbed = perturbed.gt_rank <= effective_k
        return MetricResult(
            value_clean=float(in_topk_clean),
            value_perturbed=float(in_topk_perturbed),
            degradation=float(in_topk_clean and not in_topk_perturbed),
        )
