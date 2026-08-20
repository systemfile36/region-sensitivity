"""Secondary continuous-change (continuous) metrics: gt_prob_drop, gt_logit_drop, margin_drop, loss_increase, gt_rank_worsening."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ssat.core.adapter.types import AdapterSpec
from ssat.metrics.normalize import NormalizedOutput
from ssat.metrics.registry import MetricResult

LOSS_EPSILON = 1e-12


class GtProbDrop:
    """Measure the drop in ground-truth class probability after perturbation."""

    name = "gt_prob_drop"
    requires: tuple[str, ...] = ("gt_prob",)
    higher_is_better = True
    kind: Literal["continuous"] = "continuous"

    def available_when(self, adapter_spec: AdapterSpec) -> bool:
        """Always available: gt_prob never depends on output_kind."""

        return True

    def compute(self, clean: NormalizedOutput, perturbed: NormalizedOutput) -> MetricResult:
        return MetricResult(
            value_clean=clean.gt_prob,
            value_perturbed=perturbed.gt_prob,
            degradation=clean.gt_prob - perturbed.gt_prob,
        )


class GtLogitDrop:
    """Measure the drop in ground-truth class logit after perturbation."""

    name = "gt_logit_drop"
    requires: tuple[str, ...] = ("gt_logit",)
    higher_is_better = True
    kind: Literal["continuous"] = "continuous"

    def available_when(self, adapter_spec: AdapterSpec) -> bool:
        """Only available when output_kind is logits; gt_logit is None otherwise."""

        return adapter_spec.output_kind == "logits"

    def compute(self, clean: NormalizedOutput, perturbed: NormalizedOutput) -> MetricResult:
        # available_when gates this metric out whenever gt_logit would be None.
        clean_gt_logit = clean.gt_logit
        perturbed_gt_logit = perturbed.gt_logit
        assert clean_gt_logit is not None and perturbed_gt_logit is not None
        return MetricResult(
            value_clean=clean_gt_logit,
            value_perturbed=perturbed_gt_logit,
            degradation=clean_gt_logit - perturbed_gt_logit,
        )


class MarginDrop:
    """Measure the drop in classification margin after perturbation."""

    name = "margin_drop"
    requires: tuple[str, ...] = ("margin",)
    higher_is_better = True
    kind: Literal["continuous"] = "continuous"

    def available_when(self, adapter_spec: AdapterSpec) -> bool:
        """Only available when output_kind is logits; margin is None otherwise."""

        return adapter_spec.output_kind == "logits"

    def compute(self, clean: NormalizedOutput, perturbed: NormalizedOutput) -> MetricResult:
        clean_margin = clean.margin
        perturbed_margin = perturbed.margin
        assert clean_margin is not None and perturbed_margin is not None
        return MetricResult(
            value_clean=clean_margin,
            value_perturbed=perturbed_margin,
            degradation=clean_margin - perturbed_margin,
        )


class LossIncrease:
    """Measure the increase in cross-entropy loss after perturbation."""

    name = "loss_increase"
    requires: tuple[str, ...] = ("gt_prob",)
    higher_is_better = False
    kind: Literal["continuous"] = "continuous"

    def available_when(self, adapter_spec: AdapterSpec) -> bool:
        """Always available: gt_prob never depends on output_kind."""

        return True

    def compute(self, clean: NormalizedOutput, perturbed: NormalizedOutput) -> MetricResult:
        clean_loss = _cross_entropy(clean.gt_prob)
        perturbed_loss = _cross_entropy(perturbed.gt_prob)
        return MetricResult(
            value_clean=clean_loss,
            value_perturbed=perturbed_loss,
            degradation=perturbed_loss - clean_loss,
        )


class GtRankWorsening:
    """Measure the worsening in ground-truth rank after perturbation."""

    name = "gt_rank_worsening"
    requires: tuple[str, ...] = ("gt_rank",)
    higher_is_better = False
    kind: Literal["continuous"] = "continuous"

    def available_when(self, adapter_spec: AdapterSpec) -> bool:
        """Always available: gt_rank never depends on output_kind."""

        return True

    def compute(self, clean: NormalizedOutput, perturbed: NormalizedOutput) -> MetricResult:
        return MetricResult(
            value_clean=float(clean.gt_rank),
            value_perturbed=float(perturbed.gt_rank),
            degradation=float(perturbed.gt_rank - clean.gt_rank),
        )


def _cross_entropy(gt_prob: float) -> float:
    """Compute -log(gt_prob), floored away from 0 to keep the result finite.

    Log-sum-exp-stabilized softmax (normalize.py) makes gt_prob underflowing
    to exactly 0.0 an extreme, practically unreachable edge case; the floor
    exists so a future such case yields a large finite loss instead of ``inf``
    poisoning downstream aggregates.
    """

    return float(-np.log(max(gt_prob, LOSS_EPSILON)))
