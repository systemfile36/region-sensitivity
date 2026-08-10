"""L1 hand-calculated tests for the 2nd-priority (continuous change) built-in metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ssat.core.adapter.types import AdapterSpec
from ssat.metrics.builtin_metrics import (
    DEFAULT_TOPK,
    FlipCorrectToWrong,
    FlipWrongToCorrect,
    GtLogitDrop,
    GtProbDrop,
    GtRankWorsening,
    LOSS_EPSILON,
    LossIncrease,
    MarginDrop,
    PredChanged,
    TopkExit,
    default_metric_registry,
)
from ssat.metrics.builtin_metrics.continuous import _cross_entropy
from ssat.metrics.normalize import normalize_output


def _adapter_spec(*, output_kind: str = "logits") -> AdapterSpec:
    return AdapterSpec(model_id="model", deterministic=True, output_kind=output_kind)


def _derived(values: list[float], *, gt_label: int, output_kind: str = "logits"):
    return normalize_output(
        np.array(values, dtype=np.float64),
        gt_label=gt_label,
        adapter_spec=_adapter_spec(output_kind=output_kind),
    )


# gt_label is fixed at 0. clean=[ln3, 0] gives softmax [0.75, 0.25]; perturbed=[0, ln3]
# gives softmax [0.25, 0.75] — hand-calculable via the standard 2-class softmax formula,
# independent of the implementation under test.
_LN3 = math.log(3.0)
_CLEAN = [_LN3, 0.0]
_PERTURBED = [0.0, _LN3]


def test_gt_prob_drop_matches_hand_calculated_softmax() -> None:
    clean = _derived(_CLEAN, gt_label=0)
    perturbed = _derived(_PERTURBED, gt_label=0)

    result = GtProbDrop().compute(clean, perturbed)

    assert result.value_clean == pytest.approx(0.75)
    assert result.value_perturbed == pytest.approx(0.25)
    assert result.degradation == pytest.approx(0.5)


def test_gt_logit_drop_matches_hand_calculated_logits() -> None:
    clean = _derived(_CLEAN, gt_label=0)
    perturbed = _derived(_PERTURBED, gt_label=0)

    result = GtLogitDrop().compute(clean, perturbed)

    assert result.value_clean == pytest.approx(_LN3)
    assert result.value_perturbed == pytest.approx(0.0)
    assert result.degradation == pytest.approx(_LN3)


def test_margin_drop_matches_hand_calculated_margins() -> None:
    clean = _derived(_CLEAN, gt_label=0)
    perturbed = _derived(_PERTURBED, gt_label=0)

    result = MarginDrop().compute(clean, perturbed)

    assert result.value_clean == pytest.approx(_LN3)
    assert result.value_perturbed == pytest.approx(-_LN3)
    assert result.degradation == pytest.approx(2 * _LN3)


def test_loss_increase_matches_hand_calculated_cross_entropy() -> None:
    clean = _derived(_CLEAN, gt_label=0)
    perturbed = _derived(_PERTURBED, gt_label=0)

    result = LossIncrease().compute(clean, perturbed)

    assert result.value_clean == pytest.approx(-math.log(0.75))
    assert result.value_perturbed == pytest.approx(-math.log(0.25))
    assert result.degradation == pytest.approx(_LN3)


def test_loss_increase_epsilon_floors_zero_probability_to_a_finite_value() -> None:
    # gt_prob=0.0 cannot occur through normalize_output's stabilized softmax, but the
    # floor is exercised directly to confirm it prevents -log(0) == inf.
    assert math.isfinite(_cross_entropy(0.0))
    assert _cross_entropy(0.0) == pytest.approx(-math.log(LOSS_EPSILON))


def _rank_logits(*, num_classes: int, gt_rank: int) -> list[float]:
    """Build logits so the gt class (index 0) lands at exactly ``gt_rank``."""

    values = [0.0] * num_classes
    for index in range(1, gt_rank):
        values[index] = 5.0
    for index in range(gt_rank, num_classes):
        values[index] = -5.0
    return values


def test_gt_rank_worsening_reports_a_positive_amount_when_rank_worsens() -> None:
    clean = _derived(_rank_logits(num_classes=6, gt_rank=1), gt_label=0)
    perturbed = _derived(_rank_logits(num_classes=6, gt_rank=3), gt_label=0)

    result = GtRankWorsening().compute(clean, perturbed)

    assert result.value_clean == pytest.approx(1.0)
    assert result.value_perturbed == pytest.approx(3.0)
    assert result.degradation == pytest.approx(2.0)


def test_gt_rank_worsening_reports_a_negative_amount_when_rank_improves() -> None:
    clean = _derived(_rank_logits(num_classes=6, gt_rank=3), gt_label=0)
    perturbed = _derived(_rank_logits(num_classes=6, gt_rank=1), gt_label=0)

    result = GtRankWorsening().compute(clean, perturbed)

    assert result.degradation == pytest.approx(-2.0)


@pytest.mark.parametrize("output_kind", ["logits", "probs"])
def test_gt_prob_drop_loss_increase_and_rank_worsening_are_always_available(
    output_kind: str,
) -> None:
    spec = _adapter_spec(output_kind=output_kind)

    assert GtProbDrop().available_when(spec) is True
    assert LossIncrease().available_when(spec) is True
    assert GtRankWorsening().available_when(spec) is True


def test_gt_logit_drop_and_margin_drop_are_available_only_when_output_kind_is_logits() -> None:
    logits_spec = _adapter_spec(output_kind="logits")
    probs_spec = _adapter_spec(output_kind="probs")

    assert GtLogitDrop().available_when(logits_spec) is True
    assert MarginDrop().available_when(logits_spec) is True
    assert GtLogitDrop().available_when(probs_spec) is False
    assert MarginDrop().available_when(probs_spec) is False


def test_default_metric_registry_registers_every_built_in_metric() -> None:
    registry = default_metric_registry()

    assert set(registry.names) == {
        "flip_correct_to_wrong",
        "flip_wrong_to_correct",
        "pred_changed",
        "topk_exit",
        "gt_prob_drop",
        "gt_logit_drop",
        "margin_drop",
        "loss_increase",
        "gt_rank_worsening",
    }
    assert len(registry.names) == len(set(registry.names))


def test_default_metric_registry_reuses_the_1st_priority_metrics_unmodified() -> None:
    # Sanity check that this stage's registry wiring did not disturb the flip
    # metrics already covered by test_metrics_flip_metrics.py.
    registry = default_metric_registry()

    assert isinstance(registry.names, tuple)
    for cls in (FlipCorrectToWrong, FlipWrongToCorrect, PredChanged):
        assert cls().name in registry.names
    assert TopkExit(k=DEFAULT_TOPK).name in registry.names
