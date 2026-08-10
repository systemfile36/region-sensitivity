"""L1 hand-calculated tests for the 1st-priority (error-flip) built-in metrics."""

from __future__ import annotations

import numpy as np
import pytest

from ssat.core.adapter.types import AdapterSpec
from ssat.metrics.builtin_metrics import (
    FlipCorrectToWrong,
    FlipWrongToCorrect,
    PredChanged,
    TopkExit,
)
from ssat.metrics.normalize import normalize_output


def _adapter_spec() -> AdapterSpec:
    return AdapterSpec(model_id="model", deterministic=True)


def _derived(values: list[float], *, gt_label: int):
    return normalize_output(
        np.array(values, dtype=np.float64), gt_label=gt_label, adapter_spec=_adapter_spec()
    )


# gt_label is fixed at 0; "correct" logits rank class 0 first, "wrong" ranks it second.
_CORRECT = [2.0, 0.0]
_WRONG = [0.0, 2.0]


@pytest.mark.parametrize(
    ("clean_logits", "perturbed_logits", "expected_flip_c2w", "expected_flip_w2c"),
    [
        (_CORRECT, _CORRECT, 0.0, 0.0),
        (_CORRECT, _WRONG, 1.0, 0.0),
        (_WRONG, _CORRECT, 0.0, 1.0),
        (_WRONG, _WRONG, 0.0, 0.0),
    ],
)
def test_flip_metrics_match_hand_calculated_degradation(
    clean_logits: list[float],
    perturbed_logits: list[float],
    expected_flip_c2w: float,
    expected_flip_w2c: float,
) -> None:
    clean = _derived(clean_logits, gt_label=0)
    perturbed = _derived(perturbed_logits, gt_label=0)
    clean_correct = clean_logits == _CORRECT
    perturbed_correct = perturbed_logits == _CORRECT

    c2w = FlipCorrectToWrong().compute(clean, perturbed)
    w2c = FlipWrongToCorrect().compute(clean, perturbed)

    assert c2w.value_clean == pytest.approx(float(clean_correct))
    assert c2w.value_perturbed == pytest.approx(float(perturbed_correct))
    assert c2w.degradation == pytest.approx(expected_flip_c2w)
    assert w2c.value_clean == pytest.approx(float(clean_correct))
    assert w2c.value_perturbed == pytest.approx(float(perturbed_correct))
    assert w2c.degradation == pytest.approx(expected_flip_w2c)


def test_flip_correct_to_wrong_is_fixed_false_when_clean_is_already_wrong() -> None:
    clean = _derived(_WRONG, gt_label=0)
    perturbed = _derived(_WRONG, gt_label=0)

    result = FlipCorrectToWrong().compute(clean, perturbed)

    assert result.degradation == pytest.approx(0.0)


def test_pred_changed_is_false_when_top1_is_unchanged() -> None:
    clean = _derived(_CORRECT, gt_label=0)
    perturbed = _derived(_CORRECT, gt_label=0)

    result = PredChanged().compute(clean, perturbed)

    assert result.value_clean == pytest.approx(0.0)
    assert result.value_perturbed == pytest.approx(0.0)
    assert result.degradation == pytest.approx(0.0)


def test_pred_changed_is_true_when_top1_changes() -> None:
    clean = _derived(_CORRECT, gt_label=0)
    perturbed = _derived(_WRONG, gt_label=0)

    result = PredChanged().compute(clean, perturbed)

    assert result.value_perturbed == pytest.approx(1.0)
    assert result.degradation == pytest.approx(1.0)


def _rank_logits(*, num_classes: int, gt_rank: int) -> list[float]:
    """Build logits so the gt class (index 0) lands at exactly ``gt_rank``."""

    values = [0.0] * num_classes
    for index in range(1, gt_rank):
        values[index] = 5.0
    for index in range(gt_rank, num_classes):
        values[index] = -5.0
    return values


def test_topk_exit_detects_exit_at_the_k_boundary() -> None:
    clean = _derived(_rank_logits(num_classes=6, gt_rank=5), gt_label=0)
    perturbed = _derived(_rank_logits(num_classes=6, gt_rank=6), gt_label=0)

    result = TopkExit(k=5).compute(clean, perturbed)

    assert clean.gt_rank == 5
    assert perturbed.gt_rank == 6
    assert result.value_clean == pytest.approx(1.0)  # rank 5 is still within top-5
    assert result.value_perturbed == pytest.approx(0.0)  # rank 6 has exited
    assert result.degradation == pytest.approx(1.0)


def test_topk_exit_does_not_flag_items_that_stay_within_top_k() -> None:
    clean = _derived(_rank_logits(num_classes=6, gt_rank=5), gt_label=0)
    perturbed = _derived(_rank_logits(num_classes=6, gt_rank=5), gt_label=0)

    result = TopkExit(k=5).compute(clean, perturbed)

    assert result.degradation == pytest.approx(0.0)


def test_topk_exit_narrows_k_to_the_available_class_count() -> None:
    # 3 classes with the default k=5: effective_k = min(5, 3) = 3, so rank 3
    # (the worst possible rank in a 3-class problem) still counts as "in top-k".
    clean = _derived(_rank_logits(num_classes=3, gt_rank=1), gt_label=0)
    perturbed = _derived(_rank_logits(num_classes=3, gt_rank=3), gt_label=0)

    result = TopkExit().compute(clean, perturbed)

    assert result.value_clean == pytest.approx(1.0)
    assert result.value_perturbed == pytest.approx(1.0)
    assert result.degradation == pytest.approx(0.0)


@pytest.mark.parametrize("invalid_k", [0, -1, True])
def test_topk_exit_rejects_a_non_positive_k(invalid_k: int) -> None:
    with pytest.raises(ValueError, match="positive int"):
        TopkExit(k=invalid_k)
