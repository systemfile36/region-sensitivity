"""Tests for the metrics engine's N1 OutputNormalizer (normalize_output)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ssat.core.adapter.types import AdapterSpec
from ssat.metrics.normalize import normalize_output


def _adapter_spec(*, output_kind: str = "logits") -> AdapterSpec:
    return AdapterSpec(model_id="model", deterministic=True, output_kind=output_kind)


def test_normalize_output_matches_hand_calculated_derived_values() -> None:
    values = np.array([1.0, 2.0, 0.5, -1.0], dtype=np.float64)

    result = normalize_output(values, gt_label=1, adapter_spec=_adapter_spec())

    raw_exp = [math.exp(v) for v in values]
    total = sum(raw_exp)
    expected_prob = [v / total for v in raw_exp]
    expected_entropy = -sum(p * math.log(p) for p in expected_prob)

    assert result.prob == pytest.approx(expected_prob)
    assert result.top1_index == 1
    assert result.top1_prob == pytest.approx(expected_prob[1])
    assert result.gt_prob == pytest.approx(expected_prob[1])
    assert result.gt_rank == 1
    assert result.gt_logit == pytest.approx(2.0)
    assert result.margin == pytest.approx(2.0 - 1.0)  # gt_logit - max(other logits)
    assert result.entropy == pytest.approx(expected_entropy)
    assert result.logit == pytest.approx(values)


def test_gt_rank_is_one_indexed_and_resolves_ties_to_best_rank() -> None:
    values = np.array([0.0, 0.0], dtype=np.float64)

    result = normalize_output(values, gt_label=0, adapter_spec=_adapter_spec())

    assert result.gt_rank == 1
    assert result.prob == pytest.approx([0.5, 0.5])
    assert result.entropy == pytest.approx(math.log(2.0))


def test_margin_is_positive_when_ground_truth_is_top1() -> None:
    values = np.array([2.0, 0.0, 0.0], dtype=np.float64)

    result = normalize_output(values, gt_label=0, adapter_spec=_adapter_spec())

    assert result.margin == pytest.approx(2.0)
    assert result.gt_rank == 1


def test_margin_is_negative_when_ground_truth_is_not_top1() -> None:
    values = np.array([2.0, 0.0, 0.0], dtype=np.float64)

    result = normalize_output(values, gt_label=1, adapter_spec=_adapter_spec())

    assert result.margin == pytest.approx(-2.0)
    assert result.gt_rank == 2


def test_normalize_output_withholds_logit_derived_fields_when_output_kind_is_not_logits() -> None:
    """output_kind="probs" is unreachable with the v1 core (AdapterSpec.output_kind
    is fixed to Literal["logits"]); this mock AdapterSpec forces the branch to
    exercise the defensive code path (design §N1, plan §5 단계 2)."""

    values = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    mock_spec = _adapter_spec(output_kind="probs")

    result = normalize_output(values, gt_label=1, adapter_spec=mock_spec)

    assert result.logit is None
    assert result.gt_logit is None
    assert result.margin is None
    # prob-only derived values remain available and pass values through unchanged.
    assert result.prob == pytest.approx(values)
    assert result.top1_index == 1
    assert result.gt_prob == pytest.approx(0.5)
    assert result.gt_rank == 1
    assert result.entropy == pytest.approx(
        -sum(p * math.log(p) for p in values)
    )


def test_normalize_output_rejects_non_ndarray_values() -> None:
    with pytest.raises(TypeError, match="numpy ndarray"):
        normalize_output([1.0, 2.0], gt_label=0, adapter_spec=_adapter_spec())


def test_normalize_output_rejects_non_floating_dtype() -> None:
    values = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(TypeError, match="floating dtype"):
        normalize_output(values, gt_label=0, adapter_spec=_adapter_spec())


def test_normalize_output_rejects_non_one_dimensional_values() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="one-dimensional"):
        normalize_output(values, gt_label=0, adapter_spec=_adapter_spec())


def test_normalize_output_rejects_fewer_than_two_classes() -> None:
    values = np.array([1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="at least two classes"):
        normalize_output(values, gt_label=0, adapter_spec=_adapter_spec())


def test_normalize_output_rejects_out_of_range_gt_label() -> None:
    values = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match="index into values"):
        normalize_output(values, gt_label=2, adapter_spec=_adapter_spec())


def test_normalize_output_rejects_non_int_gt_label() -> None:
    values = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(TypeError, match="must be an int"):
        normalize_output(values, gt_label=True, adapter_spec=_adapter_spec())
