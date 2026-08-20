"""N1 OutputNormalizer: derive probability/logit/rank/margin values.

``normalize_output`` applies identically to clean and perturbed output
vectors so that later metric code never touches raw logits directly,
avoiding duplicated derivations and inconsistencies across metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ssat.core.adapter.types import AdapterSpec


@dataclass(frozen=True, slots=True)
class NormalizedOutput:
    """Carry every value derived from one model output vector for one item.

    Attributes:
        prob: Class probabilities — softmax of ``values`` when the adapter
            reports logits, or ``values`` unchanged when it already reports
            probabilities.
        logit: Original logit vector, or ``None`` when the adapter reports
            probability outputs — logit reconstruction from probabilities is
            unreliable, so logit-derived fields are withheld rather than
            approximated.
        top1_index: Index of the highest-probability class.
        top1_prob: Probability of the ``top1_index`` class.
        gt_prob: Probability assigned to the ground-truth class.
        gt_logit: Logit assigned to the ground-truth class, or ``None``
            under the same condition as ``logit``.
        gt_rank: 1-indexed rank of the ground-truth class by probability
            (rank 1 is the most likely class); ties resolve to the best
            (lowest) rank.
        margin: ``gt_logit`` minus the highest logit among the other
            classes, or ``None`` under the same condition as ``logit``.
            Positive means the ground-truth class ranks first.
        entropy: Shannon entropy, in nats, of the probability distribution.
    """

    prob: NDArray[np.floating]
    logit: NDArray[np.floating] | None
    top1_index: int
    top1_prob: float
    gt_prob: float
    gt_logit: float | None
    gt_rank: int
    margin: float | None
    entropy: float


def normalize_output(
    values: NDArray[np.floating],
    *,
    gt_label: int,
    adapter_spec: AdapterSpec,
) -> NormalizedOutput:
    """Derive every N1 value from one clean or perturbed output vector.

    When ``adapter_spec.output_kind`` is ``"logits"``, ``prob`` is the
    softmax of ``values`` and ``logit``/``gt_logit``/``margin`` come from
    ``values`` directly. Otherwise (unreachable with the v1 core, which
    fixes ``AdapterSpec.output_kind`` to ``Literal["logits"]``; guarded
    here defensively in case a future core adds ``"probs"``), ``values`` is
    already a probability distribution: ``prob`` passes it through unchanged, while
    ``logit``/``gt_logit``/``margin`` are withheld as ``None`` instead of
    being reconstructed by an unstable log transform.

    Args:
        values: One-dimensional logit or probability vector ordered by
            model class, with at least two classes.
        gt_label: Ground-truth class index into ``values``.
        adapter_spec: Only ``adapter_spec.output_kind`` is consulted.

    Raises:
        TypeError: If ``values`` is not a one-dimensional floating ndarray,
            or ``gt_label`` is not an int.
        ValueError: If ``values`` has fewer than two classes, or
            ``gt_label`` does not index into ``values``.
    """

    if not isinstance(values, np.ndarray):
        raise TypeError("values must be a numpy ndarray")
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("values must use a floating dtype")
    if values.size < 2:
        raise ValueError("values must contain at least two classes")
    if isinstance(gt_label, bool) or not isinstance(gt_label, int):
        raise TypeError("gt_label must be an int")
    if not 0 <= gt_label < values.size:
        raise ValueError("gt_label must index into values")

    logit_available = adapter_spec.output_kind == "logits"
    prob = _softmax(values) if logit_available else values
    top1_index = int(np.argmax(prob))
    top1_prob = float(prob[top1_index])
    gt_prob = float(prob[gt_label])
    gt_rank = int(np.sum(prob > gt_prob)) + 1
    entropy = _entropy(prob)

    logit = values.copy() if logit_available else None
    gt_logit = float(values[gt_label]) if logit_available else None
    margin = _margin(values, gt_label) if logit_available else None

    return NormalizedOutput(
        prob=prob,
        logit=logit,
        top1_index=top1_index,
        top1_prob=top1_prob,
        gt_prob=gt_prob,
        gt_logit=gt_logit,
        gt_rank=gt_rank,
        margin=margin,
        entropy=entropy,
    )


def _softmax(values: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute a numerically stable softmax via the log-sum-exp trick."""

    shifted = values - np.max(values)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted)


def _entropy(prob: NDArray[np.floating]) -> float:
    """Compute Shannon entropy in nats, treating 0·log(0) as 0."""

    log_prob = np.log(prob, where=prob > 0, out=np.zeros_like(prob))
    return float(-np.sum(prob * log_prob))


def _margin(values: NDArray[np.floating], gt_label: int) -> float:
    """Compute gt_logit minus the highest logit among the other classes."""

    others = np.delete(values, gt_label)
    return float(values[gt_label] - np.max(others))
