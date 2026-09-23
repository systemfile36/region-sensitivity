"""Agreement, rank-correlation, and discrimination statistics without scipy/sklearn.

Every revision-1 experiment compares two labelings of the same anchors
(e.g. reliability grades under two threshold settings) or two score vectors
(e.g. region profiles under two protocols). These helpers keep those
comparisons identical across experiments.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

GRADE_ORDER: tuple[str, ...] = ("unreliable", "low", "moderate", "high")
"""Reliability grades from worst to best; the position is the ordinal value."""

_GRADE_RANK = {grade: rank for rank, grade in enumerate(GRADE_ORDER)}


def grade_ordinal(grades: pd.Series) -> pd.Series:
    """Map grade strings onto ``GRADE_ORDER`` positions.

    Raises:
        ValueError: If any value is not one of ``GRADE_ORDER``.
    """

    ordinal = grades.map(_GRADE_RANK)
    if ordinal.isna().any():
        unknown = sorted(set(grades[ordinal.isna()].astype(str)))
        raise ValueError(f"unknown grade value(s): {unknown}")
    return ordinal.astype(np.int64)


def weighted_kappa(
    a: Sequence[int] | np.ndarray, b: Sequence[int] | np.ndarray, n_categories: int
) -> float | None:
    """Return Cohen's kappa with linear weights over ordinal categories ``0..n-1``.

    Returns:
        The weighted kappa, or ``None`` when chance agreement is already
        perfect (e.g. both raters use a single category), where kappa is
        undefined.
    """

    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("a and b must be non-empty and the same length")
    if n_categories < 2:
        raise ValueError("n_categories must be at least 2")
    observed = np.zeros((n_categories, n_categories), dtype=np.float64)
    np.add.at(observed, (a, b), 1.0)
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    index = np.arange(n_categories)
    weights = 1.0 - np.abs(index[:, None] - index[None, :]) / (n_categories - 1)
    p_observed = float((weights * observed).sum())
    p_expected = float((weights * expected).sum())
    if np.isclose(p_expected, 1.0):
        return None
    return (p_observed - p_expected) / (1.0 - p_expected)


def cohen_kappa(a: Sequence[object], b: Sequence[object]) -> float | None:
    """Return unweighted Cohen's kappa for two nominal labelings.

    Returns:
        Kappa, or ``None`` when chance agreement is perfect.
    """

    a = pd.Series(list(a), dtype=object)
    b = pd.Series(list(b), dtype=object)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("a and b must be non-empty and the same length")
    p_observed = float((a.values == b.values).mean())
    share_a = a.value_counts(normalize=True)
    share_b = b.value_counts(normalize=True)
    p_expected = float(share_a.mul(share_b, fill_value=0.0).sum())
    if np.isclose(p_expected, 1.0):
        return None
    return (p_observed - p_expected) / (1.0 - p_expected)


def grade_agreement(a: pd.Series, b: pd.Series) -> dict[str, float | int | None]:
    """Summarize how grades change from labeling ``a`` to labeling ``b``.

    Both series are aligned on their index (the anchor key); only anchors
    present in both are compared.

    Returns:
        ``n``, ``agreement`` (exact match share), ``kappa_linear``, and the
        shares ``unchanged``, ``delta_1`` (one ordinal step), ``delta_ge2``
        (two or more steps), ``up`` (``b`` better than ``a``), ``down``.
    """

    joined = pd.concat({"a": a, "b": b}, axis=1, join="inner")
    n = len(joined)
    if n == 0:
        raise ValueError("a and b share no index values")
    ordinal_a = grade_ordinal(joined["a"]).to_numpy()
    ordinal_b = grade_ordinal(joined["b"]).to_numpy()
    delta = ordinal_b - ordinal_a
    return {
        "n": n,
        "agreement": float((delta == 0).mean()),
        "kappa_linear": weighted_kappa(ordinal_a, ordinal_b, len(GRADE_ORDER)),
        "unchanged": float((delta == 0).mean()),
        "delta_1": float((np.abs(delta) == 1).mean()),
        "delta_ge2": float((np.abs(delta) >= 2).mean()),
        "up": float((delta > 0).mean()),
        "down": float((delta < 0).mean()),
    }


def transition_matrix(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Return grade transition counts, rows ``from`` (``a``) and columns ``to`` (``b``).

    Every grade in ``GRADE_ORDER`` appears on both axes, including empty ones.
    """

    joined = pd.concat({"a": a, "b": b}, axis=1, join="inner")
    grade_ordinal(joined["a"])
    grade_ordinal(joined["b"])
    counts = pd.crosstab(joined["a"], joined["b"])
    counts = counts.reindex(index=list(GRADE_ORDER), columns=list(GRADE_ORDER), fill_value=0)
    counts.index.name = "from_grade"
    counts.columns.name = "to_grade"
    return counts.astype(np.int64)


def rankdata(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return 1-based ranks with ties sharing their average rank."""

    return pd.Series(np.asarray(values, dtype=np.float64)).rank(method="average").to_numpy()


def spearman(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float | None:
    """Return Spearman's rho over pairs where both values are finite.

    Returns:
        Rho, or ``None`` with fewer than two pairs or a constant input.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 2:
        return None
    rank_x = rankdata(x[keep])
    rank_y = rankdata(y[keep])
    if np.ptp(rank_x) == 0 or np.ptp(rank_y) == 0:
        return None
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def auroc(scores: Sequence[float] | np.ndarray, labels: Sequence[bool] | np.ndarray) -> float | None:
    """Return the area under the ROC curve via the Mann-Whitney rank formula.

    Ties between a positive and a negative score count as one half.

    Returns:
        AUROC, or ``None`` when either class is empty.
    """

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same length")
    n_positive = int(labels.sum())
    n_negative = int(labels.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return None
    ranks = rankdata(scores)
    rank_sum = float(ranks[labels].sum())
    u_statistic = rank_sum - n_positive * (n_positive + 1) / 2.0
    return u_statistic / (n_positive * n_negative)
