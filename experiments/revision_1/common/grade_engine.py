"""Vectorized re-implementation of ``ssat.analysis.reliability`` grading.

``grade_anchors`` reproduces the v1.0.0 flags and grade from a per-anchor
feature table (built by ``a1_threshold/build_features.py``) while exposing the
thresholds that ``compute_reliability`` hardcodes or leaves implicit: the z
threshold, a sign deadband, the multi-strategy count, the CI level, the
control-std ddof, and a control-std floor. With ``GradeParams()`` it must
match the stored ``reliability.parquet`` exactly (A1 parity gate).

Flags use int8 codes (``TRUE``/``FALSE``/``UNAVAILABLE``); grades use the
ordinal index of ``agreement.GRADE_ORDER``.

Feature table columns:

- key: ``sample_id``, ``region_key``, ``invert_mask``, ``metric_name``
- ``op__<op>``: per-operator anchor value (NaN when the op is missing)
- ``cond<i>__excess``, ``cond<i>__std0``, ``cond<i>__n``, ``cond<i>__available``:
  per-condition control comparison (``available`` is ``control_available``)
- ``ci<level>__low``, ``ci<level>__high``: region CI broadcast to the anchor
  (NaN when the region has no interval)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from experiments.revision_1.common.agreement import GRADE_ORDER

TRUE = np.int8(1)
FALSE = np.int8(0)
UNAVAILABLE = np.int8(-1)
FLAG_LABELS = {1: "true", 0: "false", -1: "unavailable"}
GRADE_CODES = {grade: np.int8(index) for index, grade in enumerate(GRADE_ORDER)}

CiLevel = Literal["90", "95", "99", "999"]
CI_PERCENTILES: dict[str, tuple[float, float]] = {
    "90": (5.0, 95.0),
    "95": (2.5, 97.5),
    "99": (0.5, 99.5),
    "999": (0.05, 99.95),
}
FLAG_COLUMNS = ("sign_consistent", "exceeds_control", "multi_strategy", "ci_excludes_zero")


@dataclass(frozen=True)
class GradeParams:
    """Grading thresholds; the defaults reproduce ``ssat`` v1.0.0.

    Attributes:
        z_threshold: ``exceeds_control`` is TRUE when the max condition z exceeds it.
        sign_alpha: Sign deadband as a multiple of ``RunScales.sign_scale``.
        min_multi_strategy: Minimum count of the dominant operator sign.
        ci_level: Bootstrap CI level key of ``CI_PERCENTILES``.
        ddof: Delta degrees of freedom applied to the stored (ddof 0) control std.
        std_floor_beta: Control-std floor as a multiple of ``RunScales.std_scale``.
        zero_abstain: Treat zero signs as abstentions (plan section 3.7).
    """

    z_threshold: float = 2.0
    sign_alpha: float = 0.0
    min_multi_strategy: int = 2
    ci_level: CiLevel = "95"
    ddof: int = 0
    std_floor_beta: float = 0.0
    zero_abstain: bool = False

    def as_dict(self) -> dict[str, object]:
        """Return the parameters as a plain dict."""

        return asdict(self)


@dataclass(frozen=True)
class RunScales:
    """Per-run, per-metric scales that turn relative thresholds into absolute ones.

    Attributes:
        sign_scale: ``S_run``, the median ``|v|`` over target (anchor, op) values.
        std_scale: ``B_run``, the median ddof-0 control std over evaluable conditions.
    """

    sign_scale: float
    std_scale: float


def op_columns(features: pd.DataFrame) -> list[str]:
    """Return the ``op__<op>`` columns in table order."""

    return [column for column in features.columns if column.startswith("op__")]


def condition_indices(features: pd.DataFrame) -> list[int]:
    """Return the condition indices ``i`` of the ``cond<i>__*`` columns."""

    return sorted(
        {int(column[4:].split("__", 1)[0]) for column in features.columns if column.startswith("cond")}
    )


def compute_scales(features: pd.DataFrame) -> RunScales:
    """Compute ``RunScales`` from a feature table (independent of any sweep setting)."""

    values = features[op_columns(features)].to_numpy(dtype=float)
    magnitudes = np.abs(values[~np.isnan(values)])
    stds = []
    for index in condition_indices(features):
        usable = features[f"cond{index}__available"].to_numpy(bool) & (
            features[f"cond{index}__n"].to_numpy() >= 2
        )
        stds.append(features[f"cond{index}__std0"].to_numpy(dtype=float)[usable])
    std_values = np.concatenate(stds) if stds else np.array([])
    return RunScales(
        sign_scale=float(np.median(magnitudes)) if magnitudes.size else float("nan"),
        std_scale=float(np.median(std_values)) if std_values.size else float("nan"),
    )


def max_z(features: pd.DataFrame, params: GradeParams, scales: RunScales) -> np.ndarray:
    """Return each anchor's max evaluable condition z (NaN when none is evaluable)."""

    indices = condition_indices(features)
    if not indices:
        return np.full(len(features), np.nan)
    floor = params.std_floor_beta * scales.std_scale if params.std_floor_beta > 0 else 0.0
    columns = []
    for index in indices:
        excess = features[f"cond{index}__excess"].to_numpy(dtype=float)
        std0 = features[f"cond{index}__std0"].to_numpy(dtype=float)
        n = features[f"cond{index}__n"].to_numpy(dtype=float)
        available = features[f"cond{index}__available"].to_numpy(bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            std = std0 if params.ddof == 0 else std0 * np.sqrt(n / (n - params.ddof))
            if floor > 0:
                std = np.maximum(std, floor)
            valid = available & (n >= 2) & (std != 0) & np.isfinite(std) & np.isfinite(excess)
            columns.append(np.where(valid, excess / std, np.nan))
    stacked = np.column_stack(columns)
    best = np.where(np.isnan(stacked), -np.inf, stacked).max(axis=1)
    return np.where(np.isnan(stacked).all(axis=1), np.nan, best)


def exceeds_flag(z: np.ndarray, threshold: float) -> np.ndarray:
    """Return ``exceeds_control`` codes for max-z values."""

    return np.where(np.isnan(z), UNAVAILABLE, np.where(z > threshold, TRUE, FALSE)).astype(np.int8)


def sign_flags(
    features: pd.DataFrame, params: GradeParams, scales: RunScales
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sign_consistent, multi_strategy)`` codes."""

    values = features[op_columns(features)].to_numpy(dtype=float)
    present = ~np.isnan(values)
    epsilon = params.sign_alpha * scales.sign_scale if params.sign_alpha > 0 else 0.0
    positive = (present & (values > epsilon)).sum(axis=1)
    negative = (present & (values < -epsilon)).sum(axis=1)
    n_present = present.sum(axis=1)
    zero = n_present - positive - negative
    missing = n_present == 0

    if params.zero_abstain:
        nonzero = positive + negative
        consistent = np.where(
            missing | (nonzero == 0),
            UNAVAILABLE,
            np.where((positive == nonzero) | (negative == nonzero), TRUE, FALSE),
        )
        dominant = np.maximum(positive, negative)
    else:
        dominant = np.maximum(np.maximum(positive, negative), zero)
        consistent = np.where(missing, UNAVAILABLE, np.where(dominant == n_present, TRUE, FALSE))
    multi = np.where(missing, UNAVAILABLE, np.where(dominant >= params.min_multi_strategy, TRUE, FALSE))
    return consistent.astype(np.int8), multi.astype(np.int8)


def ci_flag(features: pd.DataFrame, level: str) -> np.ndarray:
    """Return ``ci_excludes_zero`` codes for one CI level."""

    low = features[f"ci{level}__low"].to_numpy(dtype=float)
    high = features[f"ci{level}__high"].to_numpy(dtype=float)
    return np.where(
        np.isnan(low) | np.isnan(high), UNAVAILABLE, np.where((low > 0) | (high < 0), TRUE, FALSE)
    ).astype(np.int8)


def combine_grade(
    sign_consistent: np.ndarray,
    exceeds_control: np.ndarray,
    multi_strategy: np.ndarray,
    ci_excludes_zero: np.ndarray,
) -> np.ndarray:
    """Vectorized ``ssat.analysis.reliability._grade``; returns grade codes."""

    core = np.column_stack([exceeds_control, multi_strategy, ci_excludes_zero])
    sign_true = sign_consistent == TRUE
    high = sign_true & (core == TRUE).all(axis=1)
    moderate = sign_true & (core == TRUE).any(axis=1) & ~(core == FALSE).any(axis=1)
    grade = np.where(
        sign_consistent == FALSE,
        GRADE_CODES["unreliable"],
        np.where(high, GRADE_CODES["high"], np.where(moderate, GRADE_CODES["moderate"], GRADE_CODES["low"])),
    )
    return grade.astype(np.int8)


def grade_anchors(features: pd.DataFrame, params: GradeParams, *, scales: RunScales) -> pd.DataFrame:
    """Return per-anchor flag codes, grade code, and ``max_z`` under ``params``.

    Args:
        features: Feature table (module docstring).
        params: Thresholds to grade with.
        scales: The run's ``RunScales`` from ``compute_scales``.

    Returns:
        Frame aligned with ``features`` with columns ``sign_consistent``,
        ``exceeds_control``, ``multi_strategy``, ``ci_excludes_zero``
        (int8 flag codes), ``grade`` (int8 index into ``GRADE_ORDER``), and
        ``max_z``.
    """

    z = max_z(features, params, scales)
    exceeds = exceeds_flag(z, params.z_threshold)
    consistent, multi = sign_flags(features, params, scales)
    ci = ci_flag(features, params.ci_level)
    return pd.DataFrame(
        {
            "sign_consistent": consistent,
            "exceeds_control": exceeds,
            "multi_strategy": multi,
            "ci_excludes_zero": ci,
            "grade": combine_grade(consistent, exceeds, multi, ci),
            "max_z": z,
        },
        index=features.index,
    )


def flag_labels(codes: np.ndarray | pd.Series) -> np.ndarray:
    """Map flag codes to ``FlagValue`` strings."""

    return np.array([FLAG_LABELS[int(code)] for code in np.asarray(codes)], dtype=object)


def grade_labels(codes: np.ndarray | pd.Series) -> np.ndarray:
    """Map grade codes to ``ReliabilityGrade`` strings."""

    return np.asarray(GRADE_ORDER, dtype=object)[np.asarray(codes, dtype=int)]


def grade_codes(labels: Sequence[str] | pd.Series) -> np.ndarray:
    """Map ``ReliabilityGrade`` strings to grade codes."""

    return np.array([GRADE_CODES[label] for label in labels], dtype=np.int8)


def flag_codes(labels: Sequence[str] | pd.Series) -> np.ndarray:
    """Map ``FlagValue`` strings to flag codes."""

    lookup = {label: np.int8(code) for code, label in FLAG_LABELS.items()}
    return np.array([lookup[label] for label in labels], dtype=np.int8)
