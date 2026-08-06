"""Private validation and compositing helpers for perturbation operators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real
from typing import Any

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ssat.core.perturb.base import PerturbationError


def composite(
    source: NDArray[np.uint8],
    candidate: NDArray[np.uint8],
    mask: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    """Copy candidate values into selected source pixels.

    Args:
        source: Original source array.
        candidate: Full-frame perturbation candidate.
        mask: Source-space pixels to replace.

    Returns:
        A new array containing the masked composite.
    """

    result = source.copy()
    np.copyto(result, candidate, where=mask[np.newaxis, :, :, np.newaxis])
    return result


def fill_value(value: Any, channels: int, op_name: str) -> NDArray[np.uint8]:
    """Normalize a scalar or per-channel fill value.

    Args:
        value: Scalar or sequence in the uint8 pixel range.
        channels: Number of source channels.
        op_name: Operation name included in validation errors.

    Returns:
        A uint8 vector whose length equals ``channels``.

    Raises:
        PerturbationError: If the value is invalid or misaligned.
    """

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(value)
        if len(values) != channels:
            raise PerturbationError(
                f"{op_name}.value must contain exactly {channels} channels"
            )
    else:
        values = (value,) * channels

    normalized: list[int] = []
    for item in values:
        if (
            isinstance(item, bool)
            or not isinstance(item, Real)
            or not isfinite(float(item))
            or not 0.0 <= float(item) <= 255.0
        ):
            raise PerturbationError(
                f"{op_name}.value must contain finite values within [0, 255]"
            )
        normalized.append(int(np.rint(float(item))))
    return np.asarray(normalized, dtype=np.uint8)


def validate_config_fill_value(value: Any, op_name: str) -> None:
    """Validate a config-time scalar or channel-list fill value.

    Args:
        value: User-supplied JSON fill value.
        op_name: Operation name included in validation errors.

    Raises:
        PerturbationError: If the value is empty, nonnumeric, or out of range.
    """

    values = value if isinstance(value, list) else [value]
    if not values:
        raise PerturbationError(f"{op_name}.value must not be empty")
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise PerturbationError(
                f"{op_name}.value must contain only numeric values"
            )
        if not isfinite(float(item)) or not 0.0 <= float(item) <= 255.0:
            raise PerturbationError(
                f"{op_name}.value values must be finite and within [0, 255]"
            )


def positive_real(value: Any, field_name: str) -> float:
    """Validate a finite positive numeric parameter.

    Args:
        value: Candidate numeric parameter.
        field_name: Logical field included in an error.

    Returns:
        The validated value as a float.

    Raises:
        PerturbationError: If the value is not finite and positive.
    """

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise PerturbationError(f"{field_name} must be a finite positive number")
    return float(value)


def require_rng(rng: Generator | None, op_name: str) -> Generator:
    """Require an item-local generator for a stochastic operation.

    Args:
        rng: Candidate NumPy generator.
        op_name: Operation name included in an error.

    Returns:
        The validated generator.

    Raises:
        PerturbationError: If no NumPy generator was supplied.
    """

    if rng is None or not isinstance(rng, Generator):
        raise PerturbationError(f"{op_name} requires a numpy Generator")
    return rng


def require_keys(
    params: Mapping[str, Any],
    expected: set[str],
    op_name: str,
) -> None:
    """Require an exact operation parameter key set.

    Args:
        params: Candidate parameter mapping.
        expected: Exact accepted field names.
        op_name: Operation name included in an error.

    Raises:
        PerturbationError: If the mapping contains missing or extra keys.
    """

    if set(params) != expected:
        fields = ", ".join(sorted(expected))
        raise PerturbationError(
            f"{op_name} params must contain exactly: {fields}"
        )
