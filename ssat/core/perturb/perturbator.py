"""Validated facade for registered source-space perturbation operators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ssat.core.perturb.base import PerturbationError, PerturbationOperator
from ssat.core.perturb.dispatch import dispatch_operator
from ssat.core.perturb.factory import build_operators
from ssat.core.types import PerturbationOp


class Perturbator:
    """Validate inputs and dispatch work to registered operators.

    Args:
        operators: Optional operators in explicit dispatch-priority order.

    Raises:
        TypeError: If an item does not implement ``PerturbationOperator``.
        ValueError: If the operator collection is empty.
    """

    def __init__(
        self,
        operators: Sequence[PerturbationOperator] | None = None,
    ) -> None:
        resolved = tuple(build_operators() if operators is None else operators)
        if not resolved:
            raise ValueError("operators must not be empty")
        if any(not isinstance(operator, PerturbationOperator) for operator in resolved):
            raise TypeError("operators must contain PerturbationOperator values")
        self._operators = resolved

    def apply(
        self,
        array: NDArray[np.uint8],
        mask: NDArray[np.bool_],
        op: PerturbationOp,
        params: Mapping[str, Any],
        rng: Generator | None = None,
    ) -> NDArray[np.uint8]:
        """Apply the first registered operator supporting an operation.

        Args:
            array: Original pixels in ``(T, H, W, C)`` uint8 layout.
            mask: ``(H, W)`` boolean selection in source pixel space.
            op: Supported perturbation operation.
            params: Fully resolved operation-specific parameters.
            rng: Item-local generator required by stochastic operations.

        Returns:
            A new uint8 array with only selected pixels replaced.

        Raises:
            PerturbationError: If an input, operation, operator result, or
                operation-specific parameter is invalid.
        """

        self._validate_inputs(array, mask, op, params)
        result = dispatch_operator(
            self._operators,
            array,
            mask,
            op,
            params,
            rng,
        )
        if (
            not isinstance(result, np.ndarray)
            or result.dtype != np.uint8
            or result.shape != array.shape
        ):
            raise PerturbationError("perturbation produced an invalid output array")
        return result

    @staticmethod
    def _validate_inputs(
        array: NDArray[np.uint8],
        mask: NDArray[np.bool_],
        op: PerturbationOp,
        params: Mapping[str, Any],
    ) -> None:
        """Validate inputs shared by every perturbation operator.

        Args:
            array: Candidate source pixel array.
            mask: Candidate source-space selection mask.
            op: Candidate operation enum.
            params: Candidate operation parameter mapping.

        Raises:
            PerturbationError: If an input violates the public contract.
        """

        if not isinstance(array, np.ndarray):
            raise PerturbationError("array must be a numpy ndarray")
        if array.dtype != np.uint8 or array.ndim != 4:
            raise PerturbationError("array must be (T, H, W, C) uint8")
        if any(dimension <= 0 for dimension in array.shape):
            raise PerturbationError("array dimensions must be positive")
        if not isinstance(mask, np.ndarray):
            raise PerturbationError("mask must be a numpy ndarray")
        if mask.dtype != np.bool_ or mask.shape != array.shape[1:3]:
            raise PerturbationError("mask must be (H, W) bool matching array")
        if not isinstance(op, PerturbationOp):
            raise PerturbationError("op must be a PerturbationOp")
        if not isinstance(params, Mapping):
            raise PerturbationError("params must be a mapping")
