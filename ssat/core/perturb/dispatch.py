"""Ordered dispatch for registered perturbation operators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ssat.core.perturb.base import PerturbationError, PerturbationOperator
from ssat.core.types import PerturbationOp


def find_operator(
    operators: Sequence[PerturbationOperator],
    op: PerturbationOp,
) -> PerturbationOperator:
    """Return the first registered operator supporting an operation.

    Args:
        operators: Operators in explicit dispatch-priority order.
        op: Perturbation operation requested by a work item or config.

    Returns:
        The first supporting operator.

    Raises:
        PerturbationError: If support detection fails or no operator supports
            ``op``.
    """

    for operator in operators:
        try:
            supported = operator.supports(op)
        except PerturbationError:
            raise
        except Exception as error:
            raise PerturbationError(
                f"operator support check failed op={op.value} "
                f"operator={operator.__class__.__name__}"
            ) from error
        if not isinstance(supported, bool):
            raise PerturbationError(
                f"operator supports() must return bool: {operator.__class__.__name__}"
            )
        if supported:
            return operator
    raise PerturbationError(f"unsupported perturbation op={op.value}")


def dispatch_operator(
    operators: Sequence[PerturbationOperator],
    array: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    op: PerturbationOp,
    params: Mapping[str, Any],
    rng: Generator | None = None,
) -> NDArray[np.uint8]:
    """Execute the first registered operator supporting an operation.

    Args:
        operators: Operators in explicit dispatch-priority order.
        array: Validated source pixels.
        mask: Validated source-space selection mask.
        op: Perturbation operation requested by a work item.
        params: Fully resolved operation-specific parameters.
        rng: Optional item-local generator.

    Returns:
        The first supporting operator's output array.

    Raises:
        PerturbationError: If support detection or execution fails, or no
            registered operator supports ``op``.
    """

    operator = find_operator(operators, op)
    try:
        return operator.apply(array, mask, params, rng)
    except PerturbationError:
        raise
    except Exception as error:
        raise PerturbationError(
            f"operator execution failed op={op.value} "
            f"operator={operator.__class__.__name__}"
        ) from error
