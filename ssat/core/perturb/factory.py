"""Factory for deterministic perturbation operator construction."""

from __future__ import annotations

from collections.abc import Sequence

from ssat.core.perturb.base import PerturbationOperator
from ssat.core.perturb.operators import (
    BlurOperator,
    ConstantFillOperator,
    GaussianNoiseOperator,
    MeanFillOperator,
    PatchShuffleOperator,
)

OperatorType = type[PerturbationOperator]

_DEFAULT_OPERATOR_TYPES: tuple[OperatorType, ...] = (
    ConstantFillOperator,
    MeanFillOperator,
    BlurOperator,
    GaussianNoiseOperator,
    PatchShuffleOperator,
)


class OperatorFactory:
    """Register operator classes and build fresh instances in stable order.

    Args:
        operator_types: Optional operator classes registered at construction.
    """

    def __init__(self, operator_types: Sequence[OperatorType] = ()) -> None:
        self._operator_types: list[OperatorType] = []
        for operator_type in operator_types:
            self.register(operator_type)

    def register(self, operator_type: OperatorType) -> None:
        """Append one concrete operator class to the factory.

        Args:
            operator_type: ``PerturbationOperator`` subclass to instantiate.

        Raises:
            TypeError: If ``operator_type`` does not implement the operator
                inheritance contract.
            ValueError: If the same class is already registered.
        """

        if not isinstance(operator_type, type) or not issubclass(
            operator_type, PerturbationOperator
        ):
            raise TypeError("operator_type must be a PerturbationOperator subclass")
        if operator_type in self._operator_types:
            raise ValueError(f"operator type already registered: {operator_type.__name__}")
        self._operator_types.append(operator_type)

    def build_operators(self) -> list[PerturbationOperator]:
        """Construct all registered operators in registration order.

        Returns:
            Fresh operator instances in deterministic dispatch order.
        """

        return [operator_type() for operator_type in self._operator_types]


def build_operators() -> list[PerturbationOperator]:
    """Build a fresh list containing every built-in operator.

    Returns:
        Stateless built-in operators in deterministic dispatch order.
    """

    return OperatorFactory(_DEFAULT_OPERATOR_TYPES).build_operators()
