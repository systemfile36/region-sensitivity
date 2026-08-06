"""Tests for perturbation operator registration and ordered dispatch."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from numpy.random import Generator
from numpy.typing import NDArray

from ssat.core.perturb import (
    OperatorFactory,
    PerturbationError,
    PerturbationOperator,
    Perturbator,
    build_operators,
)
from ssat.core.perturb.operators import (
    BlurOperator,
    ConstantFillOperator,
    GaussianNoiseOperator,
    MeanFillOperator,
    PatchShuffleOperator,
)
from ssat.core.types import PerturbationOp


class CustomConstantOperator(PerturbationOperator):
    """Provide a simple custom constant operation for injection tests."""

    def supports(self, op: PerturbationOp) -> bool:
        """Support only constant fill.

        Args:
            op: Requested perturbation operation.

        Returns:
            ``True`` only for constant fill.
        """

        return op is PerturbationOp.CONSTANT_FILL

    def apply(
        self,
        array: NDArray[np.uint8],
        mask: NDArray[np.bool_],
        params: Mapping[str, Any],
        rng: Generator | None = None,
    ) -> NDArray[np.uint8]:
        """Return a custom fixed fill result.

        Args:
            array: Validated source array.
            mask: Validated selection mask.
            params: Ignored custom parameters.
            rng: Ignored item-local generator.

        Returns:
            A copied array containing value 99 inside the mask.
        """

        result = array.copy()
        result[:, mask, :] = 99
        return result


class RecordingOperator(PerturbationOperator):
    """Record support and apply calls for dispatch-order tests.

    Args:
        name: Identifier appended to the shared event list.
        supported: Whether ``supports`` should return true.
        events: Mutable event sink owned by the test.
        value: Fill value produced by ``apply``.
    """

    def __init__(
        self,
        name: str,
        supported: bool,
        events: list[str],
        value: int,
    ) -> None:
        self._name = name
        self._supported = supported
        self._events = events
        self._value = value

    def supports(self, op: PerturbationOp) -> bool:
        """Record and return the configured support decision.

        Args:
            op: Requested perturbation operation.

        Returns:
            The support decision configured by the test.
        """

        self._events.append(f"supports:{self._name}")
        return self._supported

    def apply(
        self,
        array: NDArray[np.uint8],
        mask: NDArray[np.bool_],
        params: Mapping[str, Any],
        rng: Generator | None = None,
    ) -> NDArray[np.uint8]:
        """Record execution and return the configured fill result.

        Args:
            array: Validated source array.
            mask: Validated selection mask.
            params: Ignored test parameters.
            rng: Ignored item-local generator.

        Returns:
            A copied array filled with the configured value inside the mask.
        """

        self._events.append(f"apply:{self._name}")
        result = array.copy()
        result[:, mask, :] = self._value
        return result


class InvalidOutputOperator(CustomConstantOperator):
    """Return an invalid dtype to exercise facade output validation."""

    def apply(
        self,
        array: NDArray[np.uint8],
        mask: NDArray[np.bool_],
        params: Mapping[str, Any],
        rng: Generator | None = None,
    ) -> NDArray[np.uint8]:
        """Return a float array that violates the operator contract.

        Args:
            array: Validated source array.
            mask: Validated selection mask.
            params: Ignored test parameters.
            rng: Ignored item-local generator.

        Returns:
            An intentionally invalid float array.
        """

        return array.astype(np.float32)  # type: ignore[return-value]


class RaisingSupportOperator(CustomConstantOperator):
    """Raise unexpectedly while checking operation support."""

    def supports(self, op: PerturbationOp) -> bool:
        """Raise a representative support-discovery failure.

        Args:
            op: Requested perturbation operation.

        Raises:
            RuntimeError: Always, to exercise the dispatch boundary.
        """

        raise RuntimeError("support failed")


class RaisingApplyOperator(CustomConstantOperator):
    """Raise unexpectedly while executing a supported operation."""

    def apply(
        self,
        array: NDArray[np.uint8],
        mask: NDArray[np.bool_],
        params: Mapping[str, Any],
        rng: Generator | None = None,
    ) -> NDArray[np.uint8]:
        """Raise a representative operator execution failure.

        Args:
            array: Validated source array.
            mask: Validated selection mask.
            params: Ignored test parameters.
            rng: Ignored item-local generator.

        Raises:
            RuntimeError: Always, to exercise the dispatch boundary.
        """

        raise RuntimeError("apply failed")


def _array() -> NDArray[np.uint8]:
    """Create a small source array for dispatch tests."""

    return np.zeros((1, 2, 2, 3), dtype=np.uint8)


def _mask() -> NDArray[np.bool_]:
    """Create a full source-space selection mask."""

    return np.ones((2, 2), dtype=np.bool_)


def test_operator_contract_is_abstract() -> None:
    """The common operator contract cannot be instantiated directly."""

    with pytest.raises(TypeError):
        PerturbationOperator()


@pytest.mark.parametrize(
    ("operator_type", "supported_op"),
    [
        (ConstantFillOperator, PerturbationOp.CONSTANT_FILL),
        (MeanFillOperator, PerturbationOp.MEAN_FILL),
        (BlurOperator, PerturbationOp.BLUR),
        (GaussianNoiseOperator, PerturbationOp.GAUSSIAN_NOISE),
        (PatchShuffleOperator, PerturbationOp.PATCH_SHUFFLE),
    ],
)
def test_builtin_operators_support_exactly_one_operation(
    operator_type: type[PerturbationOperator],
    supported_op: PerturbationOp,
) -> None:
    """Each built-in advertises only its own operation."""

    operator = operator_type()
    assert operator.supports(supported_op)
    assert all(
        not operator.supports(op)
        for op in PerturbationOp
        if op is not supported_op
    )


def test_default_factory_builds_fresh_operators_in_stable_order() -> None:
    """Default builds preserve registration order without sharing instances."""

    first = build_operators()
    second = build_operators()
    expected = [
        ConstantFillOperator,
        MeanFillOperator,
        BlurOperator,
        GaussianNoiseOperator,
        PatchShuffleOperator,
    ]

    assert [type(operator) for operator in first] == expected
    assert [type(operator) for operator in second] == expected
    assert all(left is not right for left, right in zip(first, second, strict=True))


def test_factory_validates_registration_and_builds_custom_operator() -> None:
    """Factories enforce class contracts and create registered custom types."""

    factory = OperatorFactory()
    factory.register(CustomConstantOperator)
    first = factory.build_operators()
    second = factory.build_operators()

    assert len(first) == 1
    assert isinstance(first[0], CustomConstantOperator)
    assert first[0] is not second[0]
    with pytest.raises(ValueError, match="already registered"):
        factory.register(CustomConstantOperator)
    with pytest.raises(TypeError, match="subclass"):
        factory.register(object)  # type: ignore[arg-type]


def test_custom_factory_output_can_be_injected_into_perturbator() -> None:
    """A custom factory can replace default dispatch without facade changes."""

    factory = OperatorFactory((CustomConstantOperator,))
    perturbator = Perturbator(factory.build_operators())

    result = perturbator.apply(
        _array(),
        _mask(),
        PerturbationOp.CONSTANT_FILL,
        {},
    )

    assert np.all(result == 99)


def test_dispatch_stops_after_first_supporting_operator() -> None:
    """Registration order defines priority and prevents later calls."""

    events: list[str] = []
    operators = (
        RecordingOperator("skip", False, events, 1),
        RecordingOperator("first", True, events, 2),
        RecordingOperator("later", True, events, 3),
    )

    result = Perturbator(operators).apply(
        _array(),
        _mask(),
        PerturbationOp.CONSTANT_FILL,
        {},
    )

    assert np.all(result == 2)
    assert events == ["supports:skip", "supports:first", "apply:first"]


def test_unsupported_operation_and_empty_operator_list_fail_clearly() -> None:
    """Missing dispatch coverage and unusable construction fail early."""

    unsupported = RecordingOperator("none", False, [], 0)
    with pytest.raises(PerturbationError, match="unsupported perturbation op=blur"):
        Perturbator((unsupported,)).apply(
            _array(),
            _mask(),
            PerturbationOp.BLUR,
            {"sigma": 1.0},
        )
    with pytest.raises(ValueError, match="must not be empty"):
        Perturbator(())
    with pytest.raises(TypeError, match="PerturbationOperator"):
        Perturbator((object(),))  # type: ignore[arg-type]


def test_invalid_operator_output_is_rejected_by_facade() -> None:
    """Perturbator validates custom results after successful dispatch."""

    with pytest.raises(PerturbationError, match="invalid output"):
        Perturbator((InvalidOutputOperator(),)).apply(
            _array(),
            _mask(),
            PerturbationOp.CONSTANT_FILL,
            {},
        )


@pytest.mark.parametrize(
    ("operator", "message"),
    [
        (RaisingSupportOperator(), "support check failed"),
        (RaisingApplyOperator(), "execution failed"),
    ],
)
def test_unexpected_operator_errors_preserve_their_causes(
    operator: PerturbationOperator,
    message: str,
) -> None:
    """Dispatch exposes one error type while retaining operator causes."""

    with pytest.raises(PerturbationError, match=message) as captured:
        Perturbator((operator,)).apply(
            _array(),
            _mask(),
            PerturbationOp.CONSTANT_FILL,
            {},
        )
    assert isinstance(captured.value.__cause__, RuntimeError)
