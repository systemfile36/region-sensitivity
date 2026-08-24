"""Generic registration ledger shared by the region/perturb/plan factories."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, TypeVar

T = TypeVar("T")


class StrategyRegistry(Generic[T]):
    """Hold a stable-order, duplicate-free list of registered strategy classes.

    ``RegionMaskGeneratorFactory``, ``RegionFamilyExpanderFactory``, and
    ``OperatorFactory`` each wrap one of these internally to share the
    identical type-check/duplicate-check registration policy, while keeping
    their own public class name and ``build()``/``build_operators()`` shape.

    Args:
        base_type: Common base class every registered type must subclass.
        type_label: Parameter name used in the ``TypeError`` message.
        item_label: Noun used in the duplicate-registration ``ValueError``.
        strategy_types: Optional types registered immediately at construction.
    """

    def __init__(
        self,
        base_type: type,
        *,
        type_label: str,
        item_label: str,
        strategy_types: Sequence[type[T]] = (),
    ) -> None:
        self._base_type = base_type
        self._type_label = type_label
        self._item_label = item_label
        self._strategy_types: list[type[T]] = []
        for strategy_type in strategy_types:
            self.register(strategy_type)

    def register(self, strategy_type: type[T]) -> None:
        """Append one strategy class, enforcing the shared type/duplicate policy.

        Args:
            strategy_type: Subclass of ``base_type`` to register.

        Raises:
            TypeError: If ``strategy_type`` is not a subclass of ``base_type``.
            ValueError: If the same class is already registered.
        """

        if not isinstance(strategy_type, type) or not issubclass(
            strategy_type, self._base_type
        ):
            raise TypeError(
                f"{self._type_label} must be a {self._base_type.__name__} subclass"
            )
        if strategy_type in self._strategy_types:
            raise ValueError(
                f"{self._item_label} already registered: {strategy_type.__name__}"
            )
        self._strategy_types.append(strategy_type)

    @property
    def registered_types(self) -> tuple[type[T], ...]:
        """Return registered types in stable registration order."""

        return tuple(self._strategy_types)
