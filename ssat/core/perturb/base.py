"""Abstract contract shared by source-space perturbation operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ssat.core.types import PerturbationOp


class PerturbationError(ValueError):
    """Indicate that a perturbation cannot be applied safely."""


class PerturbationOperator(ABC):
    """Define the dispatch and execution contract for one perturbation."""

    @abstractmethod
    def supports(self, op: PerturbationOp) -> bool:
        """Return whether this operator can execute an operation.

        Args:
            op: Perturbation operation requested by a work item.

        Returns:
            ``True`` when this operator supports the requested operation.
        """

    @abstractmethod
    def validate_config(self, params: Mapping[str, Any]) -> None:
        """Validate user parameters before runtime resolution.

        Args:
            params: User-supplied operation parameter mapping.

        Raises:
            PerturbationError: If the configuration is invalid.
        """

    def requires_dataset_stats(self) -> bool:
        """Return whether config resolution needs dataset channel statistics.

        Returns:
            ``False`` unless a concrete operator overrides the requirement.
        """

        return False

    def resolve_config_params(
        self,
        params: Mapping[str, Any],
        channel_mean: tuple[float, ...] | None = None,
    ) -> dict[str, Any]:
        """Resolve runtime parameters from validated user configuration.

        Args:
            params: Validated user parameter mapping.
            channel_mean: Optional dataset mean required by derived operators.

        Returns:
            Runtime parameter mapping consumed by :meth:`apply`.
        """

        return dict(params)

    @abstractmethod
    def apply(
        self,
        array: NDArray[np.uint8],
        mask: NDArray[np.bool_],
        params: Mapping[str, Any],
        rng: Generator | None = None,
    ) -> NDArray[np.uint8]:
        """Apply the operator inside a source-space mask.

        Args:
            array: Validated pixels in ``(T, H, W, C)`` uint8 layout.
            mask: Validated ``(H, W)`` boolean selection mask.
            params: Fully resolved operation-specific parameters.
            rng: Item-local generator, ignored by deterministic operators.

        Returns:
            A new uint8 array containing the masked perturbation.

        Raises:
            PerturbationError: If parameters or required runtime inputs are
                invalid.
        """
