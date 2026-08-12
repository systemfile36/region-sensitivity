"""Abstract contracts and shared state for region mask generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ssat.core.region.types import RegionSpec


class RegionResolutionError(ValueError):
    """Indicate that a concrete region cannot be materialized safely."""


class ExplicitMaskCache:
    """Store decoded explicit masks in bounded least-recently-used order.

    Args:
        max_size: Positive maximum number of masks retained by one resolver.

    Raises:
        ValueError: If ``max_size`` is not a positive integer.
    """

    def __init__(self, max_size: int) -> None:
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("explicit_cache_size must be a positive integer")
        self._max_size = max_size
        self._entries: OrderedDict[str, NDArray[np.bool_]] = OrderedDict()

    def get(self, digest: str) -> NDArray[np.bool_] | None:
        """Return and refresh a cached mask without copying it.

        Args:
            digest: Verified SHA-256 cache key.

        Returns:
            The immutable cached mask, or ``None`` when absent.
        """

        mask = self._entries.get(digest)
        if mask is not None:
            self._entries.move_to_end(digest)
        return mask

    def put(self, digest: str, mask: NDArray[np.bool_]) -> None:
        """Store an immutable mask and evict the oldest entries.

        Args:
            digest: Verified SHA-256 cache key.
            mask: Read-only boolean mask to retain.
        """

        self._entries[digest] = mask
        self._entries.move_to_end(digest)
        while len(self._entries) > self._max_size:
            self._entries.popitem(last=False)


TargetMaskResolver = Callable[[int, int, RegionSpec], NDArray[np.bool_]]


@dataclass(frozen=True, slots=True)
class MaskResolutionContext:
    """Provide shared cache and recursive target resolution to generators.

    Attributes:
        explicit_cache: Bounded cache shared by explicit mask generators.
        resolve_target: Callback dispatching an embedded deterministic target.
    """

    explicit_cache: ExplicitMaskCache
    resolve_target: TargetMaskResolver


class RegionMaskGenerator(ABC):
    """Define support detection and mask generation for one region kind.

    Args:
        context: Shared services supplied by the owning resolver.
    """

    def __init__(self, context: MaskResolutionContext) -> None:
        self._context = context

    @abstractmethod
    def supports(self, spec: RegionSpec) -> bool:
        """Return whether this generator supports a concrete region recipe.

        Args:
            spec: Concrete region recipe produced during planning.

        Returns:
            ``True`` when this generator can materialize ``spec``.
        """

    @abstractmethod
    def get_mask(
        self,
        height: int,
        width: int,
        spec: RegionSpec,
        rng: Generator | None = None,
    ) -> NDArray[np.bool_]:
        """Materialize one source-space boolean mask.

        Args:
            height: Positive source image height.
            width: Positive source image width.
            spec: Concrete region recipe to materialize.
            rng: Item-local generator required by stochastic generators.

        Returns:
            A boolean mask in ``(H, W)`` layout, broadcast across every
            source frame by the perturb/runtime layers. Generators that
            need a different selection per frame (e.g. a future
            skeleton-tracking provider) may instead return ``(T, H, W)``;
            ``RegionResolver`` validates ``T`` against the source sample.

        Raises:
            RegionResolutionError: If the recipe or runtime input is invalid.
        """


def mean_frame_area(mask: NDArray[np.bool_]) -> float:
    """Measure a mask's nonzero pixel count under the shared time-axis rule.

    A ``(H, W)`` mask reports its exact pixel count. A ``(T, H, W)`` mask
    reports the mean nonzero count across frames, so area and ratio
    statistics stay comparable whether a region is broadcast across every
    frame or varies per frame.

    Args:
        mask: A ``(H, W)`` or ``(T, H, W)`` boolean mask.

    Returns:
        The (possibly fractional) mean per-frame nonzero pixel count.
    """

    if mask.ndim == 2:
        return float(np.count_nonzero(mask))
    return float(mask.reshape(mask.shape[0], -1).sum(axis=1).mean())
