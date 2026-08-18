"""Name-based transform configuration and registry (MMCV/MMAction2-style).

Mirrors :mod:`ssat.core.source.provider` and :mod:`ssat.core.adapter.provider`:
each concrete transform type is registered by name in a
:class:`TransformRegistry`, and callers build an ordered :class:`Pipeline`
from a list of ``{"type": ..., **kwargs}`` step configurations instead of a
fixed if/elif dispatch table (contrast
:func:`ssat.core.adapter.preprocessing.parse_preprocessing_ops`). This module
is the generic registry engine only -- concrete built-in transforms live in
:mod:`ssat.core.adapter.transforms`, which this module does not import.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray


class TransformError(ValueError):
    """Indicate invalid transform registration, configuration, or execution."""


class BaseTransform(ABC):
    """Apply one deterministic pixel/mask operation inside a registry-built pipeline.

    Attributes:
        type_name: Registry key this transform is registered under by
            default.
        mask_supported: Whether ``apply_mask`` preserves the ``(H, W)`` /
            ``(T, H, W)`` mask contract 1:1. ``False`` for ops that break
            that geometry (e.g. ``TenCrop``'s batch expansion) -- the owning
            :class:`Pipeline` reports ``mask_supported=False`` as soon as any
            step sets this to ``False``.
    """

    type_name: ClassVar[str]
    mask_supported: ClassVar[bool] = True

    @abstractmethod
    def apply_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
        """Transform one ``(B, T, H, W, C)`` batch."""

    def apply_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Apply matching geometry to a ``(H, W)`` or ``(T, H, W)`` mask.

        The default is identity, which is correct for photometric-only ops.
        Geometry ops must override this; ops with ``mask_supported = False``
        never have this called -- :class:`Pipeline.apply_mask` raises
        instead of calling through to them.
        """

        return mask


class TransformRegistry:
    """Instance-local name registry with explicit transform-class registration."""

    def __init__(self) -> None:
        self._transforms: dict[str, type[BaseTransform]] = {}

    @property
    def names(self) -> tuple[str, ...]:
        """Return every registered transform type name."""

        return tuple(self._transforms)

    def register(self, transform_cls: type[BaseTransform], *, name: str | None = None) -> None:
        """Register one :class:`BaseTransform` subclass under ``name`` or its ``type_name``.

        Raises:
            TypeError: If ``transform_cls`` is not a ``BaseTransform`` subclass.
            TransformError: If the resolved name is empty or already registered.
        """

        if not (isinstance(transform_cls, type) and issubclass(transform_cls, BaseTransform)):
            raise TypeError("transform_cls must be a BaseTransform subclass")
        key = name if name is not None else getattr(transform_cls, "type_name", None)
        if not key:
            raise TransformError("transform name must not be empty")
        if key in self._transforms:
            raise TransformError(f"transform already registered: {key}")
        self._transforms[key] = transform_cls

    def register_module(
        self, name: str | None = None
    ) -> Callable[[type[BaseTransform]], type[BaseTransform]]:
        """Return a decorator that registers a class the way :meth:`register` does.

        Usage:
            TRANSFORMS = default_transform_registry()

            @TRANSFORMS.register_module()
            class MyOp(BaseTransform):
                type_name = "MyOp"
                ...
        """

        def _decorator(transform_cls: type[BaseTransform]) -> type[BaseTransform]:
            self.register(transform_cls, name=name)
            return transform_cls

        return _decorator

    def build(self, cfg: Mapping[str, Any]) -> BaseTransform:
        """Build one transform from a ``{"type": ..., **kwargs}`` mapping.

        ``kwargs`` are passed straight into the registered class's
        constructor -- each built-in is a frozen dataclass whose
        ``__post_init__`` validates its own fields, so no separate pydantic
        config model is required per op.

        Raises:
            TransformError: If ``cfg`` is not a mapping, ``type`` is missing,
                empty, or unregistered, or construction fails for any reason.
        """

        if not isinstance(cfg, Mapping):
            raise TransformError("transform step configuration must be a mapping")
        raw = dict(cfg)
        type_name = raw.pop("type", None)
        if not isinstance(type_name, str) or not type_name:
            raise TransformError("transform step must set a non-empty 'type'")
        transform_cls = self._transforms.get(type_name)
        if transform_cls is None:
            known = ", ".join(self._transforms) or "none"
            raise TransformError(
                f"unknown transform type {type_name!r}; registered transforms: {known}"
            )
        try:
            return transform_cls(**raw)
        except TransformError:
            raise
        except Exception as error:
            raise TransformError(f"failed to build transform {type_name!r}: {error}") from error


@dataclass(frozen=True, slots=True)
class Pipeline:
    """Compose registry-built transforms into one ordered callable (mmcv ``Compose``)."""

    config: tuple[dict[str, Any], ...]
    transforms: tuple[BaseTransform, ...]

    @property
    def mask_supported(self) -> bool:
        """Return whether every step preserves the mask geometry contract."""

        return all(transform.mask_supported for transform in self.transforms)

    def __call__(self, batch: NDArray[Any]) -> NDArray[Any]:
        """Apply every step's ``apply_batch`` in order."""

        result = batch
        for transform in self.transforms:
            result = transform.apply_batch(result)
        return np.ascontiguousarray(result)

    def apply_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Apply every step's ``apply_mask`` in order.

        Raises:
            TransformError: If any step has ``mask_supported = False``.
        """

        if not self.mask_supported:
            raise TransformError("pipeline contains a transform without mask support")
        result = mask
        for transform in self.transforms:
            result = transform.apply_mask(result)
        return result

    def describe(self) -> str:
        """Create a concise stable description from the ordered step types."""

        return " -> ".join(str(step.get("type")) for step in self.config) or "identity"


def build_pipeline(config: Sequence[Mapping[str, Any]], registry: TransformRegistry) -> Pipeline:
    """Build one :class:`Pipeline` from an MMAction2-style step-list config.

    Raises:
        TransformError: If ``config`` is not a sequence of steps, is empty,
            or any step fails to build.
    """

    if not isinstance(config, Sequence) or isinstance(config, (str, bytes)):
        raise TransformError("pipeline configuration must be a sequence of steps")
    if not config:
        raise TransformError("pipeline configuration must not be empty")
    steps = tuple(dict(step) for step in config)
    transforms = tuple(registry.build(step) for step in steps)
    return Pipeline(config=steps, transforms=transforms)
