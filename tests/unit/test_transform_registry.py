from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
import pytest

from ssat.core.adapter.transform_registry import (
    BaseTransform,
    Pipeline,
    TransformError,
    TransformRegistry,
    build_pipeline,
)


@dataclass(frozen=True, slots=True)
class _AddOne(BaseTransform):
    """Add ``amount`` to every pixel; identity mask geometry."""

    type_name: ClassVar[str] = "AddOne"
    amount: int = 1

    def apply_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
        return batch + self.amount


@dataclass(frozen=True, slots=True)
class _DropFirstFrame(BaseTransform):
    """Drop the first frame on the T axis; mirrors it on (T, H, W) masks too."""

    type_name: ClassVar[str] = "DropFirstFrame"

    def apply_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
        return batch[:, 1:]

    def apply_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        return mask[1:] if mask.ndim == 3 else mask


@dataclass(frozen=True, slots=True)
class _NoMaskGeometry(BaseTransform):
    """A transform that cannot express its geometry on a mask at all."""

    type_name: ClassVar[str] = "NoMaskGeometry"
    mask_supported: ClassVar[bool] = False

    def apply_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
        return batch


def _registry(*classes: type[BaseTransform]) -> TransformRegistry:
    registry = TransformRegistry()
    for cls in classes:
        registry.register(cls)
    return registry


def _batch(shape: tuple[int, ...] = (2, 3, 4, 4, 3)) -> NDArray[np.uint8]:
    return np.zeros(shape, dtype=np.uint8)


# --- TransformRegistry.register -------------------------------------------------


def test_register_rejects_duplicate_empty_or_invalid_transforms() -> None:
    registry = TransformRegistry()
    registry.register(_AddOne)
    with pytest.raises(TransformError, match="already registered"):
        registry.register(_AddOne)
    with pytest.raises(TypeError, match="must be a BaseTransform subclass"):
        registry.register(object)  # type: ignore[arg-type]
    with pytest.raises(TransformError, match="must not be empty"):
        registry.register(_AddOne, name="")


def test_register_under_explicit_name_does_not_use_type_name() -> None:
    registry = TransformRegistry()
    registry.register(_AddOne, name="aliased")
    assert registry.names == ("aliased",)


# --- TransformRegistry.register_module ------------------------------------------


def test_register_module_decorator_registers_and_returns_class_unchanged() -> None:
    registry = TransformRegistry()

    @registry.register_module()
    class _Custom(BaseTransform):
        type_name: ClassVar[str] = "Custom"

        def apply_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
            return batch

    assert registry.names == ("Custom",)
    assert registry.build({"type": "Custom"}).__class__ is _Custom


def test_register_module_decorator_accepts_explicit_name() -> None:
    registry = TransformRegistry()

    @registry.register_module(name="renamed")
    class _Custom(BaseTransform):
        type_name: ClassVar[str] = "Custom"

        def apply_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
            return batch

    assert registry.names == ("renamed",)


# --- TransformRegistry.build -----------------------------------------------------


def test_build_rejects_non_mapping_config() -> None:
    registry = _registry(_AddOne)
    with pytest.raises(TransformError, match="must be a mapping"):
        registry.build(["type", "AddOne"])  # type: ignore[arg-type]


def test_build_rejects_missing_or_empty_type() -> None:
    registry = _registry(_AddOne)
    with pytest.raises(TransformError, match="non-empty 'type'"):
        registry.build({"amount": 1})
    with pytest.raises(TransformError, match="non-empty 'type'"):
        registry.build({"type": ""})


def test_build_unknown_type_lists_registered_transform_names() -> None:
    registry = _registry(_AddOne, _DropFirstFrame)
    with pytest.raises(TransformError, match="AddOne, DropFirstFrame"):
        registry.build({"type": "Nope"})


def test_build_wraps_constructor_errors_as_transform_error() -> None:
    registry = _registry(_AddOne)
    with pytest.raises(TransformError, match="failed to build transform 'AddOne'"):
        registry.build({"type": "AddOne", "unknown_kwarg": 1})


def test_build_passes_kwargs_into_constructor() -> None:
    registry = _registry(_AddOne)
    transform = registry.build({"type": "AddOne", "amount": 5})
    assert isinstance(transform, _AddOne)
    assert transform.amount == 5


# --- build_pipeline / Pipeline ----------------------------------------------------


def test_build_pipeline_rejects_empty_config() -> None:
    registry = _registry(_AddOne)
    with pytest.raises(TransformError, match="must not be empty"):
        build_pipeline([], registry)


def test_build_pipeline_rejects_non_sequence_config() -> None:
    registry = _registry(_AddOne)
    with pytest.raises(TransformError, match="sequence of steps"):
        build_pipeline({"type": "AddOne"}, registry)  # type: ignore[arg-type]


def test_pipeline_applies_steps_in_order_and_describes_them() -> None:
    registry = _registry(_AddOne, _DropFirstFrame)
    pipeline = build_pipeline(
        [{"type": "AddOne", "amount": 2}, {"type": "DropFirstFrame"}], registry
    )
    assert isinstance(pipeline, Pipeline)
    assert pipeline.describe() == "AddOne -> DropFirstFrame"

    batch = _batch()
    result = pipeline(batch)
    assert result.shape == (2, 2, 4, 4, 3)
    assert np.all(result == 2)


def test_pipeline_mask_supported_reflects_every_step() -> None:
    registry = _registry(_AddOne, _NoMaskGeometry)
    all_supported = build_pipeline([{"type": "AddOne"}], registry)
    assert all_supported.mask_supported is True

    mixed = build_pipeline([{"type": "AddOne"}, {"type": "NoMaskGeometry"}], registry)
    assert mixed.mask_supported is False
    with pytest.raises(TransformError, match="without mask support"):
        mixed.apply_mask(np.zeros((4, 4), dtype=np.bool_))


def test_pipeline_apply_mask_mirrors_batch_geometry() -> None:
    registry = _registry(_DropFirstFrame)
    pipeline = build_pipeline([{"type": "DropFirstFrame"}], registry)
    mask = np.zeros((3, 4, 4), dtype=np.bool_)
    mask[0] = True
    transformed = pipeline.apply_mask(mask)
    assert transformed.shape == (2, 4, 4)
    assert not transformed.any()


def test_end_to_end_custom_transform_registration() -> None:
    """The registration path documented for the custom-transform extension point."""

    registry = TransformRegistry()

    @registry.register_module()
    class Invert(BaseTransform):
        type_name: ClassVar[str] = "Invert"

        def apply_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
            return 255 - batch

    pipeline = build_pipeline([{"type": "Invert"}], registry)
    batch = np.full((1, 1, 2, 2, 3), 10, dtype=np.uint8)
    result = pipeline(batch)
    assert np.all(result == 245)
