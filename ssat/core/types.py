"""Shared constants, enums, and JSON-compatible value types."""

from __future__ import annotations

from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import TypeAlias

SCHEMA_VERSION = "1.0.0"

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
FrozenJsonValue: TypeAlias = JsonScalar | tuple["FrozenJsonValue", ...] | MappingProxyType


class RegionKind(str, Enum):
    """Enumerate mask recipes understood by the region layer."""

    GRID = "grid"
    BBOX_PARTITION = "bbox_partition"
    EXPLICIT = "explicit"
    RANDOM_AREA_MATCH = "random_area_match"


class PerturbationOp(str, Enum):
    """Enumerate v1 deletion-style perturbation operations."""

    CONSTANT_FILL = "constant_fill"
    MEAN_FILL = "mean_fill"
    BLUR = "blur"
    GAUSSIAN_NOISE = "gaussian_noise"
    PATCH_SHUFFLE = "patch_shuffle"


class ItemStatus(str, Enum):
    """Represent terminal statuses written for clean and perturbed items."""

    OK = "ok"
    LOAD_FAILED = "load_failed"
    PREDICT_FAILED = "predict_failed"
    SKIPPED_OOM = "skipped_oom"


def validate_json_value(value: JsonValue, *, path: str = "params") -> None:
    """Validate a value against the canonical JSON parameter contract.

    Args:
        value: Scalar, list, or string-keyed mapping to validate recursively.
        path: Logical field path included in validation errors.

    Raises:
        ValueError: If the value is non-finite or not JSON-compatible.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string mapping key")
            validate_json_value(item, path=f"{path}.{key}")
        return
    raise ValueError(f"{path} contains unsupported value type {type(value).__name__}")


def freeze_json_mapping(value: dict[str, JsonValue]) -> MappingProxyType:
    """Create an immutable view of JSON-compatible parameters.

    Lists become tuples and mappings become read-only mapping proxies. Runtime
    work specifications use this form to prevent identity-changing mutation.

    Args:
        value: Parameter mapping to validate and freeze recursively.

    Returns:
        A recursively immutable mapping proxy.

    Raises:
        ValueError: If a nested value is not JSON-compatible.
    """

    validate_json_value(value)

    def freeze(item: JsonValue) -> FrozenJsonValue:
        """Recursively convert mutable JSON containers to immutable forms."""

        if isinstance(item, dict):
            return MappingProxyType({key: freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item

    frozen = freeze(value)
    if not isinstance(frozen, MappingProxyType):  # pragma: no cover - defensive
        raise TypeError("expected a mapping")
    return frozen
