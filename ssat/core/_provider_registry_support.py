"""Provider-registry helpers shared by the adapter and source registries."""

from __future__ import annotations

from typing import get_args

from pydantic import BaseModel
from pydantic_core import PydanticUndefined


def validate_config_literal(
    config_model: type[BaseModel],
    field_name: str,
    provider_name: str,
    *,
    error: type[Exception],
) -> None:
    """Require ``config_model``'s discriminator field to resolve to ``provider_name``.

    Guards against a registered provider's config class drifting out of sync
    with the provider's own ``name`` -- previously enforced only by human
    convention, matched independently at each provider's definition site. A
    mismatch here otherwise surfaces later as a confusing Pydantic validation
    error inside ``parse()`` instead of a clear registration-time one, and
    permanently makes the provider unreachable by name.

    Accepts both the built-in ``Literal["x"] = "x"`` style and a plain
    ``str = "x"`` default, so user-registered providers are not forced into
    ``Literal`` typing to pass this check.

    Args:
        config_model: Provider-specific Pydantic config class.
        field_name: Name of the discriminator field (``"provider"`` for
            adapters, ``"kind"`` for sources).
        provider_name: Expected field value, taken from ``provider.name``.
        error: Exception type to raise on mismatch -- the registry's own
            domain error, e.g. ``AdapterProviderError``/``SourceProviderError``.

    Raises:
        Exception: An instance of ``error`` if the field is missing, its
            ``Literal`` values (if any) exclude ``provider_name``, or its
            default (if any) differs from ``provider_name``.
    """

    field = config_model.model_fields.get(field_name)
    if field is None:
        raise error(f"provider {provider_name!r} config_model has no {field_name!r} field")

    allowed = get_args(field.annotation)
    if allowed and provider_name not in allowed:
        raise error(
            f"provider {provider_name!r} config_model.{field_name} must allow "
            f"{provider_name!r}, found Literal{allowed!r}"
        )

    default = field.default
    if default is not PydanticUndefined and default != provider_name:
        raise error(
            f"provider {provider_name!r} config_model.{field_name} default must "
            f"be {provider_name!r}, found {default!r}"
        )
