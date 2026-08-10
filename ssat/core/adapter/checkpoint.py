"""Shared local checkpoint loading for built-in torch providers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ssat.core.adapter.base import AdapterError


def load_state_dict_checkpoint(
    model: Any,
    path: str | Path,
    *,
    state_dict_key: str | None = None,
    strict: bool = True,
) -> None:
    """Load a trusted local state dict with torch's restricted loader."""

    import torch

    checkpoint_path = Path(path)
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if state_dict_key is not None:
            if not isinstance(payload, Mapping) or state_dict_key not in payload:
                raise AdapterError(
                    f"checkpoint has no state_dict_key {state_dict_key!r}"
                )
            payload = payload[state_dict_key]
        if not isinstance(payload, Mapping):
            raise AdapterError("checkpoint payload must be a state-dict mapping")
        model.load_state_dict(payload, strict=strict)
    except AdapterError:
        raise
    except Exception as error:
        raise AdapterError(
            f"failed to load checkpoint {checkpoint_path}"
        ) from error
