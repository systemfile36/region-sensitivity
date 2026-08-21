"""Restricted conversion helpers for the project's MMAction2 TSM checkpoint."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
import builtins
from pathlib import Path
import sys
import types
from typing import Any, Iterator

from ssat.core.adapter.base import AdapterError

_ALLOWED_UNSAFE_GLOBALS = {
    "builtins.getattr",
    "mmengine.logging.history_buffer.HistoryBuffer",
    "numpy._core.multiarray._reconstruct",
    "numpy._core.multiarray.scalar",
    "numpy.core.multiarray._reconstruct",
    "numpy.core.multiarray.scalar",
    "numpy.dtype",
    "numpy.ndarray",
}


class _HistoryBuffer:
    """Minimal data-only compatibility shell used by torch's restricted loader."""

    def __init__(
        self,
        log_history: Any = None,
        count_history: Any = None,
        max_length: int = 1_000_000,
    ) -> None:
        self._log_history = log_history
        self._count_history = count_history
        self._max_length = max_length

    @property
    def current(self) -> Any:
        return self._log_history[-1]

    def mean(self, window_size: int | None = None) -> float:
        return 0.0

    def min(self, window_size: int | None = None) -> float:
        return 0.0

    def max(self, window_size: int | None = None) -> float:
        return 0.0


_HistoryBuffer.__name__ = "HistoryBuffer"
_HistoryBuffer.__qualname__ = "HistoryBuffer"
_HistoryBuffer.__module__ = "mmengine.logging.history_buffer"


@contextmanager
def _history_buffer_module() -> Iterator[None]:
    names = (
        "mmengine",
        "mmengine.logging",
        "mmengine.logging.history_buffer",
    )
    previous = {name: sys.modules.get(name) for name in names}
    mmengine = types.ModuleType("mmengine")
    logging = types.ModuleType("mmengine.logging")
    history = types.ModuleType("mmengine.logging.history_buffer")
    history.HistoryBuffer = _HistoryBuffer
    logging.history_buffer = history
    mmengine.logging = logging
    sys.modules[names[0]] = mmengine
    sys.modules[names[1]] = logging
    sys.modules[names[2]] = history
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_mmaction_checkpoint_restricted(path: str | Path) -> Mapping[str, Any]:
    """Load a known MMEngine checkpoint without importing MMEngine or pickle code."""

    import numpy as np
    import torch

    checkpoint_path = Path(path)
    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path))
    unexpected = sorted(unsafe - _ALLOWED_UNSAFE_GLOBALS)
    if unexpected:
        raise AdapterError(
            "checkpoint contains unapproved pickle globals: " + ", ".join(unexpected)
        )
    numpy_core = getattr(np, "_core", np.core)
    safe = [
        builtins.getattr,
        _HistoryBuffer,
        numpy_core.multiarray.scalar,
        numpy_core.multiarray._reconstruct,
        np.dtype,
        np.ndarray,
        type(np.dtype(np.float16)),
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
        type(np.dtype(np.int64)),
        type(np.dtype(np.uint64)),
    ]
    try:
        with _history_buffer_module(), torch.serialization.safe_globals(safe):
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
    except Exception as error:
        raise AdapterError(
            f"restricted MMAction checkpoint load failed: {checkpoint_path}"
        ) from error
    if not isinstance(payload, Mapping):
        raise AdapterError("MMAction checkpoint payload must be a mapping")
    return payload


def mmaction_tsm_key_to_native(key: str) -> str:
    """Map one MMAction2 ResNetTSM state key to the native torchvision model."""

    if key == "backbone.conv1.conv.weight":
        return "backbone.conv1.weight"
    if key.startswith("backbone.conv1.bn."):
        return "backbone.bn1." + key.removeprefix("backbone.conv1.bn.")
    if key.startswith("cls_head.fc_cls."):
        return "backbone.fc." + key.removeprefix("cls_head.fc_cls.")
    for conv_index in (1, 2, 3):
        marker = f".conv{conv_index}.conv."
        if marker in key:
            return key.replace(marker, f".conv{conv_index}.")
        marker = f".conv{conv_index}.bn."
        if marker in key:
            return key.replace(marker, f".bn{conv_index}.")
    if ".downsample.conv." in key:
        return key.replace(".downsample.conv.", ".downsample.0.")
    if ".downsample.bn." in key:
        return key.replace(".downsample.bn.", ".downsample.1.")
    raise AdapterError(f"unsupported MMAction TSM state key: {key}")


def convert_mmaction_tsm_state_dict(state_dict: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """Convert and collision-check the complete TSM state dictionary."""

    converted: OrderedDict[str, Any] = OrderedDict()
    for source_key, tensor in state_dict.items():
        if not isinstance(source_key, str):
            raise AdapterError("MMAction state-dict keys must be strings")
        target_key = mmaction_tsm_key_to_native(source_key)
        if target_key in converted:
            raise AdapterError(f"checkpoint key collision after conversion: {target_key}")
        converted[target_key] = tensor
    return converted
