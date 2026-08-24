"""Shared boilerplate for the four torch-backed model adapters.

Extracts the device resolution, seeded construction, OOM cleanup, and
``predict()`` exception-wrapping shapes that ``TorchvisionAdapter``,
``TorchvisionVideoAdapter``, ``TorchvisionTSMAdapter``, and ``TimmAdapter``
previously each reimplemented identically. Deliberately does *not* impose one
template-method ``predict()`` on every adapter: ``TorchvisionTSMAdapter``
combines its preprocess and inference stages into a single wrapped call and
leaves its decode call unwrapped, a pre-existing divergence from the other
three that this mixin preserves via the ``_wrap_inference`` helpers below
rather than normalizing away.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any, TypeVar

from ssat.core.adapter.base import AdapterError, AdapterOutOfMemoryError
from ssat.core.adapter.output_decoder import OutputDecoder

T = TypeVar("T")


class _TorchAdapterMixin:
    """Mix into a ``ModelAdapter`` subclass to share torch-specific boilerplate.

    Listed before ``ModelAdapter`` in a subclass's base list so its concrete
    :meth:`cleanup_after_oom` takes MRO precedence over ``ModelAdapter``'s
    framework-neutral no-op default.
    """

    _model: Any
    _device: Any

    @staticmethod
    def _validate_init_seed(init_seed: int) -> None:
        """Require ``init_seed`` to fit the documented 64-bit unsigned range."""

        if isinstance(init_seed, bool) or not 0 <= init_seed <= 2**63 - 1:
            raise ValueError("init_seed must be between 0 and 2**63 - 1")

    @staticmethod
    def _seeded_model_init(init_seed: int, build_model: Callable[[], T]) -> T:
        """Construct a model inside a forked RNG seeded with ``init_seed``.

        Args:
            init_seed: Seed applied only for the duration of construction.
            build_model: Zero-argument callable that builds and returns the model.

        Returns:
            The model returned by ``build_model``.
        """

        import torch

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            return build_model()

    def _place_on_device(self, model: Any, device: Any) -> None:
        """Resolve the target device and move ``model`` onto it.

        Sets ``self._model``/``self._device`` on success.

        Raises:
            AdapterError: If moving the model to the resolved device fails.
        """

        import torch

        resolved_device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        try:
            self._model = model.eval().to(resolved_device)
        except Exception as error:
            raise AdapterError(f"failed to move model to device {resolved_device}") from error
        self._device = resolved_device

    @staticmethod
    def _finalize_output_decoder(
        output_decoder: OutputDecoder | None, default: OutputDecoder
    ) -> OutputDecoder:
        """Apply the ``output_decoder or default`` fallback and validate the result.

        Raises:
            TypeError: If the resolved decoder does not implement ``OutputDecoder``.
        """

        decoder = output_decoder or default
        if not isinstance(decoder, OutputDecoder):
            raise TypeError("output_decoder must implement OutputDecoder")
        return decoder

    def cleanup_after_oom(self) -> None:
        """Release Python and CUDA allocations before a smaller retry."""

        import torch

        gc.collect()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def _wrap_stage(fn: Callable[[], T], error_message: str) -> T:
        """Run ``fn``, wrapping any non-``AdapterError`` failure in ``AdapterError``.

        Matches the preprocess/decode stage shape shared by every adapter:
        an existing ``AdapterError`` passes through unwrapped.
        """

        try:
            return fn()
        except Exception as error:
            if isinstance(error, AdapterError):
                raise
            raise AdapterError(error_message) from error

    @staticmethod
    def _wrap_inference(
        fn: Callable[[], T],
        *,
        oom_message: str,
        error_message: str,
        passthrough_adapter_error: bool = False,
    ) -> T:
        """Run ``fn``, translating a device OOM and wrapping other failures.

        Args:
            fn: Zero-argument callable performing (and possibly preceded by,
                for ``TorchvisionTSMAdapter``) the forward pass.
            oom_message: Message for the translated ``AdapterOutOfMemoryError``.
            error_message: Message for any other wrapped failure.
            passthrough_adapter_error: When ``True``, an ``AdapterError`` raised
                by ``fn`` (e.g. from preprocessing folded into the same call)
                passes through unwrapped instead of being re-wrapped. Only
                ``TorchvisionTSMAdapter`` sets this, since it is the one
                adapter that folds preprocessing into this same wrapped call.
        """

        import torch

        try:
            return fn()
        except torch.cuda.OutOfMemoryError as error:
            raise AdapterOutOfMemoryError(oom_message) from error
        except Exception as error:
            if passthrough_adapter_error and isinstance(error, AdapterError):
                raise
            raise AdapterError(error_message) from error
