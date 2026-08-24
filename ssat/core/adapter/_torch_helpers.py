"""Private helpers shared by torch-backed image classification adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ssat.core.adapter.base import AdapterError


def resolve_torchvision_weights(weights_enum: Any, weights: Any) -> Any:
    """Resolve a convenient string weight selector against one enum.

    Shared by ``TorchvisionAdapter`` and ``TorchvisionVideoAdapter``, the
    only two adapters with a torchvision weights-enum concept.
    """

    if weights is None:
        return None
    if isinstance(weights, str):
        if weights == "DEFAULT":
            return weights_enum.DEFAULT
        try:
            return weights_enum[weights]
        except KeyError as error:
            raise ValueError(f"unknown torchvision weights {weights!r}") from error
    if not isinstance(weights, weights_enum):
        raise TypeError(f"weights must be None, a string, or {weights_enum.__name__}")
    return weights


def validate_image_classifier_batch(batch: NDArray[np.uint8]) -> None:
    """Require the v1 image classification specialization of THWC."""

    if batch.shape[1] != 1:
        raise AdapterError("image classification adapters require T=1")
    if batch.shape[-1] != 3:
        raise AdapterError("image classification adapters require three RGB channels")


def transform_mask_with_ops(
    mask: NDArray[np.bool_],
    ops: Sequence[Any],
) -> NDArray[np.bool_]:
    """Apply known torchvision/timm geometry and ignore photometric ops."""

    import torch
    from torchvision.transforms import CenterCrop as TorchCenterCrop
    from torchvision.transforms import Resize as TorchResize
    from torchvision.transforms import functional as functional
    from torchvision.transforms.functional import InterpolationMode

    tensor = torch.from_numpy(mask.astype(np.uint8, copy=False)).unsqueeze(0)
    ignored_names = {
        "ConvertImageDtype",
        "MaybeToTensor",
        "Normalize",
        "PILToTensor",
        "ToDtype",
        "ToImage",
        "ToTensor",
    }
    for op in ops:
        if isinstance(op, TorchResize):
            tensor = functional.resize(
                tensor,
                op.size,
                interpolation=InterpolationMode.NEAREST,
                max_size=op.max_size,
                antialias=False,
            )
        elif isinstance(op, TorchCenterCrop):
            tensor = functional.center_crop(tensor, op.size)
        elif type(op).__name__ == "ResizeKeepRatio" and type(op).__module__.startswith(
            "timm."
        ):
            if op.random_scale_prob > 0 or op.random_aspect_prob > 0:
                raise AdapterError("nondeterministic ResizeKeepRatio cannot transform masks")
            size = op.get_params(
                tensor,
                op.size,
                op.longest,
                op.random_scale_prob,
                op.random_scale_range,
                op.random_scale_area,
                op.random_aspect_prob,
                op.random_aspect_range,
            )
            tensor = functional.resize(
                tensor,
                size,
                interpolation=InterpolationMode.NEAREST,
                antialias=False,
            )
        elif type(op).__name__ == "CenterCropOrPad" and type(op).__module__.startswith(
            "timm."
        ):
            tensor = op(tensor)
        elif type(op).__name__ in ignored_names:
            continue
        else:
            raise AdapterError(
                f"unsupported preprocessing op for mask geometry: {type(op).__name__}"
            )
    result = tensor.squeeze(0).cpu().numpy()
    return np.asarray(result, dtype=np.bool_)
