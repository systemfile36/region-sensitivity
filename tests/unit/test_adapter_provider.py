from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from ssat.core.adapter import (
    AdapterProvider,
    AdapterProviderError,
    AdapterProviderRegistry,
    CallableAdapter,
    ProviderConfig,
    TimmProviderConfig,
    TorchvisionProviderConfig,
    TorchvisionVideoProviderConfig,
    default_adapter_provider_registry,
)
from ssat.application import ApplicationError, ApplicationErrorCode, AuditApplication, EstimateRequest
from ssat.utils.io import sha256_file


class _CustomConfig(ProviderConfig):
    provider: Literal["custom"] = "custom"
    model_id: str


class _CustomProvider(AdapterProvider):
    name = "custom"
    config_model = _CustomConfig

    def build(self, config: ProviderConfig, *, base_dir: Path) -> CallableAdapter:
        assert isinstance(config, _CustomConfig)
        return CallableAdapter(
            lambda batch: np.zeros((len(batch), 2), dtype=np.float32),
            model_id=config.model_id,
        )


def test_default_registry_is_fresh_and_contains_only_builtins() -> None:
    first = default_adapter_provider_registry()
    second = default_adapter_provider_registry()
    assert first is not second
    assert first.names == ("torchvision", "torchvision_video", "timm")


def test_custom_provider_requires_explicit_registration(tmp_path: Path) -> None:
    registry = AdapterProviderRegistry()
    with pytest.raises(AdapterProviderError, match="unknown adapter provider"):
        registry.parse({"provider": "custom", "model_id": "example"})

    registry.register(_CustomProvider())
    config = registry.parse({"provider": "custom", "model_id": "example"})
    assert registry.build(config, base_dir=tmp_path).describe().model_id == "example"
    with pytest.raises(AdapterProviderError, match="already registered"):
        registry.register(_CustomProvider())


def test_builtin_checkpoint_selectors_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        TorchvisionProviderConfig(
            model_name="resnet18",
            weights="DEFAULT",
            checkpoint={"path": "model.pt"},
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        TimmProviderConfig(
            model_name="resnet18",
            pretrained=True,
            checkpoint={"path": "model.pt"},
        )


def test_video_checkpoint_selector_is_mutually_exclusive_with_weights() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        TorchvisionVideoProviderConfig(
            model_name="r3d_18",
            weights="DEFAULT",
            checkpoint={"path": "model.pt"},
        )


def test_unknown_provider_fields_are_rejected() -> None:
    registry = default_adapter_provider_registry()
    with pytest.raises(AdapterProviderError, match="invalid 'torchvision'"):
        registry.parse(
            {
                "provider": "torchvision",
                "model_name": "resnet18",
                "typo": True,
            }
        )


def test_application_preserves_provider_error_category(tmp_path: Path) -> None:
    config = {
        "source": {
            "kind": "image_manifest",
            "manifest": Path(__file__).parents[1]
            / "fixtures"
            / "synthetic_classification"
            / "manifest.json",
        },
        "adapter": {"provider": "missing"},
        "regions": [{"region_id": "r", "kind": "grid", "params": {"rows": 1, "cols": 1}}],
        "perturbations": [{"op": "constant_fill", "params": {"value": 0}}],
    }
    with pytest.raises(ApplicationError) as caught:
        AuditApplication().estimate(EstimateRequest(config, base_dir=tmp_path))
    assert caught.value.code is ApplicationErrorCode.PROVIDER


def test_torchvision_provider_loads_local_checkpoint_without_network(
    tmp_path: Path,
) -> None:
    import torch
    from torchvision import models

    checkpoint = tmp_path / "squeezenet.pt"
    torch.save({"state_dict": models.squeezenet1_0(weights=None).state_dict()}, checkpoint)
    registry = default_adapter_provider_registry()
    config = registry.parse(
        {
            "provider": "torchvision",
            "model_name": "squeezenet1_0",
            "device": "cpu",
            "checkpoint": {
                "path": checkpoint.name,
                "state_dict_key": "state_dict",
            },
        }
    )
    adapter = registry.build(config, base_dir=tmp_path)
    assert adapter.describe().weights_hash == sha256_file(checkpoint)
    assert adapter.describe().weights_id == "checkpoint:squeezenet.pt"
    outputs = adapter.predict(np.zeros((1, 1, 64, 64, 3), dtype=np.uint8))
    assert outputs[0].logits.shape == (1000,)


_CROP_FREE_OPS = (
    {"op": "resize", "size": [224, 224]},
    {"op": "to_float"},
    {"op": "normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    {"op": "channels_first"},
    {"op": "squeeze_time"},
)


def _grid_cell_mask(row: int, col: int, *, size: int = 32, cells: int = 4) -> np.ndarray:
    """Build one cell of a ``cells``x``cells`` grid over a ``size``x``size`` frame."""

    step = size // cells
    mask = np.zeros((size, size), dtype=np.bool_)
    mask[row * step : (row + 1) * step, col * step : (col + 1) * step] = True
    return mask


def test_declared_preprocessing_keeps_equal_regions_equal_in_model_space() -> None:
    """A configured pipeline can avoid the preset crop that unequalizes regions.

    The stock ImageNet preset resizes the short edge to 256 then center-crops
    to 224, so on a small square frame the corner cells lose far more area
    than the middle ones -- nominally equal grid cells arrive at the model
    with unequal areas, which confounds any comparison between regions. This
    pins both halves of that: the preset really does distort, and a declared
    crop-free pipeline really does keep every cell identical.
    """

    registry = default_adapter_provider_registry()
    base = {"provider": "torchvision", "model_name": "squeezenet1_0", "device": "cpu"}

    preset = registry.build(registry.parse(dict(base)), base_dir=Path.cwd())
    declared = registry.build(
        registry.parse({**base, "preprocessing": _CROP_FREE_OPS}), base_dir=Path.cwd()
    )

    cells = [_grid_cell_mask(row, col) for row in range(4) for col in range(4)]
    preset_areas = {int(preset.transform_mask(cell).sum()) for cell in cells}
    declared_areas = {int(declared.transform_mask(cell).sum()) for cell in cells}

    assert len(preset_areas) > 1
    assert declared_areas == {56 * 56}


def test_declared_preprocessing_changes_the_recorded_fingerprint() -> None:
    """Runs made with different preprocessing must not look identical in a manifest."""

    registry = default_adapter_provider_registry()
    base = {"provider": "torchvision", "model_name": "squeezenet1_0", "device": "cpu"}

    preset = registry.build(registry.parse(dict(base)), base_dir=Path.cwd()).describe()
    declared = registry.build(
        registry.parse({**base, "preprocessing": _CROP_FREE_OPS}), base_dir=Path.cwd()
    ).describe()

    assert preset.preprocessing_fingerprint != declared.preprocessing_fingerprint
    assert declared.mask_transform_available


def test_declared_preprocessing_runs_prediction_end_to_end() -> None:
    """The declarative pipeline yields numpy, so predict must adopt it as a tensor."""

    registry = default_adapter_provider_registry()
    adapter = registry.build(
        registry.parse(
            {
                "provider": "torchvision",
                "model_name": "squeezenet1_0",
                "device": "cpu",
                "preprocessing": _CROP_FREE_OPS,
            }
        ),
        base_dir=Path.cwd(),
    )

    outputs = adapter.predict(np.zeros((2, 1, 32, 32, 3), dtype=np.uint8))

    assert len(outputs) == 2
    assert outputs[0].logits.shape == (1000,)


@pytest.mark.parametrize(
    "preprocessing, message",
    [
        ((), "must not be empty"),
        (({"op": "nope"},), "unknown preprocessing op"),
        (({"op": "resize"},), "missing 'size'"),
    ],
)
def test_malformed_preprocessing_is_rejected_at_parse_time(
    preprocessing: tuple, message: str
) -> None:
    """A bad op list must fail when the config loads, not mid-audit."""

    registry = default_adapter_provider_registry()
    with pytest.raises(AdapterProviderError, match=message):
        registry.parse(
            {
                "provider": "torchvision",
                "model_name": "squeezenet1_0",
                "preprocessing": preprocessing,
            }
        )


def test_torchvision_video_provider_loads_local_checkpoint_without_network(
    tmp_path: Path,
) -> None:
    import torch
    from torchvision.models import video

    checkpoint = tmp_path / "r3d18.pt"
    torch.save({"state_dict": video.r3d_18(weights=None).state_dict()}, checkpoint)
    registry = default_adapter_provider_registry()
    config = registry.parse(
        {
            "provider": "torchvision_video",
            "model_name": "r3d_18",
            "device": "cpu",
            "resize_size": 40,
            "crop_size": 32,
            "checkpoint": {
                "path": checkpoint.name,
                "state_dict_key": "state_dict",
            },
        }
    )
    adapter = registry.build(config, base_dir=tmp_path)
    assert adapter.describe().weights_hash == sha256_file(checkpoint)
    assert adapter.describe().weights_id == "checkpoint:r3d18.pt"
    outputs = adapter.predict(np.zeros((1, 4, 40, 40, 3), dtype=np.uint8))
    assert outputs[0].logits.shape == (400,)
