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
