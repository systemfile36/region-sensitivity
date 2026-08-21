from pathlib import Path

import numpy as np

from ssat.core.adapter import default_adapter_provider_registry


def test_timm_squash_is_crop_free_and_changes_fingerprint() -> None:
    registry = default_adapter_provider_registry()
    base = {"provider": "timm", "model_name": "mobilenetv2_050", "device": "cpu"}
    exact = registry.build(registry.parse(base), base_dir=Path.cwd())
    squash = registry.build(
        registry.parse({**base, "geometry_mode": "squash"}), base_dir=Path.cwd()
    )
    mask = np.zeros((32, 32), dtype=np.bool_)
    mask[:8, :8] = True
    assert exact.describe().preprocessing_fingerprint != squash.describe().preprocessing_fingerprint
    assert squash.transform_mask(mask).shape == (224, 224)
    assert int(squash.transform_mask(mask).sum()) == 56 * 56


def test_timm_squash_runs_prediction_without_pretrained_weights() -> None:
    registry = default_adapter_provider_registry()
    adapter = registry.build(
        registry.parse(
            {
                "provider": "timm",
                "model_name": "mobilenetv2_050",
                "device": "cpu",
                "geometry_mode": "squash",
            }
        ),
        base_dir=Path.cwd(),
    )
    outputs = adapter.predict(np.zeros((1, 1, 32, 48, 3), dtype=np.uint8))
    assert outputs[0].logits.shape == (1000,)
