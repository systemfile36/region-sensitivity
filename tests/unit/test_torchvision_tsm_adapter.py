from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from ssat.core.adapter import AdapterError, TorchvisionTSMAdapter
from ssat.core.adapter.mmaction_checkpoint import (
    convert_mmaction_tsm_state_dict,
    load_mmaction_checkpoint_restricted,
    mmaction_tsm_key_to_native,
)
from ssat.core.adapter.torchvision_tsm_adapter import TemporalShift


class _UnapprovedCheckpointObject:
    pass


def test_temporal_shift_moves_channel_folds_in_opposite_directions() -> None:
    import torch

    source = torch.arange(1 * 3 * 4, dtype=torch.float32).reshape(3, 4, 1, 1)
    shifted = TemporalShift.shift(source, num_segments=3, shift_div=4).reshape(3, 4)
    assert shifted[:, 0].tolist() == [4.0, 8.0, 0.0]
    assert shifted[:, 1].tolist() == [0.0, 1.0, 5.0]
    assert shifted[:, 2:].tolist() == source.reshape(3, 4)[:, 2:].tolist()


def test_temporal_shift_rejects_incomplete_segment_batches() -> None:
    import torch

    with pytest.raises(ValueError, match="divisible"):
        TemporalShift.shift(torch.zeros((7, 8, 1, 1)), 8, 8)


@pytest.mark.parametrize(
    "source,target",
    [
        ("backbone.conv1.conv.weight", "backbone.conv1.weight"),
        ("backbone.conv1.bn.running_mean", "backbone.bn1.running_mean"),
        ("backbone.layer2.0.conv1.conv.net.weight", "backbone.layer2.0.conv1.net.weight"),
        ("backbone.layer2.0.conv2.bn.weight", "backbone.layer2.0.bn2.weight"),
        ("backbone.layer4.0.downsample.conv.weight", "backbone.layer4.0.downsample.0.weight"),
        ("cls_head.fc_cls.weight", "backbone.fc.weight"),
    ],
)
def test_mmaction_checkpoint_key_mapping(source: str, target: str) -> None:
    assert mmaction_tsm_key_to_native(source) == target


def test_checkpoint_conversion_rejects_unknown_keys_and_collisions() -> None:
    with pytest.raises(AdapterError, match="unsupported"):
        convert_mmaction_tsm_state_dict(OrderedDict([("other.weight", object())]))


def test_restricted_mmaction_loader_rejects_unapproved_globals(tmp_path) -> None:
    import torch

    checkpoint = tmp_path / "unapproved.pt"
    torch.save({"payload": _UnapprovedCheckpointObject()}, checkpoint)
    with pytest.raises(AdapterError, match="unapproved pickle globals"):
        load_mmaction_checkpoint_restricted(checkpoint)


def test_tsm_adapter_preprocessing_modes_have_distinct_geometry() -> None:
    exact = TorchvisionTSMAdapter(
        num_segments=2, num_classes=3, device="cpu", preprocessing="mmaction2_val"
    )
    crop_free = TorchvisionTSMAdapter(
        num_segments=2, num_classes=3, device="cpu", preprocessing="crop_free"
    )
    mask = np.zeros((2, 32, 32), dtype=np.bool_)
    mask[:, :8, :8] = True
    assert exact.describe().preprocessing_fingerprint != crop_free.describe().preprocessing_fingerprint
    assert exact.transform_mask(mask).shape == (2, 224, 224)
    assert crop_free.transform_mask(mask).shape == (2, 224, 224)
    assert exact.transform_mask(mask).sum() != crop_free.transform_mask(mask).sum()


def test_tsm_adapter_requires_configured_segment_count() -> None:
    adapter = TorchvisionTSMAdapter(num_segments=2, num_classes=3, device="cpu")
    with pytest.raises(AdapterError, match="requires T=2"):
        adapter.predict(np.zeros((1, 1, 32, 32, 3), dtype=np.uint8))
