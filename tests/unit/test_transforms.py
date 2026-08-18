"""Tests for the v1 registry-addressable built-in transforms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest

from ssat.core.adapter import preprocessing as _decl
from ssat.core.adapter.transform_registry import TransformError, build_pipeline
from ssat.core.adapter.transforms import (
    CenterCrop,
    FormatShape,
    Normalize,
    PipelinePreprocessor,
    Resize,
    SampleFrames,
    TenCrop,
    ToFloat,
    default_transform_registry,
)


def _clip_batch(t: int, height: int = 4, width: int = 4) -> NDArray[np.uint8]:
    return np.arange(t * height * width * 3, dtype=np.uint8).reshape(1, t, height, width, 3)


def _batch(count: int = 2, height: int = 10, width: int = 14) -> NDArray[np.uint8]:
    return np.arange(count * height * width * 3, dtype=np.uint8).reshape(
        count, 1, height, width, 3
    )


# --- SampleFrames -----------------------------------------------------------------


def test_sample_frames_centers_the_clip_for_even_and_odd_totals() -> None:
    even = SampleFrames(clip_len=2).apply_batch(_clip_batch(6))
    assert np.array_equal(even, _clip_batch(6)[:, 2:4])  # start = (6 - 2) // 2 = 2

    odd = SampleFrames(clip_len=3).apply_batch(_clip_batch(7))
    assert np.array_equal(odd, _clip_batch(7)[:, 2:5])  # start = (7 - 3) // 2 = 2


def test_sample_frames_rejects_clip_len_exceeding_available_frames() -> None:
    with pytest.raises(TransformError, match="exceeds available frames"):
        SampleFrames(clip_len=5).apply_batch(_clip_batch(4))


@pytest.mark.parametrize("clip_len", [0, -1, True])
def test_sample_frames_rejects_non_positive_or_bool_clip_len(clip_len: object) -> None:
    with pytest.raises(ValueError, match="positive int"):
        SampleFrames(clip_len=clip_len)  # type: ignore[arg-type]


def test_sample_frames_slices_per_frame_masks_and_leaves_shared_masks_untouched() -> None:
    transform = SampleFrames(clip_len=2)
    per_frame = np.zeros((6, 4, 4), dtype=np.bool_)
    per_frame[2] = True  # lands inside the centered [2:4) slice
    sliced = transform.apply_mask(per_frame)
    assert sliced.shape == (2, 4, 4)
    assert sliced[0].all()
    assert not sliced[1].any()

    shared = np.zeros((4, 4), dtype=np.bool_)
    assert transform.apply_mask(shared) is shared


# --- Resize -------------------------------------------------------------------------


def test_resize_scale_with_wildcard_axis_matches_short_edge_int_form() -> None:
    """scale=(-1, 50) pins the H=100 short edge to 50, matching bare int 50."""

    batch = np.zeros((1, 1, 100, 200, 3), dtype=np.uint8)
    resized = Resize(scale=(-1, 50)).apply_batch(batch)
    assert resized.shape[2:4] == (50, 100)
    assert np.array_equal(resized, Resize(scale=50).apply_batch(batch))


def test_resize_scale_explicit_pair_is_not_axis_swapped() -> None:
    """mmaction (w, h) = (300, 150) must produce (height, width) = (150, 300)."""

    batch = np.zeros((1, 1, 20, 20, 3), dtype=np.uint8)
    resized = Resize(scale=(300, 150)).apply_batch(batch)
    assert resized.shape[2:4] == (150, 300)


def test_resize_rejects_both_axes_as_wildcard() -> None:
    with pytest.raises(ValueError, match="cannot set both"):
        Resize(scale=(-1, -1))


def test_resize_apply_mask_delegates_to_declarative_engine() -> None:
    mask = np.zeros((100, 200), dtype=np.bool_)
    mask[:, :50] = True
    transformed = Resize(scale=(-1, 50)).apply_mask(mask)
    expected = _decl.transform_mask_geometry(mask, (_decl.Resize(50),))
    assert np.array_equal(transformed, expected)


# --- CenterCrop / ToFloat / Normalize (pure delegation) ------------------------------


def test_center_crop_matches_direct_declarative_call() -> None:
    batch = _batch(count=1, height=4, width=6)
    op = CenterCrop(crop_size=8)  # larger than input -> exercises padding
    result = op.apply_batch(batch)
    expected = _decl.apply_preprocessing(batch, (_decl.CenterCrop(8),))
    assert np.array_equal(result, expected)

    mask = np.zeros((4, 6), dtype=np.bool_)
    mask[:, :3] = True
    assert np.array_equal(
        op.apply_mask(mask), _decl.transform_mask_geometry(mask, (_decl.CenterCrop(8),))
    )


def test_center_crop_rejects_non_positive_crop_size() -> None:
    with pytest.raises(ValueError, match="positive int"):
        CenterCrop(crop_size=0)


def test_to_float_matches_direct_declarative_call() -> None:
    batch = _batch(count=1, height=2, width=2)
    result = ToFloat(scale=0.5).apply_batch(batch)
    expected = _decl.apply_preprocessing(batch, (_decl.ToFloat(0.5),))
    assert np.array_equal(result, expected)


def test_normalize_matches_direct_declarative_call() -> None:
    batch = _batch(count=1, height=2, width=2).astype(np.float32)
    op = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    result = op.apply_batch(batch)
    expected = _decl.apply_preprocessing(
        batch, (_decl.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),)
    )
    assert np.array_equal(result, expected)


def test_normalize_rejects_mismatched_or_invalid_mean_std() -> None:
    with pytest.raises(ValueError):
        Normalize(mean=(0.5, 0.5), std=(0.5,))
    with pytest.raises(ValueError):
        Normalize(mean=(0.5,), std=(0.0,))


# --- TenCrop --------------------------------------------------------------------------


def test_ten_crop_produces_ten_ordered_mirrored_views_per_sample() -> None:
    height, width, crop = 6, 8, 4
    batch = np.arange(2 * 1 * height * width * 3, dtype=np.uint8).reshape(2, 1, height, width, 3)
    result = TenCrop(crop_size=crop).apply_batch(batch)
    assert result.shape == (20, 1, crop, crop, 3)

    origins = (
        (0, 0),
        (0, width - crop),
        (height - crop, 0),
        (height - crop, width - crop),
        ((height - crop) // 2, (width - crop) // 2),
    )
    for sample in range(2):
        for view_index, (top, left) in enumerate(origins):
            plain = batch[sample : sample + 1, :, top : top + crop, left : left + crop, :]
            mirrored = plain[:, :, :, ::-1, :]
            base = sample * 10 + view_index * 2
            assert np.array_equal(result[base : base + 1], plain)
            assert np.array_equal(result[base + 1 : base + 2], mirrored)


def test_ten_crop_pads_undersized_input_like_center_crop() -> None:
    batch = np.full((1, 1, 3, 3, 3), 7, dtype=np.uint8)
    result = TenCrop(crop_size=5).apply_batch(batch)
    assert result.shape == (10, 1, 5, 5, 3)


def test_ten_crop_apply_mask_always_raises() -> None:
    with pytest.raises(TransformError, match="mask_supported=False"):
        TenCrop(crop_size=4).apply_mask(np.zeros((6, 8), dtype=np.bool_))


def test_ten_crop_rejects_non_positive_crop_size() -> None:
    with pytest.raises(ValueError, match="positive int"):
        TenCrop(crop_size=0)


# --- FormatShape ------------------------------------------------------------------------


def test_format_shape_nchw_requires_singleton_time_axis() -> None:
    single = _batch(count=1, height=2, width=2)
    result = FormatShape(input_format="NCHW").apply_batch(single)
    assert result.shape == (1, 3, 2, 2)

    multi = np.zeros((1, 2, 2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="T=1"):
        FormatShape(input_format="NCHW").apply_batch(multi)


def test_format_shape_ntchw_keeps_time_axis() -> None:
    multi = np.zeros((1, 4, 2, 2, 3), dtype=np.uint8)
    result = FormatShape(input_format="NTCHW").apply_batch(multi)
    assert result.shape == (1, 4, 3, 2, 2)


def test_format_shape_rejects_unknown_input_format() -> None:
    with pytest.raises(ValueError, match='"NCHW" or "NTCHW"'):
        FormatShape(input_format="NCTHW")  # type: ignore[arg-type]


# --- default_transform_registry ----------------------------------------------------------


def test_default_transform_registry_registers_exactly_the_v1_built_ins() -> None:
    registry = default_transform_registry()
    assert set(registry.names) == {
        "SampleFrames",
        "Resize",
        "CenterCrop",
        "TenCrop",
        "ToFloat",
        "Normalize",
        "FormatShape",
    }


# --- PipelinePreprocessor ------------------------------------------------------------------


def test_pipeline_preprocessor_round_trips_pixels_and_masks() -> None:
    preprocessor = PipelinePreprocessor(
        [
            {"type": "Resize", "scale": [-1, 8]},
            {"type": "CenterCrop", "crop_size": 6},
            {"type": "ToFloat"},
            {"type": "FormatShape", "input_format": "NCHW"},
        ]
    )
    spec = preprocessor.describe()
    assert spec.kind == "pipeline"
    assert spec.deterministic is True
    assert spec.mask_transform_available is True
    assert spec.fingerprint is not None

    batch = _batch(count=1, height=4, width=6)
    prepared = preprocessor.transform_batch(batch)
    assert prepared.shape == (1, 3, 6, 6)

    mask = np.zeros((4, 6), dtype=np.bool_)
    mask[:, :3] = True
    transformed = preprocessor.transform_mask(mask)
    assert transformed is not None
    assert transformed.shape == (6, 6)


def test_pipeline_preprocessor_with_ten_crop_reports_mask_unavailable() -> None:
    preprocessor = PipelinePreprocessor([{"type": "TenCrop", "crop_size": 4}])
    assert preprocessor.describe().mask_transform_available is False
    assert preprocessor.transform_mask(np.zeros((6, 8), dtype=np.bool_)) is None


def test_pipeline_preprocessor_fingerprint_is_canonical_and_sensitive() -> None:
    steps = [{"type": "Resize", "scale": 8}, {"type": "CenterCrop", "crop_size": 6}]
    changed_steps = [{"type": "Resize", "scale": 9}, {"type": "CenterCrop", "crop_size": 6}]
    typed = PipelinePreprocessor(steps)
    same = PipelinePreprocessor(list(steps))
    changed = PipelinePreprocessor(changed_steps)
    assert typed.describe().fingerprint == same.describe().fingerprint
    assert typed.describe().fingerprint != changed.describe().fingerprint


def test_pipeline_preprocessor_accepts_a_custom_registry() -> None:
    from ssat.core.adapter.transform_registry import BaseTransform

    registry = default_transform_registry()

    @registry.register_module()
    class Invert(BaseTransform):
        type_name = "Invert"

        def apply_batch(self, batch: NDArray[np.uint8]) -> NDArray[np.uint8]:
            return 255 - batch

    preprocessor = PipelinePreprocessor([{"type": "Invert"}], registry=registry)
    batch = np.full((1, 1, 2, 2, 3), 10, dtype=np.uint8)
    assert np.all(preprocessor.transform_batch(batch) == 245)


def test_build_pipeline_rejects_unregistered_type_with_registered_names() -> None:
    registry = default_transform_registry()
    with pytest.raises(TransformError, match="SampleFrames"):
        build_pipeline([{"type": "Nope"}], registry)
