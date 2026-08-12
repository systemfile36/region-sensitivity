"""Tests for deterministic mask-aware perturbation operations."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from ssat.core.perturb import PerturbationError, Perturbator, derive
from ssat.core.types import PerturbationOp


def _array() -> np.ndarray:
    """Create a non-uniform RGB source array."""

    return np.arange(1 * 6 * 7 * 3, dtype=np.uint8).reshape(1, 6, 7, 3)


def _mask() -> np.ndarray:
    """Create a nontrivial source-space test mask."""

    mask = np.zeros((6, 7), dtype=np.bool_)
    mask[1:5, 2:6] = True
    return mask


def _two_frame_array() -> np.ndarray:
    """Create a two-frame non-uniform RGB source array."""

    return np.arange(2 * 6 * 7 * 3, dtype=np.uint8).reshape(2, 6, 7, 3)


def _per_frame_mask() -> np.ndarray:
    """Create a (T, H, W) mask selecting a different region per frame."""

    mask = np.zeros((2, 6, 7), dtype=np.bool_)
    mask[0, 1:5, 2:6] = True
    mask[1, 0:2, 0:2] = True
    return mask


@pytest.mark.parametrize(
    ("op", "params", "seed"),
    [
        (PerturbationOp.CONSTANT_FILL, {"value": 17}, None),
        (PerturbationOp.MEAN_FILL, {"value": [10.2, 20.5, 30.8]}, None),
        (PerturbationOp.BLUR, {"sigma": 1.0}, None),
        (PerturbationOp.GAUSSIAN_NOISE, {"sigma": 20.0}, 3),
        (PerturbationOp.PATCH_SHUFFLE, {"patch_size": 2}, 0),
    ],
)
def test_operations_preserve_pixels_outside_mask_and_inputs(
    op: PerturbationOp,
    params: Mapping[str, Any],
    seed: int | None,
) -> None:
    """Every operation returns a copy and composites only selected pixels."""

    array = _array()
    original = array.copy()
    mask = _mask()
    original_mask = mask.copy()
    rng = np.random.default_rng(seed) if seed is not None else None

    result = Perturbator().apply(array, mask, op, params, rng)

    assert result is not array
    assert result.dtype == np.uint8
    assert np.array_equal(result[:, ~mask, :], array[:, ~mask, :])
    assert np.array_equal(array, original)
    assert np.array_equal(mask, original_mask)


@pytest.mark.parametrize(
    ("op", "params", "seed"),
    [
        (PerturbationOp.CONSTANT_FILL, {"value": 17}, None),
        (PerturbationOp.MEAN_FILL, {"value": [10.2, 20.5, 30.8]}, None),
        (PerturbationOp.BLUR, {"sigma": 1.0}, None),
        (PerturbationOp.GAUSSIAN_NOISE, {"sigma": 20.0}, 3),
        (PerturbationOp.PATCH_SHUFFLE, {"patch_size": 2}, 0),
    ],
)
def test_operations_support_per_frame_masks(
    op: PerturbationOp,
    params: Mapping[str, Any],
    seed: int | None,
) -> None:
    """A (T, H, W) mask selects an independent region in each frame."""

    array = _two_frame_array()
    original = array.copy()
    mask = _per_frame_mask()
    rng = np.random.default_rng(seed) if seed is not None else None

    result = Perturbator().apply(array, mask, op, params, rng)

    assert result is not array
    assert result.dtype == np.uint8
    assert np.array_equal(result[0][~mask[0]], array[0][~mask[0]])
    assert np.array_equal(result[1][~mask[1]], array[1][~mask[1]])
    assert np.array_equal(array, original)


def test_per_frame_fill_applies_only_within_each_frames_mask() -> None:
    """(T, H, W) masks fill exactly the selected region in each frame."""

    array = np.zeros((2, 3, 3, 3), dtype=np.uint8)
    mask = np.zeros((2, 3, 3), dtype=np.bool_)
    mask[0, 0, :] = True
    mask[1, :, 0] = True

    result = Perturbator().apply(array, mask, PerturbationOp.CONSTANT_FILL, {"value": 9})

    assert np.all(result[0, 0, :, :] == 9)
    assert np.all(result[0, 1:, :, :] == 0)
    assert np.all(result[1, :, 0, :] == 9)
    assert np.all(result[1, :, 1:, :] == 0)


def test_per_frame_mask_frame_count_must_match_array() -> None:
    """A (T, H, W) mask with a mismatched frame count is rejected."""

    array = _array()  # T = 1
    mask = np.zeros((2, 6, 7), dtype=np.bool_)
    with pytest.raises(PerturbationError, match="matching"):
        Perturbator().apply(array, mask, PerturbationOp.CONSTANT_FILL, {"value": 0})


def test_fill_rounds_scalar_and_channel_values() -> None:
    """Fill operations broadcast scalars and round per-channel values."""

    array = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    mask = np.array([[True, False], [False, True]], dtype=np.bool_)
    perturbator = Perturbator()

    scalar = perturbator.apply(
        array, mask, PerturbationOp.CONSTANT_FILL, {"value": 12.6}
    )
    channels = perturbator.apply(
        array,
        mask,
        PerturbationOp.MEAN_FILL,
        {"value": [10.2, 20.5, 30.8]},
    )

    assert np.all(scalar[:, mask, :] == 13)
    assert np.all(channels[:, mask, :] == np.array([10, 20, 31]))


def test_complement_mask_changes_the_opposite_region() -> None:
    """Caller-side inversion naturally selects the exact complementary area."""

    array = _array()
    mask = _mask()
    perturbator = Perturbator()
    inside = perturbator.apply(
        array, mask, PerturbationOp.CONSTANT_FILL, {"value": 0}
    )
    outside = perturbator.apply(
        array, ~mask, PerturbationOp.CONSTANT_FILL, {"value": 0}
    )

    assert np.array_equal(inside[:, ~mask, :], array[:, ~mask, :])
    assert np.array_equal(outside[:, mask, :], array[:, mask, :])
    assert np.all(inside[:, mask, :] == 0)
    assert np.all(outside[:, ~mask, :] == 0)


def test_gaussian_noise_is_deterministic_and_clipped() -> None:
    """Noise uses only the supplied generator and remains in uint8 range."""

    array = np.full((1, 4, 4, 3), 250, dtype=np.uint8)
    mask = np.ones((4, 4), dtype=np.bool_)
    perturbator = Perturbator()
    first = perturbator.apply(
        array,
        mask,
        PerturbationOp.GAUSSIAN_NOISE,
        {"sigma": 100.0},
        np.random.default_rng(7),
    )
    repeated = perturbator.apply(
        array,
        mask,
        PerturbationOp.GAUSSIAN_NOISE,
        {"sigma": 100.0},
        np.random.default_rng(7),
    )
    different = perturbator.apply(
        array,
        mask,
        PerturbationOp.GAUSSIAN_NOISE,
        {"sigma": 100.0},
        np.random.default_rng(8),
    )

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, different)
    assert first.min() >= 0
    assert first.max() <= 255
    assert np.any(first == 255)


def test_patch_shuffle_moves_tiles_and_preserves_partial_edges() -> None:
    """Tile permutation is shared across frames and excludes partial edges."""

    first_frame = np.zeros((5, 5, 1), dtype=np.uint8)
    first_frame[0:2, 0:2] = 10
    first_frame[0:2, 2:4] = 20
    first_frame[2:4, 0:2] = 30
    first_frame[2:4, 2:4] = 40
    first_frame[4, :] = 91
    first_frame[:, 4] = 92
    array = np.stack((first_frame, first_frame + 100), axis=0)
    mask = np.ones((5, 5), dtype=np.bool_)

    result = Perturbator().apply(
        array,
        mask,
        PerturbationOp.PATCH_SHUFFLE,
        {"patch_size": 2},
        np.random.default_rng(0),
    )

    assert [int(result[0, row, col, 0]) for row, col in ((0, 0), (0, 2), (2, 0), (2, 2))] == [30, 10, 20, 40]
    assert np.array_equal(result[:, 4, :, :], array[:, 4, :, :])
    assert np.array_equal(result[:, :, 4, :], array[:, :, 4, :])
    assert np.all(result[1, :4, :4, 0] - result[0, :4, :4, 0] == 100)


def test_patch_shuffle_is_noop_when_no_tile_can_move() -> None:
    """A patch larger than the image produces a safe copied no-op result."""

    array = _array()
    result = Perturbator().apply(
        array,
        _mask(),
        PerturbationOp.PATCH_SHUFFLE,
        {"patch_size": 20},
        np.random.default_rng(0),
    )

    assert result is not array
    assert np.array_equal(result, array)


@pytest.mark.parametrize(
    ("op", "params", "error"),
    [
        (PerturbationOp.CONSTANT_FILL, {}, "exactly"),
        (PerturbationOp.CONSTANT_FILL, {"value": [1, 2]}, "3 channels"),
        (PerturbationOp.MEAN_FILL, {"value": True}, "finite values"),
        (PerturbationOp.BLUR, {"sigma": 0}, "positive"),
        (PerturbationOp.GAUSSIAN_NOISE, {"sigma": 1, "extra": 2}, "exactly"),
        (PerturbationOp.PATCH_SHUFFLE, {"patch_size": True}, "positive integer"),
    ],
)
def test_operation_params_are_revalidated(
    op: PerturbationOp,
    params: Mapping[str, Any],
    error: str,
) -> None:
    """Runtime validation rejects missing, extra, mistyped, and ranged params."""

    rng = np.random.default_rng(1)
    with pytest.raises(PerturbationError, match=error):
        Perturbator().apply(_array(), _mask(), op, params, rng)


@pytest.mark.parametrize(
    "op, params",
    [
        (PerturbationOp.GAUSSIAN_NOISE, {"sigma": 1.0}),
        (PerturbationOp.PATCH_SHUFFLE, {"patch_size": 2}),
    ],
)
def test_stochastic_operations_require_generator(
    op: PerturbationOp,
    params: Mapping[str, Any],
) -> None:
    """Stochastic operations cannot fall back to global random state."""

    with pytest.raises(PerturbationError, match="requires a numpy Generator"):
        Perturbator().apply(_array(), _mask(), op, params)


def test_array_and_mask_contracts_are_enforced() -> None:
    """Perturbations reject invalid dtype, layout, and mask alignment."""

    array = _array()
    perturbator = Perturbator()
    with pytest.raises(PerturbationError, match="uint8"):
        perturbator.apply(
            array.astype(np.float32),
            _mask(),
            PerturbationOp.CONSTANT_FILL,
            {"value": 0},
        )
    with pytest.raises(PerturbationError, match="matching"):
        perturbator.apply(
            array,
            np.ones((2, 2), dtype=np.bool_),
            PerturbationOp.CONSTANT_FILL,
            {"value": 0},
        )


def test_derive_has_stable_regression_value_and_item_sensitivity() -> None:
    """Seed derivation is stable and sensitive to every identity component."""

    expected = 239766286650576752744321539599534570936
    assert derive(17, "a" * 64, 3) == expected
    assert derive(17, "a" * 64, 3) == expected
    assert derive(18, "a" * 64, 3) != expected
    assert derive(17, "b" * 64, 3) != expected
    assert derive(17, "a" * 64, 4) != expected


@pytest.mark.parametrize(
    "args",
    [(-1, "a" * 64, 0), (0, "invalid", 0), (0, "A" * 64, 0), (0, "a" * 64, -1)],
)
def test_derive_rejects_invalid_identity_inputs(args: tuple[Any, ...]) -> None:
    """Seed derivation validates its complete public identity contract."""

    with pytest.raises(ValueError):
        derive(*args)


def test_perturbations_do_not_change_numpy_global_rng_state() -> None:
    """Local generators leave NumPy's legacy global RNG state untouched."""

    np.random.seed(1234)
    before = np.random.get_state()
    perturbator = Perturbator()
    perturbator.apply(
        _array(),
        _mask(),
        PerturbationOp.GAUSSIAN_NOISE,
        {"sigma": 5.0},
        np.random.default_rng(derive(1, "a" * 64, 0)),
    )
    perturbator.apply(
        _array(),
        _mask(),
        PerturbationOp.PATCH_SHUFFLE,
        {"patch_size": 2},
        np.random.default_rng(derive(1, "b" * 64, 0)),
    )
    after = np.random.get_state()

    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
