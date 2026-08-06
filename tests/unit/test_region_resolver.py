"""Tests for concrete region mask materialization."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ssat.core.region import RegionResolutionError, RegionResolver, RegionSpec
from ssat.core.types import RegionKind, thaw_json_value
from ssat.utils.io import sha256_file


def _grid(
    rows: int,
    cols: int,
    row_index: int,
    col_index: int,
) -> RegionSpec:
    """Create a concrete grid cell recipe for tests."""

    return RegionSpec(
        region_id="grid",
        region_instance_id=f"grid/r{row_index}/c{col_index}",
        kind=RegionKind.GRID,
        params={
            "rows": rows,
            "cols": cols,
            "row_index": row_index,
            "col_index": col_index,
        },
    )


def _explicit(path: Path, digest: str | None = None) -> RegionSpec:
    """Create an explicit region recipe for tests."""

    return RegionSpec(
        region_id="mask",
        region_instance_id="mask",
        kind=RegionKind.EXPLICIT,
        ref=path.as_posix(),
        ref_hash=digest or sha256_file(path),
    )


def _control(target: RegionSpec) -> RegionSpec:
    """Embed a concrete target recipe in a random control region."""

    return RegionSpec(
        region_id="control:grid:0",
        region_instance_id=f"control:{target.region_instance_id}:0:0",
        kind=RegionKind.RANDOM_AREA_MATCH,
        params={
            "target_region": {
                "region_id": target.region_id,
                "region_instance_id": target.region_instance_id,
                "kind": target.kind.value,
                "params": thaw_json_value(target.params),
                "ref": target.ref,
                "ref_hash": target.ref_hash,
            },
            "control_request_index": 0,
            "control_index": 0,
        },
    )


def test_grid_cells_use_integer_boundaries_and_cover_image() -> None:
    """Non-square grid cells cover every pixel once in row-major recipes."""

    resolver = RegionResolver()
    masks = [
        resolver.resolve((1, 5, 7, 3), _grid(2, 3, row, col))[0]
        for row in range(2)
        for col in range(3)
    ]

    coverage = np.sum(np.stack(masks), axis=0)
    assert np.all(coverage == 1)
    assert np.array_equal(
        np.argwhere(masks[1]),
        np.array([[0, 2], [0, 3], [1, 2], [1, 3]]),
    )

    _, meta = resolver.resolve((1, 5, 7, 3), _grid(2, 3, 1, 2))
    assert meta.intended_area_px == 9
    assert meta.intended_area_ratio == 9 / 35
    assert meta.generator_kind == "grid"
    assert meta.generator_version == "1.0.0"


def test_grid_supports_more_cells_than_source_pixels() -> None:
    """Tiny images remain exactly covered even when some cells are empty."""

    resolver = RegionResolver()
    masks = [
        resolver.resolve((1, 2, 2, 3), _grid(3, 3, row, col))[0]
        for row in range(3)
        for col in range(3)
    ]
    assert np.all(np.sum(np.stack(masks), axis=0) == 1)
    assert sum(int(mask.any()) for mask in masks) == 4


def test_explicit_mask_is_verified_decoded_and_cache_safe(tmp_path: Path) -> None:
    """Explicit masks use nonzero pixels and return cache-isolated arrays."""

    path = tmp_path / "mask.png"
    pixels = np.array([[0, 2, 0], [255, 0, 7]], dtype=np.uint8)
    Image.fromarray(pixels, mode="L").save(path)
    resolver = RegionResolver(explicit_cache_size=1)
    spec = _explicit(path)

    first, meta = resolver.resolve((1, 2, 3, 3), spec)
    first[:] = False
    second, _ = resolver.resolve((1, 2, 3, 3), spec)

    assert np.array_equal(second, pixels != 0)
    assert meta.intended_area_px == 3
    assert first is not second


def test_explicit_mask_rejects_hash_shape_and_multichannel(tmp_path: Path) -> None:
    """Explicit masks fail instead of being resized or silently converted."""

    grayscale = tmp_path / "gray.png"
    Image.fromarray(np.ones((2, 3), dtype=np.uint8), mode="L").save(grayscale)
    rgb = tmp_path / "rgb.png"
    Image.fromarray(np.ones((2, 3, 3), dtype=np.uint8), mode="RGB").save(rgb)
    resolver = RegionResolver()

    with pytest.raises(RegionResolutionError, match="ref_hash mismatch"):
        resolver.resolve((1, 2, 3, 3), _explicit(grayscale, "a" * 64))
    with pytest.raises(RegionResolutionError, match="does not match"):
        resolver.resolve((1, 3, 3, 3), _explicit(grayscale))
    with pytest.raises(RegionResolutionError, match="single-channel"):
        resolver.resolve((1, 2, 3, 3), _explicit(rgb))


def test_random_area_match_is_exact_and_deterministic() -> None:
    """Random controls uniformly select exactly the concrete target area."""

    resolver = RegionResolver()
    control = _control(_grid(2, 2, 0, 0))

    first, first_meta = resolver.resolve(
        (1, 5, 7, 3), control, np.random.default_rng(42)
    )
    repeated, _ = resolver.resolve(
        (1, 5, 7, 3), control, np.random.default_rng(42)
    )
    different, _ = resolver.resolve(
        (1, 5, 7, 3), control, np.random.default_rng(43)
    )

    assert first.sum() == 6
    assert first_meta.intended_area_px == 6
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, different)


def test_random_area_match_resolves_embedded_explicit_target(tmp_path: Path) -> None:
    """A control can reconstruct an explicit target without a registry."""

    path = tmp_path / "target.png"
    pixels = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.uint8)
    Image.fromarray(pixels, mode="L").save(path)
    resolver = RegionResolver()

    mask, meta = resolver.resolve(
        (1, 2, 3, 3),
        _control(_explicit(path)),
        np.random.default_rng(5),
    )

    assert mask.sum() == 3
    assert meta.intended_area_px == 3


def test_random_area_match_validates_rng_and_rejects_nested_target() -> None:
    """Control recipes require item-local RNG and cannot recursively nest."""

    resolver = RegionResolver()
    control = _control(_grid(1, 1, 0, 0))
    with pytest.raises(RegionResolutionError, match="requires a numpy Generator"):
        resolver.resolve((1, 3, 3, 3), control)

    nested = _control(control)
    with pytest.raises(RegionResolutionError, match="nested"):
        resolver.resolve((1, 3, 3, 3), nested, np.random.default_rng(1))


@pytest.mark.parametrize(
    "shape",
    [(3, 3), (1, 0, 3, 3), (True, 2, 3, 3)],
)
def test_invalid_shapes_are_rejected(shape: tuple[int, ...]) -> None:
    """Region resolution accepts only positive THWC shapes."""

    with pytest.raises(RegionResolutionError, match="shape"):
        RegionResolver().resolve(shape, _grid(1, 1, 0, 0))  # type: ignore[arg-type]


def test_reserved_region_kind_is_not_materialized() -> None:
    """Future annotation kinds fail explicitly until generators exist."""

    spec = RegionSpec(
        region_id="people",
        region_instance_id="people/torso",
        kind=RegionKind.SKELETON_PARTS,
    )
    with pytest.raises(RegionResolutionError, match="not implemented"):
        RegionResolver().resolve((1, 3, 3, 3), spec)
