"""Tests for region mask generator registration and ordered dispatch."""

import numpy as np
import pytest
from numpy.random import Generator
from numpy.typing import NDArray

from ssat.core.region import (
    MaskResolutionContext,
    RegionMaskGenerator,
    RegionMaskGeneratorFactory,
    RegionResolutionError,
    RegionResolver,
    RegionSpec,
    build_mask_generators,
)
from ssat.core.region.mask_base import ExplicitMaskCache
from ssat.core.region.mask_generators import (
    ExplicitMaskGenerator,
    GridMaskGenerator,
    RandomAreaMatchMaskGenerator,
)
from ssat.core.types import RegionKind


def _context() -> MaskResolutionContext:
    """Create shared services for standalone generator tests."""

    return MaskResolutionContext(
        explicit_cache=ExplicitMaskCache(2),
        resolve_target=lambda height, width, spec: np.zeros(
            (height, width), dtype=np.bool_
        ),
    )


def _spec(kind: RegionKind = RegionKind.GRID) -> RegionSpec:
    """Create a minimal concrete region recipe."""

    params = (
        {"rows": 1, "cols": 1, "row_index": 0, "col_index": 0}
        if kind is RegionKind.GRID
        else {}
    )
    return RegionSpec(
        region_id="region",
        region_instance_id="region/instance",
        kind=kind,
        params=params,
        ref="/tmp/mask.png" if kind is RegionKind.EXPLICIT else None,
        ref_hash="a" * 64 if kind is RegionKind.EXPLICIT else None,
    )


class RecordingMaskGenerator(RegionMaskGenerator):
    """Record support and generation calls for priority tests.

    Args:
        context: Shared mask resolution services.
        name: Identifier appended to the event sink.
        supported: Configured support decision.
        events: Mutable event sink owned by the test.
        value: Boolean value returned for every mask pixel.
    """

    def __init__(
        self,
        context: MaskResolutionContext,
        name: str = "custom",
        supported: bool = True,
        events: list[str] | None = None,
        value: bool = True,
    ) -> None:
        super().__init__(context)
        self._name = name
        self._supported = supported
        self._events = events if events is not None else []
        self._value = value

    def supports(self, spec: RegionSpec) -> bool:
        """Record and return the configured support decision.

        Args:
            spec: Concrete region recipe.

        Returns:
            The decision configured by the test.
        """

        self._events.append(f"supports:{self._name}")
        return self._supported

    def get_mask(
        self,
        height: int,
        width: int,
        spec: RegionSpec,
        rng: Generator | None = None,
    ) -> NDArray[np.bool_]:
        """Record execution and return a uniform mask.

        Args:
            height: Source image height.
            width: Source image width.
            spec: Concrete region recipe.
            rng: Ignored item-local generator.

        Returns:
            Uniform boolean mask configured by the test.
        """

        self._events.append(f"get_mask:{self._name}")
        return np.full((height, width), self._value, dtype=np.bool_)


class PerFrameMaskGenerator(RegionMaskGenerator):
    """Return a caller-supplied mask, used to probe the (T, H, W) contract.

    Args:
        context: Shared mask resolution services.
        mask: Mask returned unchanged for every request.
    """

    def __init__(self, context: MaskResolutionContext, mask: NDArray[np.bool_]) -> None:
        super().__init__(context)
        self._mask = mask

    def supports(self, spec: RegionSpec) -> bool:
        """Support every recipe so it can stand in for any reserved kind."""

        return True

    def get_mask(
        self,
        height: int,
        width: int,
        spec: RegionSpec,
        rng: Generator | None = None,
    ) -> NDArray[np.bool_]:
        """Return the configured mask, ignoring the requested geometry."""

        return self._mask


class InvalidMaskGenerator(RecordingMaskGenerator):
    """Return a mask with an invalid dtype."""

    def get_mask(
        self,
        height: int,
        width: int,
        spec: RegionSpec,
        rng: Generator | None = None,
    ) -> NDArray[np.bool_]:
        """Return an intentionally invalid uint8 mask.

        Args:
            height: Source image height.
            width: Source image width.
            spec: Concrete region recipe.
            rng: Ignored item-local generator.

        Returns:
            Invalid uint8 output used by the test.
        """

        return np.ones((height, width), dtype=np.uint8)  # type: ignore[return-value]


class RaisingMaskGenerator(RecordingMaskGenerator):
    """Raise unexpectedly during mask generation."""

    def get_mask(
        self,
        height: int,
        width: int,
        spec: RegionSpec,
        rng: Generator | None = None,
    ) -> NDArray[np.bool_]:
        """Raise a representative generator failure.

        Args:
            height: Source image height.
            width: Source image width.
            spec: Concrete region recipe.
            rng: Ignored item-local generator.

        Raises:
            RuntimeError: Always, to test error translation.
        """

        raise RuntimeError("mask failed")


class RaisingMaskSupportGenerator(RecordingMaskGenerator):
    """Raise unexpectedly during mask support detection."""

    def supports(self, spec: RegionSpec) -> bool:
        """Raise a representative support-discovery failure.

        Args:
            spec: Concrete region recipe.

        Raises:
            RuntimeError: Always, to test error translation.
        """

        raise RuntimeError("mask support failed")


def test_region_mask_generator_contract_is_abstract() -> None:
    """The mask generator contract cannot be instantiated directly."""

    with pytest.raises(TypeError):
        RegionMaskGenerator(_context())


@pytest.mark.parametrize(
    ("generator_type", "kind"),
    [
        (GridMaskGenerator, RegionKind.GRID),
        (ExplicitMaskGenerator, RegionKind.EXPLICIT),
        (RandomAreaMatchMaskGenerator, RegionKind.RANDOM_AREA_MATCH),
    ],
)
def test_builtin_mask_generators_support_exact_kind(
    generator_type: type[RegionMaskGenerator],
    kind: RegionKind,
) -> None:
    """Every built-in generator advertises only its concrete kind."""

    generator = generator_type(_context())
    assert generator.supports(_spec(kind))
    assert not generator.supports(_spec(RegionKind.SKELETON_PARTS))


def test_mask_factory_builds_fresh_generators_in_stable_order() -> None:
    """Default mask builds preserve order without sharing instances."""

    first = build_mask_generators(_context())
    second = build_mask_generators(_context())
    expected = [
        GridMaskGenerator,
        ExplicitMaskGenerator,
        RandomAreaMatchMaskGenerator,
    ]
    assert [type(generator) for generator in first] == expected
    assert [type(generator) for generator in second] == expected
    assert all(left is not right for left, right in zip(first, second, strict=True))


def test_mask_factory_validates_registration() -> None:
    """Mask factories reject invalid and duplicate generator classes."""

    factory = RegionMaskGeneratorFactory((RecordingMaskGenerator,))
    assert isinstance(factory.build(_context())[0], RecordingMaskGenerator)
    with pytest.raises(ValueError, match="already registered"):
        factory.register(RecordingMaskGenerator)
    with pytest.raises(TypeError, match="subclass"):
        factory.register(object)  # type: ignore[arg-type]


def test_mask_dispatch_stops_after_first_supporting_generator() -> None:
    """Generator registration order defines mask override priority."""

    events: list[str] = []
    context = _context()
    generators = (
        RecordingMaskGenerator(context, "skip", False, events, False),
        RecordingMaskGenerator(context, "first", True, events, True),
        RecordingMaskGenerator(context, "later", True, events, False),
    )
    mask, _ = RegionResolver(mask_generators=generators).resolve(
        (1, 2, 3, 1), _spec()
    )

    assert np.all(mask)
    assert events == ["supports:skip", "supports:first", "get_mask:first"]


def test_mask_facade_rejects_invalid_output_and_preserves_errors() -> None:
    """The resolver validates custom masks and retains execution causes."""

    context = _context()
    with pytest.raises(RegionResolutionError, match="invalid output"):
        RegionResolver(
            mask_generators=(InvalidMaskGenerator(context),)
        ).resolve((1, 2, 2, 1), _spec())

    with pytest.raises(RegionResolutionError, match="execution failed") as captured:
        RegionResolver(
            mask_generators=(RaisingMaskGenerator(context),)
        ).resolve((1, 2, 2, 1), _spec())
    assert isinstance(captured.value.__cause__, RuntimeError)

    with pytest.raises(RegionResolutionError, match="support check failed") as captured:
        RegionResolver(
            mask_generators=(RaisingMaskSupportGenerator(context),)
        ).resolve((1, 2, 2, 1), _spec())
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_mask_facade_rejects_empty_and_invalid_generator_lists() -> None:
    """The resolver rejects unusable custom generator collections."""

    with pytest.raises(ValueError, match="must not be empty"):
        RegionResolver(mask_generators=())
    with pytest.raises(TypeError, match="RegionMaskGenerator"):
        RegionResolver(mask_generators=(object(),))  # type: ignore[arg-type]


def test_resolver_accepts_and_measures_per_frame_masks() -> None:
    """A (T, H, W) mask is validated and reduced to a mean per-frame area."""

    context = _context()
    mask = np.zeros((3, 2, 2), dtype=np.bool_)
    mask[0] = True  # 4 px
    mask[1, 0, :] = True  # 2 px
    # frame 2 stays empty -> mean = (4 + 2 + 0) / 3 = 2
    resolver = RegionResolver(mask_generators=(PerFrameMaskGenerator(context, mask),))

    resolved, meta = resolver.resolve((3, 2, 2, 1), _spec())

    assert resolved.shape == (3, 2, 2)
    assert meta.intended_area_px == 2
    assert meta.intended_area_ratio == 2 / 4


def test_resolver_rejects_per_frame_mask_with_wrong_frame_count() -> None:
    """A (T, H, W) mask must match the source sample's frame count."""

    context = _context()
    mask = np.zeros((2, 2, 2), dtype=np.bool_)
    resolver = RegionResolver(mask_generators=(PerFrameMaskGenerator(context, mask),))

    with pytest.raises(RegionResolutionError, match="frame count"):
        resolver.resolve((3, 2, 2, 1), _spec())


def test_resolver_rejects_invalid_mask_rank() -> None:
    """Masks outside (H, W)/(T, H, W) are rejected as invalid output."""

    context = _context()
    mask = np.zeros((2,), dtype=np.bool_)
    resolver = RegionResolver(mask_generators=(PerFrameMaskGenerator(context, mask),))

    with pytest.raises(RegionResolutionError, match="invalid output"):
        resolver.resolve((3, 2, 2, 1), _spec())


def test_random_area_match_rejects_per_frame_targets() -> None:
    """Embedded targets that resolve to (T, H, W) are not supported yet."""

    resolver = RegionResolver()
    per_frame_mask = np.ones((3, 2, 2), dtype=np.bool_)
    bound_context = MaskResolutionContext(
        explicit_cache=ExplicitMaskCache(1),
        resolve_target=resolver._resolve_target,
    )
    # Append a permissive generator behind the resolver's real dispatch chain
    # so the embedded SKELETON_PARTS target resolves to a (T, H, W) mask.
    resolver._mask_generators = resolver._mask_generators + (
        PerFrameMaskGenerator(bound_context, per_frame_mask),
    )
    target = RegionSpec(
        region_id="skeleton",
        region_instance_id="skeleton/torso",
        kind=RegionKind.SKELETON_PARTS,
    )
    control = RegionSpec(
        region_id="control:skeleton:0",
        region_instance_id="control:skeleton/torso:0:0",
        kind=RegionKind.RANDOM_AREA_MATCH,
        params={
            "target_region": {
                "region_id": target.region_id,
                "region_instance_id": target.region_instance_id,
                "kind": target.kind.value,
                "params": {},
                "ref": None,
                "ref_hash": None,
            },
            "control_request_index": 0,
            "control_index": 0,
        },
    )

    with pytest.raises(RegionResolutionError, match="per-frame"):
        resolver.resolve((3, 2, 2, 1), control, np.random.default_rng(0))
