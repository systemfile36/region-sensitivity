"""Tests for family expander registration and ordered dispatch."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from ssat.core.config.schema import ResolvedRegionConfig
from ssat.core.plan import (
    RegionExpander,
    RegionExpansionContext,
    RegionExpansionError,
    RegionFamilyExpander,
    RegionFamilyExpanderFactory,
    build_family_expanders,
)
from ssat.core.plan.region_expanders import (
    ExplicitRegionExpander,
    GridRegionExpander,
    SampleDependentRegionExpander,
)
from ssat.core.region.types import RegionSpec
from ssat.core.source.types import SampleMeta
from ssat.core.types import RegionKind


def _sample() -> SampleMeta:
    """Create lightweight source metadata for expansion tests."""

    return SampleMeta("sample", Path("sample.png"))


def _family(kind: RegionKind = RegionKind.GRID) -> ResolvedRegionConfig:
    """Create a minimal resolved family recipe."""

    params = {"rows": 1, "cols": 1} if kind is RegionKind.GRID else {}
    return ResolvedRegionConfig(
        region_id="family",
        kind=kind,
        params=params,
        ref=Path("/tmp/mask.png") if kind is RegionKind.EXPLICIT else None,
        ref_hash="a" * 64 if kind is RegionKind.EXPLICIT else None,
    )


class RecordingFamilyExpander(RegionFamilyExpander):
    """Record support and expansion calls for priority tests.

    Args:
        context: Shared region expansion services.
        name: Identifier appended to the event sink.
        supported: Configured support decision.
        events: Mutable event sink owned by the test.
    """

    def __init__(
        self,
        context: RegionExpansionContext,
        name: str = "custom",
        supported: bool = True,
        events: list[str] | None = None,
    ) -> None:
        super().__init__(context)
        self._name = name
        self._supported = supported
        self._events = events if events is not None else []

    def supports(self, family: ResolvedRegionConfig) -> bool:
        """Record and return the configured support decision.

        Args:
            family: Resolved region-family recipe.

        Returns:
            The decision configured by the test.
        """

        self._events.append(f"supports:{self._name}")
        return self._supported

    def expand(
        self,
        sample: SampleMeta,
        family: ResolvedRegionConfig,
    ) -> Sequence[RegionSpec]:
        """Record execution and return one valid concrete region.

        Args:
            sample: Lightweight source metadata.
            family: Resolved region-family recipe.

        Returns:
            One concrete region belonging to ``family``.
        """

        self._events.append(f"expand:{self._name}")
        return (
            RegionSpec(
                region_id=family.region_id,
                region_instance_id=f"{family.region_id}/{self._name}",
                kind=family.kind,
                params=family.params,
            ),
        )


class InvalidFamilyExpander(RecordingFamilyExpander):
    """Return a value outside the concrete region contract."""

    def expand(
        self,
        sample: SampleMeta,
        family: ResolvedRegionConfig,
    ) -> Sequence[RegionSpec]:
        """Return an intentionally invalid expansion value.

        Args:
            sample: Lightweight source metadata.
            family: Resolved region-family recipe.

        Returns:
            An invalid sequence used by the test.
        """

        return (object(),)  # type: ignore[return-value]


class RaisingFamilyExpander(RecordingFamilyExpander):
    """Raise unexpectedly during family expansion."""

    def expand(
        self,
        sample: SampleMeta,
        family: ResolvedRegionConfig,
    ) -> Sequence[RegionSpec]:
        """Raise a representative expansion failure.

        Args:
            sample: Lightweight source metadata.
            family: Resolved region-family recipe.

        Raises:
            RuntimeError: Always, to test error translation.
        """

        raise RuntimeError("expansion failed")


class RaisingFamilySupportExpander(RecordingFamilyExpander):
    """Raise unexpectedly during family support detection."""

    def supports(self, family: ResolvedRegionConfig) -> bool:
        """Raise a representative support-discovery failure.

        Args:
            family: Resolved region-family recipe.

        Raises:
            RuntimeError: Always, to test error translation.
        """

        raise RuntimeError("expansion support failed")


def test_region_family_expander_contract_is_abstract() -> None:
    """The family expander contract cannot be instantiated directly."""

    with pytest.raises(TypeError):
        RegionFamilyExpander(RegionExpansionContext())


@pytest.mark.parametrize(
    ("expander_type", "kind"),
    [
        (GridRegionExpander, RegionKind.GRID),
        (ExplicitRegionExpander, RegionKind.EXPLICIT),
        (SampleDependentRegionExpander, RegionKind.SKELETON_PARTS),
    ],
)
def test_builtin_family_expanders_support_expected_kinds(
    expander_type: type[RegionFamilyExpander],
    kind: RegionKind,
) -> None:
    """Built-in expanders advertise only their family category."""

    expander = expander_type(RegionExpansionContext())
    assert expander.supports(_family(kind))
    assert not expander.supports(_family(RegionKind.RANDOM_AREA_MATCH))


def test_family_factory_builds_fresh_expanders_in_stable_order() -> None:
    """Default family builds preserve order without shared instances."""

    first = build_family_expanders(RegionExpansionContext())
    second = build_family_expanders(RegionExpansionContext())
    expected = [
        GridRegionExpander,
        ExplicitRegionExpander,
        SampleDependentRegionExpander,
    ]
    assert [type(expander) for expander in first] == expected
    assert [type(expander) for expander in second] == expected
    assert all(left is not right for left, right in zip(first, second, strict=True))


def test_family_factory_validates_registration() -> None:
    """Family factories reject invalid and duplicate expander classes."""

    factory = RegionFamilyExpanderFactory((RecordingFamilyExpander,))
    assert isinstance(
        factory.build(RegionExpansionContext())[0],
        RecordingFamilyExpander,
    )
    with pytest.raises(ValueError, match="already registered"):
        factory.register(RecordingFamilyExpander)
    with pytest.raises(TypeError, match="subclass"):
        factory.register(object)  # type: ignore[arg-type]


def test_family_dispatch_stops_after_first_supporting_expander() -> None:
    """Expander registration order defines planning override priority."""

    events: list[str] = []
    context = RegionExpansionContext()
    expanders = (
        RecordingFamilyExpander(context, "skip", False, events),
        RecordingFamilyExpander(context, "first", True, events),
        RecordingFamilyExpander(context, "later", True, events),
    )
    result = RegionExpander(family_expanders=expanders).expand(
        _sample(), _family()
    )

    assert result[0].region_instance_id == "family/first"
    assert events == ["supports:skip", "supports:first", "expand:first"]


def test_family_facade_rejects_invalid_output_and_collections() -> None:
    """RegionExpander validates strategy output and construction inputs."""

    context = RegionExpansionContext()
    with pytest.raises(RegionExpansionError, match="RegionSpec"):
        RegionExpander(
            family_expanders=(InvalidFamilyExpander(context),)
        ).expand(_sample(), _family())
    with pytest.raises(ValueError, match="must not be empty"):
        RegionExpander(family_expanders=())
    with pytest.raises(TypeError, match="RegionFamilyExpander"):
        RegionExpander(family_expanders=(object(),))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("expander", "message"),
    [
        (RaisingFamilySupportExpander(RegionExpansionContext()), "support check"),
        (RaisingFamilyExpander(RegionExpansionContext()), "execution failed"),
    ],
)
def test_family_dispatch_preserves_unexpected_error_causes(
    expander: RegionFamilyExpander,
    message: str,
) -> None:
    """Family dispatch exposes one error type while retaining its cause."""

    with pytest.raises(RegionExpansionError, match=message) as captured:
        RegionExpander(family_expanders=(expander,)).expand(
            _sample(), _family()
        )
    assert isinstance(captured.value.__cause__, RuntimeError)
