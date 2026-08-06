"""Unit tests for deterministic region-family expansion."""

from pathlib import Path

import pytest

from ssat.core.config.schema import ResolvedRegionConfig
from ssat.core.plan.region_expander import RegionExpander, RegionExpansionError
from ssat.core.region.types import RegionSpec
from ssat.core.source.types import SampleMeta
from ssat.core.types import RegionKind


def _sample() -> SampleMeta:
    return SampleMeta(sample_id="sample", path=Path("sample.png"))


@pytest.mark.parametrize(
    ("rows", "cols", "expected_ids"),
    [
        (1, 1, ("grid/r0/c0",)),
        (2, 2, ("grid/r0/c0", "grid/r0/c1", "grid/r1/c0", "grid/r1/c1")),
        (
            2,
            3,
            (
                "grid/r0/c0",
                "grid/r0/c1",
                "grid/r0/c2",
                "grid/r1/c0",
                "grid/r1/c1",
                "grid/r1/c2",
            ),
        ),
    ],
)
def test_grid_expands_in_row_major_order(
    rows: int,
    cols: int,
    expected_ids: tuple[str, ...],
) -> None:
    family = ResolvedRegionConfig(
        region_id="grid",
        kind=RegionKind.GRID,
        params={"rows": rows, "cols": cols},
    )

    instances = RegionExpander().expand(_sample(), family)

    assert tuple(instance.region_instance_id for instance in instances) == expected_ids
    assert all(instance.region_id == "grid" for instance in instances)
    assert instances[-1].params == {
        "rows": rows,
        "cols": cols,
        "row_index": rows - 1,
        "col_index": cols - 1,
    }


def test_explicit_family_expands_to_one_instance(tmp_path: Path) -> None:
    ref = (tmp_path / "mask.png").resolve()
    family = ResolvedRegionConfig(
        region_id="mask",
        kind=RegionKind.EXPLICIT,
        params={"threshold": 1},
        ref=ref,
        ref_hash="a" * 64,
    )

    instances = RegionExpander().expand(_sample(), family)

    assert len(instances) == 1
    assert instances[0].region_instance_id == "mask"
    assert instances[0].ref == ref.as_posix()
    assert instances[0].ref_hash == "a" * 64


@pytest.mark.parametrize(
    "kind",
    [RegionKind.BBOX_PARTITION, RegionKind.SKELETON_PARTS, RegionKind.GT_BBOX],
)
def test_sample_dependent_kind_requires_provider(kind: RegionKind) -> None:
    family = ResolvedRegionConfig(region_id="future", kind=kind)

    with pytest.raises(RegionExpansionError, match="SampleRegionProvider"):
        RegionExpander().expand(_sample(), family)


def test_sample_region_provider_can_supply_future_instances() -> None:
    class Provider:
        """Return one representative provider-backed region."""

        def expand(
            self,
            sample: SampleMeta,
            family: ResolvedRegionConfig,
        ) -> tuple[RegionSpec, ...]:
            """Create one stable sample-dependent region."""

            return (
                RegionSpec(
                    region_id=family.region_id,
                    region_instance_id=f"{family.region_id}/{sample.sample_id}/person0",
                    kind=family.kind,
                    params={"person_index": 0},
                ),
            )

    family = ResolvedRegionConfig(
        region_id="people",
        kind=RegionKind.SKELETON_PARTS,
    )

    instances = RegionExpander(Provider()).expand(_sample(), family)

    assert instances[0].region_instance_id == "people/sample/person0"
