"""Unit tests for deterministic plan construction and materialization."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from ssat.core.adapter.types import AdapterSpec
from ssat.core.config.schema import (
    ControlConfig,
    DumpConfig,
    PerturbationConfig,
    ResolvedConfig,
    ResolvedRegionConfig,
    RuntimeConfig,
)
from ssat.core.plan import PlanBuildError, PlanBuilder
from ssat.core.plan.types import WorkItem
from ssat.core.source.types import SampleMeta
from ssat.core.types import PerturbationOp, RegionKind


class FakeSampleSource:
    """Sample source that records metadata and pixel loading calls."""

    def __init__(self, samples: Sequence[SampleMeta]) -> None:
        self.samples = samples
        self.list_calls = 0
        self.load_calls: list[str] = []

    def list_samples(self) -> Sequence[SampleMeta]:
        """Return the configured metadata sequence."""
        self.list_calls += 1
        return self.samples

    def load(self, sample_id: str) -> Any:
        """Fail if PlanBuilder attempts to load sample pixels."""
        self.load_calls.append(sample_id)
        raise AssertionError("PlanBuilder must not load sample pixels")


class FailingSampleSource(FakeSampleSource):
    """Sample source whose metadata listing always fails."""

    def list_samples(self) -> Sequence[SampleMeta]:
        """Raise a representative source failure."""
        self.list_calls += 1
        raise RuntimeError("catalog unavailable")


def _sample(sample_id: str) -> SampleMeta:
    return SampleMeta(sample_id=sample_id, path=f"data/{sample_id}.npy")


def _resolved_config(
    tmp_path: Path,
    *,
    variants_per_chunk: int = 10,
    regions: tuple[ResolvedRegionConfig, ...] | None = None,
    perturbations: tuple[PerturbationConfig, ...] | None = None,
    controls: tuple[ControlConfig, ...] = (),
) -> ResolvedConfig:
    return ResolvedConfig(
        config_base_dir=tmp_path.resolve(),
        regions=regions
        or (
            ResolvedRegionConfig(
                region_id="region-a",
                kind=RegionKind.GRID,
                params={"rows": 2, "cols": 2},
            ),
        ),
        perturbations=perturbations
        or (
            PerturbationConfig(
                op=PerturbationOp.BLUR,
                params={"sigma": 1.0},
                seed_salts=(0,),
            ),
        ),
        controls=controls,
        runtime=RuntimeConfig(variants_per_chunk=variants_per_chunk),
        dump=DumpConfig(),
        adapter_spec=AdapterSpec(model_id="fake-model", deterministic=True),
    )


def _all_items(builder: PlanBuilder) -> tuple[WorkItem, ...]:
    return tuple(
        item
        for metadata in builder.enumerate()
        for item in builder.materialize(metadata.chunk_id).items
    )


def test_sample_order_is_stable_and_source_is_listed_once(tmp_path: Path) -> None:
    source = FakeSampleSource((_sample("z"), _sample("a")))
    builder = PlanBuilder(_resolved_config(tmp_path), source)

    first = builder.enumerate()
    second = builder.enumerate()
    clean = builder.enumerate_clean()

    assert [sample.sample_id for sample in clean] == ["a", "z"]
    assert [chunk.sample_id for chunk in first] == ["a", "z"]
    assert first == second
    assert source.list_calls == 1
    assert source.load_calls == []


def test_normal_items_follow_region_perturbation_seed_order(tmp_path: Path) -> None:
    regions = (
        ResolvedRegionConfig(
            region_id="r1",
            kind=RegionKind.GRID,
            params={"rows": 1, "cols": 2},
        ),
        ResolvedRegionConfig(
            region_id="r2",
            kind=RegionKind.GRID,
            params={"rows": 1, "cols": 1},
        ),
    )
    perturbations = (
        PerturbationConfig(
            op=PerturbationOp.BLUR,
            params={"sigma": 1.0},
            seed_salts=(7, 3),
        ),
        PerturbationConfig(
            op=PerturbationOp.PATCH_SHUFFLE,
            params={"patch_size": 4},
            seed_salts=(5,),
        ),
    )
    builder = PlanBuilder(
        _resolved_config(
            tmp_path,
            regions=regions,
            perturbations=perturbations,
        ),
        FakeSampleSource((_sample("s"),)),
    )

    observed = [
        (item.region_spec.region_instance_id, item.perturb_op.value, item.seed_salt)
        for item in _all_items(builder)
    ]

    assert observed == [
        ("r1/r0/c0", "blur", 7),
        ("r1/r0/c0", "blur", 3),
        ("r1/r0/c0", "patch_shuffle", 5),
        ("r1/r0/c1", "blur", 7),
        ("r1/r0/c1", "blur", 3),
        ("r1/r0/c1", "patch_shuffle", 5),
        ("r2/r0/c0", "blur", 7),
        ("r2/r0/c0", "blur", 3),
        ("r2/r0/c0", "patch_shuffle", 5),
    ]


@pytest.mark.parametrize(
    ("variants_per_chunk", "expected_sizes"),
    [
        (3, [3, 3]),
        (4, [4, 2]),
        (1, [1, 1, 1, 1, 1, 1]),
    ],
)
def test_chunks_split_without_empty_chunks(
    tmp_path: Path,
    variants_per_chunk: int,
    expected_sizes: list[int],
) -> None:
    config = _resolved_config(
        tmp_path,
        variants_per_chunk=variants_per_chunk,
        regions=(
            ResolvedRegionConfig(
                region_id="r1",
                kind=RegionKind.GRID,
                params={"rows": 1, "cols": 2},
            ),
        ),
        perturbations=(
            PerturbationConfig(
                op=PerturbationOp.BLUR,
                params={"sigma": 1.0},
                seed_salts=(0, 1, 2),
            ),
        ),
    )
    builder = PlanBuilder(config, FakeSampleSource((_sample("s"),)))

    chunks = builder.enumerate()

    assert [len(chunk.item_ids) for chunk in chunks] == expected_sizes


def test_builders_with_different_source_order_have_identical_plan(
    tmp_path: Path,
) -> None:
    config = _resolved_config(tmp_path, variants_per_chunk=1)
    first = PlanBuilder(
        config,
        FakeSampleSource((_sample("b"), _sample("a"))),
    )
    second = PlanBuilder(
        config,
        FakeSampleSource((_sample("a"), _sample("b"))),
    )

    assert first.enumerate() == second.enumerate()
    assert first.enumerate_clean() == second.enumerate_clean()


def test_config_order_changes_chunks_but_preserves_work_item_identity(
    tmp_path: Path,
) -> None:
    regions = (
        ResolvedRegionConfig(
            region_id="r1",
            kind=RegionKind.GRID,
            params={"rows": 1, "cols": 1},
        ),
        ResolvedRegionConfig(
            region_id="r2",
            kind=RegionKind.GRID,
            params={"rows": 1, "cols": 1},
        ),
    )
    perturbations = (
        PerturbationConfig(
            op=PerturbationOp.BLUR,
            params={"sigma": 1.0},
        ),
        PerturbationConfig(
            op=PerturbationOp.PATCH_SHUFFLE,
            params={"patch_size": 4},
        ),
    )
    first = PlanBuilder(
        _resolved_config(
            tmp_path,
            variants_per_chunk=2,
            regions=regions,
            perturbations=perturbations,
        ),
        FakeSampleSource((_sample("s"),)),
    )
    second = PlanBuilder(
        _resolved_config(
            tmp_path,
            variants_per_chunk=2,
            regions=tuple(reversed(regions)),
            perturbations=perturbations,
        ),
        FakeSampleSource((_sample("s"),)),
    )

    assert first.enumerate() != second.enumerate()
    assert {item.item_id for item in _all_items(first)} == {
        item.item_id for item in _all_items(second)
    }


def test_materialize_reproduces_metadata_ids_and_chunk_id(tmp_path: Path) -> None:
    builder = PlanBuilder(
        _resolved_config(tmp_path, variants_per_chunk=2),
        FakeSampleSource((_sample("s"),)),
    )

    for metadata in builder.enumerate():
        chunk = builder.materialize(metadata.chunk_id)
        assert chunk.chunk_id == metadata.chunk_id
        assert chunk.sample_id == metadata.sample_id
        assert tuple(item.item_id for item in chunk.items) == metadata.item_ids


def test_grid_cells_multiply_perturbation_variants(tmp_path: Path) -> None:
    config = _resolved_config(
        tmp_path,
        regions=(
            ResolvedRegionConfig(
                region_id="grid",
                kind=RegionKind.GRID,
                params={"rows": 2, "cols": 2},
            ),
        ),
        perturbations=(
            PerturbationConfig(
                op=PerturbationOp.BLUR,
                params={"sigma": 1.0},
            ),
            PerturbationConfig(
                op=PerturbationOp.CONSTANT_FILL,
                params={"value": 0},
            ),
        ),
    )
    builder = PlanBuilder(config, FakeSampleSource((_sample("s"),)))

    items = _all_items(builder)

    assert len(items) == 8
    assert [item.region_spec.region_instance_id for item in items] == [
        "grid/r0/c0",
        "grid/r0/c0",
        "grid/r0/c1",
        "grid/r0/c1",
        "grid/r1/c0",
        "grid/r1/c0",
        "grid/r1/c1",
        "grid/r1/c1",
    ]


def test_materialize_rejects_unknown_chunk_id(tmp_path: Path) -> None:
    builder = PlanBuilder(
        _resolved_config(tmp_path),
        FakeSampleSource((_sample("s"),)),
    )

    with pytest.raises(PlanBuildError, match="unknown chunk_id"):
        builder.materialize("f" * 64)


def test_materialize_detects_config_mutation_after_enumeration(
    tmp_path: Path,
) -> None:
    config = _resolved_config(tmp_path)
    builder = PlanBuilder(config, FakeSampleSource((_sample("s"),)))
    metadata = builder.enumerate()[0]
    config.perturbations[0].params["sigma"] = 2.0

    with pytest.raises(PlanBuildError, match="item_ids do not match"):
        builder.materialize(metadata.chunk_id)


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        ((), "at least one sample"),
        ((_sample("same"), _sample("same")), "duplicate sample_id"),
        ((object(),), "must return SampleMeta"),
    ],
)
def test_invalid_source_metadata_is_rejected(
    tmp_path: Path,
    samples: Sequence[Any],
    message: str,
) -> None:
    source = FakeSampleSource(samples)  # type: ignore[arg-type]
    builder = PlanBuilder(_resolved_config(tmp_path), source)

    with pytest.raises(PlanBuildError, match=message):
        builder.enumerate()


def test_source_listing_error_is_wrapped(tmp_path: Path) -> None:
    builder = PlanBuilder(
        _resolved_config(tmp_path),
        FailingSampleSource(()),
    )

    with pytest.raises(PlanBuildError, match=r"list_samples\(\) failed") as error:
        builder.enumerate_clean()

    assert isinstance(error.value.__cause__, RuntimeError)


def test_controls_are_appended_after_normal_items_with_full_product(
    tmp_path: Path,
) -> None:
    target = ResolvedRegionConfig(
        region_id="target",
        kind=RegionKind.GRID,
        params={"rows": 2, "cols": 2},
    )
    perturbations = (
        PerturbationConfig(
            op=PerturbationOp.BLUR,
            params={"sigma": 1.0},
            seed_salts=(0, 1),
        ),
        PerturbationConfig(
            op=PerturbationOp.PATCH_SHUFFLE,
            params={"patch_size": 4},
            seed_salts=(9,),
            invert_mask=True,
        ),
    )
    controls = ControlConfig(
        match_area_of="target",
        n_samples=2,
    )
    builder = PlanBuilder(
        _resolved_config(
            tmp_path,
            regions=(target,),
            perturbations=perturbations,
            controls=(controls,),
        ),
        FakeSampleSource((_sample("s"),)),
    )

    items = _all_items(builder)
    normal_count = 12
    control_items = items[normal_count:]

    assert len(items) == 36
    assert all(not item.is_control for item in items[:normal_count])
    assert all(item.is_control for item in control_items)
    assert len({item.item_id for item in items}) == len(items)
    assert [
        (
            item.region_spec.params["target_region"]["region_instance_id"],
            item.region_spec.params["control_index"],
            item.perturb_op.value,
            item.seed_salt,
        )
        for item in control_items
    ] == [
        (target_id, control_index, op, seed)
        for target_id in (
            "target/r0/c0",
            "target/r0/c1",
            "target/r1/c0",
            "target/r1/c1",
        )
        for control_index in (0, 1)
        for op, seed in (("blur", 0), ("blur", 1), ("patch_shuffle", 9))
    ]
    target_recipe = control_items[0].region_spec.params["target_region"]
    assert target_recipe == {
        "region_id": "target",
        "region_instance_id": "target/r0/c0",
        "kind": "grid",
        "params": {"rows": 2, "cols": 2, "row_index": 0, "col_index": 0},
        "ref": None,
        "ref_hash": None,
    }
    assert len({item.region_spec.region_instance_id for item in control_items}) == 8
    assert control_items[-1].invert_mask is True


def test_duplicate_control_requests_are_distinguished_by_request_ordinal(
    tmp_path: Path,
) -> None:
    request = ControlConfig(match_area_of="region-a", n_samples=1)
    config = _resolved_config(
        tmp_path,
        regions=(
            ResolvedRegionConfig(
                region_id="region-a",
                kind=RegionKind.GRID,
                params={"rows": 1, "cols": 1},
            ),
        ),
        controls=(request, request),
    )
    builder = PlanBuilder(config, FakeSampleSource((_sample("s"),)))

    controls = [item for item in _all_items(builder) if item.is_control]

    assert len(controls) == 2
    assert controls[0].item_id != controls[1].item_id
    assert controls[0].region_spec.region_id != controls[1].region_spec.region_id
    assert [
        item.region_spec.params["control_request_index"] for item in controls
    ] == [0, 1]


def test_no_control_configuration_creates_no_control_items(tmp_path: Path) -> None:
    builder = PlanBuilder(
        _resolved_config(tmp_path),
        FakeSampleSource((_sample("s"),)),
    )

    assert not any(item.is_control for item in _all_items(builder))
