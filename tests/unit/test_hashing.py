import math

import pytest

from ssat.core.plan.hashing import canonical_json, compute_chunk_id, compute_item_id


def identity(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "sample_id": "sample-1",
        "region_spec": {
            "region_id": "grid",
            "kind": "grid",
            "params": {"ratio": 0.1, "rows": 2},
            "ref": None,
        },
        "perturb_op": "blur",
        "perturb_params": {"sigma": 1.25},
        "invert_mask": False,
        "seed_salt": 0,
        "is_control": False,
    }
    value.update(overrides)
    return value


def test_key_order_does_not_change_item_id() -> None:
    first = identity()
    second = dict(reversed(list(first.items())))
    assert compute_item_id(first) == compute_item_id(second)


def test_v1_item_id_regression_value() -> None:
    assert compute_item_id(identity()) == (
        "56ce62890b2ea59632489954384dacf4dead3573b781d81fb4121498f946d668"
    )


def test_equivalent_float_spellings_have_same_item_id() -> None:
    assert compute_item_id(identity(weight=0.1)) == compute_item_id(identity(weight=0.10))


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("sample_id", "sample-2"),
        ("perturb_op", "gaussian_noise"),
        ("invert_mask", True),
        ("seed_salt", 1),
        ("is_control", True),
    ],
)
def test_identity_fields_change_item_id(field: str, changed: object) -> None:
    assert compute_item_id(identity()) != compute_item_id(identity(**{field: changed}))


def test_item_id_field_is_excluded() -> None:
    first = identity(item_id="a" * 64)
    second = identity(item_id="b" * 64)
    assert compute_item_id(first) == compute_item_id(second)


def test_nested_region_or_perturbation_change_changes_item_id() -> None:
    base = identity()
    changed_region = identity(
        region_spec={
            "region_id": "grid",
            "kind": "grid",
            "params": {"ratio": 0.1, "rows": 3},
            "ref": None,
        }
    )
    changed_perturbation = identity(perturb_params={"sigma": 1.5})

    assert compute_item_id(base) != compute_item_id(changed_region)
    assert compute_item_id(base) != compute_item_id(changed_perturbation)


def test_none_mapping_fields_are_omitted() -> None:
    assert canonical_json({"a": 1, "b": None}) == canonical_json({"a": 1})


def test_float_contract_uses_twelve_decimal_places() -> None:
    assert canonical_json({"value": 0.1}) == '{"value":"0.100000000000"}'
    assert canonical_json({"value": -0.0}) == '{"value":"0.000000000000"}'


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_float_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="NaN and infinity"):
        canonical_json({"value": value})


def test_schema_version_namespaces_item_ids() -> None:
    assert compute_item_id(identity(), schema_version="1") != compute_item_id(
        identity(), schema_version="2"
    )


def test_chunk_id_regression_value() -> None:
    assert compute_chunk_id("sample", 0, ("a" * 64, "b" * 64)) == (
        "60cb82d57876b7d812edccbea442778047ad80649845300edfdc19473046099a"
    )


def test_chunk_id_covers_order_ordinal_sample_and_schema_version() -> None:
    item_ids = ("a" * 64, "b" * 64)
    base = compute_chunk_id("sample", 0, item_ids)

    assert compute_chunk_id("sample", 0, tuple(reversed(item_ids))) != base
    assert compute_chunk_id("sample", 1, item_ids) != base
    assert compute_chunk_id("other", 0, item_ids) != base
    assert compute_chunk_id("sample", 0, item_ids, schema_version="2") != base
