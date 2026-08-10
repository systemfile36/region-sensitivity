from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ssat.core.config.schema import AuditConfig


EXAMPLES = Path(__file__).parents[2] / "configs" / "examples"


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.yaml")))
def test_example_configs_are_valid(path: Path) -> None:
    value = yaml.safe_load(path.read_text())
    value.pop("source")
    value.pop("adapter")
    config = AuditConfig.model_validate(value)
    assert config.schema_version == "1.0.0"


def minimal_config() -> dict:
    return {
        "regions": [{"region_id": "grid", "kind": "grid", "params": {}}],
        "perturbations": [{"op": "blur", "params": {"sigma": 1.0}}],
    }


def test_unknown_key_is_rejected() -> None:
    value = minimal_config()
    value["typo"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AuditConfig.model_validate(value)


def test_duplicate_region_ids_are_rejected() -> None:
    value = minimal_config()
    value["regions"].append({"region_id": "grid", "kind": "grid"})
    with pytest.raises(ValidationError, match="region_id values must be unique"):
        AuditConfig.model_validate(value)


def test_control_must_reference_known_region() -> None:
    value = minimal_config()
    value["controls"] = [{"match_area_of": "missing", "n_samples": 1}]
    with pytest.raises(ValidationError, match="unknown region_id"):
        AuditConfig.model_validate(value)


def test_explicit_region_requires_ref() -> None:
    value = minimal_config()
    value["regions"] = [{"region_id": "mask", "kind": "explicit"}]
    with pytest.raises(ValidationError, match="explicit regions require ref"):
        AuditConfig.model_validate(value)


def test_procedural_region_rejects_ref() -> None:
    value = minimal_config()
    value["regions"][0]["ref"] = "mask.png"
    with pytest.raises(ValidationError, match="procedural regions cannot define"):
        AuditConfig.model_validate(value)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), object()])
def test_params_reject_non_json_values(bad_value: object) -> None:
    value = minimal_config()
    value["perturbations"][0]["params"] = {"bad": bad_value}
    with pytest.raises(ValidationError):
        AuditConfig.model_validate(value)


def test_runtime_defaults_and_bounds() -> None:
    config = AuditConfig.model_validate(minimal_config())
    assert config.runtime.variants_per_chunk == 16
    assert config.runtime.target_batch_size == 32
    assert config.runtime.num_workers == 0

    value = minimal_config()
    value["runtime"] = {"variants_per_chunk": 0}
    with pytest.raises(ValidationError):
        AuditConfig.model_validate(value)


def test_config_is_frozen() -> None:
    config = AuditConfig.model_validate(minimal_config())
    with pytest.raises(ValidationError):
        config.runtime = config.runtime
