import hashlib
import io
import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

from ssat.core.adapter.types import AdapterSpec
from ssat.core.config.resolver import ConfigResolutionError, ConfigResolver
from ssat.core.config.schema import AuditConfig, ResolvedConfig
from ssat.core.config.stats import DatasetStatsError, compute_dataset_stats
from ssat.core.source.types import LoadError, LoadedSample, SampleMeta


class FakeAdapter:
    def __init__(self, *, deterministic: bool = True, result: object | None = None) -> None:
        self.deterministic = deterministic
        self.result = result
        self.describe_calls = 0

    def describe(self) -> object:
        self.describe_calls += 1
        if self.result is not None:
            return self.result
        return AdapterSpec(model_id="fake-model", deterministic=self.deterministic)


class FakeSource:
    def __init__(
        self,
        samples: list[SampleMeta] | None = None,
        results: dict[str, LoadedSample | LoadError] | None = None,
    ) -> None:
        self.samples = samples or []
        self.results = results or {}
        self.list_calls = 0
        self.load_calls: list[str] = []

    def list_samples(self) -> list[SampleMeta]:
        self.list_calls += 1
        return list(self.samples)

    def load(self, sample_id: str) -> LoadedSample | LoadError:
        self.load_calls.append(sample_id)
        return self.results[sample_id]


class RaisingSource(FakeSource):
    def list_samples(self) -> list[SampleMeta]:
        raise RuntimeError("list failed")


def make_loaded(sample_id: str, array: np.ndarray) -> LoadedSample:
    return LoadedSample(
        array=array,
        sample_id=sample_id,
        original_shape=array.shape,
        content_hash="a" * 64,
    )


def make_config(
    perturbation: dict | None = None,
    *,
    dataset_stats: dict | None = None,
    allow_nondeterministic: bool = False,
) -> dict:
    config = {
        "regions": [{"region_id": "grid", "kind": "grid", "params": {}}],
        "perturbations": [perturbation or {"op": "blur", "params": {"sigma": 1.0}}],
        "runtime": {"allow_nondeterministic": allow_nondeterministic},
    }
    if dataset_stats is not None:
        config["dataset_stats"] = dataset_stats
    return config


def capture_logger() -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    logger = logging.Logger("ssat.test.resolver")
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger, stream


def test_yaml_input_resolves_explicit_ref_and_round_trips(tmp_path: Path) -> None:
    mask = tmp_path / "masks" / "foreground.bin"
    mask.parent.mkdir()
    mask.write_bytes(b"mask-content")
    config_path = tmp_path / "audit.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "regions": [
                    {
                        "region_id": "foreground",
                        "kind": "explicit",
                        "ref": "masks/foreground.bin",
                    }
                ],
                "perturbations": [{"op": "blur", "params": {"sigma": 1.5}}],
            }
        ),
        encoding="utf-8",
    )

    resolved = ConfigResolver().resolve(config_path, FakeAdapter(), FakeSource())
    expected_hash = hashlib.sha256(b"mask-content").hexdigest()
    assert resolved.config_source == config_path.resolve()
    assert resolved.config_base_dir == tmp_path.resolve()
    assert resolved.regions[0].ref == mask.resolve()
    assert resolved.regions[0].ref_hash == expected_hash

    restored = ResolvedConfig.model_validate_json(resolved.model_dump_json())
    assert restored == resolved


def test_mapping_relative_ref_requires_base_dir(tmp_path: Path) -> None:
    config = make_config()
    config["regions"] = [
        {"region_id": "mask", "kind": "explicit", "ref": "mask.bin"}
    ]

    with pytest.raises(ConfigResolutionError, match="base_dir is required"):
        ConfigResolver().resolve(config, FakeAdapter(), FakeSource())

    mask = tmp_path / "mask.bin"
    mask.write_bytes(b"mask")
    resolved = ConfigResolver().resolve(
        config,
        FakeAdapter(),
        FakeSource(),
        base_dir=tmp_path,
    )
    assert resolved.regions[0].ref == mask.resolve()


def test_explicit_hash_is_verified_and_normalized(tmp_path: Path) -> None:
    mask = tmp_path / "mask.bin"
    mask.write_bytes(b"mask")
    actual_hash = hashlib.sha256(b"mask").hexdigest()
    config = make_config()
    config["regions"] = [
        {
            "region_id": "mask",
            "kind": "explicit",
            "ref": str(mask),
            "ref_hash": actual_hash.upper(),
        }
    ]
    resolved = ConfigResolver().resolve(config, FakeAdapter(), FakeSource())
    assert resolved.regions[0].ref_hash == actual_hash

    config["regions"][0]["ref_hash"] = "0" * 64
    with pytest.raises(ConfigResolutionError, match="ref_hash mismatch"):
        ConfigResolver().resolve(config, FakeAdapter(), FakeSource())


def test_missing_explicit_ref_is_rejected(tmp_path: Path) -> None:
    config = make_config()
    config["regions"] = [
        {"region_id": "mask", "kind": "explicit", "ref": str(tmp_path / "missing")}
    ]
    with pytest.raises(ConfigResolutionError, match="ref does not exist"):
        ConfigResolver().resolve(config, FakeAdapter(), FakeSource())


def test_nondeterministic_adapter_is_rejected_by_default() -> None:
    with pytest.raises(ConfigResolutionError, match="is nondeterministic"):
        ConfigResolver().resolve(make_config(), FakeAdapter(deterministic=False), FakeSource())


def test_nondeterministic_adapter_can_be_allowed_with_warning() -> None:
    logger, stream = capture_logger()
    resolved = ConfigResolver(logger).resolve(
        make_config(allow_nondeterministic=True),
        FakeAdapter(deterministic=False),
        FakeSource(),
    )
    assert resolved.adapter_spec.deterministic is False
    assert "WARNING adapter.nondeterministic_allowed model_id=fake-model" in stream.getvalue()


def test_adapter_describe_must_return_adapter_spec() -> None:
    with pytest.raises(ConfigResolutionError, match="must return AdapterSpec"):
        ConfigResolver().resolve(make_config(), FakeAdapter(result={}), FakeSource())


def test_precomputed_stats_bypass_source_and_resolve_mean_fill() -> None:
    source = FakeSource()
    resolved = ConfigResolver().resolve(
        make_config(
            {"op": "mean_fill", "params": {}},
            dataset_stats={"channel_mean": [1.0, 2.0, 3.0]},
        ),
        FakeAdapter(),
        source,
    )
    assert source.list_calls == 0
    assert resolved.perturbations[0].params == {"value": [1.0, 2.0, 3.0]}


def test_non_mean_fill_does_not_compute_stats() -> None:
    source = FakeSource()
    resolved = ConfigResolver().resolve(make_config(), FakeAdapter(), source)
    assert source.list_calls == 0
    assert resolved.dataset_stats is None


def test_mean_fill_computes_weighted_stats_and_skips_load_errors() -> None:
    first = np.array([[[[0, 10, 20]]]], dtype=np.uint8)
    second = np.full((1, 1, 3, 3), [100, 110, 120], dtype=np.uint8)
    samples = [
        SampleMeta("z", Path("z")),
        SampleMeta("a", Path("a")),
        SampleMeta("broken", Path("broken")),
    ]
    source = FakeSource(
        samples,
        {
            "a": make_loaded("a", first),
            "broken": LoadError("broken", "decode_error", "bad image"),
            "z": make_loaded("z", second),
        },
    )
    logger, stream = capture_logger()
    resolved = ConfigResolver(logger).resolve(
        make_config({"op": "mean_fill", "params": {}}),
        FakeAdapter(),
        source,
    )
    assert source.load_calls == ["a", "broken", "z"]
    assert resolved.dataset_stats is not None
    assert resolved.dataset_stats.channel_mean == pytest.approx((75.0, 85.0, 95.0))
    assert resolved.perturbations[0].params["value"] == pytest.approx([75.0, 85.0, 95.0])
    assert "dataset_stats.sample_skipped sample_id=broken" in stream.getvalue()
    assert "samples_used=2 samples_skipped=1" in stream.getvalue()


@pytest.mark.parametrize(
    "source",
    [
        FakeSource(),
        FakeSource(
            [SampleMeta("bad", Path("bad"))],
            {"bad": LoadError("bad", "decode_error", "bad")},
        ),
    ],
)
def test_stats_fail_when_no_sample_can_be_used(source: FakeSource) -> None:
    with pytest.raises(ConfigResolutionError, match="dataset statistics resolution failed"):
        ConfigResolver().resolve(
            make_config({"op": "mean_fill", "params": {}}),
            FakeAdapter(),
            source,
        )


def test_stats_reject_duplicate_ids() -> None:
    source = FakeSource(
        [SampleMeta("same", Path("a")), SampleMeta("same", Path("b"))]
    )
    with pytest.raises(DatasetStatsError, match="duplicate sample_id"):
        compute_dataset_stats(source)


def test_stats_reject_channel_mismatch() -> None:
    rgb = np.zeros((1, 1, 1, 3), dtype=np.uint8)
    gray = np.zeros((1, 1, 1, 1), dtype=np.uint8)
    source = FakeSource(
        [SampleMeta("rgb", Path("rgb")), SampleMeta("gray", Path("gray"))],
        {"rgb": make_loaded("rgb", rgb), "gray": make_loaded("gray", gray)},
    )
    with pytest.raises(DatasetStatsError, match="channels"):
        compute_dataset_stats(source)


def test_stats_reject_loaded_sample_id_mismatch() -> None:
    array = np.zeros((1, 1, 1, 3), dtype=np.uint8)
    source = FakeSource(
        [SampleMeta("expected", Path("sample"))],
        {"expected": make_loaded("other", array)},
    )
    with pytest.raises(DatasetStatsError, match="does not match"):
        compute_dataset_stats(source)


def test_source_exception_is_wrapped_by_resolver() -> None:
    with pytest.raises(ConfigResolutionError, match="dataset statistics resolution failed") as exc:
        ConfigResolver().resolve(
            make_config({"op": "mean_fill", "params": {}}),
            FakeAdapter(),
            RaisingSource(),
        )
    assert isinstance(exc.value.__cause__, DatasetStatsError)


@pytest.mark.parametrize(
    "perturbation",
    [
        {"op": "constant_fill", "params": {"value": 0}},
        {"op": "constant_fill", "params": {"value": [0, 127.5, 255]}},
        {"op": "mean_fill", "params": {}},
        {"op": "blur", "params": {"sigma": 1}},
        {"op": "gaussian_noise", "params": {"sigma": 0.5}},
        {"op": "patch_shuffle", "params": {"patch_size": 8}},
    ],
)
def test_valid_v1_perturbation_params(perturbation: dict) -> None:
    resolved = ConfigResolver().resolve(
        make_config(
            perturbation,
            dataset_stats={"channel_mean": [1.0, 2.0, 3.0]},
        ),
        FakeAdapter(),
        FakeSource(),
    )
    assert resolved.perturbations[0].op.value == perturbation["op"]


@pytest.mark.parametrize(
    "perturbation",
    [
        {"op": "constant_fill", "params": {}},
        {"op": "constant_fill", "params": {"value": []}},
        {"op": "constant_fill", "params": {"value": True}},
        {"op": "constant_fill", "params": {"value": 256}},
        {"op": "mean_fill", "params": {"value": [1, 2, 3]}},
        {"op": "blur", "params": {"sigma": 0}},
        {"op": "blur", "params": {"sigma": True}},
        {"op": "gaussian_noise", "params": {"sigma": 1, "extra": 2}},
        {"op": "patch_shuffle", "params": {"patch_size": 0}},
        {"op": "patch_shuffle", "params": {"patch_size": True}},
    ],
)
def test_invalid_v1_perturbation_params(perturbation: dict) -> None:
    with pytest.raises(ConfigResolutionError):
        ConfigResolver().resolve(
            make_config(
                perturbation,
                dataset_stats={"channel_mean": [1.0, 2.0, 3.0]},
            ),
            FakeAdapter(),
            FakeSource(),
        )


def test_audit_config_model_input_is_supported(tmp_path: Path) -> None:
    config = AuditConfig.model_validate(make_config())
    resolved = ConfigResolver().resolve(
        config,
        FakeAdapter(),
        FakeSource(),
        base_dir=tmp_path,
    )
    assert resolved.config_source is None
    assert resolved.config_base_dir == tmp_path.resolve()
