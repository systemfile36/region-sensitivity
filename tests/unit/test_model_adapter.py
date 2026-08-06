"""Tests for framework-neutral and built-in model adapters."""

from __future__ import annotations

import numpy as np
import pytest
import subprocess
import sys

from ssat.core.adapter import (
    AdapterError,
    AdapterOutOfMemoryError,
    AdapterSpec,
    CallableAdapter,
    CallablePreprocessor,
    CenterCrop,
    ChannelsFirst,
    DeclarativeAdapter,
    DeclarativePreprocessor,
    IdentityPreprocessor,
    LogitsOutputDecoder,
    ModelAdapter,
    Normalize,
    OutputDecoder,
    PreprocessingSpec,
    Preprocessor,
    RawOutput,
    Resize,
    SqueezeTime,
    TimmAdapter,
    ToFloat,
    TorchvisionAdapter,
)


def _batch(count: int = 2, height: int = 10, width: int = 14) -> np.ndarray:
    """Create a small RGB image-classification batch."""

    return np.arange(count * height * width * 3, dtype=np.uint8).reshape(
        count, 1, height, width, 3
    )


def test_model_adapter_contract_is_abstract_and_callable_describes_itself() -> None:
    """The base is abstract and callable metadata is a valid AdapterSpec."""

    with pytest.raises(TypeError):
        ModelAdapter()

    adapter = CallableAdapter(
        lambda batch: np.zeros((len(batch), 3), dtype=np.float32),
        model_id="callable:test",
        max_batch_size=4,
        class_names=("a", "b", "c"),
    )
    spec = adapter.describe()
    assert spec.model_id == "callable:test"
    assert spec.deterministic is True
    assert spec.input_layout == "THWC_uint8"
    assert spec.output_kind == "logits"
    assert spec.max_batch_size == 4
    assert spec.adapter_kind == "callable"
    assert spec.mask_transform_available is False


def test_preprocessor_contracts_and_callable_metadata() -> None:
    """Preprocessors own deterministic pixel and mask transformation metadata."""

    with pytest.raises(TypeError):
        Preprocessor()

    batch = _batch(count=1)
    mask = np.zeros((4, 5), dtype=np.bool_)
    identity = IdentityPreprocessor()
    assert identity.transform_batch(batch) is batch
    assert identity.transform_mask(mask) is None
    assert identity.describe().mask_transform_available is False

    callable_preprocessor = CallablePreprocessor(
        transform_batch_fn=lambda value: value.astype(np.float32),
        transform_mask_fn=np.logical_not,
        deterministic=False,
        description="custom pipeline",
        fingerprint="a" * 64,
    )
    assert callable_preprocessor.transform_batch(batch).dtype == np.float32
    assert np.array_equal(callable_preprocessor.transform_mask(mask), ~mask)
    assert callable_preprocessor.describe() == PreprocessingSpec(
        kind="callable",
        deterministic=False,
        description="custom pipeline",
        fingerprint="a" * 64,
        mask_transform_available=True,
    )


def test_declarative_preprocessing_fingerprint_is_canonical_and_sensitive() -> None:
    """Equivalent op syntax hashes equally and a changed op changes identity."""

    typed = DeclarativePreprocessor((Resize((8, 12)), CenterCrop(6), ToFloat()))
    mapped = DeclarativePreprocessor(
        (
            {"op": "resize", "size": [8, 12]},
            {"op": "center_crop", "size": 6},
            {"op": "to_float"},
        )
    )
    changed = DeclarativePreprocessor((Resize((8, 13)), CenterCrop(6), ToFloat()))
    assert typed.describe().fingerprint == mapped.describe().fingerprint
    assert typed.describe().fingerprint != changed.describe().fingerprint
    assert typed.describe().mask_transform_available is True


def test_logits_output_decoder_supports_all_v1_input_forms() -> None:
    """The default decoder accepts tensors, arrays, mappings, and sequences."""

    import torch

    decoder = LogitsOutputDecoder()
    expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    forms = (
        expected,
        torch.from_numpy(expected),
        {"logits": expected},
        [RawOutput(expected[0]), expected[1]],
    )
    for value in forms:
        outputs = decoder.decode(value, batch_size=2, class_names=("a", "b"))
        assert np.array_equal(np.stack([item.logits for item in outputs]), expected)

    scores_decoder = LogitsOutputDecoder(logits_key="scores")
    assert len(scores_decoder.decode({"scores": expected}, batch_size=2)) == 2


@pytest.mark.parametrize(
    ("value", "batch_size", "message"),
    [
        (np.ones((2,), dtype=np.float32), 1, r"\(B, classes\)"),
        (np.ones((1, 2), dtype=np.int64), 1, "floating dtype"),
        (np.ones((1, 2), dtype=np.float32), 2, "returned 1 outputs"),
        ({"scores": np.ones((1, 2), dtype=np.float32)}, 1, "no 'logits' key"),
    ],
)
def test_logits_output_decoder_rejects_invalid_results(
    value: object,
    batch_size: int,
    message: str,
) -> None:
    """Decoder errors identify malformed output and alignment failures."""

    with pytest.raises(AdapterError, match=message):
        LogitsOutputDecoder().decode(value, batch_size=batch_size)

    with pytest.raises(TypeError):
        OutputDecoder()


def test_logits_output_decoder_validates_class_names() -> None:
    """Declared class names remain aligned with the decoded logits vector."""

    with pytest.raises(AdapterError, match="does not match class_names"):
        LogitsOutputDecoder().decode(
            np.ones((1, 3), dtype=np.float32),
            batch_size=1,
            class_names=("a", "b"),
        )


def test_callable_adapter_normalizes_outputs_and_checks_alignment() -> None:
    """Callable results contain exactly one valid logits vector per input."""

    batch = _batch()
    adapter = CallableAdapter(
        lambda value: np.ones((len(value), 3), dtype=np.float32),
        model_id="callable:test",
    )
    outputs = adapter.predict(batch)
    assert len(outputs) == len(batch)
    assert all(output.logits.shape == (3,) for output in outputs)

    short = CallableAdapter(
        lambda value: np.ones((len(value) - 1, 3), dtype=np.float32),
        model_id="callable:short",
    )
    with pytest.raises(AdapterError, match="returned 1 outputs"):
        short.predict(batch)

    capped = CallableAdapter(
        lambda value: np.ones((len(value), 3), dtype=np.float32),
        model_id="callable:capped",
        max_batch_size=1,
    )
    with pytest.raises(AdapterError, match="exceeds max_batch_size"):
        capped.predict(batch)


def test_callable_without_mask_transform_reports_unavailable() -> None:
    """Returning None selects the effective_area_available=false runtime path."""

    adapter = CallableAdapter(
        lambda batch: np.zeros((len(batch), 2), dtype=np.float32),
        model_id="callable:no-mask-transform",
    )
    mask = np.ones((4, 5), dtype=np.bool_)
    assert adapter.transform_mask(mask) is None


def test_callable_composes_preprocessor_and_custom_decoder() -> None:
    """Injected components drive prediction, provenance, and determinism."""

    preprocessor = CallablePreprocessor(
        transform_batch_fn=lambda batch: batch[:, 0],
        deterministic=False,
        description="remove time",
        fingerprint="b" * 64,
    )
    adapter = CallableAdapter(
        lambda batch: {"scores": np.ones((len(batch), 2), dtype=np.float32)},
        model_id="callable:composed",
        preprocessor=preprocessor,
        output_decoder=LogitsOutputDecoder(logits_key="scores"),
        model_name="custom-net",
        weights_id="checkpoint-v1",
        weights_hash="c" * 64,
    )
    assert len(adapter.predict(_batch())) == 2
    spec = adapter.describe()
    assert spec.deterministic is False
    assert spec.preprocessing_fingerprint == "b" * 64
    assert spec.model_name == "custom-net"
    assert spec.weights_hash == "c" * 64

    with pytest.raises(ValueError, match="cannot be used together"):
        CallableAdapter(
            lambda batch: batch,
            model_id="callable:conflict",
            preprocessor=IdentityPreprocessor(),
            transform_mask_fn=lambda mask: mask,
        )


def test_declarative_adapter_applies_pixels_but_only_geometry_to_mask() -> None:
    """Photometric/layout ops affect prediction input but preserve binary masks."""

    seen: list[np.ndarray] = []

    def predict(prepared: np.ndarray) -> np.ndarray:
        seen.append(prepared)
        return np.zeros((prepared.shape[0], 2), dtype=np.float32)

    adapter = DeclarativeAdapter(
        predict,
        model_id="declarative:test",
        preprocessing_ops=(
            Resize((8, 12)),
            CenterCrop((6, 8)),
            ToFloat(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ChannelsFirst(),
            SqueezeTime(),
        ),
    )
    outputs = adapter.predict(_batch(count=1, height=4, width=6))
    assert len(outputs) == 1
    assert seen[0].shape == (1, 3, 6, 8)
    assert seen[0].dtype == np.float32

    mask = np.zeros((4, 6), dtype=np.bool_)
    mask[:, :3] = True
    transformed = adapter.transform_mask(mask)
    assert transformed.shape == (6, 8)
    assert transformed.dtype == np.bool_
    assert transformed.sum() == 24


def test_declarative_adapter_accepts_mapping_ops_and_rejects_unknown_ops() -> None:
    """JSON-style declarations resolve to the same strict operation contract."""

    adapter = DeclarativeAdapter(
        lambda batch: np.zeros((len(batch), 1), dtype=np.float32),
        model_id="declarative:mapping",
        preprocessing_ops=(
            {"op": "resize", "size": [8, 8]},
            {"op": "center_crop", "size": 6},
        ),
    )
    assert adapter.transform_mask(np.ones((4, 5), dtype=np.bool_)).shape == (6, 6)

    with pytest.raises(ValueError, match="unknown preprocessing op"):
        DeclarativeAdapter(
            lambda batch: batch,
            model_id="declarative:invalid",
            preprocessing_ops=({"op": "random_crop", "size": 4},),
        )


@pytest.mark.parametrize(
    ("adapter_type", "model_name"),
    [
        (TorchvisionAdapter, "squeezenet1_0"),
        (TimmAdapter, "resnet18"),
    ],
)
def test_framework_adapter_runs_by_model_name_and_transforms_mask(
    adapter_type: type[ModelAdapter],
    model_name: str,
) -> None:
    """Each model zoo adapter initializes without downloads and runs inference."""

    adapter = adapter_type(model_name, device="cpu")
    spec = adapter.describe()
    assert spec.model_id
    assert spec.deterministic is True
    assert spec.preprocessing_desc
    assert spec.adapter_kind in {"torchvision", "timm"}
    assert spec.model_name == model_name
    assert spec.weights_id == "none:init_seed=0"
    assert spec.preprocessing_fingerprint is not None
    assert spec.mask_transform_available is True

    outputs = adapter.predict(_batch(count=1, height=32, width=48))
    assert len(outputs) == 1
    assert outputs[0].logits.ndim == 1
    assert outputs[0].logits.size > 1

    mask = np.zeros((100, 200), dtype=np.bool_)
    mask[:, :100] = True
    transformed = adapter.transform_mask(mask)
    assert transformed.shape == (224, 224)
    assert transformed.dtype == np.bool_
    assert transformed.sum() == 224 * 112


def test_framework_adapters_reject_video_input() -> None:
    """v1 model-zoo implementations make the image-only constraint explicit."""

    adapter = TorchvisionAdapter("squeezenet1_0", device="cpu")
    video = np.zeros((1, 2, 16, 16, 3), dtype=np.uint8)
    with pytest.raises(AdapterError, match="require T=1"):
        adapter.predict(video)


def test_untrained_framework_model_initialization_is_reproducible() -> None:
    """Local initialization seeds produce stable models without changing torch RNG."""

    import torch

    torch.manual_seed(1234)
    before = torch.random.get_rng_state()
    first = TorchvisionAdapter("squeezenet1_0", device="cpu", init_seed=7)
    after = torch.random.get_rng_state()
    second = TorchvisionAdapter("squeezenet1_0", device="cpu", init_seed=7)

    assert torch.equal(before, after)
    assert first.describe().model_id.endswith(":init_seed=7")
    first_parameter = next(first._model.parameters())
    second_parameter = next(second._model.parameters())
    assert torch.equal(first_parameter, second_parameter)


@pytest.mark.parametrize(
    ("adapter_type", "model_name"),
    [
        (TorchvisionAdapter, "squeezenet1_0"),
        (TimmAdapter, "resnet18"),
    ],
)
def test_framework_adapters_translate_torch_oom(
    adapter_type: type[ModelAdapter],
    model_name: str,
) -> None:
    """Runtime can recover OOM without importing a framework-specific exception."""

    import torch

    class OomModel(torch.nn.Module):
        def forward(self, batch: object) -> object:
            raise torch.cuda.OutOfMemoryError("injected oom")

    adapter = adapter_type(model_name, device="cpu")
    adapter._model = OomModel()
    with pytest.raises(AdapterOutOfMemoryError, match="out of memory"):
        adapter.predict(_batch(count=1, height=32, width=48))


def test_adapter_provenance_hashes_require_lowercase_sha256() -> None:
    """Artifact and preprocessing identities use one strict manifest format."""

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        AdapterSpec(
            model_id="bad-hash",
            deterministic=True,
            weights_hash="A" * 64,
        )


def test_adapter_contract_imports_remain_framework_lazy() -> None:
    """Importing adapter contracts does not load cv2, torch, or timm."""

    code = (
        "import sys; import ssat.core.adapter.types; "
        "assert 'cv2' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert 'timm' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
