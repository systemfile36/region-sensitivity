"""Unit tests for execution-aware cost estimation and sanity checks."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ssat.core.adapter import CallableAdapter, CallablePreprocessor
from ssat.core.adapter.base import AdapterOutOfMemoryError
from ssat.core.config.schema import (
    DumpConfig,
    PerturbationConfig,
    ResolvedConfig,
    ResolvedRegionConfig,
    RuntimeConfig,
)
from ssat.core.estimate import (
    AdvisoryCode,
    CostEstimator,
    DumpSizeAssumptions,
    EstimateOptions,
    EstimationError,
    EstimationLimits,
    LimitKind,
    SanityCheck,
)
from ssat.core.estimate.estimator import _select_evenly
from ssat.core.perturb import Perturbator
from ssat.core.plan import PlanBuilder
from ssat.core.source.types import LoadError, LoadedSample, SampleMeta
from ssat.core.types import ItemStatus, PerturbationOp, RegionKind


class StepClock:
    def __init__(self, step: float = 2.0) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


class WorkClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class MemorySource:
    def __init__(
        self,
        *,
        labels: tuple[int | None, ...] = (0, 1, 2, 0),
        failed_ids: set[str] | None = None,
    ) -> None:
        self.samples = tuple(
            SampleMeta(f"sample-{index}", "unused", label)
            for index, label in enumerate(labels)
        )
        self.failed_ids = failed_ids or set()

    def list_samples(self) -> list[SampleMeta]:
        return list(self.samples)

    def load(self, sample_id: str) -> LoadedSample | LoadError:
        if sample_id in self.failed_ids:
            return LoadError(sample_id, "injected", "load failed")
        index = int(sample_id.rsplit("-", 1)[1])
        label = self.samples[index].gt_label
        channel = 0 if label is None or not 0 <= label < 3 else label
        array = np.zeros((1, 4, 4, 3), dtype=np.uint8)
        array[..., channel] = 255
        return LoadedSample(array, sample_id, array.shape, "a" * 64, label)


class HalfFailPerturbator(Perturbator):
    def __init__(self) -> None:
        pass

    def apply(
        self,
        array: np.ndarray,
        mask: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> np.ndarray:
        if bool(mask[0, 0]):
            raise RuntimeError("injected prepare failure")
        return array.copy()


class AlwaysFailPerturbator(Perturbator):
    def __init__(self) -> None:
        pass

    def apply(self, *args: object, **kwargs: object) -> np.ndarray:
        raise RuntimeError("injected prepare failure")


class EmptyResume:
    def pending_clean_samples(self, samples: object, *, retry_failed: bool) -> tuple:
        return ()

    def pending_chunks(self, chunks: object, *, retry_failed: bool) -> tuple:
        return ()


class PartialResume:
    def pending_clean_samples(self, samples: object, *, retry_failed: bool) -> tuple:
        return tuple(samples)[:1]

    def pending_chunks(self, chunks: object, *, retry_failed: bool) -> tuple:
        return tuple(chunks)[-2:]


def _predict_channels(batch: np.ndarray) -> np.ndarray:
    return batch.astype(np.float32).sum(axis=(1, 2, 3))


def _adapter(
    predict_fn=_predict_channels,
    *,
    wrong_preprocessing: bool = False,
    cleanup_after_oom_fn=None,
) -> CallableAdapter:
    if wrong_preprocessing:
        preprocessor = CallablePreprocessor(
            transform_batch_fn=lambda batch: batch[..., [1, 2, 0]],
            transform_mask_fn=lambda mask: mask,
            description="wrong-channel-order",
        )
        return CallableAdapter(
            predict_fn,
            model_id="classifier",
            class_names=("zero", "one", "two"),
            preprocessor=preprocessor,
            cleanup_after_oom_fn=cleanup_after_oom_fn,
        )
    return CallableAdapter(
        predict_fn,
        model_id="classifier",
        class_names=("zero", "one", "two"),
        transform_mask_fn=lambda mask: mask,
        cleanup_after_oom_fn=cleanup_after_oom_fn,
    )


def _config(
    tmp_path: Path,
    adapter: CallableAdapter,
    *,
    target_batch_size: int = 4,
    variants_per_chunk: int = 3,
) -> ResolvedConfig:
    return ResolvedConfig(
        config_base_dir=tmp_path.resolve(),
        regions=(
            ResolvedRegionConfig(
                region_id="grid",
                kind=RegionKind.GRID,
                params={"rows": 1, "cols": 2},
            ),
        ),
        perturbations=(
            PerturbationConfig(
                op=PerturbationOp.GAUSSIAN_NOISE,
                params={"sigma": 1.0},
                seed_salts=(0, 1),
            ),
        ),
        runtime=RuntimeConfig(
            variants_per_chunk=variants_per_chunk,
            target_batch_size=target_batch_size,
        ),
        dump=DumpConfig(),
        adapter_spec=adapter.describe(),
    )


def test_estimate_options_validate_defaults_and_ranges() -> None:
    options = EstimateOptions()
    assert options.max_profile_chunks == 20
    assert options.limits.max_pending_items == 1_000_000
    assert options.dump_size.compression_ratio == 0.6

    with pytest.raises(ValueError, match="max_profile_chunks"):
        EstimateOptions(max_profile_chunks=0)
    with pytest.raises(ValueError, match="minimum_accuracy"):
        EstimateOptions(minimum_accuracy=1.1)
    with pytest.raises(ValueError, match="compression_ratio"):
        DumpSizeAssumptions(compression_ratio=0.0)
    with pytest.raises(ValueError, match="max_pending_items"):
        EstimationLimits(max_pending_items=0)


@pytest.mark.parametrize(
    ("values", "limit", "expected"),
    [
        ((0, 1, 2), 5, (0, 1, 2)),
        (tuple(range(10)), 3, (0, 4, 9)),
        (tuple(range(10)), 1, (4,)),
    ],
)
def test_even_selection_spans_sequence(
    values: tuple[int, ...],
    limit: int,
    expected: tuple[int, ...],
) -> None:
    assert _select_evenly(values, limit) == expected


def test_sanity_check_detects_wrong_preprocessing(tmp_path: Path) -> None:
    source = MemorySource()
    correct = _adapter()
    correct_config = _config(tmp_path, correct)
    good = SanityCheck(
        minimum_accuracy=0.75,
        clock=StepClock(),
    ).run(correct_config, source.samples, source, correct)

    wrong = _adapter(wrong_preprocessing=True)
    wrong_config = _config(tmp_path, wrong)
    bad = SanityCheck(
        minimum_accuracy=0.75,
        clock=StepClock(),
    ).run(wrong_config, source.samples, source, wrong)

    assert good.accuracy == 1.0
    assert good.passed is True
    assert bad.accuracy == 0.0
    assert bad.passed is False
    assert AdvisoryCode.SANITY_ACCURACY_BELOW_MINIMUM in {
        advisory.code for advisory in bad.advisories
    }


def test_sanity_excludes_unlabeled_invalid_and_nonfinite_outputs(
    tmp_path: Path,
) -> None:
    source = MemorySource(labels=(0, None, 9, 0), failed_ids={"sample-3"})

    def nonfinite(batch: np.ndarray) -> np.ndarray:
        output = _predict_channels(batch)
        if len(output) > 1:
            output[1, 0] = np.nan
        return output

    adapter = _adapter(nonfinite)
    config = _config(tmp_path, adapter)
    result = SanityCheck(
        minimum_accuracy=0.5,
        clock=StepClock(),
    ).run(config, source.samples, source, adapter)

    assert result.terminal_samples == 4
    assert result.labeled_predictions == 1
    assert result.correct_predictions == 1
    assert result.invalid_logit_predictions == 1
    assert result.invalid_label_predictions == 1
    assert result.status_counts[ItemStatus.LOAD_FAILED] == 1
    assert result.accuracy == 1.0


def test_unlabeled_sanity_is_informational_without_minimum(tmp_path: Path) -> None:
    source = MemorySource(labels=(None, None))
    adapter = _adapter()
    config = _config(tmp_path, adapter)
    report = CostEstimator(clock=StepClock()).estimate(
        config,
        PlanBuilder(config, source),
        source,
        adapter,
    )

    assert report.sanity is not None
    assert report.sanity.accuracy is None
    assert report.sanity.passed is None
    assert report.confirmation_required is False
    assert AdvisoryCode.SANITY_NO_LABELED_OUTPUTS in {
        advisory.code for advisory in report.advisories
    }


def test_cost_estimator_reports_exact_counts_time_calls_and_dump_formula(
    tmp_path: Path,
) -> None:
    source = MemorySource()
    adapter = _adapter()
    config = _config(tmp_path, adapter)
    builder = PlanBuilder(config, source)

    report = CostEstimator(clock=StepClock()).estimate(
        config,
        builder,
        source,
        adapter,
    )

    raw = 4 * (3 * 4 + 128) + 16 * (3 * 4 + 384 + 96)
    assert report.total_clean_samples == 4
    assert report.total_chunks == 8
    assert report.total_perturbed_items == 16
    assert report.pending_perturbed_items == 16
    assert report.profile is not None
    assert report.profile.selected_chunks == 8
    assert report.profile.terminal_items == 16
    assert report.sanity is not None
    assert report.estimated_remaining_seconds == pytest.approx(4.0)
    assert report.estimated_inference_calls == 5
    assert report.estimated_total_dump_bytes == 16 * 1024 + int(np.ceil(0.6 * raw))
    assert report.estimated_remaining_dump_bytes == int(np.ceil(0.6 * raw))
    assert report.confirmation_required is False


def test_profile_extrapolation_matches_virtual_full_execution_cost(
    tmp_path: Path,
) -> None:
    clock = WorkClock()

    class TimedSource(MemorySource):
        def load(self, sample_id: str) -> LoadedSample | LoadError:
            clock.advance(1.0)
            return super().load(sample_id)

    source = TimedSource(labels=(0, 1, 2, 0, 1, 2))

    def timed_predict(batch: np.ndarray) -> np.ndarray:
        clock.advance(2.0 * len(batch))
        return _predict_channels(batch)

    adapter = _adapter(timed_predict)
    config = _config(tmp_path, adapter)
    report = CostEstimator(
        EstimateOptions(max_profile_chunks=2, max_sanity_samples=2),
        clock=clock,
    ).estimate(config, PlanBuilder(config, source), source, adapter)

    # Full virtual work: clean 6 * (load 1 + infer 2), plus perturbed
    # 12 chunk loads and 24 item inferences.
    assert report.estimated_remaining_seconds == pytest.approx(78.0)


def test_limits_use_strict_greater_than_and_recommend_fraction(tmp_path: Path) -> None:
    source = MemorySource()
    adapter = _adapter()
    config = _config(tmp_path, adapter)
    builder = PlanBuilder(config, source)
    exact = EstimateOptions(
        limits=EstimationLimits(
            max_pending_items=16,
            max_estimated_seconds=4.0,
            max_remaining_dump_bytes=10**9,
        )
    )
    exact_report = CostEstimator(exact, clock=StepClock()).estimate(
        config, builder, source, adapter
    )
    assert exact_report.exceedances == ()

    exceeded = EstimateOptions(
        limits=EstimationLimits(
            max_pending_items=8,
            max_estimated_seconds=None,
            max_remaining_dump_bytes=None,
        )
    )
    report = CostEstimator(exceeded, clock=StepClock()).estimate(
        config, builder, source, adapter
    )
    assert [item.kind for item in report.exceedances] == [LimitKind.PENDING_ITEMS]
    assert report.recommended_sample_fraction == 0.5
    assert report.confirmation_required is True


def test_profile_collects_partial_failures_and_memory_recommendation(
    tmp_path: Path,
) -> None:
    source = MemorySource()
    adapter = _adapter()
    config = _config(tmp_path, adapter)
    builder = PlanBuilder(config, source)
    options = EstimateOptions(prepared_chunk_budget_bytes=50)

    report = CostEstimator(options, clock=StepClock()).estimate(
        config,
        builder,
        source,
        adapter,
        perturbator=HalfFailPerturbator(),
    )

    assert report.profile is not None
    assert report.profile.status_counts[ItemStatus.PREPARE_FAILED] == 8
    assert report.profile.status_counts[ItemStatus.OK] == 8
    assert report.recommended_variants_per_chunk == 1
    assert AdvisoryCode.PROFILE_PARTIAL_FAILURES in {
        advisory.code for advisory in report.advisories
    }
    assert AdvisoryCode.PREPARED_CHUNK_MEMORY_HIGH in {
        advisory.code for advisory in report.advisories
    }


def test_profile_rejects_zero_successful_inference(tmp_path: Path) -> None:
    source = MemorySource()
    adapter = _adapter()
    config = _config(tmp_path, adapter)
    builder = PlanBuilder(config, source)

    with pytest.raises(EstimationError, match="no successful inference"):
        CostEstimator(clock=StepClock()).estimate(
            config,
            builder,
            source,
            adapter,
            perturbator=AlwaysFailPerturbator(),
        )


def test_profile_recovers_oom_and_counts_cleanup(tmp_path: Path) -> None:
    cleanup_calls: list[None] = []

    def oom_above_one(batch: np.ndarray) -> np.ndarray:
        if len(batch) > 1:
            raise AdapterOutOfMemoryError("injected")
        return _predict_channels(batch)

    source = MemorySource()
    adapter = _adapter(
        oom_above_one,
        cleanup_after_oom_fn=lambda: cleanup_calls.append(None),
    )
    config = _config(tmp_path, adapter)
    report = CostEstimator(
        EstimateOptions(max_sanity_samples=1),
        clock=StepClock(),
    ).estimate(
        config,
        PlanBuilder(config, source),
        source,
        adapter,
    )

    assert report.profile is not None
    assert report.profile.oom_events > 0
    assert report.profile.final_batch_size == 1
    assert cleanup_calls


def test_completed_resume_skips_model_calls(tmp_path: Path) -> None:
    calls: list[int] = []

    def predict(batch: np.ndarray) -> np.ndarray:
        calls.append(len(batch))
        return _predict_channels(batch)

    source = MemorySource()
    adapter = _adapter(predict)
    config = _config(tmp_path, adapter)
    report = CostEstimator(clock=StepClock()).estimate(
        config,
        PlanBuilder(config, source),
        source,
        adapter,
        resume_index=EmptyResume(),
    )

    assert calls == []
    assert report.pending_clean_samples == 0
    assert report.pending_chunks == 0
    assert report.pending_perturbed_items == 0
    assert report.profile is None
    assert report.sanity is None
    assert report.estimated_remaining_seconds == 0.0
    assert report.estimated_remaining_dump_bytes == 0


def test_resume_filter_controls_pending_counts(tmp_path: Path) -> None:
    source = MemorySource()
    adapter = _adapter()
    config = _config(tmp_path, adapter)
    report = CostEstimator(clock=StepClock()).estimate(
        config,
        PlanBuilder(config, source),
        source,
        adapter,
        resume_index=PartialResume(),
    )

    assert report.total_clean_samples == 4
    assert report.pending_clean_samples == 1
    assert report.total_chunks == 8
    assert report.pending_chunks == 2
    assert report.pending_perturbed_items == 4


def test_estimate_import_keeps_torch_lazy() -> None:
    code = "import sys; import ssat.core.estimate; assert 'torch' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
