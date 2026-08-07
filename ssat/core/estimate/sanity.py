"""Bounded clean accuracy and throughput sanity checks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
import logging
import math
import time

import numpy as np

from ssat.core.adapter.base import ModelAdapter
from ssat.core.config.schema import ResolvedConfig
from ssat.core.estimate.errors import EstimationError
from ssat.core.estimate.measurement import (
    _CountingAdapter,
    _complete_counts,
    _elapsed,
    _merge_class_count,
    _new_batch_size_state,
    _select_evenly,
    _validate_provenance,
)
from ssat.core.estimate.types import Advisory, AdvisoryCode, SanityCheckResult
from ssat.core.runtime.pipeline import (
    iter_clean_preparation_results,
    iter_prediction_batches,
)
from ssat.core.runtime.types import BatchSizeState, CleanInferenceItem
from ssat.core.source.base import SampleSource
from ssat.core.source.types import LoadError, SampleMeta
from ssat.core.types import ItemStatus
from ssat.utils.logger_factory import get_logger


class SanityCheck:
    """Measure clean throughput and optional labeled top-1 accuracy.

    Args:
        max_samples: Positive maximum number of clean samples to measure.
        minimum_accuracy: Optional inclusive top-1 accuracy threshold.
        clock: Monotonic callable used to measure elapsed time.
        logger: Optional event logger for accuracy warnings.

    Raises:
        ValueError: If a numeric option is outside its accepted range.
        TypeError: If ``clock`` is not callable.
    """

    def __init__(
        self,
        *,
        max_samples: int = 20,
        minimum_accuracy: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize a bounded clean sanity check.

        Args:
            max_samples: Positive maximum number of samples to measure.
            minimum_accuracy: Optional inclusive top-1 threshold.
            clock: Monotonic callable used for elapsed time.
            logger: Optional event logger for threshold warnings.

        Raises:
            ValueError: If a numeric option is outside its accepted range.
            TypeError: If ``clock`` is not callable.
        """

        if (
            isinstance(max_samples, bool)
            or not isinstance(max_samples, int)
            or max_samples <= 0
        ):
            raise ValueError("max_samples must be a positive integer")
        if minimum_accuracy is not None and (
            isinstance(minimum_accuracy, bool)
            or not isinstance(minimum_accuracy, (int, float))
            or not math.isfinite(minimum_accuracy)
            or not 0.0 <= minimum_accuracy <= 1.0
        ):
            raise ValueError("minimum_accuracy must be between 0 and 1")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._max_samples = max_samples
        self._minimum_accuracy = minimum_accuracy
        self._clock = clock
        self._logger = logger or get_logger(__name__)

    def run(
        self,
        config: ResolvedConfig,
        samples: Sequence[SampleMeta],
        sample_source: SampleSource,
        adapter: ModelAdapter,
        *,
        batch_size_state: BatchSizeState | None = None,
    ) -> SanityCheckResult:
        """Run a bounded clean pass without writing dump records.

        Args:
            config: Fully resolved audit configuration.
            samples: Ordered clean sample candidates.
            sample_source: Source used by clean workers.
            adapter: Model adapter measured by the sanity pass.
            batch_size_state: Optional adaptive state shared with profiling.

        Returns:
            Clean throughput, output validity, and optional accuracy metrics.

        Raises:
            EstimationError: If selection, execution, or output validation fails.
            TypeError: If provenance inputs have invalid types.
        """

        _validate_provenance(config, adapter)
        selected = _select_evenly(tuple(samples), self._max_samples)
        if not selected:
            raise EstimationError("sanity check requires at least one sample")

        status_counts: Counter[ItemStatus] = Counter()
        terminal = 0
        successful = 0
        labeled = 0
        correct = 0
        unlabeled = 0
        invalid_labels = 0
        invalid_logits = 0
        class_count: int | None = None
        state = batch_size_state or _new_batch_size_state(
            config,
            adapter.describe(),
        )
        starting_batch_size = state.current_size
        starting_oom_events = state.oom_events
        counting_adapter = _CountingAdapter(adapter)

        def clean_items() -> Iterable[CleanInferenceItem]:
            """Convert shared clean results into measured inference inputs.

            Yields:
                Successfully loaded clean inference items.
            """

            nonlocal terminal
            for result in iter_clean_preparation_results(
                tuple(selected),
                sample_source,
                num_workers=config.runtime.num_workers,
            ):
                if isinstance(result, LoadError):
                    terminal += 1
                    status_counts[ItemStatus.LOAD_FAILED] += 1
                    continue
                yield result

        # Measure loading and inference together to model end-to-end throughput.
        started = self._clock()
        try:
            for outcome in iter_prediction_batches(
                clean_items(),
                counting_adapter,
                state,
            ):
                if outcome.error is not None:
                    raise outcome.error
                for result in outcome.results:
                    terminal += 1
                    status_counts[result.status] += 1
                    if result.status is not ItemStatus.OK or result.output is None:
                        continue
                    output = result.output.logits
                    if output.size == 0 or not np.all(np.isfinite(output)):
                        invalid_logits += 1
                        continue
                    class_count = _merge_class_count(
                        class_count,
                        int(output.size),
                    )
                    successful += 1
                    sample = result.item.sample
                    if sample.gt_label is None:
                        unlabeled += 1
                    elif not 0 <= sample.gt_label < output.size:
                        invalid_labels += 1
                    else:
                        labeled += 1
                        correct += int(np.argmax(output) == sample.gt_label)
        except EstimationError:
            raise
        except Exception as error:
            raise EstimationError("clean sanity execution failed") from error
        elapsed = _elapsed(self._clock() - started)

        accuracy = None if labeled == 0 else correct / labeled
        passed = (
            None
            if self._minimum_accuracy is None
            else accuracy is not None and accuracy >= self._minimum_accuracy
        )
        advisories: list[Advisory] = []
        failures = terminal - status_counts[ItemStatus.OK]
        if failures or invalid_labels or invalid_logits:
            advisories.append(
                Advisory(
                    AdvisoryCode.SANITY_PARTIAL_FAILURES,
                    "Clean sanity check contained failed or invalid outputs.",
                )
            )
        if accuracy is None:
            advisories.append(
                Advisory(
                    AdvisoryCode.SANITY_NO_LABELED_OUTPUTS,
                    "No successful labeled output was available for top-1 accuracy.",
                )
            )
        if passed is False:
            advisories.append(
                Advisory(
                    AdvisoryCode.SANITY_ACCURACY_BELOW_MINIMUM,
                    "Clean top-1 accuracy is unavailable or below the requested minimum.",
                )
            )
            self._logger.warning("estimate.sanity_accuracy_below_minimum")

        return SanityCheckResult(
            selected_samples=len(selected),
            terminal_samples=terminal,
            successful_predictions=successful,
            labeled_predictions=labeled,
            correct_predictions=correct,
            unlabeled_predictions=unlabeled,
            invalid_label_predictions=invalid_labels,
            invalid_logit_predictions=invalid_logits,
            status_counts=_complete_counts(status_counts),
            elapsed_seconds=elapsed,
            items_per_second=terminal / elapsed,
            inference_calls=counting_adapter.inference_calls,
            oom_events=state.oom_events - starting_oom_events,
            initial_batch_size=starting_batch_size,
            final_batch_size=state.current_size,
            class_count=class_count,
            accuracy=accuracy,
            minimum_accuracy=self._minimum_accuracy,
            passed=passed,
            advisories=tuple(advisories),
        )
