"""Bounded preprocessing/effective-area consistency checks."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

from ssat.core.adapter.base import ModelAdapter
from ssat.core.config.schema import ResolvedConfig
from ssat.core.estimate.errors import EstimationError
from ssat.core.estimate.measurement import _select_evenly, _validate_provenance
from ssat.core.estimate.types import Advisory, AdvisoryCode, AreaSanityResult
from ssat.core.perturb.rng import derive
from ssat.core.plan.builder import PlanBuilder
from ssat.core.plan.types import WorkItem
from ssat.core.region.mask_base import mean_frame_area
from ssat.core.region.resolver import RegionResolver
from ssat.core.region.types import RegionMeta
from ssat.core.source.base import SampleSource
from ssat.core.source.types import LoadError, LoadedSample, SampleMeta


class AreaSanityCheck:
    """Check whether preprocessing preserves intended region-area fractions.

    The check is bounded independently of perturbed profiling. Representative
    samples are selected evenly, while perturbation/seed duplicates of the
    same region geometry are removed before applying the per-sample cap.

    Args:
        max_samples: Maximum number of representative source samples.
        max_regions_per_sample: Maximum region geometries per selected sample.
        max_relative_deviation: Inclusive PASS threshold.
    """

    def __init__(
        self,
        *,
        max_samples: int = 3,
        max_regions_per_sample: int = 256,
        max_relative_deviation: float = 0.05,
    ) -> None:
        for name, value in (
            ("max_samples", max_samples),
            ("max_regions_per_sample", max_regions_per_sample),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(max_relative_deviation, bool)
            or not isinstance(max_relative_deviation, (int, float))
            or not math.isfinite(max_relative_deviation)
            or max_relative_deviation < 0.0
        ):
            raise ValueError(
                "max_relative_deviation must be non-negative and finite"
            )
        self._max_samples = max_samples
        self._max_regions_per_sample = max_regions_per_sample
        self._max_relative_deviation = float(max_relative_deviation)

    def run(
        self,
        config: ResolvedConfig,
        samples: Sequence[SampleMeta],
        plan_builder: PlanBuilder,
        sample_source: SampleSource,
        adapter: ModelAdapter,
        *,
        region_resolver: RegionResolver | None = None,
    ) -> AreaSanityResult:
        """Run the bounded check without perturbation or model inference."""

        _validate_provenance(config, adapter)
        selected_samples = _select_evenly(tuple(samples), self._max_samples)
        if not selected_samples:
            raise EstimationError("area sanity check requires at least one sample")

        if not adapter.describe().mask_transform_available:
            advisory = Advisory(
                AdvisoryCode.AREA_SANITY_UNAVAILABLE,
                "Region area consistency is unavailable because the adapter "
                "does not expose model-space mask geometry.",
            )
            return AreaSanityResult(
                selected_samples=len(selected_samples),
                candidate_regions=0,
                evaluated_regions=0,
                failed_regions=0,
                maximum_relative_deviation=None,
                tolerance=self._max_relative_deviation,
                passed=None,
                coverage_truncated=False,
                worst_sample_id=None,
                worst_region_id=None,
                worst_region_instance_id=None,
                worst_intended_area_ratio=None,
                worst_effective_area_ratio=None,
                advisories=(advisory,),
            )

        resolver = region_resolver or RegionResolver()
        chunks_by_sample: dict[str, list[str]] = {
            sample.sample_id: [] for sample in selected_samples
        }
        for chunk in plan_builder.enumerate():
            if chunk.sample_id in chunks_by_sample:
                chunks_by_sample[chunk.sample_id].append(chunk.chunk_id)

        candidate_regions = 0
        evaluated_regions = 0
        failed_regions = 0
        deviation_failures = 0
        evaluation_failures = 0
        coverage_truncated = False
        maximum_relative_deviation: float | None = None
        worst_rank = -1.0
        worst_sample_id: str | None = None
        worst_item: WorkItem | None = None
        worst_intended_ratio: float | None = None
        worst_effective_ratio: float | None = None

        for sample in selected_samples:
            candidates = self._deduplicated_items(
                plan_builder,
                tuple(chunks_by_sample[sample.sample_id]),
            )
            candidate_regions += len(candidates)
            if len(candidates) > self._max_regions_per_sample:
                coverage_truncated = True
            selected_items = _select_evenly(
                candidates,
                self._max_regions_per_sample,
            )

            loaded = self._load_sample(sample_source, sample.sample_id)
            if loaded is None:
                failed_regions += len(selected_items)
                evaluation_failures += len(selected_items)
                continue

            for item in selected_items:
                try:
                    intended_ratio, effective_ratio = self._measure_item(
                        config,
                        loaded,
                        item,
                        resolver,
                        adapter,
                    )
                except Exception:
                    failed_regions += 1
                    evaluation_failures += 1
                    continue

                evaluated_regions += 1
                deviation = _relative_deviation(
                    intended_ratio,
                    effective_ratio,
                )
                rank = deviation if math.isfinite(deviation) else math.inf
                if rank > worst_rank:
                    worst_rank = rank
                    worst_sample_id = sample.sample_id
                    worst_item = item
                    worst_intended_ratio = intended_ratio
                    worst_effective_ratio = effective_ratio
                if math.isfinite(deviation) and (
                    maximum_relative_deviation is None
                    or deviation > maximum_relative_deviation
                ):
                    maximum_relative_deviation = deviation
                if deviation > self._max_relative_deviation:
                    failed_regions += 1
                    deviation_failures += 1

        advisories: list[Advisory] = []
        if deviation_failures:
            advisories.append(
                Advisory(
                    AdvisoryCode.AREA_SANITY_DEVIATION_EXCEEDED,
                    "One or more effective region-area ratios exceed the "
                    "configured relative-deviation tolerance.",
                )
            )
        if evaluation_failures:
            advisories.append(
                Advisory(
                    AdvisoryCode.AREA_SANITY_PARTIAL_FAILURES,
                    "Some region geometries could not be evaluated for area "
                    "consistency.",
                )
            )
        if coverage_truncated:
            advisories.append(
                Advisory(
                    AdvisoryCode.AREA_SANITY_COVERAGE_TRUNCATED,
                    "Region area consistency used bounded region coverage; "
                    "increase max_area_sanity_regions_per_sample for more coverage.",
                )
            )

        return AreaSanityResult(
            selected_samples=len(selected_samples),
            candidate_regions=candidate_regions,
            evaluated_regions=evaluated_regions,
            failed_regions=failed_regions,
            maximum_relative_deviation=maximum_relative_deviation,
            tolerance=self._max_relative_deviation,
            passed=failed_regions == 0,
            coverage_truncated=coverage_truncated,
            worst_sample_id=worst_sample_id,
            worst_region_id=(
                None if worst_item is None else worst_item.region_spec.region_id
            ),
            worst_region_instance_id=(
                None
                if worst_item is None
                else worst_item.region_spec.region_instance_id
            ),
            worst_intended_area_ratio=worst_intended_ratio,
            worst_effective_area_ratio=worst_effective_ratio,
            advisories=tuple(advisories),
        )

    @staticmethod
    def _deduplicated_items(
        plan_builder: PlanBuilder,
        chunk_ids: tuple[str, ...],
    ) -> tuple[WorkItem, ...]:
        """Return the first planned item for each region geometry."""

        unique: dict[tuple[str, str, bool, bool], WorkItem] = {}
        for chunk_id in chunk_ids:
            for item in plan_builder.materialize(chunk_id).items:
                key = (
                    item.region_spec.region_id,
                    item.region_spec.region_instance_id,
                    item.invert_mask,
                    item.is_control,
                )
                unique.setdefault(key, item)
        return tuple(unique.values())

    @staticmethod
    def _load_sample(
        sample_source: SampleSource,
        sample_id: str,
    ) -> LoadedSample | None:
        """Load one sample, normalizing recoverable and raised failures."""

        try:
            loaded = sample_source.load(sample_id)
        except Exception:
            return None
        if isinstance(loaded, LoadError):
            return None
        if not isinstance(loaded, LoadedSample):
            return None
        if loaded.sample_id != sample_id:
            return None
        return loaded

    @staticmethod
    def _measure_item(
        config: ResolvedConfig,
        loaded: LoadedSample,
        item: WorkItem,
        resolver: RegionResolver,
        adapter: ModelAdapter,
    ) -> tuple[float, float]:
        """Measure source- and model-space area ratios for one geometry."""

        seed = derive(config.runtime.global_seed, item.item_id, item.seed_salt)
        mask, region_meta = resolver.resolve(
            loaded.original_shape,
            item.region_spec,
            np.random.default_rng(seed),
        )
        if item.invert_mask:
            mask = np.logical_not(mask)
            height, width = mask.shape[-2:]
            area = mean_frame_area(mask)
            region_meta = RegionMeta(
                intended_area_px=int(round(area)),
                intended_area_ratio=area / (height * width),
                generator_kind=region_meta.generator_kind,
                generator_version=region_meta.generator_version,
                confidence=region_meta.confidence,
            )

        transformed = adapter.transform_mask(mask)
        if transformed is None:
            raise TypeError("adapter returned no transformed mask")
        if (
            not isinstance(transformed, np.ndarray)
            or transformed.dtype != np.bool_
            or transformed.ndim not in (2, 3)
        ):
            raise TypeError("adapter returned an invalid transformed mask")
        height, width = transformed.shape[-2:]
        effective_ratio = mean_frame_area(transformed) / (height * width)
        return region_meta.intended_area_ratio, effective_ratio


def _relative_deviation(intended: float, effective: float) -> float:
    """Return relative drift with an explicit zero-area contract."""

    if intended == 0.0:
        return 0.0 if effective == 0.0 else math.inf
    return abs(effective - intended) / intended


__all__ = ["AreaSanityCheck"]
