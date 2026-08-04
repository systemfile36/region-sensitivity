"""Deterministic dataset-statistics computation for configuration resolution."""

from __future__ import annotations

import logging

import numpy as np

from ssat.core.config.schema import DatasetStats
from ssat.core.source.base import SampleSource
from ssat.core.source.types import LoadError, LoadedSample
from ssat.utils.logger_factory import get_logger


class DatasetStatsError(ValueError):
    """Indicate that deterministic dataset statistics could not be computed."""


def compute_dataset_stats(
    sample_source: SampleSource,
    *,
    logger: logging.Logger | None = None,
) -> DatasetStats:
    """Compute channel means from every readable source sample.

    Samples are processed in ``sample_id`` order. Recoverable ``LoadError``
    values are logged and excluded, while contract violations stop resolution.

    Args:
        sample_source: Source providing sample metadata and decoded uint8 arrays.
        logger: Optional logger used for audit events.

    Returns:
        Dataset-wide channel means accumulated in float64.

    Raises:
        DatasetStatsError: If source enumeration or deterministic aggregation
            cannot produce a valid result.
    """

    event_logger = logger or get_logger(__name__)
    try:
        samples = list(sample_source.list_samples())
    except Exception as error:
        raise DatasetStatsError("sample_source.list_samples() failed") from error

    sample_ids = [sample.sample_id for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise DatasetStatsError("sample_source returned duplicate sample_id values")
    if not samples:
        raise DatasetStatsError("dataset statistics require at least one sample")

    totals: np.ndarray | None = None
    channel_count: int | None = None
    pixel_count = 0
    samples_used = 0
    samples_skipped = 0

    for sample in sorted(samples, key=lambda item: item.sample_id):
        try:
            loaded = sample_source.load(sample.sample_id)
        except Exception as error:
            raise DatasetStatsError(
                f"sample_source.load() raised for sample_id={sample.sample_id!r}"
            ) from error

        if isinstance(loaded, LoadError):
            if loaded.sample_id != sample.sample_id:
                raise DatasetStatsError(
                    "load error sample_id does not match requested sample_id "
                    f"({loaded.sample_id!r} != {sample.sample_id!r})"
                )
            samples_skipped += 1
            event_logger.warning(
                "dataset_stats.sample_skipped sample_id=%s error_type=%s message=%r",
                loaded.sample_id,
                loaded.error_type,
                loaded.message,
            )
            continue
        if not isinstance(loaded, LoadedSample):
            raise DatasetStatsError(
                "sample_source.load() must return LoadedSample or LoadError"
            )
        if loaded.sample_id != sample.sample_id:
            raise DatasetStatsError(
                "loaded sample_id does not match requested sample_id "
                f"({loaded.sample_id!r} != {sample.sample_id!r})"
            )

        current_channels = loaded.array.shape[-1]
        current_pixels = int(np.prod(loaded.array.shape[:-1], dtype=np.int64))
        if current_channels <= 0 or current_pixels <= 0:
            raise DatasetStatsError(
                f"sample_id={sample.sample_id!r} has an empty spatial/channel dimension"
            )
        if channel_count is None:
            channel_count = current_channels
            totals = np.zeros(channel_count, dtype=np.float64)
        elif current_channels != channel_count:
            raise DatasetStatsError(
                f"sample_id={sample.sample_id!r} has {current_channels} channels; "
                f"expected {channel_count}"
            )

        flattened = loaded.array.reshape(-1, current_channels)
        totals += flattened.sum(axis=0, dtype=np.float64)
        pixel_count += current_pixels
        samples_used += 1

    if samples_used == 0 or totals is None or pixel_count == 0:
        raise DatasetStatsError("all samples failed while computing dataset statistics")

    channel_mean = tuple(float(value) for value in totals / pixel_count)
    event_logger.info(
        "dataset_stats.computed samples_used=%d samples_skipped=%d pixels=%d channels=%d",
        samples_used,
        samples_skipped,
        pixel_count,
        len(channel_mean),
    )
    return DatasetStats(channel_mean=channel_mean)
