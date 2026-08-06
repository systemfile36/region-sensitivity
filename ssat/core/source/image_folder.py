"""Static image source backed by explicit sample metadata."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from ssat.core.source.types import LoadError, LoadedSample, SampleMeta
from ssat.utils.io import sha256_file


class ImageFolderSource:
    """Load RGB images referenced by an explicit sample catalog.

    Args:
        samples: Lightweight metadata containing stable IDs and image paths.

    Raises:
        TypeError: If the catalog contains a value other than ``SampleMeta``.
        ValueError: If the catalog contains duplicate sample IDs.
    """

    def __init__(self, samples: Sequence[SampleMeta]) -> None:
        catalog = tuple(samples)
        if any(not isinstance(sample, SampleMeta) for sample in catalog):
            raise TypeError("samples must contain only SampleMeta values")

        sample_ids = tuple(sample.sample_id for sample in catalog)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("samples must not contain duplicate sample_id values")

        self._samples = catalog
        self._sample_by_id = {sample.sample_id: sample for sample in catalog}

    def list_samples(self) -> list[SampleMeta]:
        """Return a fresh metadata list without loading image pixels.

        Returns:
            Sample metadata in the order supplied to the constructor.
        """

        return list(self._samples)

    def load(self, sample_id: str) -> LoadedSample | LoadError:
        """Decode one catalog image into the source pixel-space contract.

        Args:
            sample_id: Stable identifier of the image to load.

        Returns:
            An RGB ``LoadedSample`` or a recoverable ``LoadError`` value.
        """

        sample = self._sample_by_id.get(sample_id)
        if sample is None:
            return LoadError(
                sample_id=sample_id,
                error_type="sample_not_found",
                message=f"unknown sample_id: {sample_id!r}",
            )

        try:
            content_hash = sha256_file(sample.path)
            with Image.open(sample.path) as image:
                rgb_image = image.convert("RGB")
                array = np.asarray(rgb_image, dtype=np.uint8).copy()
            array = array[np.newaxis, ...]
            return LoadedSample(
                array=array,
                sample_id=sample.sample_id,
                original_shape=tuple(array.shape),
                content_hash=content_hash,
                gt_label=sample.gt_label,
            )
        except FileNotFoundError as error:
            return self._load_error(sample_id, "file_not_found", error)
        except UnidentifiedImageError as error:
            return self._load_error(sample_id, "decode_error", error)
        except OSError as error:
            return self._load_error(sample_id, "io_error", error)
        except Exception as error:  # pragma: no cover - defensive value boundary
            return self._load_error(sample_id, "load_error", error)

    @staticmethod
    def _load_error(sample_id: str, error_type: str, error: Exception) -> LoadError:
        """Convert an image loading exception into a recoverable value.

        Args:
            sample_id: Identifier requested by the caller.
            error_type: Stable machine-readable failure category.
            error: Original exception raised while loading the image.

        Returns:
            A ``LoadError`` preserving the exception type and message.
        """

        detail = str(error) or error.__class__.__name__
        return LoadError(
            sample_id=sample_id,
            error_type=error_type,
            message=f"{error.__class__.__name__}: {detail}",
        )
