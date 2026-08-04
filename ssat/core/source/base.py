"""Framework-independent sample source interface."""

from __future__ import annotations

from typing import Protocol

from ssat.core.source.types import LoadError, LoadedSample, SampleMeta


class SampleSource(Protocol):
    """List dataset samples and load decoded source pixels by identifier."""

    def list_samples(self) -> list[SampleMeta]:
        """Return lightweight metadata for every audit candidate."""

    def load(self, sample_id: str) -> LoadedSample | LoadError:
        """Load one sample or return a recoverable failure value."""

