"""Minimal adapter interface required during configuration resolution."""

from __future__ import annotations

from typing import Protocol

from ssat.core.adapter.types import AdapterSpec


class SupportsDescribe(Protocol):
    """Provide deterministic adapter metadata without running inference."""

    def describe(self) -> AdapterSpec:
        """Return the adapter contract inspected by ConfigResolver."""

