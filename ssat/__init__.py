"""Spatial Sensitivity Audit Toolkit."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version

from ssat.core.types import SCHEMA_VERSION

try:
    # pyproject.toml's [project].version is the single source of truth;
    # read it back from installed package metadata instead of duplicating
    # the literal here so the two can't drift.
    __version__ = _installed_version("ssat")
except PackageNotFoundError:  # pragma: no cover - only when run from an uninstalled checkout
    __version__ = "0.0.0+unknown"

__all__ = ["SCHEMA_VERSION", "__version__"]
