"""Import and error-hierarchy checks for the metrics-engine scaffold."""

from __future__ import annotations

import importlib

import ssat.metrics
from ssat.metrics.errors import MetricsCorruptionError, MetricsError, MetricsSchemaError


def test_metrics_package_exports_every_public_symbol() -> None:
    for name in ssat.metrics.__all__:
        assert hasattr(ssat.metrics, name), f"missing export: {name}"


def test_metrics_errors_share_a_common_base() -> None:
    assert issubclass(MetricsSchemaError, MetricsError)
    assert issubclass(MetricsCorruptionError, MetricsError)
    assert issubclass(MetricsError, RuntimeError)


def test_stub_modules_import_successfully() -> None:
    for module_name in (
        "ssat.metrics.registry",
        "ssat.metrics.aggregate",
        "ssat.metrics.store",
        "ssat.metrics.viz",
        "ssat.metrics.viz.mask_check",
        "ssat.metrics.viz.heatmap",
        "ssat.metrics.viz.ranking",
    ):
        importlib.import_module(module_name)
