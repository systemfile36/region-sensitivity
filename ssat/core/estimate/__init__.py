"""Expose execution cost estimation and clean sanity-check contracts."""

from ssat.core.estimate.errors import EstimationError
from ssat.core.estimate.area_sanity import AreaSanityCheck
from ssat.core.estimate.estimator import CostEstimator
from ssat.core.estimate.sanity import SanityCheck
from ssat.core.estimate.types import (
    Advisory,
    AdvisoryCode,
    AreaSanityResult,
    DumpSizeAssumptions,
    EstimateOptions,
    EstimateReport,
    EstimationLimits,
    LimitExceedance,
    LimitKind,
    ProfileResult,
    SanityCheckResult,
)

__all__ = [
    "Advisory",
    "AdvisoryCode",
    "AreaSanityCheck",
    "AreaSanityResult",
    "CostEstimator",
    "DumpSizeAssumptions",
    "EstimateOptions",
    "EstimateReport",
    "EstimationError",
    "EstimationLimits",
    "LimitExceedance",
    "LimitKind",
    "ProfileResult",
    "SanityCheck",
    "SanityCheckResult",
]
