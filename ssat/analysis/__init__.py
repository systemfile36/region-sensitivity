"""Control/stability analysis contracts for the SSAT toolkit."""

from ssat.analysis.control import compare_to_controls
from ssat.analysis.errors import (
    AnalysisCorruptionError,
    AnalysisError,
    AnalysisSchemaError,
)
from ssat.analysis.indexer import ComparisonIndex, ComparisonIndexer
from ssat.analysis.interval import compute_intervals
from ssat.analysis.reader import AnalysisReader
from ssat.analysis.stability import (
    compute_jitter_stability,
    compute_seed_stability,
    compute_strategy_stability,
)
from ssat.analysis.strategy_profile import compute_strategy_profile
from ssat.analysis.types import (
    Alignment,
    AnchorKey,
    AnchorRow,
    AvailableAnalyses,
    ConditionKey,
    ControlComparisonRow,
    ControlPairRow,
    CoverageReport,
    FlagValue,
    IntervalRow,
    MatchMethod,
    RankCorrelationRow,
    ReliabilityGrade,
    ReliabilityRow,
    SeedStabilityRow,
    StrategyProfileRow,
    StrategyStabilityRow,
)

__all__ = [
    "Alignment",
    "AnalysisCorruptionError",
    "AnalysisError",
    "AnalysisReader",
    "AnalysisSchemaError",
    "AnchorKey",
    "AnchorRow",
    "AvailableAnalyses",
    "ComparisonIndex",
    "ComparisonIndexer",
    "ConditionKey",
    "ControlComparisonRow",
    "ControlPairRow",
    "CoverageReport",
    "FlagValue",
    "IntervalRow",
    "MatchMethod",
    "RankCorrelationRow",
    "ReliabilityGrade",
    "ReliabilityRow",
    "SeedStabilityRow",
    "StrategyProfileRow",
    "StrategyStabilityRow",
    "compare_to_controls",
    "compute_intervals",
    "compute_jitter_stability",
    "compute_seed_stability",
    "compute_strategy_profile",
    "compute_strategy_stability",
]
