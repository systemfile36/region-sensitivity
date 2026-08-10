"""Execution-layer public API."""

from ssat.core.runtime.batching import BatchSplitter, Rebatcher
from ssat.core.runtime.errors import (
    RuntimeCancelledError,
    RuntimeContractError,
    RuntimeExecutionError,
)
from ssat.core.runtime.execution import run_audit
from ssat.core.runtime.processors import ChunkProcessor, CleanProcessor
from ssat.core.runtime.types import (
    BatchSizeState,
    CleanInferenceItem,
    ExecutionSummary,
    FailedChunk,
    InferenceBatch,
    InferenceItem,
    ItemMeta,
    PerturbedInferenceItem,
    PredictionResult,
    PreparedChunk,
)

__all__ = [
    "BatchSizeState",
    "BatchSplitter",
    "ChunkProcessor",
    "CleanInferenceItem",
    "CleanProcessor",
    "ExecutionSummary",
    "FailedChunk",
    "InferenceBatch",
    "InferenceItem",
    "ItemMeta",
    "PerturbedInferenceItem",
    "PredictionResult",
    "PreparedChunk",
    "Rebatcher",
    "RuntimeContractError",
    "RuntimeCancelledError",
    "RuntimeExecutionError",
    "run_audit",
]
