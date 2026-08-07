"""Stable schemas and public persistence contracts for raw audit dumps."""

from ssat.core.dump.errors import (
    DumpCorruptionError,
    DumpError,
    DumpSchemaError,
)
from ssat.core.dump.reader import DumpReader
from ssat.core.dump.schema import CLEAN_SCHEMA, INDEX_SCHEMA, PERTURBED_SCHEMA
from ssat.core.dump.types import (
    CleanDumpRecord,
    EnvironmentSpec,
    PerturbedDumpRecord,
    ResumeEvent,
    RunManifest,
)
from ssat.core.dump.writer import DumpWriter

__all__ = [
    "CLEAN_SCHEMA",
    "INDEX_SCHEMA",
    "PERTURBED_SCHEMA",
    "CleanDumpRecord",
    "DumpCorruptionError",
    "DumpError",
    "DumpReader",
    "DumpSchemaError",
    "DumpWriter",
    "EnvironmentSpec",
    "PerturbedDumpRecord",
    "ResumeEvent",
    "RunManifest",
]
