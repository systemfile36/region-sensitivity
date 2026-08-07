"""Public failures raised while persisting or reading audit dumps."""


class DumpError(RuntimeError):
    """Base error for the raw dump subsystem."""


class DumpSchemaError(DumpError):
    """Indicate an unsupported or malformed versioned schema."""


class DumpCorruptionError(DumpError):
    """Indicate inconsistent files or references within a dump directory."""
