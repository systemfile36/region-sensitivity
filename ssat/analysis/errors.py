"""Public failures raised while computing or persisting control/stability analysis."""


class AnalysisError(RuntimeError):
    """Base error for the control/stability analysis subsystem."""


class AnalysisSchemaError(AnalysisError):
    """Indicate an unsupported or malformed versioned analysis schema."""


class AnalysisCorruptionError(AnalysisError):
    """Indicate inconsistent or missing records required by an analysis step."""
