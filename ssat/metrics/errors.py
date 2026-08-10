"""Public failures raised while computing or persisting audit metrics."""


class MetricsError(RuntimeError):
    """Base error for the metrics subsystem."""


class MetricsSchemaError(MetricsError):
    """Indicate an unsupported or malformed versioned metrics schema."""


class MetricsCorruptionError(MetricsError):
    """Indicate inconsistent or missing dump records required by a metric."""


class MetricsRegistryError(MetricsError):
    """Indicate invalid metric registration, such as a duplicate name."""
