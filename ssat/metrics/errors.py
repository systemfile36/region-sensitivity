"""Public failures raised while computing or persisting audit metrics."""


class MetricsError(RuntimeError):
    """Base error for the metrics subsystem."""


class MetricsSchemaError(MetricsError):
    """Indicate an unsupported or malformed versioned metrics schema."""


class MetricsCorruptionError(MetricsError):
    """Indicate inconsistent or missing dump records required by a metric."""


class MetricsRegistryError(MetricsError):
    """Indicate invalid metric registration, such as a duplicate name."""


class DebugVizError(MetricsError):
    """Indicate that a DebugViz view cannot be reconstructed from a dump.

    Raised when a dump lacks the information a view needs to reproduce a
    mask or perturbation deterministically — for example an explicit region
    (whose ``ref``/``ref_hash`` are not preserved past config resolution, see
    metrics/viz/mask_check.py), a missing ``source_provenance``, or a source
    image that fails to load.
    """
