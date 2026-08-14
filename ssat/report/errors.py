"""Public failures raised while assembling or rendering a report."""


class ReportError(RuntimeError):
    """Base error for the reporting layer."""


class ReportSchemaError(ReportError):
    """Indicate an unsupported or malformed versioned source schema."""


class ReportDataError(ReportError):
    """Indicate missing or inconsistent data required to assemble a report.

    Examples: a requested ``metric_name`` absent from every source store, or
    an ``analysis_dir`` whose ``source_metrics_manifest_hash`` no longer
    matches the metrics store it is paired with.
    """
