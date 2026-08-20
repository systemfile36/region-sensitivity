# Application API and UI Integration

`ssat.application` exposes the same UI-independent service used by the Typer CLI. It does not read stdin, write presentation text, or display prompts. A web application, notebook, worker, or alternate CLI can therefore reuse SSAT's validation, preflight, execution, inspection, metrics, analysis, and reporting policies.

## Two-stage execution

An audit is deliberately split into preparation and execution so a UI can display cost and sanity information before asking for approval.

```python
from pathlib import Path

from ssat.application import AuditApplication, RunRequest

application = AuditApplication()
with application.prepare_run(
    RunRequest("audit.yaml", Path("/dumps/run-001")),
    event_sink=publish_event,
) as prepared:
    # Render prepared.estimate.to_dict() and obtain approval when required.
    result = application.execute_run(
        prepared,
        confirmation_granted=user_approved,
        event_sink=publish_event,
        cancellation=job_cancellation_token,
    )
```

`PreparedRun` is a process-local, one-shot object. It owns adapter resources and should be used as a context manager or explicitly closed. Immediately before execution, SSAT checks that the configuration file, source provenance file, adapter identity, and output dump snapshot have not changed since preflight. An output lock rejects concurrent writers to the same dump.

Do not serialize `PreparedRun` into a queue. A distributed worker should receive a serializable job specification and create its own `RunRequest` and `PreparedRun`.

## Confirmation

`prepared.confirmation_required` mirrors `prepared.estimate.report.confirmation_required`. If it is true, `execute_run` raises `ApplicationError` with code `confirmation_required` unless `confirmation_granted=True` is passed. Passing confirmation for a run that does not require it is harmless.

The estimate is bounded by `EstimateOptions`; it does not execute the complete audit. It profiles selected clean samples and perturbation chunks and reports estimated remaining time, dump size, pending work, advisories, recommendations, and optional clean top-1 sanity accuracy.

## Events and cancellation

```python
from ssat.application import ApplicationEvent, CancellationToken

events: list[ApplicationEvent] = []
token = CancellationToken()

prepared = application.prepare_run(request, event_sink=events.append)
result = application.execute_run(
    prepared,
    confirmation_granted=True,
    event_sink=events.append,
    cancellation=token,
)
```

`ApplicationEvent` contains `kind`, `phase`, optional `completed` and `total` counters, and an optional `message`. Current service emissions use lifecycle kinds such as `started`, `progress`, `completed`, and `cancelled`; they do not include logits or a complete configuration. Exceptions raised by an event sink are logged and do not abort the audit.

`CancellationToken.cancel()` is thread-safe. Execution stops at the next safe runtime boundary, flushes the durable fragment, and returns a `RunResult` whose status is `cancelled`. The same configuration and output path can later resume the dump.

## Standalone operations

Every public result provides `to_dict()` with JSON-compatible paths, enums, and dataclasses.

```python
from pathlib import Path

from ssat.application import (
    AnalyzeRequest,
    ComputeMetricsRequest,
    EstimateRequest,
    ExportLabelsRequest,
    InspectRequest,
    RebuildIndexRequest,
    ReportRequest,
)

dump = Path("/dumps/run-001")

estimate = application.estimate(EstimateRequest("audit.yaml"))
summary = application.inspect(InspectRequest(dump))
metrics = application.compute_metrics(ComputeMetricsRequest(dump))
analysis = application.analyze(AnalyzeRequest(dump))
report = application.generate_report(ReportRequest(dump))
labels = application.export_labels(ExportLabelsRequest(report.report_dir, csv=True))
rebuilt = application.rebuild_index(RebuildIndexRequest(dump))
```

Default output locations are co-located with the dump:

| Operation | Default output |
| --- | --- |
| `compute_metrics` | `<dump>/metrics` |
| `analyze` | `<dump>/analysis` |
| `generate_report` | `<dump>/report` |
| `export_labels` | `<report_dir>/labels` |

`generate_report` requires a metrics store. If the requested/default analysis directory does not exist, it still creates a report and marks analysis-derived sections unavailable. `analyze` itself requires an existing compatible metrics store.

## Error handling

Application-boundary failures are raised as `ApplicationError` with a stable `ApplicationErrorCode`:

- `config_error` and `provider_error` for invalid configuration or provider construction.
- `output_error`, `output_locked`, and `stale_preflight` for unsafe output state.
- `confirmation_required` for missing required approval.
- `execution_error` for runtime failure.
- `dump_corruption`, `metrics_error`, `analysis_error`, `report_error`, and `export_labels_error` for downstream operations.

Per-sample load, preparation, prediction, and out-of-memory failures are normally persisted as item statuses rather than raised as application errors, subject to runtime policy such as `fail_fast`.

## Custom adapter and source providers

Providers are registered explicitly; SSAT does not scan modules or entry points.

```python
from ssat.application import AuditApplication
from ssat.core.adapter import default_adapter_provider_registry
from ssat.core.source import default_source_provider_registry

adapter_registry = default_adapter_provider_registry()
adapter_registry.register(MyAdapterProvider())

source_registry = default_source_provider_registry()
source_registry.register(MySourceProvider())

application = AuditApplication(
    adapter_registry,
    source_registry=source_registry,
)
```

A custom source provider must return both a `SampleSource` and file-backed `SourceProvenance` containing the resolved provenance path and its SHA-256 hash. See [Configuration Reference](CONFIG_REFERENCE.md#custom-source-providers) for an example.

The stock `ssat` executable always creates the default registries. To expose custom providers through a CLI, construct an application factory and pass it to `ssat.cli.create_app`:

```python
from ssat.cli import create_app

cli = create_app(application_factory=lambda: application)
cli()
```

Registry extension points are Python APIs and should be version-pinned while SSAT remains alpha.
