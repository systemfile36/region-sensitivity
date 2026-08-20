# Logging Policy

SSAT uses the `ssat` logger hierarchy managed by `ssat.utils.logger_factory`. Importing the library is silent because the package installs a `NullHandler`; an embedding application must opt into handlers. The stock CLI enables UTC console logging and can add a UTF-8 file handler.

```bash
ssat --log-level DEBUG --log-file /tmp/ssat.log estimate audit.yaml
```

Python applications can configure the same behavior:

```python
from ssat.utils.logger_factory import configure_logging

configure_logging("INFO", "/tmp/ssat.log")
```

## Levels

- `DEBUG`: resolved paths, hashes, and diagnostic decisions.
- `INFO`: lifecycle events, adapter validation, planning, and dataset-statistics summaries.
- `WARNING`: explicitly permitted nondeterminism, preflight advisories, recoverable skipped samples, and callback failures.
- `ERROR`: an unrecoverable failure emitted at an outer application or CLI boundary, with traceback information where configured.

Messages use stable `event key=value` text where practical. Per-sample success messages are avoided so long audits remain readable.

## Data-handling rules

Logs may include an identifier, model ID, aggregate count, hash, or path needed to diagnose a failure. They must not contain:

- raw image or video arrays;
- logits or other bulk model output;
- complete user configurations;
- credentials, tokens, or secrets;
- repetitive success records for individual samples.

Paths and sample identifiers can still be sensitive in some environments. Applications that process confidential datasets should select a protected log destination, restrict access, and apply an appropriate retention policy.

## Embedding behavior

Repeated calls to `configure_logging` replace only handlers owned by SSAT's logger factory. Handlers installed by an embedding application and handlers on the root logger are left untouched. Event-sink callbacks are separate from logging; an exception from a sink is logged as a warning and does not terminate the audit.
