# Logging Policy

SSAT uses the `ssat` logger hierarchy managed by `ssat.utils.logger_factory`.
Library imports remain silent through a `NullHandler`; applications explicitly
enable UTC console logging and an optional UTF-8 file with `configure_logging`.

## Levels

- `DEBUG`: resolved paths, hashes, and other diagnostic decisions.
- `INFO`: lifecycle events, adapter validation, and dataset-statistics summaries.
- `WARNING`: explicitly allowed nondeterminism and recoverable skipped samples.
- `ERROR`: emitted once at the outer CLI/application boundary with a stack trace.

Messages use stable `event key=value` text where practical. Logs may include an
identifier or path needed to diagnose a failure, but must not contain raw arrays,
logits, complete user configurations, credentials, or other bulk/sensitive data.
Per-sample success messages are avoided to keep long audit logs useful.

Repeated calls to `configure_logging` replace only handlers owned by the factory.
Handlers installed by an embedding application and the root logger are untouched.
