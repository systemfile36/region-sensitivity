"""Thin Typer adapter over the UI-independent application service."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer

from ssat import __version__
from ssat.application import (
    AnalyzeRequest,
    ApplicationError,
    AuditApplication,
    ComputeMetricsRequest,
    EstimateRequest,
    ExportLabelsRequest,
    InspectRequest,
    RebuildIndexRequest,
    ReportRequest,
    RunRequest,
)
from ssat.core.adapter import AdapterProviderRegistry
from ssat.core.estimate import EstimateOptions
from ssat.metrics.aggregate import DEFAULT_PRIMARY_METRIC
from ssat.presentation import (
    format_analysis,
    format_dump_summary,
    format_estimate,
    format_export_labels,
    format_metrics,
    format_rebuild,
    format_report,
    format_run,
    json_text,
)
from ssat.utils.logger_factory import configure_logging, get_logger


def create_app(
    adapter_registry: AdapterProviderRegistry | None = None,
    *,
    application_factory: Callable[[], AuditApplication] | None = None,
) -> typer.Typer:
    """Create an injectable CLI app for built-in or explicitly registered providers."""

    app = typer.Typer(
        name="ssat",
        no_args_is_help=True,
        invoke_without_command=True,
        help="Spatial Sensitivity Audit Toolkit",
    )
    service = (
        application_factory()
        if application_factory is not None
        else AuditApplication(adapter_registry)
    )

    @app.callback()
    def callback(
        log_level: str = typer.Option("INFO", help="Python logging level."),
        log_file: Path | None = typer.Option(None, help="Optional UTF-8 log file."),
        version: bool = typer.Option(
            False,
            "--version",
            is_eager=True,
            help="Show the SSAT version and exit.",
        ),
    ) -> None:
        if version:
            typer.echo(__version__)
            raise typer.Exit()
        configure_logging(log_level, log_file)

    @app.command("run")
    def run_command(
        config: Path = typer.Argument(..., exists=True, dir_okay=False),
        output: Path = typer.Option(..., "--output", "-o"),
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip required confirmation."),
        minimum_accuracy: float | None = typer.Option(
            None,
            min=0.0,
            max=1.0,
            help="Optional minimum clean top-1 accuracy.",
        ),
        max_area_relative_deviation: float = typer.Option(
            0.05,
            "--max-area-relative-deviation",
            min=0.0,
            help="Maximum relative preprocessing/effective-area deviation.",
        ),
    ) -> None:
        try:
            request = RunRequest(
                config,
                output,
                estimate_options=EstimateOptions(
                    minimum_accuracy=minimum_accuracy,
                    max_area_relative_deviation=max_area_relative_deviation,
                ),
            )
            with service.prepare_run(request) as prepared:
                typer.echo(format_estimate(prepared.estimate))
                approved = yes
                if prepared.confirmation_required and not approved:
                    approved = typer.confirm("Proceed with this audit?", default=False)
                    if not approved:
                        raise typer.Abort()
                result = service.execute_run(
                    prepared,
                    confirmation_granted=approved,
                )
            typer.echo(format_run(result))
        except (typer.Abort, typer.Exit):
            raise
        except Exception as error:
            _fail(error)

    @app.command("estimate")
    def estimate_command(
        config: Path = typer.Argument(..., exists=True, dir_okay=False),
        dump: Path | None = typer.Option(None, "--dump"),
        minimum_accuracy: float | None = typer.Option(
            None,
            min=0.0,
            max=1.0,
            help="Optional minimum clean top-1 accuracy.",
        ),
        max_area_relative_deviation: float = typer.Option(
            0.05,
            "--max-area-relative-deviation",
            min=0.0,
            help="Maximum relative preprocessing/effective-area deviation.",
        ),
        json_output: bool = typer.Option(False, "--json", help="Emit stable JSON."),
    ) -> None:
        try:
            result = service.estimate(
                EstimateRequest(
                    config,
                    dump,
                    options=EstimateOptions(
                        minimum_accuracy=minimum_accuracy,
                        max_area_relative_deviation=max_area_relative_deviation,
                    ),
                )
            )
            typer.echo(json_text(result) if json_output else format_estimate(result))
        except Exception as error:
            _fail(error)

    @app.command("metrics")
    def metrics_command(
        dump: Path = typer.Argument(..., exists=True, file_okay=False),
        metrics_dir: Path | None = typer.Option(
            None, "--metrics-dir", help="Defaults to <dump>/metrics."
        ),
        primary_metric: str = typer.Option(
            DEFAULT_PRIMARY_METRIC,
            "--primary-metric",
            help="Registered metric name to record as this run's vulnerability-score source.",
        ),
        json_output: bool = typer.Option(False, "--json", help="Emit stable JSON."),
    ) -> None:
        try:
            result = service.compute_metrics(
                ComputeMetricsRequest(dump, metrics_dir, primary_metric)
            )
            typer.echo(json_text(result) if json_output else format_metrics(result))
        except Exception as error:
            _fail(error)

    @app.command("analyze")
    def analyze_command(
        dump: Path = typer.Argument(..., exists=True, file_okay=False),
        metrics_dir: Path | None = typer.Option(
            None, "--metrics-dir", help="Defaults to <dump>/metrics."
        ),
        analysis_dir: Path | None = typer.Option(
            None, "--analysis-dir", help="Defaults to <dump>/analysis."
        ),
        primary_metric: str = typer.Option(
            DEFAULT_PRIMARY_METRIC,
            "--primary-metric",
            help="Registered metric name the fill-strategy/reliability comparisons are scoped to.",
        ),
        json_output: bool = typer.Option(False, "--json", help="Emit stable JSON."),
    ) -> None:
        try:
            result = service.analyze(
                AnalyzeRequest(dump, metrics_dir, analysis_dir, primary_metric)
            )
            typer.echo(json_text(result) if json_output else format_analysis(result))
        except Exception as error:
            _fail(error)

    @app.command("report")
    def report_command(
        dump: Path = typer.Argument(..., exists=True, file_okay=False),
        metrics_dir: Path | None = typer.Option(
            None, "--metrics-dir", help="Defaults to <dump>/metrics."
        ),
        analysis_dir: Path | None = typer.Option(
            None,
            "--analysis-dir",
            help=(
                "Defaults to <dump>/analysis. A directory that does not exist "
                "generates a report with analysis-derived sections marked "
                "unavailable, not an error."
            ),
        ),
        report_dir: Path | None = typer.Option(
            None, "--report-dir", help="Defaults to <dump>/report."
        ),
        primary_metric: str = typer.Option(
            DEFAULT_PRIMARY_METRIC,
            "--primary-metric",
            help="Registered metric name every report section is scoped to.",
        ),
        top_k: int = typer.Option(
            20, "--top-k", help="Most-vulnerable samples to feature in the gallery."
        ),
        bottom_k: int = typer.Option(
            20, "--bottom-k", help="Most-robust samples to feature in the gallery."
        ),
        json_output: bool = typer.Option(False, "--json", help="Emit stable JSON."),
    ) -> None:
        try:
            result = service.generate_report(
                ReportRequest(
                    dump,
                    metrics_dir,
                    analysis_dir,
                    report_dir,
                    primary_metric,
                    top_k,
                    bottom_k,
                )
            )
            typer.echo(json_text(result) if json_output else format_report(result))
        except Exception as error:
            _fail(error)

    @app.command("export-labels")
    def export_labels_command(
        report_dir: Path = typer.Argument(..., exists=True, file_okay=False),
        output_dir: Path | None = typer.Option(
            None, "--output-dir", help="Defaults to <report_dir>/labels."
        ),
        include_non_clean_correct: bool = typer.Option(
            False,
            "--include-non-clean-correct",
            help=(
                "Also emit samples the clean prediction already got wrong "
                "(default: only clean-correct samples, so every positive "
                "label is a corruption-induced failure)."
            ),
        ),
        csv_output: bool = typer.Option(
            False, "--csv", help="Also write labels.csv alongside labels.jsonl."
        ),
        json_output: bool = typer.Option(False, "--json", help="Emit stable JSON."),
    ) -> None:
        try:
            result = service.export_labels(
                ExportLabelsRequest(
                    report_dir,
                    output_dir,
                    only_clean_correct=not include_non_clean_correct,
                    csv=csv_output,
                )
            )
            typer.echo(json_text(result) if json_output else format_export_labels(result))
        except Exception as error:
            _fail(error)

    @app.command("inspect")
    def inspect_command(
        dump: Path = typer.Argument(..., exists=True, file_okay=False),
        json_output: bool = typer.Option(False, "--json", help="Emit stable JSON."),
    ) -> None:
        try:
            result = service.inspect(InspectRequest(dump))
            typer.echo(json_text(result) if json_output else format_dump_summary(result))
        except Exception as error:
            _fail(error)

    @app.command("rebuild-index")
    def rebuild_index_command(
        dump: Path = typer.Argument(..., exists=True, file_okay=False),
    ) -> None:
        try:
            typer.echo(format_rebuild(service.rebuild_index(RebuildIndexRequest(dump))))
        except Exception as error:
            _fail(error)

    return app


def _fail(error: Exception) -> None:
    logger = get_logger("cli")
    if isinstance(error, ApplicationError):
        logger.error("cli.failed code=%s message=%s", error.code.value, error)
        typer.echo(f"Error [{error.code.value}]: {error}", err=True)
    else:
        logger.error("cli.failed", exc_info=error)
        typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=1)


app = create_app()


def main() -> None:
    app()
