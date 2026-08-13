from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from typer.testing import CliRunner

from ssat.cli import create_app
from ssat.core.adapter import (
    AdapterProvider,
    AdapterProviderRegistry,
    CallableAdapter,
    ProviderConfig,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_classification"


class _CliConfig(ProviderConfig):
    provider: Literal["cli_fixture"] = "cli_fixture"


class _CliProvider(AdapterProvider):
    name = "cli_fixture"
    config_model = _CliConfig

    def build(self, config: ProviderConfig, *, base_dir: Path) -> CallableAdapter:
        return CallableAdapter(
            lambda batch: np.zeros((len(batch), 2), dtype=np.float32),
            model_id="cli-fixture",
            transform_mask_fn=lambda mask: mask,
        )


def _app():
    registry = AdapterProviderRegistry()
    registry.register(_CliProvider())
    return create_app(registry)


def _write_two_class_manifest(path: Path) -> None:
    """Write a subset of the shared FIXTURE manifest restricted to gt_label in {0, 1}.

    The shared manifest also has a real gt_label=2 class and two
    intentionally corrupted (gt_label=None) entries (used elsewhere to
    exercise the "load_failed" status path). _CliProvider's fixture adapter
    only scores 2 classes, so feeding it a gt_label it can't index into
    would fail metrics computation with an error unrelated to what
    test_cli_metrics_command actually checks -- this subset keeps that test
    focused on the metrics command itself.
    """

    fixture_manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    samples = [
        {
            "sample_id": sample["sample_id"],
            "path": str(FIXTURE / sample["path"]),
            "gt_label": sample["gt_label"],
        }
        for sample in fixture_manifest["samples"]
        if sample["gt_label"] in (0, 1)
    ]
    path.write_text(json.dumps({"samples": samples}), encoding="utf-8")


def _write_config(path: Path, *, manifest: Path = FIXTURE / "manifest.json") -> None:
    path.write_text(
        "\n".join(
            [
                "source:",
                "  kind: image_manifest",
                f"  manifest: {manifest}",
                "adapter:",
                "  provider: cli_fixture",
                "regions:",
                "  - region_id: whole",
                "    kind: grid",
                "    params: {rows: 1, cols: 1}",
                "perturbations:",
                "  - op: constant_fill",
                "    params: {value: 0}",
                "runtime:",
                "  variants_per_chunk: 1",
                "  target_batch_size: 8",
                "  num_workers: 0",
                "dump:",
                "  flush_every: 8",
            ]
        ),
        encoding="utf-8",
    )


def _write_config_with_controls(path: Path, *, manifest: Path) -> None:
    """Like ``_write_config`` but with a 2x2 grid, controls, and repeat seeds.

    ``_write_config``'s single ``whole`` region has nothing to compare
    against, so it cannot exercise the ``analyze`` command's control/seed-
    stability output meaningfully.
    """

    path.write_text(
        "\n".join(
            [
                "source:",
                "  kind: image_manifest",
                f"  manifest: {manifest}",
                "adapter:",
                "  provider: cli_fixture",
                "regions:",
                "  - region_id: grid",
                "    kind: grid",
                "    params: {rows: 2, cols: 2}",
                "perturbations:",
                "  - op: constant_fill",
                "    params: {value: 0}",
                "    seed_salts: [0, 1]",
                "controls:",
                "  - match_area_of: grid",
                "    n_samples: 2",
                "runtime:",
                "  variants_per_chunk: 1",
                "  target_batch_size: 8",
                "  num_workers: 0",
                "dump:",
                "  flush_every: 8",
            ]
        ),
        encoding="utf-8",
    )


def test_cli_analyze_command(tmp_path: Path) -> None:
    manifest = tmp_path / "two_class_manifest.json"
    _write_two_class_manifest(manifest)
    config = tmp_path / "audit.yaml"
    output = tmp_path / "dump"
    _write_config_with_controls(config, manifest=manifest)
    runner = CliRunner()
    app = _app()

    run = runner.invoke(app, ["run", str(config), "--output", str(output)])
    assert run.exit_code == 0, run.output

    metrics = runner.invoke(app, ["metrics", str(output)])
    assert metrics.exit_code == 0, metrics.output

    analyze = runner.invoke(app, ["analyze", str(output), "--json"])
    assert analyze.exit_code == 0, analyze.output
    payload = json.loads(analyze.stdout)
    assert Path(payload["analysis_dir"]) == (output / "analysis").resolve()
    assert payload["available_analyses"]["control_comparison"] is True
    assert payload["n_reliability_rows"] > 0
    assert (output / "analysis" / "analysis_manifest.json").is_file()

    text_analyze = runner.invoke(
        app, ["analyze", str(output), "--analysis-dir", str(tmp_path / "analysis-again")]
    )
    assert text_analyze.exit_code == 0, text_analyze.output
    assert "SSAT control/stability analysis computed" in text_analyze.stdout

    missing_metrics = runner.invoke(app, ["analyze", str(tmp_path / "no-such-dump")])
    assert missing_metrics.exit_code != 0


def test_cli_json_run_inspect_and_rebuild(tmp_path: Path) -> None:
    config = tmp_path / "audit.yaml"
    output = tmp_path / "dump"
    _write_config(config)
    runner = CliRunner()
    app = _app()

    estimate = runner.invoke(app, ["estimate", str(config), "--json"])
    assert estimate.exit_code == 0, estimate.output
    payload = json.loads(estimate.stdout)
    assert payload["report"]["pending_perturbed_items"] == 20
    assert set(payload["report"]["profile"]["status_counts"]) >= {"ok", "load_failed"}

    run = runner.invoke(app, ["run", str(config), "--output", str(output)])
    assert run.exit_code == 0, run.output
    assert "SSAT run completed" in run.stdout

    inspect = runner.invoke(app, ["inspect", str(output), "--json"])
    assert inspect.exit_code == 0, inspect.output
    summary = json.loads(inspect.stdout)
    assert summary["clean"]["rows"] == 20
    assert summary["perturbed"]["rows"] == 20

    rebuild = runner.invoke(app, ["rebuild-index", str(output)])
    assert rebuild.exit_code == 0, rebuild.output
    assert "indexed items: 20" in rebuild.stdout


def test_cli_metrics_command(tmp_path: Path) -> None:
    manifest = tmp_path / "two_class_manifest.json"
    _write_two_class_manifest(manifest)
    config = tmp_path / "audit.yaml"
    output = tmp_path / "dump"
    _write_config(config, manifest=manifest)
    runner = CliRunner()
    app = _app()

    run = runner.invoke(app, ["run", str(config), "--output", str(output)])
    assert run.exit_code == 0, run.output

    metrics = runner.invoke(app, ["metrics", str(output), "--json"])
    assert metrics.exit_code == 0, metrics.output
    payload = json.loads(metrics.stdout)
    assert payload["primary_metric"] == "margin_drop"
    assert len(payload["metric_names"]) == 9
    assert Path(payload["metrics_dir"]) == (output / "metrics").resolve()
    assert (output / "metrics" / "metrics_manifest.json").is_file()

    # A non-JSON invocation should print the same information as readable text.
    text_metrics = runner.invoke(
        app, ["metrics", str(output), "--metrics-dir", str(tmp_path / "metrics-again")]
    )
    assert text_metrics.exit_code == 0, text_metrics.output
    assert "SSAT metrics computed" in text_metrics.stdout
    assert "primary metric: margin_drop" in text_metrics.stdout

    bad_metric = runner.invoke(app, ["metrics", str(output), "--primary-metric", "nope"])
    assert bad_metric.exit_code == 1
    assert "metrics_error" in bad_metric.stderr


def test_cli_help_and_version() -> None:
    runner = CliRunner()
    app = _app()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    version = runner.invoke(app, ["--version"])
    assert version.exit_code == 0
    assert version.stdout.strip() == "1.0.0"


def test_cli_confirmation_and_yes_only_control_prompt(tmp_path: Path) -> None:
    config = tmp_path / "audit.yaml"
    rejected_output = tmp_path / "rejected"
    accepted_output = tmp_path / "accepted"
    _write_config(config)
    runner = CliRunner()
    app = _app()

    rejected = runner.invoke(
        app,
        [
            "run",
            str(config),
            "--output",
            str(rejected_output),
            "--minimum-accuracy",
            "1.0",
        ],
        input="n\n",
    )
    assert rejected.exit_code == 1
    assert not rejected_output.exists()
    assert "Proceed with this audit?" in rejected.stdout

    accepted = runner.invoke(
        app,
        [
            "run",
            str(config),
            "--output",
            str(accepted_output),
            "--minimum-accuracy",
            "1.0",
            "--yes",
        ],
    )
    assert accepted.exit_code == 0, accepted.output
    assert "confirmation required: yes" in accepted.stdout
    assert "Proceed with this audit?" not in accepted.stdout
