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


def _write_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "source:",
                "  kind: image_manifest",
                f"  manifest: {FIXTURE / 'manifest.json'}",
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
