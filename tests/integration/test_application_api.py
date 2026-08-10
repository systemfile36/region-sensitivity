from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Literal

import numpy as np
import pytest

from ssat.application import (
    ApplicationError,
    ApplicationErrorCode,
    ApplicationEvent,
    AuditApplication,
    CancellationToken,
    InspectRequest,
    RebuildIndexRequest,
    RunRequest,
)
from ssat.application.locking import output_lock
from ssat.core.adapter import (
    AdapterProvider,
    AdapterProviderRegistry,
    CallableAdapter,
    ProviderConfig,
)
from ssat.core.estimate import EstimateOptions, EstimationLimits


FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_classification"


class _FixtureConfig(ProviderConfig):
    provider: Literal["fixture"] = "fixture"
    model_id: str = "application-fixture"


class _FixtureProvider(AdapterProvider):
    name = "fixture"
    config_model = _FixtureConfig

    def build(self, config: ProviderConfig, *, base_dir: Path) -> CallableAdapter:
        assert isinstance(config, _FixtureConfig)

        def predict(batch: np.ndarray) -> np.ndarray:
            means = batch.astype(np.float32).mean(axis=(1, 2, 3, 4))
            return np.stack((means, -means), axis=1)

        return CallableAdapter(
            predict,
            model_id=config.model_id,
            class_names=("positive", "negative"),
            transform_mask_fn=lambda mask: mask.copy(),
        )


def _application() -> AuditApplication:
    registry = AdapterProviderRegistry()
    registry.register(_FixtureProvider())
    return AuditApplication(registry, code_version="application-test")


def _config(manifest: Path = FIXTURE / "manifest.json") -> dict:
    return {
        "source": {"kind": "image_manifest", "manifest": str(manifest)},
        "adapter": {"provider": "fixture"},
        "regions": [
            {"region_id": "whole", "kind": "grid", "params": {"rows": 1, "cols": 1}}
        ],
        "perturbations": [{"op": "constant_fill", "params": {"value": 0}}],
        "runtime": {
            "variants_per_chunk": 1,
            "target_batch_size": 4,
            "num_workers": 0,
        },
        "dump": {"flush_every": 5},
    }


def test_application_prepare_execute_resume_inspect_and_rebuild(tmp_path: Path) -> None:
    application = _application()
    output = tmp_path / "dump"
    events: list[ApplicationEvent] = []
    with application.prepare_run(
        RunRequest(_config(), output, base_dir=tmp_path),
        event_sink=events.append,
    ) as prepared:
        assert not output.exists()
        assert prepared.mode == "create"
        result = application.execute_run(prepared, event_sink=events.append)

    assert result.status == "completed"
    assert result.summary is not None
    assert result.summary.records_written == 40
    summary = application.inspect(InspectRequest(output))
    assert summary.clean.rows == 20
    assert summary.perturbed.rows == 20
    assert summary.manifest_counts_match
    rebuilt = application.rebuild_index(RebuildIndexRequest(output))
    assert rebuilt.indexed_items == 20
    assert rebuilt.summary.total_counts_by_status == summary.total_counts_by_status
    assert any(event.kind == "progress" for event in events)

    with application.prepare_run(
        RunRequest(_config(), output, base_dir=tmp_path)
    ) as resumed:
        assert resumed.mode == "resume"
        resumed_result = application.execute_run(resumed)
    assert resumed_result.summary is not None
    assert resumed_result.summary.records_written == 0


def test_confirmation_is_application_policy_and_does_not_create_dump(
    tmp_path: Path,
) -> None:
    output = tmp_path / "not-created"
    request = RunRequest(
        _config(),
        output,
        base_dir=tmp_path,
        estimate_options=EstimateOptions(
            limits=EstimationLimits(max_pending_items=1)
        ),
    )
    with _application().prepare_run(request) as prepared:
        assert prepared.confirmation_required
        with pytest.raises(ApplicationError) as caught:
            prepared._application.execute_run(prepared)
    assert caught.value.code is ApplicationErrorCode.CONFIRMATION_REQUIRED
    assert not output.exists()


def test_source_change_makes_preflight_stale(tmp_path: Path) -> None:
    source_manifest = tmp_path / "manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "one",
                        "path": str(FIXTURE / "images" / "sample_000.png"),
                        "gt_label": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    application = _application()
    with application.prepare_run(
        RunRequest(_config(source_manifest), tmp_path / "dump", base_dir=tmp_path)
    ) as prepared:
        source_manifest.write_text(
            source_manifest.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ApplicationError) as caught:
            application.execute_run(prepared)
    assert caught.value.code is ApplicationErrorCode.STALE_PREFLIGHT


def test_cancellation_flushes_partial_dump_for_resume(tmp_path: Path) -> None:
    application = _application()
    token = CancellationToken()
    output = tmp_path / "cancelled"

    def receive(event: ApplicationEvent) -> None:
        if event.kind == "progress":
            token.cancel()

    with application.prepare_run(
        RunRequest(_config(), output, base_dir=tmp_path)
    ) as prepared:
        result = application.execute_run(
            prepared,
            event_sink=receive,
            cancellation=token,
        )
    assert result.status == "cancelled"
    partial = application.inspect(InspectRequest(output))
    assert 0 < partial.clean.rows < 20

    with application.prepare_run(
        RunRequest(_config(), output, base_dir=tmp_path)
    ) as resumed:
        completed = application.execute_run(resumed)
    assert completed.status == "completed"
    assert application.inspect(InspectRequest(output)).clean.rows == 20


def test_application_import_does_not_load_cli_frameworks() -> None:
    code = (
        "import sys; import ssat.application; "
        "assert 'typer' not in sys.modules; assert 'click' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_application_rejects_concurrent_writer(tmp_path: Path) -> None:
    application = _application()
    output = tmp_path / "locked"
    with application.prepare_run(
        RunRequest(_config(), output, base_dir=tmp_path)
    ) as prepared:
        with output_lock(output):
            with pytest.raises(ApplicationError) as caught:
                application.execute_run(prepared)
    assert caught.value.code is ApplicationErrorCode.LOCKED
