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
    ComputeMetricsRequest,
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
from ssat.metrics.store import load_metrics


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


def _write_two_class_manifest(path: Path) -> None:
    """Write a subset of the shared FIXTURE manifest restricted to gt_label in {0, 1}.

    The shared manifest also has a real gt_label=2 class and two
    intentionally corrupted (gt_label=None) entries (used elsewhere to
    exercise the "load_failed" status path). _FixtureProvider's adapter
    above only scores 2 classes ("positive"/"negative"), so a gt_label it
    can't index into would fail metrics computation with an error unrelated
    to what the compute_metrics tests below actually check -- this subset
    keeps those tests focused on compute_metrics itself.
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


def test_application_compute_metrics_persists_every_builtin_metric(
    tmp_path: Path,
) -> None:
    application = _application()
    manifest = tmp_path / "two_class_manifest.json"
    _write_two_class_manifest(manifest)
    output = tmp_path / "dump"
    with application.prepare_run(
        RunRequest(_config(manifest), output, base_dir=tmp_path)
    ) as prepared:
        application.execute_run(prepared)

    result = application.compute_metrics(ComputeMetricsRequest(output))

    assert result.dump == output.resolve()
    assert result.metrics_dir == output.resolve() / "metrics"
    assert result.primary_metric == "margin_drop"
    # default_metric_registry() always registers all 9 v1 built-ins, and
    # this fixture adapter's output_kind is "logits" (the only value v1
    # supports -- ssat/core/adapter/types.py), so every metric's
    # available_when() is True and none get silently dropped for this run.
    assert sorted(result.metric_names) == sorted(
        [
            "flip_correct_to_wrong",
            "flip_wrong_to_correct",
            "pred_changed",
            "topk_exit",
            "gt_prob_drop",
            "gt_logit_drop",
            "margin_drop",
            "loss_increase",
            "gt_rank_worsening",
        ]
    )
    # One item_metrics row per (perturbed item, metric): 12 perturbed items
    # (the two-class manifest subset has 6 gt_label=0 + 6 gt_label=1
    # samples, one region, one perturbation each) x 9 metrics.
    assert result.n_item_metric_rows == 12 * 9

    # Reload through the public metrics-engine API to confirm the files this
    # method wrote are actually well-formed, not just that it returned a
    # plausible-looking summary.
    item_metrics, aggregation, manifest = load_metrics(result.metrics_dir)
    assert len(item_metrics) == result.n_item_metric_rows
    assert {metric.name for metric in manifest.registered_metrics} == set(
        result.metric_names
    )
    assert any(row.metric_name == "margin_drop" for row in aggregation.region_metrics)


def test_application_compute_metrics_rejects_unknown_primary_metric(
    tmp_path: Path,
) -> None:
    application = _application()
    manifest = tmp_path / "two_class_manifest.json"
    _write_two_class_manifest(manifest)
    output = tmp_path / "dump"
    with application.prepare_run(
        RunRequest(_config(manifest), output, base_dir=tmp_path)
    ) as prepared:
        application.execute_run(prepared)

    with pytest.raises(ApplicationError) as caught:
        application.compute_metrics(
            ComputeMetricsRequest(output, primary_metric="not_a_real_metric")
        )
    assert caught.value.code is ApplicationErrorCode.METRICS
    # Failing fast (before the expensive compute_item_metrics scan) means no
    # metrics directory should have been created at all.
    assert not (output / "metrics").exists()


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
