from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Literal

import numpy as np
import pytest

from ssat.analysis import load_analysis
from ssat.application import (
    AnalyzeRequest,
    ApplicationError,
    ApplicationErrorCode,
    ApplicationEvent,
    AuditApplication,
    CancellationToken,
    ComputeMetricsRequest,
    EstimateRequest,
    ExportLabelsRequest,
    InspectRequest,
    RebuildIndexRequest,
    ReportRequest,
    RunRequest,
)
from ssat.application.locking import output_lock
from ssat.core.adapter import (
    AdapterProvider,
    AdapterProviderRegistry,
    CallableAdapter,
    ProviderConfig,
)
from ssat.core.config import SourceProvenance
from ssat.core.estimate import EstimateOptions, EstimationLimits
from ssat.core.source import (
    ImageFolderSource,
    SampleMeta,
    SourceProvider,
    SourceProviderConfig,
    SourceProviderRegistry,
)
from ssat.metrics.store import load_metrics
from ssat.utils.io import sha256_file


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


class _EchoSourceConfig(SourceProviderConfig):
    kind: Literal["test_echo"] = "test_echo"
    manifest: Path


class _EchoSourceProvider(SourceProvider):
    """Build an in-memory sample list while still citing a real manifest file.

    Proves a caller-registered SourceProvider is reachable end to end and
    need not reuse image_manifest's own file-parsing path, while keeping
    SourceProvenance backed by a real file + hash -- the reproducibility
    contract this registry does not relax.
    """

    name = "test_echo"
    config_model = _EchoSourceConfig

    def build(self, config: SourceProviderConfig, *, base_dir: Path):
        assert isinstance(config, _EchoSourceConfig)
        manifest_path = config.manifest
        if not manifest_path.is_absolute():
            manifest_path = base_dir / manifest_path
        manifest_path = manifest_path.resolve(strict=True)

        fixture_manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
        samples = [
            SampleMeta(sample["sample_id"], FIXTURE / sample["path"], sample["gt_label"])
            for sample in fixture_manifest["samples"]
            if sample["gt_label"] in (0, 1)
        ]
        provenance = SourceProvenance(
            kind=config.kind,
            manifest=manifest_path,
            manifest_hash=sha256_file(manifest_path),
        )
        return ImageFolderSource(samples), provenance


def _application_with_echo_source() -> AuditApplication:
    adapter_registry = AdapterProviderRegistry()
    adapter_registry.register(_FixtureProvider())
    source_registry = SourceProviderRegistry()
    source_registry.register(_EchoSourceProvider())
    return AuditApplication(
        adapter_registry,
        source_registry=source_registry,
        code_version="application-test",
    )


def test_custom_source_provider_runs_estimate_and_run_end_to_end(tmp_path: Path) -> None:
    """A caller-registered SourceProvider must be reachable through the public API.

    Without this test the source_registry extension point could regress
    into an unused shell -- the registry accepting registration without any
    real code path ever building a sample source through it.
    """

    application = _application_with_echo_source()
    config = _config()
    config["source"] = {"kind": "test_echo", "manifest": str(FIXTURE / "manifest.json")}

    estimate = application.estimate(EstimateRequest(config, base_dir=tmp_path))
    assert estimate.report.total_perturbed_items > 0

    output = tmp_path / "dump"
    with application.prepare_run(RunRequest(config, output, base_dir=tmp_path)) as prepared:
        result = application.execute_run(prepared)

    assert result.status == "completed"
    assert result.summary is not None
    assert result.summary.records_written > 0


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


def _config_with_controls(manifest: Path) -> dict:
    """Like ``_config`` but with a 2x2 grid, controls, and repeat seeds.

    ``_config``'s single ``whole`` region has nothing to compare against, so
    it cannot exercise ``analyze()``'s control/seed-stability logic
    meaningfully -- this mirrors the config shape
    tests/integration/test_control_e2e.py uses for the same reason, built
    as a plain config dict (RunRequest's own accepted shape) instead of a
    hand-built ResolvedConfig.
    """

    return {
        "source": {"kind": "image_manifest", "manifest": str(manifest)},
        "adapter": {"provider": "fixture"},
        "regions": [{"region_id": "grid", "kind": "grid", "params": {"rows": 2, "cols": 2}}],
        "perturbations": [
            {"op": "constant_fill", "params": {"value": 0}, "seed_salts": [0, 1]}
        ],
        "controls": [{"match_area_of": "grid", "n_samples": 2}],
        "runtime": {
            "variants_per_chunk": 1,
            "target_batch_size": 4,
            "num_workers": 0,
        },
        "dump": {"flush_every": 5},
    }


def test_application_analyze_computes_and_persists_reliability(tmp_path: Path) -> None:
    application = _application()
    manifest = tmp_path / "two_class_manifest.json"
    _write_two_class_manifest(manifest)
    output = tmp_path / "dump"
    with application.prepare_run(
        RunRequest(_config_with_controls(manifest), output, base_dir=tmp_path)
    ) as prepared:
        application.execute_run(prepared)
    application.compute_metrics(ComputeMetricsRequest(output))

    result = application.analyze(AnalyzeRequest(output))

    assert result.dump == output.resolve()
    assert result.metrics_dir == output.resolve() / "metrics"
    assert result.analysis_dir == output.resolve() / "analysis"
    assert result.available_analyses.control_comparison
    assert result.available_analyses.seed_stability
    assert result.coverage_report.n_anchors > 0
    assert result.n_reliability_rows > 0
    assert set(result.grade_distribution) <= {"high", "moderate", "low", "unreliable"}

    # Reload through the public analysis-store API to confirm the persisted
    # files are well-formed, not just that AnalyzeResult looked plausible.
    (*_rest, reliability_rows, coverage_report, manifest_obj) = load_analysis(
        result.analysis_dir
    )
    assert len(reliability_rows) == result.n_reliability_rows
    assert coverage_report == result.coverage_report
    assert dict(manifest_obj.grade_distribution) == result.grade_distribution


def test_application_analyze_rejects_dump_without_metrics(tmp_path: Path) -> None:
    application = _application()
    manifest = tmp_path / "two_class_manifest.json"
    _write_two_class_manifest(manifest)
    output = tmp_path / "dump"
    with application.prepare_run(
        RunRequest(_config_with_controls(manifest), output, base_dir=tmp_path)
    ) as prepared:
        application.execute_run(prepared)

    with pytest.raises(ApplicationError) as caught:
        application.analyze(AnalyzeRequest(output))
    assert caught.value.code is ApplicationErrorCode.ANALYSIS
    assert not (output / "analysis").exists()


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


def test_application_generate_report_with_analysis_creates_report_html(tmp_path: Path) -> None:
    application = _application()
    manifest = tmp_path / "two_class_manifest.json"
    _write_two_class_manifest(manifest)
    output = tmp_path / "dump"
    with application.prepare_run(
        RunRequest(_config_with_controls(manifest), output, base_dir=tmp_path)
    ) as prepared:
        application.execute_run(prepared)
    application.compute_metrics(ComputeMetricsRequest(output))
    application.analyze(AnalyzeRequest(output))

    result = application.generate_report(ReportRequest(output))

    assert result.dump == output.resolve()
    assert result.metrics_dir == output.resolve() / "metrics"
    assert result.analysis_dir == output.resolve() / "analysis"
    assert result.report_dir == output.resolve() / "report"
    assert result.n_samples > 0
    assert result.n_regions > 0
    assert set(result.grade_distribution) <= {"high", "moderate", "low", "unreliable"}

    assert result.secondary_report_html == result.report_dir / "report_question_driven.html"
    assert result.secondary_report_html.is_file()

    report_dir = result.report_dir
    assert (report_dir / "report.html").is_file()
    assert (report_dir / "report_manifest.json").is_file()
    assert (report_dir / "assets" / "css" / "style.css").is_file()
    assert (report_dir / "assets" / "js" / "enhance.js").is_file()
    assert (report_dir / "assets" / "img" / "charts" / "vulnerability_histogram.svg").is_file()
    assert (report_dir / "assets" / "img" / "charts" / "region_bar.svg").is_file()
    # This fixture's source is an image_manifest -- source_provenance is
    # available, so R3 must have actually rendered top-K/bottom-K assets,
    # not just left the gallery empty.
    assert list((report_dir / "assets" / "img" / "heatmaps").glob("*.png"))
    assert list((report_dir / "assets" / "img" / "thumbnails").glob("*.png"))
    assert (report_dir / "data" / "report_model.json").is_file()
    assert (report_dir / "data" / "sample_rankings.csv").is_file()
    assert (report_dir / "data" / "region_summary.csv").is_file()
    assert (report_dir / "data" / "flagged_items.csv").is_file()

    html = (report_dir / "report.html").read_text(encoding="utf-8")
    assert "vulnerability_histogram.svg" in html
    assert "region_bar.svg" in html

    model = json.loads((report_dir / "data" / "report_model.json").read_text(encoding="utf-8"))
    assert model["provenance"]["run_manifest_hash"]
    assert model["provenance"]["metrics_manifest_hash"]
    assert model["provenance"]["analysis_manifest_hash"]
    assert "spatial_concentration" in model
    assert model["vulnerability_distribution"]["histogram_asset_ref"] == (
        "assets/img/charts/vulnerability_histogram.svg"
    )
    assert model["region_summary"]["chart_asset_ref"] == "assets/img/charts/region_bar.svg"


def test_application_generate_report_without_analysis_marks_sections_unavailable(
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
    application.compute_metrics(ComputeMetricsRequest(output))

    # No analyze() run at all -- analysis_dir defaults to <dump>/analysis,
    # which does not exist yet; generate_report must silently downgrade to
    # "no analysis" rather than fail (ReportRequest.analysis_dir docstring).
    result = application.generate_report(ReportRequest(output))

    assert result.analysis_dir is None
    assert result.grade_distribution == {}
    assert (result.report_dir / "report.html").is_file()
    assert result.secondary_report_html.is_file()

    model = json.loads(
        (result.report_dir / "data" / "report_model.json").read_text(encoding="utf-8")
    )
    assert model["provenance"]["analysis_dir"] is None
    assert model["provenance"]["analysis_manifest_hash"] is None
    assert model["reliability_spotlight"]["flagged_examples"] == []


def test_application_generate_report_rejects_unknown_primary_metric(tmp_path: Path) -> None:
    application = _application()
    manifest = tmp_path / "two_class_manifest.json"
    _write_two_class_manifest(manifest)
    output = tmp_path / "dump"
    with application.prepare_run(
        RunRequest(_config(manifest), output, base_dir=tmp_path)
    ) as prepared:
        application.execute_run(prepared)
    application.compute_metrics(ComputeMetricsRequest(output))

    with pytest.raises(ApplicationError) as caught:
        application.generate_report(ReportRequest(output, primary_metric="not_a_real_metric"))
    assert caught.value.code is ApplicationErrorCode.REPORT
    assert not (output / "report").exists()


def test_application_export_labels_reads_report_dir_without_rerunning_r0(
    tmp_path: Path,
) -> None:
    application = _application()
    manifest = tmp_path / "two_class_manifest.json"
    _write_two_class_manifest(manifest)
    output = tmp_path / "dump"
    with application.prepare_run(
        RunRequest(_config_with_controls(manifest), output, base_dir=tmp_path)
    ) as prepared:
        application.execute_run(prepared)
    application.compute_metrics(
        ComputeMetricsRequest(output, primary_metric="flip_correct_to_wrong")
    )
    # report_dir is deliberately outside `output` -- this test later deletes
    # the whole dump directory to prove export_labels never reopens it, and
    # ReportRequest.report_dir defaults to <dump>/report which would be
    # deleted right along with it otherwise.
    report_dir = tmp_path / "report"
    report = application.generate_report(
        ReportRequest(output, primary_metric="flip_correct_to_wrong", report_dir=report_dir)
    )

    # generate_report never runs export_labels automatically.
    default_labels_dir = report.report_dir / "labels"
    assert not default_labels_dir.exists()

    result = application.export_labels(ExportLabelsRequest(report.report_dir))

    assert result.labels_path == default_labels_dir / "labels.jsonl"
    assert result.labels_path.is_file()
    assert result.manifest_path == default_labels_dir / "labels_manifest.json"
    assert result.manifest_path.is_file()
    assert result.csv_path is None
    assert result.n_labels >= 0
    assert result.n_labels == result.n_positive + (result.n_negative_or_none or 0)

    lines = result.labels_path.read_text(encoding="utf-8").splitlines()
    meta = json.loads(lines[0])
    assert meta["is_binary_primary_metric"] is True

    # R0 is never rerun: the dump/metrics this report was built from can be
    # deleted entirely and export_labels must still succeed from report_dir
    # alone -- it only exports what was already computed.
    shutil.rmtree(output)
    csv_output_dir = tmp_path / "labels-again"
    again = application.export_labels(
        ExportLabelsRequest(report.report_dir, csv_output_dir, csv=True)
    )
    assert again.csv_path == csv_output_dir / "labels.csv"
    assert again.csv_path.is_file()


def test_application_export_labels_rejects_report_dir_without_report_model(
    tmp_path: Path,
) -> None:
    application = _application()
    empty_report_dir = tmp_path / "not-a-report"
    empty_report_dir.mkdir()

    with pytest.raises(ApplicationError) as caught:
        application.export_labels(ExportLabelsRequest(empty_report_dir))
    assert caught.value.code is ApplicationErrorCode.EXPORT_LABELS


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
