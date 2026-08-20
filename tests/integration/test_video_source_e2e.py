"""End-to-end Application API coverage for the decord-backed video source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

from ssat.application import AuditApplication, InspectRequest, RunRequest
from ssat.core.adapter import (
    AdapterProvider,
    AdapterProviderRegistry,
    CallableAdapter,
    ProviderConfig,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_video"


class _FixtureVideoConfig(ProviderConfig):
    provider: Literal["fixture_video"] = "fixture_video"


class _FixtureVideoProvider(AdapterProvider):
    """A CallableAdapter-backed provider standing in for a real video model."""

    name = "fixture_video"
    config_model = _FixtureVideoConfig

    def build(self, config: ProviderConfig, *, base_dir: Path) -> CallableAdapter:
        def predict(batch: np.ndarray) -> np.ndarray:
            means = batch.astype(np.float32).mean(axis=(1, 2, 3, 4))
            return np.stack((means, -means, np.zeros_like(means)), axis=1)

        return CallableAdapter(
            predict,
            model_id="fixture-video",
            class_names=("sweep", "pulse", "bounce"),
            transform_mask_fn=lambda mask: mask.copy(),
        )


def _application() -> AuditApplication:
    registry = AdapterProviderRegistry()
    registry.register(_FixtureVideoProvider())
    return AuditApplication(registry, code_version="video-application-test")


def _config() -> dict:
    return {
        "source": {
            "kind": "video_manifest",
            "manifest": str(FIXTURE / "manifest.json"),
            "num_frames": 4,
        },
        "adapter": {"provider": "fixture_video"},
        "regions": [
            {"region_id": "grid2x2", "kind": "grid", "params": {"rows": 2, "cols": 2}}
        ],
        "perturbations": [{"op": "constant_fill", "params": {"value": 0}}],
        "runtime": {"variants_per_chunk": 4, "target_batch_size": 4, "num_workers": 0},
        "dump": {"flush_every": 8},
    }


def test_video_source_runs_through_the_application_api_unmodified(tmp_path: Path) -> None:
    """A video_manifest source and CallableAdapter run through unmodified core.

    No ``ssat.core`` module is touched by video support: the same
    ``PlanBuilder``, ``RegionResolver``, ``Perturbator``, and runtime
    pipeline used for images process ``(T,H,W,C)`` clips with ``T>1``
    because they were already T-agnostic before this change.
    """

    application = _application()
    output = tmp_path / "dump"
    with application.prepare_run(
        RunRequest(_config(), output, base_dir=tmp_path)
    ) as prepared:
        result = application.execute_run(prepared, confirmation_granted=True)

    assert result.status == "completed"
    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    valid_count = sum(
        1 for sample in manifest["samples"] if sample["expected_status"] == "ok"
    )
    corrupt_count = len(manifest["samples"]) - valid_count

    summary = application.inspect(InspectRequest(output))
    assert summary.clean.rows == len(manifest["samples"])
    assert summary.clean.counts_by_status["ok"] == valid_count
    assert summary.clean.counts_by_status["load_failed"] == corrupt_count
    # Every sample enumerates 4 (2x2 grid) perturbed work items regardless of
    # load outcome; corrupt clips surface as load_failed instead of ok.
    assert summary.perturbed.rows == len(manifest["samples"]) * 4
    assert summary.perturbed.counts_by_status["ok"] == valid_count * 4
    assert summary.perturbed.counts_by_status["load_failed"] == corrupt_count * 4
    assert summary.manifest_counts_match
