"""End-to-end Application API coverage for YAML-configured skeleton_parts.

Mirrors tests/integration/test_video_source_e2e.py's pattern: a plain-dict
config (as CLI/YAML would produce) is run through the unmodified
AuditApplication, proving the ``skeleton_source``/``skeleton_parts`` wiring
introduced for docs/VIDEO_SKELETON_EXTENSION_ANALYSIS_v1.md stage 4 works
without any test-only injection. Rasterization/perturbation math itself is
already covered pixel-exactly by tests/unit/test_skeleton_integration.py;
this test's job is the config -> Application -> dump path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from ssat.application import AuditApplication, InspectRequest, RunRequest
from ssat.core.adapter import (
    AdapterProvider,
    AdapterProviderRegistry,
    CallableAdapter,
    ProviderConfig,
)
from ssat.core.dump import DumpReader

_WIDTH, _HEIGHT = 10, 6
_BBOXES = {
    "sample_a": [1.0, 1.0, 4.0, 3.0],  # area = 3*2 = 6
    "sample_b": [2.0, 2.0, 6.0, 5.0],  # area = 4*3 = 12
}


class _FixtureConfig(ProviderConfig):
    provider: Literal["fixture"] = "fixture"


class _FixtureProvider(AdapterProvider):
    name = "fixture"
    config_model = _FixtureConfig

    def build(self, config: ProviderConfig, *, base_dir: Path) -> CallableAdapter:
        def predict(batch: np.ndarray) -> np.ndarray:
            means = batch.astype(np.float32).mean(axis=(1, 2, 3, 4))
            return np.stack((means, -means), axis=1)

        return CallableAdapter(
            predict,
            model_id="skeleton-application-fixture",
            class_names=("positive", "negative"),
            transform_mask_fn=lambda mask: mask.copy(),
        )


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Write a two-sample image manifest and matching skeleton bbox file."""

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    samples = []
    for sample_id, fill in (("sample_a", 180), ("sample_b", 220)):
        path = images_dir / f"{sample_id}.png"
        Image.new("RGB", (_WIDTH, _HEIGHT), color=(fill, fill, fill)).save(path)
        samples.append({"sample_id": sample_id, "path": str(path), "gt_label": 0})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")

    skeleton_path = tmp_path / "skeleton.json"
    skeleton_path.write_text(
        json.dumps(
            {
                sample_id: {
                    "frame_size": [_WIDTH, _HEIGHT],
                    "parts": {"left_arm": [bbox]},
                }
                for sample_id, bbox in _BBOXES.items()
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, skeleton_path


def _config(manifest_path: Path, skeleton_path: Path) -> dict:
    return {
        "source": {"kind": "image_manifest", "manifest": str(manifest_path)},
        "adapter": {"provider": "fixture"},
        "regions": [
            {
                "region_id": "occlude_left_arm",
                "kind": "skeleton_parts",
                "params": {"body_part": "left_arm"},
            }
        ],
        "perturbations": [{"op": "constant_fill", "params": {"value": 0}}],
        "runtime": {"variants_per_chunk": 1, "target_batch_size": 4, "num_workers": 0},
        "dump": {"flush_every": 8},
        "skeleton_source": {"bbox_data": str(skeleton_path)},
    }


def _application() -> AuditApplication:
    registry = AdapterProviderRegistry()
    registry.register(_FixtureProvider())
    return AuditApplication(registry, code_version="skeleton-application-test")


def test_skeleton_parts_runs_through_the_application_api(tmp_path: Path) -> None:
    """A skeleton_source + skeleton_parts YAML config runs end to end."""

    manifest_path, skeleton_path = _write_fixture(tmp_path)
    application = _application()
    output = tmp_path / "dump"

    with application.prepare_run(
        RunRequest(_config(manifest_path, skeleton_path), output, base_dir=tmp_path)
    ) as prepared:
        result = application.execute_run(prepared)

    assert result.status == "completed"
    summary = application.inspect(InspectRequest(output))
    assert summary.clean.rows == 2
    assert summary.clean.counts_by_status["ok"] == 2
    assert summary.perturbed.rows == 2
    assert summary.perturbed.counts_by_status["ok"] == 2
    assert summary.manifest_counts_match

    perturbed = DumpReader(output).read_perturbed().to_pylist()
    area_by_sample = {row["sample_id"]: row["intended_area_px"] for row in perturbed}
    assert area_by_sample == {"sample_a": 6, "sample_b": 12}
    assert all(row["generator_kind"] == "skeleton_parts" for row in perturbed)
