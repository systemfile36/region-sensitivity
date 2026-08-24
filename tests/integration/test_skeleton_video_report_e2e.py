"""End-to-end coverage for the combined video + skeleton_parts + report path.

``test_skeleton_application_e2e.py`` proves ``skeleton_source``/
``skeleton_parts`` works through the Application API over an image source;
``test_video_source_e2e.py`` proves ``video_manifest`` works the same way.
Neither exercises the two together through ``generate_report`` -- exactly
the combination the Action Recognition (NTU/skeleton_parts) HTML report bug
lived in (empty gallery cards, and a Region Summary table exploded to one
row per (body_part, sample) instead of one row per body_part). This test
runs the real ``run`` -> ``compute_metrics`` -> ``generate_report`` sequence
against the committed synthetic video + skeleton bbox fixtures and asserts
both are fixed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

from ssat.application import AuditApplication, ComputeMetricsRequest, ReportRequest, RunRequest
from ssat.core.adapter import (
    AdapterProvider,
    AdapterProviderRegistry,
    CallableAdapter,
    ProviderConfig,
)

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_video"
_MANIFEST = _FIXTURE / "manifest.json"
_SKELETON_BBOX = _FIXTURE / "skeleton_bbox.json"


class _FixtureVideoConfig(ProviderConfig):
    provider: Literal["fixture_skeleton_video"] = "fixture_skeleton_video"


class _FixtureVideoProvider(AdapterProvider):
    """A CallableAdapter-backed provider standing in for a real action-recognition model."""

    name = "fixture_skeleton_video"
    config_model = _FixtureVideoConfig

    def build(self, config: ProviderConfig, *, base_dir: Path) -> CallableAdapter:
        def predict(batch: np.ndarray) -> np.ndarray:
            means = batch.astype(np.float32).mean(axis=(1, 2, 3, 4))
            return np.stack((means, -means, np.zeros_like(means)), axis=1)

        return CallableAdapter(
            predict,
            model_id="fixture-skeleton-video",
            class_names=("sweep", "pulse", "bounce"),
            transform_mask_fn=lambda mask: mask.copy(),
        )


def _application() -> AuditApplication:
    registry = AdapterProviderRegistry()
    registry.register(_FixtureVideoProvider())
    return AuditApplication(registry, code_version="skeleton-video-report-test")


def _config() -> dict:
    return {
        "source": {"kind": "video_manifest", "manifest": str(_MANIFEST), "num_frames": 8},
        "adapter": {"provider": "fixture_skeleton_video"},
        "skeleton_source": {"bbox_data": str(_SKELETON_BBOX)},
        "regions": [
            {
                "region_id": "occlude_left_arm",
                "kind": "skeleton_parts",
                "params": {"body_part": "left_arm", "bbox_scale": 1.15},
            }
        ],
        "perturbations": [{"op": "constant_fill", "params": {"value": 0}}],
        "runtime": {"variants_per_chunk": 4, "target_batch_size": 4, "num_workers": 0},
        "dump": {"flush_every": 8},
    }


def test_skeleton_parts_video_report_has_populated_gallery_and_aggregated_region_summary(
    tmp_path: Path,
) -> None:
    application = _application()
    output = tmp_path / "dump"

    with application.prepare_run(RunRequest(_config(), output, base_dir=tmp_path)) as prepared:
        result = application.execute_run(prepared, confirmation_granted=True)
    assert result.status == "completed"

    application.compute_metrics(ComputeMetricsRequest(output))
    report_result = application.generate_report(ReportRequest(output))

    report_dir = report_result.report_dir
    assert (report_dir / "report.html").is_file()

    # Bug 1: the gallery must actually render real frames, not empty
    # "No original"/"No heatmap" placeholders.
    heatmap_pngs = list((report_dir / "assets" / "img" / "heatmaps").glob("*.png"))
    thumbnail_pngs = list((report_dir / "assets" / "img" / "thumbnails").glob("*.png"))
    assert heatmap_pngs
    assert thumbnail_pngs

    # Bug 2: the Region Summary must be one row per body part (region_id),
    # not one row per (body_part, sample) pair.
    model = json.loads((report_dir / "data" / "report_model.json").read_text(encoding="utf-8"))
    region_rows = model["region_summary"]["rows"]
    assert len(region_rows) == 1
    row = region_rows[0]
    assert row["region_id"] == "occlude_left_arm"
    assert row["region_key"] == "occlude_left_arm"
    assert row["region_kind"] == "skeleton_parts"
    # 12 valid clips in the fixture (2 are intentionally corrupt/load_failed).
    assert row["n_valid"] == 12

    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    valid_count = sum(1 for sample in manifest["samples"] if sample["expected_status"] == "ok")
    assert row["n_valid"] == valid_count

    html = (report_dir / "report.html").read_text(encoding="utf-8")
    assert "No original" not in html
    assert "No heatmap" not in html
