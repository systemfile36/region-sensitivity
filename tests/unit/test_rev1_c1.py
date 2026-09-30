"""Unit tests for experiments/revision_1/c1_captum (command map, measurement helpers, summaries)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.revision_1.c1_captum import summarize
from experiments.revision_1.c1_captum.measure_resources import entry_bytes, workflow_order, workflow_steps


def test_workflow_steps_follow_the_command_map(tmp_path: Path) -> None:
    captum = workflow_steps("captum", tmp_path)
    assert [step for step, _, _ in captum] == ["C-1", "C-2", "C-3"]
    assert [argv[2] for _, _, argv in captum] == ["audit", "analyze", "report"]
    ssat = workflow_steps("ssat", tmp_path)
    assert [step for step, _, _ in ssat] == [f"S-{index}" for index in range(1, 8)]
    assert [stage for _, stage, _ in ssat].count("audit") == 4
    assert all(argv[argv.index("--results-dir") + 1] == str(tmp_path) for _, _, argv in ssat)
    assert [argv[argv.index("--model") + 1] for _, _, argv in ssat if "--model" in argv] == ["shortcut", "normal"] * 2
    with pytest.raises(ValueError):
        workflow_steps("other", tmp_path)


def test_workflow_order_alternates() -> None:
    assert [workflow_order(repeat) for repeat in range(3)] == [("captum", "ssat"), ("ssat", "captum"), ("captum", "ssat")]


def test_entry_bytes(tmp_path: Path) -> None:
    (tmp_path / "raw" / "sub").mkdir(parents=True)
    (tmp_path / "raw" / "sub" / "a.parquet").write_bytes(b"x" * 7)
    (tmp_path / "report.md").write_bytes(b"y" * 3)
    (tmp_path / "accuracy.json").write_bytes(b"z" * 2)
    assert entry_bytes(tmp_path) == {"files": 5, "raw": 7, "total": 12}


def _step(step: str, stage: str, elapsed: float, gpu_mib: int, util: float) -> dict:
    return {"step": step, "stage": stage, "argv": ["/py", "/x/run.py", "audit", "--config"], "elapsed_s": elapsed,
            "peak_rss_kb": 2**20, "tree_rss_peak_bytes": 3 * 2**30, "tree_rss_peak_processes": 5,
            "gpu_peak_mib": gpu_mib, "gpu_util_mean": util, "returncode": 0}


def _captum_output(root: Path, raw_rows: int) -> Path:
    (root / "analysis").mkdir(parents=True)
    (root / "analysis" / "summary.json").write_text(json.dumps({
        "raw_rows": raw_rows, "raw_canonical_hash": "r", "canonical_hash": "a", "area_sanity": {"pass": True},
        "verdicts": {"Q1_x": {"pass": True}, "Q2_x": {"pass": True}, "B_auxiliary_control": {"patch_region_rank": 2}}}))
    return root


def _ssat_output(root: Path, dumps: dict[str, int]) -> Path:
    for name, perturbed in dumps.items():
        (root / "dumps" / name / "clean").mkdir(parents=True)
        pq.write_table(pa.table({"sample_id": list(range(4))}), root / "dumps" / name / "clean" / "part.parquet")
        (root / "dumps" / name / "run_manifest.json").write_text(json.dumps({"counts_by_status": {"ok": perturbed + 4, "load_failed": 0}}))
    return root


def test_run_rows_and_resources(tmp_path: Path) -> None:
    captum = _captum_output(tmp_path / "captum", 1000)
    ssat = _ssat_output(tmp_path / "ssat", {"a": 600, "b": 400})
    records = [
        {"workflow": "captum", "repeat": 0, "output": str(captum), "gpu_idle_mib": 2000, "bytes": {"raw": 10, "files": 2, "total": 12},
         "steps": [_step("C-1", "audit", 30.0, 6000, 90.0), _step("C-2", "analysis_report", 10.0, 2000, 0.0)]},
        {"workflow": "ssat", "repeat": 0, "output": str(ssat), "gpu_idle_mib": 2200, "bytes": {"dumps": 30, "files": 5, "total": 35},
         "steps": [_step("S-1", "audit", 10.0, 3000, 2.0), _step("S-4", "audit", 30.0, 3200, 6.0), _step("S-3", "analysis_report", 20.0, 2200, 0.0)]},
    ]
    runs = summarize.run_rows(records, summarize.step_rows(records)).set_index("workflow")
    assert runs.loc["captum", "perturbed_items"] == 1000 and runs.loc["ssat", "perturbed_items"] == 1000
    assert runs.loc["ssat", "audit_s"] == 40.0 and runs.loc["ssat", "analysis_report_s"] == 20.0
    assert runs.loc["ssat", "ms_per_item"] == pytest.approx(60.0)
    assert runs.loc["ssat", "gpu_above_idle_mib"] == 1000
    assert runs.loc["ssat", "audit_gpu_util_mean"] == pytest.approx((10 * 2.0 + 30 * 6.0) / 40)
    assert runs.loc["captum", "tree_rss_peak_gib"] == pytest.approx(3.0)
    resources = summarize.resource_rows(runs.reset_index())
    assert set(resources["workflow"]) == {"captum", "ssat"} and (resources["n_repeats"] == 1).all()
    storage = summarize.storage_rows(records).set_index(["workflow", "entry"])
    assert storage.loc[("ssat", "dumps"), "share"] == pytest.approx(30 / 35)
    assert storage.loc[("captum", "raw"), "content"].startswith("raw observations")
    check = summarize.check_captum(captum, {"raw_rows": 1000, "raw_canonical_hash": "r", "analysis_canonical_hash": "a"})
    assert check["pass"]
    assert not summarize.check_captum(captum, {"raw_rows": 1000, "raw_canonical_hash": "x", "analysis_canonical_hash": "a"})["pass"]


def test_compare_verdicts_tolerates_only_q2_multiplier_noise() -> None:
    stored = {"Q1_identifies_patch_region": {"pass": True, "patch_region_rank": 1},
              "Q2_separated_from_baseline": {"pass": True, "multiplier": 100.0, "threshold": 3.0},
              "Q3_distinguishes_normal_model": {"pass": True, "patch_region_rank_in_m_normal": 16},
              "Q4_robust_to_fill_strategy": {"pass": True, "reproduced_in": ["blur"], "min_required": 2},
              "Q5_predicts_generalization_gap": {"pass": True, "margin_points": 95.75},
              "B_auxiliary_control": {"patch_region_rank_on_random_placement": 2}}
    near = copy.deepcopy(stored)
    near["Q2_separated_from_baseline"]["multiplier"] = 100.05
    result = summarize.compare_verdicts(near, stored)
    assert all(result["checks"].values()) and result["q2_multiplier_relative_difference"] == pytest.approx(5e-4)
    moved = copy.deepcopy(stored)
    moved["Q3_distinguishes_normal_model"]["patch_region_rank_in_m_normal"] = 15
    assert not summarize.compare_verdicts(moved, stored)["checks"]["q3_rank"]
