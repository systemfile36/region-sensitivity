"""Unit tests for experiments/revision_1/a4_scaling (sweep, measurement helpers, fits, figures)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.revision_1.a4_scaling import compare_plan_cache, fit_scaling, summarize
from experiments.revision_1.a4_scaling.run_scaling import loop_seconds, parse_log_events, scheduled, subdir_bytes
from experiments.revision_1.a4_scaling.sweep import AXES, SINGLE_REPEAT_MIN_N, Setting, build_config, owner_axis, schedule
from ssat.application.config import load_application_config
from ssat.core.adapter.provider import default_adapter_provider_registry


def test_planned_items_match_the_implementation_plan_table() -> None:
    items = {axis: [setting.planned_items for setting in settings] for axis, settings in AXES.items()}
    assert items["samples"] == [80_250, 160_500, 321_000, 642_000, 1_284_000]
    assert items["regions"] == [81_000, 321_000, 721_000, 1_281_000]
    assert items["controls"] == [40_500, 80_500, 160_500, 440_500, 840_500]
    assert items["perturbations"] == [32_500, 64_500, 160_500, 224_500]


def test_schedule_skips_shared_settings_and_repeats_small_n() -> None:
    measured = Counter()
    for axis in AXES:
        for setting, repeat in schedule(axis, 3):
            assert owner_axis(setting) == axis
            measured[setting] += 1
    for axis, settings in AXES.items():
        for setting in settings:
            assert measured[setting] == (1 if setting.n >= SINGLE_REPEAT_MIN_N else 3)
    first = [setting for setting, repeat in schedule("regions", 3) if repeat == 0]
    second = [setting for setting, repeat in schedule("regions", 3) if repeat == 1]
    assert second == first[1:] + first[:1]


@pytest.mark.parametrize("setting", sorted({s for settings in AXES.values() for s in settings}, key=lambda s: s.key), ids=lambda s: s.key)
def test_configs_validate_against_the_ssat_schema(setting: Setting) -> None:
    config = build_config(setting)
    loaded = load_application_config(config, default_adapter_provider_registry(), base_dir=Path("/"))
    audit = loaded.audit
    assert sum(len(op.seed_salts) for op in audit.perturbations) == setting.v
    assert audit.regions[0].params == {"rows": setting.grid, "cols": setting.grid}
    assert (audit.controls[0].n_samples if audit.controls else 0) == setting.k
    assert config["source"]["annotation_file"].endswith(f"a4_N{setting.n}.txt")


def test_log_parsing_and_subdir_bytes(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "2026-09-29T05:37:33Z INFO ssat.core.config.resolver config.resolve_started input_type=AuditConfig\n"
        "  sanity top-1 accuracy: 85.00%\n"
        "2026-09-29T05:37:41Z INFO ssat.core.runtime.execution runtime.started clean=50\n"
        "2026-09-29T05:38:37Z INFO ssat.core.runtime.execution runtime.finished clean_records=50\n",
        encoding="utf-8",
    )
    events = parse_log_events(log)
    assert events["config.resolve_started"] == "2026-09-29T05:37:33Z"
    assert loop_seconds(events) == 56.0
    assert loop_seconds({}) is None
    (tmp_path / "out" / "perturbed").mkdir(parents=True)
    (tmp_path / "out" / "perturbed" / "a.parquet").write_bytes(b"x" * 10)
    (tmp_path / "out" / "run_manifest.json").write_bytes(b"y" * 3)
    sizes = subdir_bytes(tmp_path / "out")
    assert sizes["perturbed"] == 10 and sizes["other"] == 3 and sizes["total"] == 13 and sizes["metrics"] == 0


def _record(axis: str, setting: Setting, repeat: int) -> dict:
    """Synthetic measurement with run = 30 + 0.004 * items and bytes = 1900 * items."""

    items = setting.planned_items
    phases = {phase: {"elapsed_s": base + slope * items, "peak_rss_kb": int((1 + items / 1e5) * 2**20), "returncode": 0}
              for phase, base, slope in (("run", 30.0, 0.004), ("metrics", 5.0, 0.0005), ("analyze", 4.0, 0.0004), ("report", 3.0, 0.0001))}
    sizes = {name: 0 for name in fit_scaling.SUBDIRS}
    return {"axis": axis, "level": 0, "repeat": repeat, "setting": setting.key, "n": setting.n, "grid": setting.grid,
            "regions": setting.regions, "v": setting.v, "k": setting.k, "planned_items": items, "manifest_items": items,
            "started_at": "t", "run_loop_s": 0.0036 * items, "gpu_peak_mib": 3500, "gpu_util_mean": 5.0, "phases": phases,
            "bytes_after_run": {**sizes, "total": 1900 * items}, "bytes_final": {**sizes, "total": 2300 * items}}


@pytest.fixture()
def measurements() -> pd.DataFrame:
    records = [_record(axis, setting, repeat) for axis in AXES for setting, repeat in schedule(axis, 3)]
    return fit_scaling.flatten(records)


def test_fits_recover_the_generating_cost_model(measurements: pd.DataFrame) -> None:
    fits = fit_scaling.fit_rows(measurements).set_index(["axis", "y"])
    run = fits.loc[("pooled", "run_s")]
    assert run["a"] == pytest.approx(30.0) and run["b"] == pytest.approx(0.004) and run["r2"] == pytest.approx(1.0)
    assert fits.loc[("regions", "raw_bytes"), "b"] == pytest.approx(1900.0)
    assert fits.loc[("samples", "run_s"), "n_points"] == 11
    assert fits.loc[("regions", "run_s"), "n_points"] == 12  # includes the shared N=1,000 4x4 setting
    workloads = fit_scaling.workload_rows(fit_scaling.fit_rows(measurements)).set_index("workload")
    medium = workloads.loc["medium (paper setting)"]
    assert medium["items"] == 3_210_000 and bool(medium["extrapolated"])
    assert medium["pred_run_s"] == pytest.approx(30 + 0.004 * 3_210_000)
    assert "pred_run_rss_gib" not in workloads.columns
    assert not any("run_rss" in name for name in workloads.index)


def test_setting_summary_and_flatten_checks(measurements: pd.DataFrame) -> None:
    settings = fit_scaling.setting_summary(measurements)
    assert set(settings["n_repeats"]) == {1, 3}
    assert np.allclose(settings["run_s_cv"].fillna(0), 0)
    bad = _record("samples", AXES["samples"][0], 0) | {"manifest_items": 1}
    with pytest.raises(ValueError, match="planned"):
        fit_scaling.flatten([bad])


def test_coarse_to_fine_share() -> None:
    rows = fit_scaling.coarse_to_fine_rows().set_index("strategy")
    assert rows.loc["exhaustive 8x8", "items_per_sample"] == 1 + 64 * 20
    refine = rows.loc["2x2 then 4x4 inside the top cell (8x8 resolution there)"]
    assert refine["share_of_exhaustive_8x8"] == pytest.approx((1 + 20 * 20) / (1 + 64 * 20))


def test_component_rows_sum_the_main_process_stages() -> None:
    stat = lambda ms: {"items": 10, "seconds": ms / 100, "items_per_s": 1000 / ms, "ms_per_item": ms}  # noqa: E731
    components = {"decode": stat(1.0), "preparation_pool": stat(2.0), "preparation_serial": stat(6.0),
                  "inference": {"items": 10, "preprocess": stat(2.0), "forward": stat(0.5), "effective_area_mask_transform": stat(0.5)}}
    rows = fit_scaling.component_rows(components)
    assert rows.iloc[-1]["ms_per_item"] == pytest.approx(3.0)
    assert rows.iloc[-1]["items_per_s"] == pytest.approx(1000 / 3)


def test_summarize_writes_figures(measurements: pd.DataFrame, tmp_path: Path) -> None:
    measurements.to_csv(tmp_path / "measurements.csv", index=False)
    pd.DataFrame([{"reference": "a2_imagenet_mnv2_050_exact_k20", "n": 2000, "k": 20, "seconds": 12800.0},
                  {"reference": "imagenet_mnv2_050_exact_k3", "n": 10000, "k": 3, "seconds": 11900.0}]).to_csv(
        tmp_path / "reference_points.csv", index=False)
    assert summarize.main(["--summary-dir", str(tmp_path)]) == 0
    for name in ("samples", "regions", "controls", "perturbations", "memory"):
        assert (tmp_path / f"fig_a4_{name}.pdf").stat().st_size > 0
    assert json.loads((tmp_path / "summarize.provenance.json").read_text())["inputs"]


def test_scheduled_filters_settings_and_keeps_order() -> None:
    only = frozenset({"n500_g4_v5_k0", "n500_g4_v5_k20"})
    keys = [(setting.key, repeat) for setting, repeat in scheduled("controls", 3, only)]
    assert keys == [(key, repeat) for key, repeat in ((s.key, r) for s, r in schedule("controls", 3)) if key in only]
    assert len(keys) == 6
    assert scheduled("samples", 3, only) == []
    assert scheduled("controls", 3) == list(schedule("controls", 3))


def test_compare_plan_cache_ratios_and_drift_control(tmp_path: Path) -> None:
    settings = {s.key: s for s in AXES["controls"]} | {s.key: s for s in AXES["samples"]}
    keys = ("n500_g4_v5_k0", "n500_g4_v5_k20", "n1000_g4_v5_k3")
    pre = [_record("controls", settings[key], repeat) for key in keys for repeat in range(3)]
    post = []
    for record in pre:
        # After the fix: drift control unchanged, K=20 10 % faster, K=3 5 % faster (loop and run).
        factor = {"n500_g4_v5_k0": 1.0, "n500_g4_v5_k20": 0.9, "n1000_g4_v5_k3": 0.95}[record["setting"]]
        phases = {**record["phases"], "run": {**record["phases"]["run"], "elapsed_s": record["phases"]["run"]["elapsed_s"] * factor}}
        post.append({**record, "run_loop_s": record["run_loop_s"] * factor, "phases": phases})
    rows = compare_plan_cache.comparison_rows(fit_scaling.flatten(pre), fit_scaling.flatten(post)).set_index("setting")
    assert rows.loc["n500_g4_v5_k0", "chunks_per_sample"] == 1
    assert rows.loc["n500_g4_v5_k20", "chunks_per_sample"] == 14
    assert rows.loc["n500_g4_v5_k20", "loop_ratio"] == pytest.approx(0.9)
    assert rows.loc["n1000_g4_v5_k3", "run_ratio_vs_drift_control"] == pytest.approx(0.95)
    assert rows.loc["n500_g4_v5_k20", "loop_ms_per_item_before"] == pytest.approx(3.6)
    fits = compare_plan_cache.subset_fits(fit_scaling.flatten(pre), fit_scaling.flatten(post)).set_index(["code", "time"])
    assert fits.loc[("before", "loop"), "ms_per_item"] == pytest.approx(3.6)
    assert fits.loc[("before", "loop"), "predicted_paper_setting_s"] == pytest.approx(0.0036 * 3_210_000)
    with pytest.raises(ValueError, match="not in the pre-registered"):
        compare_plan_cache.comparison_rows(fit_scaling.flatten(pre[:3]), fit_scaling.flatten(post))
