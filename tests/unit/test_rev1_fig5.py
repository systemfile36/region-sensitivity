"""Unit tests for scripts/paper_figures/rev1_fig5_design_sensitivity.py (data tables and outputs)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.paper_figures import rev1_fig5_design_sensitivity as fig5


@pytest.fixture(scope="module")
def data() -> fig5.Fig5Data:
    return fig5.load_data()


def test_thresholds_match_a1_report(data: fig5.Fig5Data) -> None:
    points = data.preset_points.set_index(["series", "preset"])
    imagenet_default = points.loc[("imagenet", "default")]
    assert imagenet_default["high_min"] == pytest.approx(13.9, abs=0.05)
    assert imagenet_default["high_max"] == pytest.approx(14.5, abs=0.05)
    assert points.loc[("imagenet", "conservative"), "high_min"] == pytest.approx(7.5, abs=0.05)
    assert points.loc[("imagenet", "lenient"), "high_max"] == pytest.approx(17.3, abs=0.05)
    assert points.loc[("imagenet", "lenient"), "z_threshold"] == 1.5
    assert set(data.thresholds["z_threshold"].agg(["min", "max"])) == {0.0, 6.0}


def test_threshold_table_rejects_inconsistent_default() -> None:
    z_curve = pd.read_csv(fig5.INPUTS["a1_z_curve"])
    table = pd.read_csv(fig5.INPUTS["a1_table"])
    table.loc[table["run"] == "synthetic_shortcut", "high_default"] += 0.01
    presets = json.loads(fig5.INPUTS["a1_protocol"].read_text(encoding="utf-8"))["presets"]
    with pytest.raises(ValueError, match="synthetic_shortcut"):
        fig5.threshold_tables(z_curve, table, presets)


def test_control_count_matches_a2_report(data: fig5.Fig5Data) -> None:
    high = data.control_count.set_index(["series", "k"])["high_share_mean"]
    assert [round(high[("imagenet_cf", k)], 1) for k in (3, 5, 10, 20)] == [14.5, 10.7, 8.0, 6.8]
    assert round(high[("imagenet_exact", 20)], 1) == 5.8
    agreement = data.control_count.set_index(["series", "k"])["agreement_mean"]
    assert round(agreement[("imagenet_cf", 10)], 1) == 97.6


def test_profile_order_and_spearman_match_cross_model(data: fig5.Fig5Data) -> None:
    reference = data.profiles[data.profiles["model"] == fig5.REFERENCE_MODEL].sort_values("position")
    assert reference["cell_type"].tolist() == ["center"] * 4 + ["edge"] * 8 + ["corner"] * 4
    values = fig5.figure_values(data)["architectures"]
    for model, value in values["spearman_with_reference"].items():
        assert value == pytest.approx(values["spearman_in_cross_model_csv"][model], abs=1e-6)


def test_cost_fit_and_references_match_a4_report(data: fig5.Fig5Data) -> None:
    assert data.time_fit["b"] * 1e3 == pytest.approx(3.80, abs=0.01)
    assert set(data.timing["group"]) == set(fig5.AXIS_GROUPS)
    assert len(data.timing) == 41
    paper = data.references.set_index("reference").loc["imagenet_mnv2_050_exact_k3"]
    assert paper["relative_error"] == pytest.approx(0.025, abs=0.001)
    a2 = data.references.set_index("reference").loc["a2_imagenet_mnv2_050_exact_k20"]
    assert a2["relative_error"] == pytest.approx(0.009, abs=0.001)
    assert paper["predicted_raw_gb"] == pytest.approx(11.5, abs=0.05)
    run_fit = fig5.cost_tables(pd.read_csv(fig5.INPUTS["a4_measurements"]), pd.read_csv(fig5.INPUTS["a4_fits"]),
                               pd.read_csv(fig5.INPUTS["a4_reference_points"]), "run")[1]
    assert run_fit["b"] * 1e3 == pytest.approx(3.84, abs=0.01)


def test_post_fix_fit_matches_a4_report_section_6(data: fig5.Fig5Data) -> None:
    assert len(data.post_timing) == 10
    assert set(data.post_timing["group"]) == {"samples", "controls"}
    assert data.post_fit["b"] * 1e3 == pytest.approx(3.58, abs=0.01)
    assert data.post_fit["predicted_paper_setting_s"] == pytest.approx(11520, abs=5)
    values = fig5.figure_values(data)["time"]
    assert values["after_fix"]["fit_ms_per_item"] == pytest.approx(3.58, abs=0.01)
    assert values["before_fix"]["fit_ms_per_item"] == pytest.approx(3.80, abs=0.01)


def test_post_fix_tables_reject_unknown_settings() -> None:
    sweep = pd.read_csv(fig5.INPUTS["a4_measurements"])
    post = pd.read_csv(fig5.INPUTS["a4_post_measurements"])
    fits = pd.read_csv(fig5.INPUTS["a4_post_fits"])
    with pytest.raises(ValueError, match="not in the A4 sweep"):
        fig5.post_fix_tables(post, fits, sweep[sweep["setting"] != "n500_g4_v5_k20"], "loop")
    with pytest.raises(ValueError, match="'after' rows"):
        fig5.post_fix_tables(post, fits[fits["code"] == "before"], sweep, "loop")


def test_axis_group_assigns_shared_settings_to_their_owner() -> None:
    assert fig5.axis_group("n500_g4_v5_k3") == "samples"
    assert fig5.axis_group("n500_g4_v5_k20") == "controls"
    assert fig5.axis_group("n1000_g8_v5_k3") == "regions"
    with pytest.raises(ValueError):
        fig5.axis_group("n7_g4_v5_k3")


def test_main_writes_requested_combinations(tmp_path) -> None:
    assert fig5.main(["--output-dir", str(tmp_path), "--versions", "core", "extended", "--layouts", "grid", "column", "panels",
                      "--styles", "mono", "--annotations", "on", "--formats", "pdf"]) == 0
    index = pd.read_csv(tmp_path / "index.csv")
    expected = {
        "core/mono/fig5_core_grid_mono.pdf",
        "core/mono/fig5_core_column_mono.pdf",
        "extended/mono/fig5_extended_grid_mono.pdf",
        "fig5_design_sensitivity.pdf",
        *(f"core/mono/panels/fig5_core_{letter}_{key}_mono.pdf" for letter, key in zip("abcd", fig5.VERSIONS["core"])),
        *(f"extended/mono/panels/fig5_extended_{letter}_{key}_mono.pdf" for letter, key in zip("abcdef", fig5.VERSIONS["extended"])),
    }
    # The extended version has no single-column layout.
    assert set(index["path"]) == expected
    assert all((tmp_path / path).stat().st_size > 0 for path in expected)
    values = json.loads((tmp_path / "fig5_values.json").read_text(encoding="utf-8"))
    assert values["time"]["measure"] == "audit loop"
    assert (tmp_path / "provenance.json").is_file()
