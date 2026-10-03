"""Fig. 5 of the revised SSAT manuscript: sensitivity of the audit design and cost.

Reads only tracked revision-1 summary CSVs (A1, A2, A3, A4); no dump, store,
or ImageNet image is opened. Every combination of the options below is
written, so the manuscript can switch layouts without re-running analyses.

* Versions: ``core`` has the four panels of the manuscript plan (thresholds,
  HIGH share vs K, four-model profiles, audit time). ``extended`` adds grade
  agreement with K=20 and storage (six panels). The audit-time panel shows
  the A4 sweep, which ran before the planner fix, and the four settings
  re-measured after it (``deviations.md`` D-013), each with its own fit.
* Layouts: ``wide`` (one row; 2x3 for ``extended``) and ``grid`` (two
  columns) at full page width, ``column`` (one column, ``core`` only) at
  single-column width, and ``panels`` (one file per panel).
* Styles: ``color`` (validated categorical palette) and ``mono`` (grayscale).
  Every series also has its own line style and marker, so both styles stay
  readable in print.
* Annotations: ``on`` adds the default-threshold, K=3, and fit-slope labels
  (the slopes only in panels at least 80 mm wide); ``off`` (``_plain``
  files) leaves them to the caption.

Outputs under ``--output-dir``: ``<version>/<style>/fig5_<version>_<layout>_<style>[_plain].<fmt>``,
``<version>/<style>/panels/...``, ``fig5_design_sensitivity.pdf`` (core, grid,
color, annotated; the manuscript's ``figures/`` copy), ``fig5_values.json``
(every plotted or annotated number, with consistency checks against the
summaries), ``index.csv``, and ``provenance.json``.

Example:
    python scripts/paper_figures/rev1_fig5_design_sensitivity.py
    python scripts/paper_figures/rev1_fig5_design_sensitivity.py --versions core --layouts grid --styles color --formats pdf
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.revision_1.a4_scaling.sweep import AXES, owner_axis  # noqa: E402
from experiments.revision_1.common.provenance import REPO_ROOT, write_json_shared, write_provenance  # noqa: E402

REV1_DIR = REPO_ROOT / "experiments" / "revision_1"
DEFAULT_OUTPUT_DIR = REV1_DIR / "results" / "paper_figures" / "fig5"
INPUTS = {
    "a1_protocol": REV1_DIR / "a1_threshold" / "protocol.json",
    "a1_z_curve": REV1_DIR / "a1_threshold" / "summary" / "z_curve.csv",
    "a1_table": REV1_DIR / "a1_threshold" / "summary" / "table_a1.csv",
    "a2_grade_vs_k": REV1_DIR / "a2_control_count" / "summary" / "grade_vs_k.csv",
    "a3_region_profile": REV1_DIR / "a3_multi_arch" / "summary" / "region_profile.csv",
    "a3_cross_model": REV1_DIR / "a3_multi_arch" / "summary" / "cross_model.csv",
    "a4_measurements": REV1_DIR / "a4_scaling" / "summary" / "measurements.csv",
    "a4_fits": REV1_DIR / "a4_scaling" / "summary" / "fits.csv",
    "a4_reference_points": REV1_DIR / "a4_scaling" / "summary" / "reference_points.csv",
    "a4_post_measurements": REV1_DIR / "a4_scaling" / "summary_plan_cache" / "measurements.csv",
    "a4_post_fits": REV1_DIR / "a4_scaling" / "summary_plan_cache" / "fits.csv",
}

# Elsevier artwork widths (double and single column).
FULL_WIDTH_MM = 190.0
SINGLE_WIDTH_MM = 90.0
COMPACT_PANEL_MM = 60.0
TWO_COLUMN_LEGEND_MM = 80.0
MM = 1 / 25.4

# Reference categorical palette, slots 1-4 and 7 (validated light mode; aqua
# and yellow sit below 3:1, so every series also has its own marker and line style).
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
AXIS_INK = "#8c8b86"
GRID_INK = "#e6e5e1"
BAND_FILL = "#f0efec"

A2_SERIES = {
    "a2_imagenet_mnv2_050_cf_k20": "imagenet_cf",
    "a2_imagenet_mnv2_050_exact_k20": "imagenet_exact",
    "a2_synthetic_shortcut_k20": "synthetic",
}
MODELS = ("mobilenetv2_050", "mobilenetv2_100", "convnext_tiny", "deit_small")
REFERENCE_MODEL = "mobilenetv2_050"
K_TICKS = (2, 3, 5, 10, 20)
PAPER_K = 3
REFERENCES = {
    "imagenet_mnv2_050_exact_k3": ("Sec. 3.2 setting", "*"),
    "a2_imagenet_mnv2_050_exact_k20": ("A2 run, $K$=20", "X"),
}
# What each reference run timed (``reference_points.csv`` ``method``): the Sec. 3.2
# run's manifest spans the audit loop, the A2 run's log the whole ``ssat run``.
# Its prediction error uses the fit of that quantity, whichever time is plotted.
REFERENCE_MEASURES = {"imagenet_mnv2_050_exact_k3": "loop", "a2_imagenet_mnv2_050_exact_k20": "run"}
AXIS_GROUPS = {
    "samples": ("samples $N$", "o"),
    "regions": ("grid", "s"),
    "controls": ("controls $K$", "^"),
    "perturbations": ("conditions $V$", "D"),
}
TIME_MEASURES = {"loop": ("run_loop_s", "audit loop"), "run": ("run_s", "ssat run")}


@dataclass(frozen=True)
class SeriesStyle:
    """Line and marker encoding of one series; ``mono`` is its grayscale color."""

    label: str
    color: str
    mono: str
    linestyle: str
    marker: str
    filled: bool = True


SERIES = {
    "imagenet": SeriesStyle("ImageNet (4 runs)", BLUE, "#000000", "-", "o"),
    "imagenet_cf": SeriesStyle("ImageNet, crop-free", BLUE, "#000000", "-", "o"),
    "imagenet_exact": SeriesStyle("ImageNet, official", BLUE, "#000000", "--", "o", filled=False),
    "synthetic": SeriesStyle("Synthetic shortcut", VIOLET, "#6e6e6e", "-.", "D"),
    "mobilenetv2_050": SeriesStyle("MobileNetV2-0.5", BLUE, "#000000", "-", "o"),
    "mobilenetv2_100": SeriesStyle("MobileNetV2-1.0", ORANGE, "#4d4d4d", "--", "s"),
    "convnext_tiny": SeriesStyle("ConvNeXt-T", AQUA, "#6e6e6e", "-.", "^"),
    "deit_small": SeriesStyle("DeiT-S", YELLOW, "#8f8f8f", ":", "D"),
    "total": SeriesStyle("all outputs", BLUE, "#000000", "-", "o"),
    "raw": SeriesStyle("raw dump", ORANGE, "#6e6e6e", "--", "s", filled=False),
    "reference": SeriesStyle("reference run", ORANGE, "#000000", "", "*"),
    "before_fix": SeriesStyle("before the fix", AXIS_INK, "#8f8f8f", "-", "o", filled=False),
    "after_fix": SeriesStyle("after the fix", BLUE, "#000000", "-", "o"),
}


@dataclass(frozen=True)
class Fig5Data:
    """Tidy tables behind every panel, in percent, hours, and gigabytes."""

    presets: dict[str, dict[str, Any]]
    thresholds: pd.DataFrame
    preset_points: pd.DataFrame
    control_count: pd.DataFrame
    profiles: pd.DataFrame
    profile_spearman: dict[str, float]
    timing: pd.DataFrame
    time_measure: str
    time_fit: dict[str, float]
    post_timing: pd.DataFrame
    post_fit: dict[str, float]
    storage_fit: dict[str, float]
    references: pd.DataFrame


@dataclass(frozen=True)
class Context:
    """Rendering options shared by the panel functions."""

    style: str
    annotate: bool
    panel_mm: float

    @property
    def compact(self) -> bool:
        return self.panel_mm < COMPACT_PANEL_MM

    def color(self, key: str) -> str:
        series = SERIES[key]
        return series.color if self.style == "color" else series.mono

    @property
    def legend_size(self) -> float:
        return 5.5 if self.compact else 6.5


# --------------------------------------------------------------------------- data


def _percent(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    frame = frame.copy()
    frame[list(columns)] = frame[list(columns)] * 100.0
    return frame


def threshold_tables(z_curve: pd.DataFrame, table_a1: pd.DataFrame, presets: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """HIGH share against the z threshold, and at the lenient/default/conservative presets.

    ImageNet is the min/mean/max over its four runs; synthetic is the shortcut run.

    Raises:
        ValueError: If the curve at the default threshold does not reproduce
            ``high_default`` of ``table_a1.csv`` for every run.
    """

    curve = z_curve[z_curve["metric"] == "margin_drop"]
    groups = {"imagenet": table_a1.loc[table_a1["dataset"] == "imagenet", "run"].tolist(), "synthetic": ["synthetic_shortcut"]}
    if len(groups["imagenet"]) != 4:
        raise ValueError(f"expected 4 ImageNet runs in table_a1, got {groups['imagenet']}")
    default_z = float(presets["default"]["z_threshold"])
    at_default = curve[np.isclose(curve["z_threshold"], default_z)].set_index("run")["high"]
    for run, expected in table_a1.set_index("run")["high_default"].items():
        if run in at_default.index and not math.isclose(at_default[run], expected, abs_tol=1e-12):
            raise ValueError(f"z_curve at z={default_z} gives HIGH {at_default[run]} for {run}, table_a1 {expected}")

    curves, points = [], []
    for series, runs in groups.items():
        rows = curve[curve["run"].isin(runs)]
        stats = rows.groupby("z_threshold")["high"].agg(high_min="min", high_mean="mean", high_max="max").reset_index()
        curves.append(stats.assign(series=series))
        table = table_a1[table_a1["run"].isin(runs)]
        for preset, column in (("lenient", "lenient_high"), ("default", "high_default"), ("conservative", "conservative_high")):
            values = table[column]
            points.append({"series": series, "preset": preset, "z_threshold": float(presets[preset]["z_threshold"]),
                           "high_min": values.min(), "high_mean": values.mean(), "high_max": values.max()})
    columns = ("high_min", "high_mean", "high_max")
    return _percent(pd.concat(curves, ignore_index=True), columns), _percent(pd.DataFrame(points), columns)


def control_count_table(grade_vs_k: pd.DataFrame) -> pd.DataFrame:
    """HIGH share and grade agreement with K=20 per K (random subsets, ddof 0, all cells).

    Raises:
        ValueError: If the K=20 HIGH share differs from ``k20_high_share``.
    """

    rows = grade_vs_k[(grade_vs_k["scheme"] == "global_random") & (grade_vs_k["ddof"] == 0)
                      & (grade_vs_k["cell_type"] == "all") & grade_vs_k["run"].isin(A2_SERIES)].copy()
    full = rows[rows["k"] == 20]
    if not np.allclose(full["high_share_mean"], full["k20_high_share"], atol=1e-12):
        raise ValueError("grade_vs_k: K=20 HIGH share does not match k20_high_share")
    rows["series"] = rows["run"].map(A2_SERIES)
    columns = [f"{stat}_{q}" for stat in ("high_share", "agreement") for q in ("mean", "p05", "p95")]
    return _percent(rows[["series", "run", "k", "n_replicates", *columns]].sort_values(["series", "k"]), columns).reset_index(drop=True)


def profile_table(region_profile: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Crop-free 16-cell profiles (z within model), ordered by the MobileNetV2-0.5 rank.

    Also returns each model's Spearman correlation with MobileNetV2-0.5 over the
    16 cell means, the statistic reported in ``cross_model.csv``.
    """

    rows = region_profile[(region_profile["population"] == "all") & (region_profile["protocol"] == "crop_free")]
    counts = rows.groupby("model")["cell"].nunique()
    if set(counts.index) != set(MODELS) or (counts != 16).any():
        raise ValueError(f"region_profile: expected 16 crop-free cells per model, got {counts.to_dict()}")
    order = rows[rows["model"] == REFERENCE_MODEL].sort_values("rank_in_model")["cell"].tolist()
    position = {cell: i for i, cell in enumerate(order)}
    table = rows.assign(position=rows["cell"].map(position)).sort_values(["model", "position"])
    ranks = table.pivot(index="cell", columns="model", values="mean_margin_drop").rank()
    spearman = {model: float(ranks[REFERENCE_MODEL].corr(ranks[model])) for model in MODELS if model != REFERENCE_MODEL}
    return table[["model", "cell", "cell_type", "position", "mean_margin_drop", "z_in_model"]].reset_index(drop=True), spearman


def axis_group(setting: str) -> str:
    """The sweep axis that owns an A4 setting (shared settings belong to the first axis listing them)."""

    for settings in AXES.values():
        for candidate in settings:
            if candidate.key == setting:
                return owner_axis(candidate)
    raise ValueError(f"A4 setting {setting} is on no sweep axis")


def _timing_frame(measurements: pd.DataFrame, column: str) -> pd.DataFrame:
    return pd.DataFrame({
        "setting": measurements["setting"],
        "group": measurements["setting"].map(axis_group),
        "items": measurements["items"],
        "hours": measurements[column] / 3600.0,
        "total_gb": measurements["total_bytes"] / 1e9,
        "raw_gb": measurements["raw_bytes"] / 1e9,
    })


def cost_tables(measurements: pd.DataFrame, fits: pd.DataFrame, reference_points: pd.DataFrame,
                time_measure: str) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], pd.DataFrame]:
    """A4 timed audits, pooled linear fits, and the two independently measured reference runs."""

    column, _ = TIME_MEASURES[time_measure]
    pooled = fits[fits["axis"] == "pooled"].set_index("y")
    time_fit = {key: float(pooled.loc[column, key]) for key in ("a", "b", "r2", "x_min", "x_max")}
    storage_fit = {"total_bytes_per_item": float(pooled.loc["total_bytes", "b"]), "raw_bytes_per_item": float(pooled.loc["raw_bytes", "b"])}
    timing = _timing_frame(measurements, column)
    refs = reference_points[reference_points["reference"].isin(REFERENCES)].copy()
    measure_fits = {measure: (float(pooled.loc[y, "a"]), float(pooled.loc[y, "b"])) for measure, (y, _) in TIME_MEASURES.items()}
    refs["fit_measure"] = refs["reference"].map(REFERENCE_MEASURES)
    refs["predicted_s"] = [measure_fits[m][0] + measure_fits[m][1] * items for m, items in zip(refs["fit_measure"], refs["items"])]
    refs["relative_error"] = refs["predicted_s"] / refs["seconds"] - 1.0
    refs["hours"] = refs["seconds"] / 3600.0
    refs["raw_gb"] = refs["raw_bytes"] / 1e9
    refs["predicted_raw_gb"] = storage_fit["raw_bytes_per_item"] * refs["items"] / 1e9
    return timing, time_fit, storage_fit, refs.reset_index(drop=True)


def post_fix_tables(measurements: pd.DataFrame, fits: pd.DataFrame, sweep: pd.DataFrame,
                    time_measure: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """The settings re-measured after the planner fix and their linear fit (``compare_plan_cache.py``).

    Raises:
        ValueError: If a re-measured setting is not in the sweep, or the fit is missing.
    """

    extra = set(measurements["setting"]) - set(sweep["setting"])
    if extra:
        raise ValueError(f"re-measured settings not in the A4 sweep: {sorted(extra)}")
    rows = fits[(fits["code"] == "after") & (fits["time"] == time_measure)]
    if len(rows) != 1:
        raise ValueError(f"summary_plan_cache/fits.csv has {len(rows)} 'after' rows for {time_measure}")
    fit = {key: float(rows.iloc[0][key]) for key in ("a", "b", "r2", "x_min", "x_max", "n_points", "predicted_paper_setting_s")}
    return _timing_frame(measurements, TIME_MEASURES[time_measure][0]), fit


def load_data(time_measure: str = "loop") -> Fig5Data:
    """Read the summary CSVs and build every panel table."""

    presets = json.loads(INPUTS["a1_protocol"].read_text(encoding="utf-8"))["presets"]
    thresholds, preset_points = threshold_tables(pd.read_csv(INPUTS["a1_z_curve"]), pd.read_csv(INPUTS["a1_table"]), presets)
    profiles, spearman = profile_table(pd.read_csv(INPUTS["a3_region_profile"]))
    sweep = pd.read_csv(INPUTS["a4_measurements"])
    timing, time_fit, storage_fit, refs = cost_tables(sweep, pd.read_csv(INPUTS["a4_fits"]),
                                                      pd.read_csv(INPUTS["a4_reference_points"]), time_measure)
    post_timing, post_fit = post_fix_tables(pd.read_csv(INPUTS["a4_post_measurements"]), pd.read_csv(INPUTS["a4_post_fits"]),
                                            sweep, time_measure)
    return Fig5Data(presets=presets, thresholds=thresholds, preset_points=preset_points,
                    control_count=control_count_table(pd.read_csv(INPUTS["a2_grade_vs_k"])),
                    profiles=profiles, profile_spearman=spearman, timing=timing, time_measure=time_measure,
                    time_fit=time_fit, post_timing=post_timing, post_fit=post_fit, storage_fit=storage_fit, references=refs)


def figure_values(data: Fig5Data) -> dict[str, Any]:
    """Every plotted or annotated number, for checking the caption and text against the figure."""

    def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return json.loads(frame.to_json(orient="records", double_precision=6))

    cross = pd.read_csv(INPUTS["a3_cross_model"])
    cross = cross[(cross["population"] == "all") & (cross["protocol"] == "crop_free")]
    reported = {}
    for model in data.profile_spearman:
        pair = cross[((cross["model_a"] == REFERENCE_MODEL) & (cross["model_b"] == model))
                     | ((cross["model_b"] == REFERENCE_MODEL) & (cross["model_a"] == model))]
        reported[model] = float(pair["profile_spearman"].iloc[0]) if len(pair) else None
    curve_default = data.thresholds[np.isclose(data.thresholds["z_threshold"], float(data.presets["default"]["z_threshold"]))]
    return {
        "units": {"shares": "percent", "time": "hours", "storage": "GB (1e9 bytes)"},
        "thresholds": {"presets": data.presets, "at_default": records(curve_default), "preset_points": records(data.preset_points),
                       "curve_endpoints": records(data.thresholds[data.thresholds["z_threshold"].isin([0.0, 6.0])])},
        "control_count": records(data.control_count),
        "architectures": {"protocol": "crop_free", "order_by": REFERENCE_MODEL,
                          "cell_order": data.profiles[data.profiles["model"] == REFERENCE_MODEL]["cell"].tolist(),
                          "spearman_with_reference": data.profile_spearman, "spearman_in_cross_model_csv": reported},
        "time": {"measure": TIME_MEASURES[data.time_measure][1],
                 "before_fix": {"source": "A4 sweep, pooled fit", "fit_seconds": data.time_fit,
                                "fit_ms_per_item": data.time_fit["b"] * 1e3, "n_measurements": int(len(data.timing))},
                 "after_fix": {"source": "D-013 re-measurement, fit over its four settings", "fit_seconds": data.post_fit,
                               "fit_ms_per_item": data.post_fit["b"] * 1e3, "n_measurements": int(len(data.post_timing)),
                               "predicted_sec32_setting_h": data.post_fit["predicted_paper_setting_s"] / 3600.0},
                 "references_before_fix": records(data.references[["reference", "items", "seconds", "fit_measure", "predicted_s",
                                                                   "relative_error", "method"]])},
        "storage": {**data.storage_fit,
                    "references": records(data.references[["reference", "items", "raw_gb", "predicted_raw_gb"]].dropna())},
    }


# --------------------------------------------------------------------------- panels


def _style_axes(ax: Axes, *, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_INK)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=AXIS_INK, labelcolor=TEXT_SECONDARY, width=0.6, length=2.5)
    ax.grid(axis=grid_axis, color=GRID_INK, linewidth=0.5)
    ax.set_axisbelow(True)


def _line(ax: Axes, ctx: Context, key: str, x: Any, y: Any, **kwargs: Any) -> None:
    series = SERIES[key]
    color = ctx.color(key)
    ax.plot(x, y, color=color, linestyle=series.linestyle, marker=series.marker, markersize=3.0 if ctx.compact else 3.8,
            markerfacecolor=color if series.filled else SURFACE, markeredgecolor=color, markeredgewidth=0.8,
            linewidth=1.2, label=series.label, **kwargs)


def _band(ax: Axes, ctx: Context, key: str, x: Any, low: Any, high: Any) -> None:
    if np.any(np.asarray(high) > np.asarray(low)):
        ax.fill_between(x, low, high, color=ctx.color(key), alpha=0.18, linewidth=0)


def _note(ax: Axes, text: str, xy: tuple[float, float], ctx: Context, **kwargs: Any) -> None:
    ax.annotate(text, xy=xy, fontsize=ctx.legend_size, color=TEXT_SECONDARY, **kwargs)


def _log_k_axis(ax: Axes, ticks: Sequence[int] = K_TICKS) -> None:
    ax.set_xscale("log")
    ax.set_xticks(list(ticks), [str(k) for k in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("matched controls per anchor $K$")


def plot_thresholds(ax: Axes, data: Fig5Data, ctx: Context) -> None:
    """HIGH share against the z threshold, with the lenient and conservative presets."""

    for key, rows in data.thresholds.groupby("series", sort=False):
        z = rows["z_threshold"].to_numpy()
        _band(ax, ctx, key, z, rows["high_min"], rows["high_max"])
        _line(ax, ctx, key, z, rows["high_mean"], markevery=[i for i, v in enumerate(z) if np.isclose(v, round(v))])
    markers = {"lenient": "^", "conservative": "v"}
    # Small horizontal offsets keep the two series' preset markers apart at the same threshold.
    offsets = {"imagenet": -0.08, "synthetic": 0.08}
    for _, point in data.preset_points[data.preset_points["preset"].isin(markers)].iterrows():
        error = [[point["high_mean"] - point["high_min"]], [point["high_max"] - point["high_mean"]]]
        ax.errorbar(point["z_threshold"] + offsets[point["series"]], point["high_mean"], yerr=error, fmt=markers[point["preset"]],
                    color=ctx.color(point["series"]),
                    markeredgecolor=TEXT_PRIMARY, markeredgewidth=0.6, markersize=4.5 if ctx.compact else 5.5, elinewidth=0.8, zorder=5)
    default_z = float(data.presets["default"]["z_threshold"])
    ax.axvline(default_z, color=AXIS_INK, linestyle=":", linewidth=0.8, zorder=1)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"z threshold $\tau_z$")
    ax.set_ylabel("HIGH share (%)")
    if ctx.annotate:
        _note(ax, f"default ({default_z:g})", (default_z, 0), ctx, xytext=(2, 2), textcoords="offset points")
    handles, _ = ax.get_legend_handles_labels()
    handles += [Line2D([], [], linestyle="", marker=marker, markerfacecolor=SURFACE, markeredgecolor=TEXT_PRIMARY, markersize=4.5,
                       label=f"{preset} preset") for preset, marker in markers.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=ctx.legend_size, frameon=False)


def _plot_control_stat(ax: Axes, data: Fig5Data, ctx: Context, stat: str, *, include_full: bool) -> None:
    for key in A2_SERIES.values():
        rows = data.control_count[data.control_count["series"] == key]
        if not include_full:
            rows = rows[rows["k"] < 20]
        _band(ax, ctx, key, rows["k"], rows[f"{stat}_p05"], rows[f"{stat}_p95"])
        _line(ax, ctx, key, rows["k"], rows[f"{stat}_mean"])
    ax.axvline(PAPER_K, color=AXIS_INK, linestyle=":", linewidth=0.8, zorder=1)
    _log_k_axis(ax, K_TICKS if include_full else [k for k in K_TICKS if k < 20])


def plot_control_high(ax: Axes, data: Fig5Data, ctx: Context) -> None:
    """HIGH share against the number of matched controls."""

    _plot_control_stat(ax, data, ctx, "high_share", include_full=True)
    ax.set_ylim(0, None)
    ax.set_ylabel("HIGH share (%)")
    if ctx.annotate:
        rows = data.control_count[data.control_count["series"] == "imagenet_cf"].set_index("k")["high_share_mean"]
        for k, offset in ((PAPER_K, (4, 2)), (20, (-4, 6))):
            _note(ax, f"{rows[k]:.1f}%", (k, rows[k]), ctx, xytext=offset, textcoords="offset points",
                  ha="left" if k == PAPER_K else "right")
        _note(ax, f"$K$={PAPER_K} (Table 3)", (PAPER_K, 0), ctx, xytext=(2, 2), textcoords="offset points")
    ax.legend(loc="upper right", fontsize=ctx.legend_size, frameon=False)


def plot_control_agreement(ax: Axes, data: Fig5Data, ctx: Context) -> None:
    """Grade agreement of random K-control subsets with all 20 controls."""

    _plot_control_stat(ax, data, ctx, "agreement", include_full=False)
    ax.set_ylim(None, 100)
    ax.set_ylabel("grade agreement with $K$=20 (%)")
    if ctx.annotate:
        rows = data.control_count[data.control_count["series"] == "imagenet_cf"].set_index("k")["agreement_mean"]
        for k in (PAPER_K, 10):
            _note(ax, f"{rows[k]:.1f}%", (k, rows[k]), ctx, xytext=(4, -8), textcoords="offset points")
    ax.legend(loc="lower right", fontsize=ctx.legend_size, frameon=False)


def plot_architectures(ax: Axes, data: Fig5Data, ctx: Context) -> None:
    """Standardized crop-free 16-cell profiles of the four classifiers."""

    reference = data.profiles[data.profiles["model"] == REFERENCE_MODEL].sort_values("position")
    types = reference["cell_type"].tolist()
    start = 0
    for i in range(1, len(types) + 1):
        if i == len(types) or types[i] != types[start]:
            if types[start] != "edge":
                ax.axvspan(start - 0.5, i - 0.5, color=BAND_FILL, linewidth=0, zorder=0)
            ax.text((start + i - 1) / 2, 1.0, types[start], transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=ctx.legend_size, color=TEXT_SECONDARY)
            start = i
    for model in MODELS:
        rows = data.profiles[data.profiles["model"] == model].sort_values("position")
        _line(ax, ctx, model, rows["position"], rows["z_in_model"])
    ax.axhline(0, color=AXIS_INK, linewidth=0.5, zorder=1)
    ax.set_xlim(-0.5, len(types) - 0.5)
    if ctx.compact:
        ax.set_xticks([])
        ax.set_xlabel("grid cell, crop-free")
    else:
        ax.set_xticks(range(len(types)), [cell.replace("/", "") for cell in reference["cell"]], rotation=90, fontsize=ctx.legend_size)
        ax.set_xlabel("grid cell, crop-free (ordered by MobileNetV2-0.5)")
    ax.set_ylabel("mean margin_drop (z within model)")
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.12 * (high - low))
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93), fontsize=ctx.legend_size, frameon=False)


def _fit_line(ax: Axes, ctx: Context, key: str, slope: float, intercept: float, x_fit_max: float, x_end: float, label: str) -> None:
    color = ctx.color(key)
    inside = np.linspace(0, x_fit_max, 50)
    beyond = np.linspace(x_fit_max, x_end, 50)
    ax.plot(inside / 1e6, intercept + slope * inside, color=color, linewidth=1.0, label=label, zorder=2)
    ax.plot(beyond / 1e6, intercept + slope * beyond, color=color, linewidth=1.0, linestyle=(0, (3, 2)), zorder=2)


def _reference_points(ax: Axes, data: Fig5Data, ctx: Context, column: str, *, label_suffix: str = "") -> None:
    for _, ref in data.references.dropna(subset=[column]).iterrows():
        label, marker = REFERENCES[ref["reference"]]
        label += label_suffix
        ax.plot(ref["items"] / 1e6, ref[column], linestyle="", marker=marker, markersize=7 if marker == "*" else 5,
                color=ctx.color("reference"), markeredgecolor=TEXT_PRIMARY, markeredgewidth=0.5, label=label, zorder=5)


def plot_time(ax: Axes, data: Fig5Data, ctx: Context) -> None:
    """Wall time against model evaluations: the sweep before the planner fix and the settings re-measured after it."""

    size = 3.0 if ctx.compact else 3.5
    after = ctx.color("after_fix")
    sweep_handles = []
    for group, (label, marker) in AXIS_GROUPS.items():
        rows = data.timing[data.timing["group"] == group]
        sweep_handles += ax.plot(rows["items"] / 1e6, rows["hours"], linestyle="", marker=marker, markersize=size, markerfacecolor=SURFACE,
                                 markeredgecolor=TEXT_SECONDARY, markeredgewidth=0.7, label=label, zorder=3)
        rows = data.post_timing[data.post_timing["group"] == group]
        ax.plot(rows["items"] / 1e6, rows["hours"], linestyle="", marker=marker, markersize=size + 0.5, markerfacecolor=after,
                markeredgecolor=SURFACE, markeredgewidth=0.4, zorder=4)
    x_end = 1.05 * data.references["items"].max()
    fit_handles = []
    for key, fit, word in (("after_fix", data.post_fit, "after"), ("before_fix", data.time_fit, "before")):
        slope_label = ctx.annotate and ctx.panel_mm >= TWO_COLUMN_LEGEND_MM
        label = f"{word}: {fit['b'] * 1e3:.1f} ms per evaluation" if slope_label else f"fit {word} the fix"
        _fit_line(ax, ctx, key, fit["b"] / 3600.0, fit["a"] / 3600.0, fit["x_max"], x_end, label)
        fit_handles.append(ax.get_lines()[-2])
    references = []
    for _, ref in data.references.iterrows():
        _, marker = REFERENCES[ref["reference"]]
        references += ax.plot(ref["items"] / 1e6, ref["hours"], linestyle="", marker=marker, markersize=7 if marker == "*" else 5,
                              color=ctx.color("reference"), markeredgecolor=TEXT_PRIMARY, markeredgewidth=0.5, zorder=5)
    ax.set_xlim(0, x_end / 1e6)
    ax.set_ylim(0, None)
    ax.set_xlabel("model evaluations (millions)")
    ax.set_ylabel(f"{TIME_MEASURES[data.time_measure][1]} wall time (h)")
    after_points = Line2D([], [], linestyle="", marker="o", markersize=size + 0.5, markerfacecolor=after, markeredgecolor=SURFACE,
                          label="after the fix")
    rest = [after_points, *fit_handles, tuple(references)]
    rest_labels = ["after the fix", *(h.get_label() for h in fit_handles), "Sec. 3.2 and A2 runs"]
    common = {"fontsize": ctx.legend_size, "frameon": False, "handletextpad": 0.3, "title_fontsize": ctx.legend_size,
              "handler_map": {tuple: HandlerTuple(ndivide=None, pad=0.3)}}
    if ctx.panel_mm < TWO_COLUMN_LEGEND_MM:
        # Narrow panels: one legend above the lines, with headroom for it.
        ax.set_ylim(0, 1.9 * ax.get_ylim()[1])
        ax.legend(handles=[*sweep_handles, *rest], labels=[*(h.get_label() for h in sweep_handles), *rest_labels],
                  loc="upper left", ncol=1 if ctx.compact else 2, columnspacing=0.8, title="sweep before the fix (open):",
                  alignment="left", **common)
        return
    # Sweep axes upper left; the re-measurement, fits, and reference runs in the empty area below the lines.
    sweep = ax.legend(handles=sweep_handles, loc="upper left", ncol=2, columnspacing=0.8, title="sweep before the fix",
                      alignment="left", **common)
    ax.add_artist(sweep)
    ax.legend(handles=rest, labels=rest_labels, loc="lower right", **common)


def plot_storage(ax: Axes, data: Fig5Data, ctx: Context) -> None:
    """Stored bytes against model evaluations: all outputs and the raw dump."""

    x_end = 1.05 * data.references["items"].max()
    for key, column, per_item in (("total", "total_gb", "total_bytes_per_item"), ("raw", "raw_gb", "raw_bytes_per_item")):
        series = SERIES[key]
        color = ctx.color(key)
        ax.plot(data.timing["items"] / 1e6, data.timing[column], linestyle="", marker=series.marker, markersize=3.0 if ctx.compact else 3.5,
                markerfacecolor=color if series.filled else SURFACE, markeredgecolor=color, markeredgewidth=0.7, label=series.label, zorder=3)
        slope = data.storage_fit[per_item] / 1e9
        _fit_line(ax, ctx, key, slope, 0.0, data.time_fit["x_max"], x_end, f"{data.storage_fit[per_item] / 1e3:.1f} kB per evaluation")
    _reference_points(ax, data, ctx, "raw_gb", label_suffix=", raw dump")
    ax.set_xlim(0, x_end / 1e6)
    ax.set_ylim(0, None)
    ax.set_xlabel("model evaluations (millions)")
    ax.set_ylabel("storage (GB)")
    # Headroom keeps the legend above the extrapolated lines.
    ax.set_ylim(0, 1.6 * ax.get_ylim()[1])
    ax.legend(loc="upper left", fontsize=ctx.legend_size, frameon=False)


@dataclass(frozen=True)
class Panel:
    title: str
    plot: Callable[[Axes, Fig5Data, Context], None]


PANELS = {
    "thresholds": Panel("Grading threshold", plot_thresholds),
    "control_high": Panel("Control count", plot_control_high),
    "control_agreement": Panel("Control count: agreement", plot_control_agreement),
    "architectures": Panel("Architectures", plot_architectures),
    "time": Panel("Audit time", plot_time),
    "storage": Panel("Storage", plot_storage),
}
VERSIONS = {
    "core": ("thresholds", "control_high", "architectures", "time"),
    "extended": ("thresholds", "control_high", "control_agreement", "architectures", "time", "storage"),
}


@dataclass(frozen=True)
class Layout:
    """Page width, row height, and columns per panel count; ``None`` means one file per panel."""

    width: str
    row_height_mm: float
    columns: dict[int, int] | None


LAYOUTS = {
    "wide": Layout("full", 58.0, {4: 4, 6: 3}),
    "grid": Layout("full", 64.0, {4: 2, 6: 2}),
    "column": Layout("single", 56.0, {4: 1}),
    "panels": Layout("single", 64.0, None),
}

RC = {
    "font.size": 7.5,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.0,
    "axes.labelcolor": TEXT_PRIMARY,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.handlelength": 2.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
}


def _font_family() -> list[str]:
    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = [name for name in ("Arial", "Helvetica", "Liberation Sans") if name in available]
    return [*preferred, "DejaVu Sans"]


def _draw(axes: Sequence[Axes], panels: Sequence[str], letters: Sequence[str], data: Fig5Data, ctx: Context) -> None:
    for ax, key, letter in zip(axes, panels, letters):
        _style_axes(ax)
        PANELS[key].plot(ax, data, ctx)
        ax.set_title(f"({letter}) {PANELS[key].title}", loc="left", fontweight="bold", color=TEXT_PRIMARY)


def _save(fig: plt.Figure, stem: Path, formats: Sequence[str], dpi: int) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi)
        path.chmod(0o644)
        paths.append(path)
    plt.close(fig)
    return paths


def render(data: Fig5Data, version: str, layout_name: str, style: str, annotate: bool, *, output_dir: Path,
           formats: Sequence[str], dpi: int, widths_mm: dict[str, float], stem: Path | None = None) -> list[dict[str, Any]]:
    """Write one version/layout/style/annotation combination and return its index rows.

    Args:
        stem: File path without suffix for a multi-panel layout; defaults to
            ``<output_dir>/<version>/<style>/fig5_<version>_<layout>_<style>[_plain]``.
    """

    panels = VERSIONS[version]
    layout = LAYOUTS[layout_name]
    letters = [chr(ord("a") + i) for i in range(len(panels))]
    width_mm = widths_mm[layout.width]
    suffix = f"{style}{'' if annotate else '_plain'}"
    base = output_dir / version / style
    jobs: list[tuple[Sequence[str], Sequence[str], int, Path]] = []
    if layout.columns is None:
        jobs = [((key,), (letter,), 1, base / "panels" / f"fig5_{version}_{letter}_{key}_{suffix}") for key, letter in zip(panels, letters)]
    elif len(panels) in layout.columns:
        jobs = [(panels, letters, layout.columns[len(panels)], stem or base / f"fig5_{version}_{layout_name}_{suffix}")]
    rows = []
    for job_panels, job_letters, ncols, stem in jobs:
        nrows = math.ceil(len(job_panels) / ncols)
        height_mm = layout.row_height_mm * nrows
        ctx = Context(style=style, annotate=annotate, panel_mm=width_mm / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(width_mm * MM, height_mm * MM), layout="constrained", squeeze=False)
        _draw(axes.flat, job_panels, job_letters, data, ctx)
        for ax in list(axes.flat)[len(job_panels):]:
            ax.set_axis_off()
        for path in _save(fig, stem, formats, dpi):
            rows.append({"version": version, "layout": layout_name, "panels": "+".join(job_panels), "style": style,
                         "annotations": "on" if annotate else "off", "format": path.suffix[1:],
                         "width_mm": width_mm, "height_mm": height_mm, "path": path.relative_to(output_dir).as_posix()})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--versions", nargs="+", choices=list(VERSIONS), default=list(VERSIONS))
    parser.add_argument("--layouts", nargs="+", choices=list(LAYOUTS), default=list(LAYOUTS))
    parser.add_argument("--styles", nargs="+", choices=("color", "mono"), default=["color", "mono"])
    parser.add_argument("--annotations", nargs="+", choices=("on", "off"), default=["on", "off"])
    parser.add_argument("--formats", nargs="+", choices=("pdf", "png", "svg"), default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=300, help="Raster resolution for png.")
    parser.add_argument("--time-measure", choices=list(TIME_MEASURES), default="loop",
                        help="A4 time plotted and fitted: the audit loop (default) or the whole `ssat run`.")
    parser.add_argument("--full-width-mm", type=float, default=FULL_WIDTH_MM)
    parser.add_argument("--single-width-mm", type=float, default=SINGLE_WIDTH_MM)
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.time_measure)
    widths = {"full": args.full_width_mm, "single": args.single_width_mm}
    rows: list[dict[str, Any]] = []
    with plt.rc_context({**RC, "font.family": _font_family()}):
        for version in args.versions:
            for layout in args.layouts:
                for style in args.styles:
                    for annotation in args.annotations:
                        rows += render(data, version, layout, style, annotation == "on", output_dir=output_dir,
                                       formats=args.formats, dpi=args.dpi, widths_mm=widths)
        if "core" in args.versions:
            rows += render(data, "core", "grid", "color", True, output_dir=output_dir, formats=["pdf"], dpi=args.dpi,
                           widths_mm=widths, stem=output_dir / "fig5_design_sensitivity")
    index = pd.DataFrame(rows)
    index.to_csv(output_dir / "index.csv", index=False)
    (output_dir / "index.csv").chmod(0o644)
    write_json_shared(output_dir / "fig5_values.json", figure_values(data))
    write_provenance(output_dir, inputs=INPUTS, extra={"args": {key: str(value) for key, value in vars(args).items()}})
    print(f"wrote {len(index)} figure files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
