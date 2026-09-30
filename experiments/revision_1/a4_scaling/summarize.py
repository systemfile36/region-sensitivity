"""A4 figures from the ``fit_scaling.py`` CSVs in ``summary/``.

* ``fig_a4_samples.pdf``: wall time per phase and throughput against N;
* ``fig_a4_regions.pdf``: wall time and storage against regions per sample;
* ``fig_a4_controls.pdf``: ``ssat run`` seconds per sample against K, with
  the A2 K=20 run as a separately measured point;
* ``fig_a4_perturbations.pdf``: wall time against V;
* ``fig_a4_memory.pdf``: peak host RSS per phase and GPU peak against items.

Points are individual repeats; lines join setting means.

Example:
    python experiments/revision_1/a4_scaling/summarize.py
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.revision_1.a4_scaling.sweep import AXES, level_of  # noqa: E402
from experiments.revision_1.common.provenance import write_provenance  # noqa: E402

SUMMARY_DIR = Path(__file__).resolve().parent / "summary"
PHASE_COLORS = {"run": "#1f77b4", "metrics": "#ff7f0e", "analyze": "#2ca02c", "report": "#d62728"}
AXIS_LABELS = {"samples": "samples N", "regions": "regions per sample R", "controls": "controls per target K",
               "perturbations": "operator-seed variants V"}


def axis_frame(frame: pd.DataFrame, axis: str) -> pd.DataFrame:
    """Rows of every setting on ``axis`` (including shared ones), with the axis level as ``x``."""

    levels = {setting.key: level_of(axis, setting) for setting in AXES[axis]}
    rows = frame[frame["setting"].isin(levels)].copy()
    rows["x"] = rows["setting"].map(levels)
    return rows.sort_values("x")


def _log_x(ax: plt.Axes, ticks: Sequence[int]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([f"{tick:,}" for tick in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())


def _phase_times(ax: plt.Axes, rows: pd.DataFrame, *, log: bool) -> None:
    for phase, color in PHASE_COLORS.items():
        ax.scatter(rows["x"], rows[f"{phase}_s"], s=10, color=color, alpha=0.6)
        means = rows.groupby("x")[f"{phase}_s"].mean()
        ax.plot(means.index, means.to_numpy(), "-", color=color, label=phase)
    means = rows.groupby("x")["run_loop_s"].mean()
    ax.plot(means.index, means.to_numpy(), "--", color=PHASE_COLORS["run"], label="run: audit loop only")
    if log:
        _log_x(ax, sorted(rows["x"].unique()))
        ax.set_yscale("log")
    ax.set_ylabel("wall time (s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False)


def plot_samples(frame: pd.DataFrame, path: Path) -> None:
    rows = axis_frame(frame, "samples")
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4))
    _phase_times(left, rows, log=True)
    for column, label, style in (("loop_items_per_s", "audit loop", "-o"), ("run_items_per_s", "whole `ssat run`", "--s")):
        means = rows.groupby("x")[column].mean()
        right.plot(means.index, means.to_numpy(), style, ms=3, label=label)
        right.scatter(rows["x"], rows[column], s=8, alpha=0.5)
    _log_x(right, sorted(rows["x"].unique()))
    right.set_ylabel("items / s (clean + perturbed)")
    right.set_ylim(bottom=0)
    right.grid(alpha=0.3)
    right.legend(fontsize=7, frameon=False)
    for ax in (left, right):
        ax.set_xlabel(AXIS_LABELS["samples"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_regions(frame: pd.DataFrame, path: Path) -> None:
    rows = axis_frame(frame, "regions")
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4))
    _phase_times(left, rows, log=False)
    for column, label in (("raw_bytes", "raw dump (after run)"), ("total_bytes", "with metrics, analysis, report")):
        means = rows.groupby("x")[column].mean() / 2**30
        right.plot(means.index, means.to_numpy(), "-o", ms=3, label=label)
    right.set_ylabel("storage (GiB)")
    right.grid(alpha=0.3)
    right.legend(fontsize=7, frameon=False)
    for ax in (left, right):
        ax.set_xlabel(AXIS_LABELS["regions"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_controls(frame: pd.DataFrame, references: pd.DataFrame, path: Path) -> None:
    rows = axis_frame(frame, "controls")
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    rows = rows.assign(per_sample=rows["run_s"] / rows["n"], loop_per_sample=rows["run_loop_s"] / rows["n"])
    for column, label, style in (("per_sample", "`ssat run`", "-o"), ("loop_per_sample", "audit loop", "--s")):
        means = rows.groupby("x")[column].mean()
        ax.plot(means.index, means.to_numpy(), style, ms=3, label=f"A4 (N=500): {label}")
    a2 = references[references["reference"].str.startswith("a2_")]
    ax.scatter(a2["k"], a2["seconds"] / a2["n"], marker="*", s=80, color="k", zorder=3, label="A2 K=20 run (N=2,000; `ssat run`)")
    ax.set_xlabel(AXIS_LABELS["controls"])
    ax.set_ylabel("seconds per sample")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_perturbations(frame: pd.DataFrame, path: Path) -> None:
    rows = axis_frame(frame, "perturbations")
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    _phase_times(ax, rows, log=False)
    ax.set_xlabel(AXIS_LABELS["perturbations"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_memory(frame: pd.DataFrame, path: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4))
    for phase, color in PHASE_COLORS.items():
        left.scatter(frame["items"], frame[f"{phase}_rss_gib"], s=10, color=color, label=phase)
    left.set_ylabel("peak host RSS (GiB)")
    right.scatter(frame["items"], frame["gpu_peak_mib"] / 1024, s=10, color="k")
    right.set_ylabel("peak GPU memory used (GiB, nvidia-smi)")
    right.set_ylim(bottom=0)
    for ax in (left, right):
        ax.set_xscale("log")
        ax.set_xlabel("items per run (clean + perturbed)")
        ax.grid(alpha=0.3)
    left.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    args = parser.parse_args(argv)
    summary = args.summary_dir
    inputs = {name: summary / f"{name}.csv" for name in ("measurements", "reference_points")}
    frame, references = (pd.read_csv(path) for path in inputs.values())
    figures = {
        "fig_a4_samples.pdf": lambda path: plot_samples(frame, path),
        "fig_a4_regions.pdf": lambda path: plot_regions(frame, path),
        "fig_a4_controls.pdf": lambda path: plot_controls(frame, references, path),
        "fig_a4_perturbations.pdf": lambda path: plot_perturbations(frame, path),
        "fig_a4_memory.pdf": lambda path: plot_memory(frame, path),
    }
    for name, plot in figures.items():
        plot(summary / name)
        (summary / name).chmod(0o644)
    write_provenance(summary, inputs=inputs, filename="summarize.provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
