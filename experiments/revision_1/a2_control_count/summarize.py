"""A2 step 3: figures from the ablation CSVs in ``summary/``.

Reads only ``summary/*.csv`` written by ``run_ablation.py`` and writes
``fig_a2_convergence.pdf`` (control-mean error and grade agreement vs K),
``fig_a2_high_share.pdf`` (HIGH share vs K under ddof 0 and 1), and
``fig_a2_flip.pdf`` (replicate flip probability vs K).

Example:
    python experiments/revision_1/a2_control_count/summarize.py
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

from experiments.revision_1.common.provenance import write_provenance  # noqa: E402

SUMMARY_DIR = Path(__file__).resolve().parent / "summary"
RUN_LABELS = {
    "a2_imagenet_mnv2_050_cf_k20": "ImageNet MNv2-0.5 crop-free",
    "a2_imagenet_mnv2_050_exact_k20": "ImageNet MNv2-0.5 exact",
    "a2_synthetic_shortcut_k20": "Synthetic M_shortcut",
}
RUN_COLORS = {
    "a2_imagenet_mnv2_050_cf_k20": "#1f77b4",
    "a2_imagenet_mnv2_050_exact_k20": "#ff7f0e",
    "a2_synthetic_shortcut_k20": "#2ca02c",
}


def _runs(frame: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(frame["run"]))
    return [run for run in RUN_LABELS if run in present] + [run for run in present if run not in RUN_LABELS]


def _log_k_axis(ax: plt.Axes, ticks: list[int]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())


def _band(ax: plt.Axes, rows: pd.DataFrame, stat: str, color: str, label: str, *, prefix: pd.DataFrame | None = None) -> None:
    rows = rows.sort_values("k")
    ax.plot(rows["k"], rows[f"{stat}_mean"], "-o", color=color, label=label, ms=3)
    ax.fill_between(rows["k"], rows[f"{stat}_p05"], rows[f"{stat}_p95"], color=color, alpha=0.2, lw=0)
    if prefix is not None:
        prefix = prefix.sort_values("k")
        ax.plot(prefix["k"], prefix[f"{stat}_mean"], "x", color=color, ms=5)


def plot_convergence(convergence: pd.DataFrame, grades: pd.DataFrame, path: Path) -> None:
    """Control-mean error (left) and grade agreement with K=20 (right) against K."""

    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4))
    for run in _runs(convergence):
        rows = convergence[(convergence["run"] == run) & (convergence["scheme"] == "global_random")].sort_values("k")
        color = RUN_COLORS.get(run, "k")
        left.plot(rows["k"], rows["median"], "-o", color=color, label=RUN_LABELS.get(run, run), ms=3)
        left.fill_between(rows["k"], rows["replicate_median_p05"], rows["replicate_median_p95"], color=color, alpha=0.2, lw=0)
        prefix = convergence[(convergence["run"] == run) & (convergence["scheme"] == "prefix")].sort_values("k")
        left.plot(prefix["k"], prefix["median"], "x", color=color, ms=5)
    _log_k_axis(left, [1, 2, 3, 5, 10])
    left.set_xlabel("K (controls per target)")
    left.set_ylabel("median |mu_K - mu_20| / sigma_20")
    left.set_title("Control baseline convergence")
    selected = grades[(grades["cell_type"] == "all") & (grades["ddof"] == 0)]
    for run in _runs(selected):
        rows = selected[selected["run"] == run]
        _band(right, rows[rows["scheme"] == "global_random"], "agreement", RUN_COLORS.get(run, "k"), RUN_LABELS.get(run, run),
              prefix=rows[rows["scheme"] == "prefix"])
    _log_k_axis(right, [2, 3, 5, 10, 20])
    right.set_xlabel("K (controls per target)")
    right.set_ylabel("grade agreement with K=20")
    right.set_title("Reliability grade (ddof 0)")
    left.legend(fontsize=7, frameon=False)
    for ax in (left, right):
        ax.grid(alpha=0.3)
    fig.text(0.5, -0.02, "line/band: random subsets (mean, 5-95 %), x: prefix {0..K-1}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_high_share(grades: pd.DataFrame, path: Path) -> None:
    """HIGH share against K for ddof 0 (solid) and 1 (dashed), random subsets."""

    selected = grades[(grades["cell_type"] == "all") & (grades["scheme"] == "global_random")]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for run in _runs(selected):
        for ddof, style in ((0, "-o"), (1, "--s")):
            rows = selected[(selected["run"] == run) & (selected["ddof"] == ddof)].sort_values("k")
            ax.plot(rows["k"], rows["high_share_mean"], style, color=RUN_COLORS.get(run, "k"), ms=3,
                    label=f"{RUN_LABELS.get(run, run)}, ddof {ddof}")
            ax.fill_between(rows["k"], rows["high_share_p05"], rows["high_share_p95"], color=RUN_COLORS.get(run, "k"), alpha=0.12, lw=0)
    _log_k_axis(ax, [2, 3, 5, 10, 20])
    ax.set_xlabel("K (controls per target)")
    ax.set_ylabel("HIGH share")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_flip(variability: pd.DataFrame, path: Path) -> None:
    """Mean pairwise flip probability and share of anchors that ever flip, against K (ddof 0)."""

    selected = variability[(variability["cell_type"] == "all") & (variability["ddof"] == 0)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.2))
    for run in _runs(selected):
        rows = selected[selected["run"] == run].sort_values("k")
        left.plot(rows["k"], rows["flip_pairwise_mean"], "-o", color=RUN_COLORS.get(run, "k"), label=RUN_LABELS.get(run, run), ms=3)
        right.plot(rows["k"], rows["flip_pairwise_share_gt0"], "-o", color=RUN_COLORS.get(run, "k"), ms=3)
    for ax, label in ((left, "mean P(two random subsets disagree)"), (right, "share of anchors whose grade can flip")):
        _log_k_axis(ax, [2, 3, 5, 10])
        ax.set_xlabel("K (controls per target)")
        ax.set_ylabel(label)
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
    inputs = {name: summary / f"{name}.csv" for name in ("control_mean_convergence", "grade_vs_k", "replicate_variability")}
    convergence, grades, variability = (pd.read_csv(path) for path in inputs.values())
    plot_convergence(convergence, grades, summary / "fig_a2_convergence.pdf")
    plot_high_share(grades, summary / "fig_a2_high_share.pdf")
    plot_flip(variability, summary / "fig_a2_flip.pdf")
    for name in ("fig_a2_convergence.pdf", "fig_a2_high_share.pdf", "fig_a2_flip.pdf"):
        (summary / name).chmod(0o644)
    write_provenance(summary, inputs=inputs, filename="summarize.provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
