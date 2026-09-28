"""A3 figures from the ``compare_models.py`` CSVs in ``summary/`` (population ``all``).

Writes ``fig_a3_profiles.pdf`` (4x4 heatmaps coloured by within-model rank,
models x protocols), ``fig_a3_grades.pdf`` (stacked grade shares), and
``fig_a3_similarity.pdf`` (model x model profile Spearman and median
per-sample Spearman per protocol; pairs outside the comparison rule are
blank).

Example:
    python experiments/revision_1/a3_multi_arch/summarize.py
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.revision_1.common.agreement import GRADE_ORDER  # noqa: E402
from experiments.revision_1.common.provenance import write_provenance  # noqa: E402

SUMMARY_DIR = Path(__file__).resolve().parent / "summary"
MODELS = ("mobilenetv2_050", "mobilenetv2_100", "convnext_tiny", "deit_small")
MODEL_LABELS = {"mobilenetv2_050": "MNv2-0.5", "mobilenetv2_100": "MNv2-1.0", "convnext_tiny": "ConvNeXt-T", "deit_small": "DeiT-S"}
PROTOCOLS = ("crop_free", "exact")
PROTOCOL_LABELS = {"crop_free": "crop-free", "exact": "exact"}
GRADE_COLORS = {"high": "#2ca02c", "moderate": "#98df8a", "low": "#ffbb78", "unreliable": "#d62728"}


def plot_profiles(profile: pd.DataFrame, path: Path) -> None:
    """4x4 grids of within-model rank (1 = largest mean margin_drop), one panel per model and protocol."""

    rows = profile[profile["population"] == "all"]
    fig, axes = plt.subplots(len(PROTOCOLS), len(MODELS), figsize=(2.2 * len(MODELS), 2.3 * len(PROTOCOLS)))
    for i, protocol in enumerate(PROTOCOLS):
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            cells = rows[(rows["protocol"] == protocol) & (rows["model"] == model)]
            grid = np.full((4, 4), np.nan)
            for _, cell in cells.iterrows():
                r, c = (int(part[1:]) for part in cell["cell"].split("/"))
                grid[r, c] = cell["rank_in_model"]
            image = ax.imshow(grid, cmap="viridis_r", vmin=1, vmax=16)
            for (r, c), value in np.ndenumerate(grid):
                if np.isfinite(value):
                    ax.text(c, r, f"{int(value)}", ha="center", va="center", fontsize=7, color="w" if value < 9 else "k")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(MODEL_LABELS[model], fontsize=9)
            if j == 0:
                ax.set_ylabel(PROTOCOL_LABELS[protocol], fontsize=9)
    fig.colorbar(image, ax=axes, shrink=0.6, label="rank of mean margin_drop within model")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_grades(grades: pd.DataFrame, path: Path) -> None:
    """Stacked grade shares per model, one panel per protocol."""

    rows = grades[(grades["population"] == "all") & (grades["cell_type"] == "all")]
    fig, axes = plt.subplots(1, len(PROTOCOLS), figsize=(8, 3), sharey=True)
    for ax, protocol in zip(axes, PROTOCOLS):
        bottom = np.zeros(len(MODELS))
        selected = rows[rows["protocol"] == protocol].set_index("model").reindex(list(MODELS))
        for grade in reversed(GRADE_ORDER):
            values = selected[grade].to_numpy()
            ax.bar([MODEL_LABELS[m] for m in MODELS], values, bottom=bottom, color=GRADE_COLORS.get(grade), label=grade.upper())
            bottom += values
        ax.set_title(PROTOCOL_LABELS[protocol])
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("share of anchors (margin_drop)")
    axes[-1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_similarity(cross: pd.DataFrame, path: Path) -> None:
    """Model x model similarity matrices: profile Spearman and median per-sample Spearman."""

    rows = cross[cross["population"] == "all"]
    stats = (("profile_spearman", "16-cell mean profile Spearman"), ("sample_spearman_p50", "median per-sample Spearman"))
    fig, axes = plt.subplots(len(PROTOCOLS), len(stats), figsize=(8, 7))
    labels = [MODEL_LABELS[m] for m in MODELS]
    for i, protocol in enumerate(PROTOCOLS):
        selected = rows[rows["protocol"] == protocol]
        for j, (stat, title) in enumerate(stats):
            ax = axes[i, j]
            matrix = np.full((len(MODELS), len(MODELS)), np.nan)
            np.fill_diagonal(matrix, 1.0)
            for _, row in selected.iterrows():
                a, b = MODELS.index(row["model_a"]), MODELS.index(row["model_b"])
                matrix[a, b] = matrix[b, a] = row[stat]
            ax.imshow(np.ma.masked_invalid(matrix), cmap="magma", vmin=0, vmax=1)
            for (a, b), value in np.ndenumerate(matrix):
                ax.text(b, a, "–" if np.isnan(value) else f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="k" if np.isnan(value) or value > 0.6 else "w")
            ax.set_xticks(range(len(MODELS)), labels, fontsize=7, rotation=30)
            ax.set_yticks(range(len(MODELS)), labels, fontsize=7)
            ax.set_title(f"{PROTOCOL_LABELS[protocol]}: {title}", fontsize=8)
    fig.text(0.5, 0.01, "–: not compared (exact geometry differs from MNv2-0.5; see protocol comparison rule)", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    args = parser.parse_args(argv)
    summary = args.summary_dir
    inputs = {name: summary / f"{name}.csv" for name in ("region_profile", "grade_distribution", "cross_model")}
    profile, grades, cross = (pd.read_csv(path) for path in inputs.values())
    figures = {"fig_a3_profiles.pdf": lambda path: plot_profiles(profile, path),
               "fig_a3_grades.pdf": lambda path: plot_grades(grades, path),
               "fig_a3_similarity.pdf": lambda path: plot_similarity(cross, path)}
    for name, plot in figures.items():
        plot(summary / name)
        (summary / name).chmod(0o644)
    write_provenance(summary, inputs=inputs, filename="summarize.provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
