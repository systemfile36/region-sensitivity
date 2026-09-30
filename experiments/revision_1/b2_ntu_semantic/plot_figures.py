#!/usr/bin/env python3
"""B2 figures from the ``evaluate_alignment.py`` outputs in ``summary/``.

- ``fig_b2_alignment.pdf``: class AUROC distribution (primary variant) and
  the permutation null distribution of the mean AUROC;
- ``fig_b2_examples.pdf``: class x group lift heatmap for the pre-registered
  top-3 and bottom-3 classes by AUROC, each cell labelled with the consensus
  rating.

Example:
    python experiments/revision_1/b2_ntu_semantic/plot_figures.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.revision_1.b2_ntu_semantic.annotations import GROUPS  # noqa: E402
from experiments.revision_1.b2_ntu_semantic.evaluate_alignment import SUMMARY_DIR, Variant, score_matrix  # noqa: E402
from experiments.revision_1.common.provenance import write_provenance  # noqa: E402


def plot_alignment(by_class: pd.DataFrame, null: pd.DataFrame, summary: dict, path: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4))
    left.hist(by_class["auroc"].dropna(), bins=[i / 10 for i in range(11)], color="#1f77b4", edgecolor="white")
    left.axvline(0.5, color="grey", ls=":", label="chance (0.5)")
    left.axvline(summary["mean_auroc"], color="k", label=f"mean {summary['mean_auroc']:.2f}")
    left.set_xlabel("class AUROC (relevant vs non-relevant groups)")
    left.set_ylabel("classes")
    left.legend(fontsize=7, frameon=False)
    right.hist(null["null_mean_auroc"].dropna(), bins=40, color="#bbbbbb")
    right.axvline(summary["mean_auroc"], color="k", label=f"observed (p = {summary['auroc_permutation_p']:.4f})")
    right.set_xlabel("mean AUROC under permuted annotations")
    right.set_ylabel("permutations")
    right.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_examples(scores: pd.DataFrame, consensus: pd.DataFrame, by_class: pd.DataFrame, summary: dict, path: Path) -> None:
    examples = summary["representative_classes"]
    labels = examples["top"] + examples["bottom"]
    matrix = score_matrix(scores, Variant("primary")).loc[labels]
    names = by_class.set_index("label_id").loc[labels]
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(labels) + 1.2))
    limit = float(abs(matrix.to_numpy()).max())
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row, label in enumerate(labels):
        for column, group in enumerate(GROUPS):
            rating = consensus.loc[label, group]
            ax.text(column, row, f"{rating:g}", ha="center", va="center", fontsize=8,
                    fontweight="bold" if rating >= 1 else "normal")
    ax.set_xticks(range(len(GROUPS)), GROUPS)
    ax.set_yticks(range(len(labels)), [f"A{label + 1:03d} {names.loc[label, 'action_name']} (AUROC {names.loc[label, 'auroc']:.2f})"
                                       for label in labels], fontsize=7)
    ax.axhline(len(examples["top"]) - 0.5, color="k", lw=1)
    fig.colorbar(image, ax=ax, label="SSAT lift (z across classes)")
    ax.set_title("cell text: consensus rating (bold = relevant)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    args = parser.parse_args(argv)
    summary_dir = args.summary_dir
    inputs = {name: summary_dir / file for name, file in (("by_class", "alignment_by_class.csv"), ("null", "permutation_null.csv"),
                                                          ("scores", "class_group_scores.csv"), ("consensus", "consensus.csv"),
                                                          ("summary", "alignment_summary.json"))}
    by_class, null, scores = (pd.read_csv(inputs[name]) for name in ("by_class", "null", "scores"))
    consensus = pd.read_csv(inputs["consensus"], index_col="label_id")
    summary = json.loads(inputs["summary"].read_text(encoding="utf-8"))
    plot_alignment(by_class, null, summary, summary_dir / "fig_b2_alignment.pdf")
    plot_examples(scores, consensus, by_class, summary, summary_dir / "fig_b2_examples.pdf")
    for name in ("fig_b2_alignment.pdf", "fig_b2_examples.pdf"):
        (summary_dir / name).chmod(0o644)
    write_provenance(summary_dir, inputs=inputs, filename="plot_figures.provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
