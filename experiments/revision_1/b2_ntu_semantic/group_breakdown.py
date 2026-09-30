#!/usr/bin/env python3
"""B2 post hoc (not pre-registered): agreement per body-part group, across classes.

For each group, the AUROC of its lift ``L[., g]`` for classes that rate the
group relevant vs classes that do not, the mean lift in each set, and how
often the group is the highest-lift group vs how often it is annotated
primary. The question is which groups carry the class-level agreement.
Reads the committed sheets (same blind guard) and
``summary/class_group_scores.csv``, and writes ``summary/group_breakdown.csv``.

Example:
    python experiments/revision_1/b2_ntu_semantic/group_breakdown.py --annotators A
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from experiments.revision_1.b2_ntu_semantic.annotations import GROUPS, require_blind_annotations
from experiments.revision_1.b2_ntu_semantic.evaluate_alignment import SUMMARY_DIR, Variant, auroc, consensus, score_matrix
from experiments.revision_1.common.provenance import write_provenance


def group_rows(rating: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """One row per group: cross-class AUROC, mean lift by relevance, top-group and primary counts."""

    top = matrix.idxmax(axis=1)
    rows = []
    for group in GROUPS:
        relevant = rating[group] >= 1
        values = matrix[group]
        rows.append({"group": group, "n_relevant_classes": int(relevant.sum()), "n_primary_classes": int((rating[group] == 2).sum()),
                     "auroc_across_classes": auroc(values[relevant].to_numpy(), values[~relevant].to_numpy()),
                     "mean_lift_relevant": float(values[relevant].mean()), "mean_lift_not_relevant": float(values[~relevant].mean()),
                     "n_classes_top_group": int((top == group).sum())})
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--annotators", nargs="+", default=["A", "B"])
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    args = parser.parse_args(argv)
    rating = consensus(require_blind_annotations(args.annotators)).sort_index()
    scores_path = args.summary_dir / "class_group_scores.csv"
    scores = pd.read_csv(scores_path)
    frames = [group_rows(rating, score_matrix(scores, Variant(run, run=run))).assign(run=run) for run in ("exact", "crop_free")]
    path = args.summary_dir / "group_breakdown.csv"
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    path.chmod(0o644)
    write_provenance(args.summary_dir, inputs={"scores": scores_path}, extra={"annotators": args.annotators, "post_hoc": True},
                     filename="group_breakdown.provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
