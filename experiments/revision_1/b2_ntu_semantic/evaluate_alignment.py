#!/usr/bin/env python3
"""B2 step 3: agreement between annotated relevant body parts and SSAT scores (implementation plan section 8.5).

Consensus rating = mean over annotators; relevant = consensus >= 1,
primary = consensus == 2 (pre-registered). Per class, the AUROC of the
score for relevant vs non-relevant groups (ties count 1/2) is computed for
classes that have both. Reported:
- annotator agreement: linear weighted kappa and raw agreement, per group
  and overall;
- primary: mean class AUROC with a class-bootstrap 95 % CI, and a
  permutation p-value (annotation rows reassigned to classes at random);
- secondary: top-1 hit rate (highest-scoring group annotated primary)
  against chance, and the mean per-class Spearman between consensus ratings
  and scores;
- robustness: crop-free run, clean-correct samples, raw and area-adjusted
  scores, single-person classes only, each annotator alone.

Refuses to run unless every annotator sheet is complete and committed
unchanged. Reads ``summary/class_group_scores.csv`` (``ssat_part_scores.py``).

Example:
    python experiments/revision_1/b2_ntu_semantic/evaluate_alignment.py --annotators A B
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.revision_1.b2_ntu_semantic.annotations import GROUPS, RATINGS, class_rows, require_blind_annotations
from experiments.revision_1.common.provenance import write_provenance

B2_DIR = Path(__file__).resolve().parent
SUMMARY_DIR = B2_DIR / "summary"
N_PERMUTATIONS = 10_000
N_BOOTSTRAP = 10_000
SEED = 20260930
N_EXAMPLES = 3
PATTERNS = np.array(list(product((False, True), repeat=len(GROUPS))))


@dataclass(frozen=True)
class Variant:
    """One analysis setting (the first entry of ``variants`` is primary)."""

    name: str
    run: str = "exact"
    population: str = "all"
    score: str = "lift"
    annotators: tuple[str, ...] | None = None  # None: all annotators
    single_person_only: bool = False


def variants(annotators: Sequence[str]) -> list[Variant]:
    rows = [Variant("primary"), Variant("crop_free", run="crop_free"), Variant("clean_correct", population="clean_correct"),
            Variant("raw_score", score="raw"), Variant("area_adjusted", score="area_adjusted_lift"),
            Variant("single_person_classes", single_person_only=True)]
    if len(annotators) > 1:
        rows += [Variant(f"annotator_{annotator}_only", annotators=(annotator,)) for annotator in annotators]
    return rows


def consensus(ratings: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean rating over annotators, indexed by label id, columns ``GROUPS``."""

    return sum(frame[list(GROUPS)].astype(float) for frame in ratings.values()) / len(ratings)


def weighted_kappa(first: Sequence[int], second: Sequence[int], categories: Sequence[int] = RATINGS) -> float:
    """Cohen's kappa with linear weights; NaN when expected disagreement is zero."""

    index = {category: position for position, category in enumerate(categories)}
    k = len(categories)
    observed = np.zeros((k, k))
    for a, b in zip(first, second, strict=True):
        observed[index[a], index[b]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    weights = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    denominator = (weights * expected).sum()
    return float("nan") if denominator == 0 else float(1 - (weights * observed).sum() / denominator)


def agreement_rows(ratings: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Weighted kappa and raw agreement per annotator pair, per group and overall."""

    rows = []
    for left, right in combinations(sorted(ratings), 2):
        a, b = ratings[left].sort_index(), ratings[right].sort_index()
        for group in (*GROUPS, "overall"):
            columns = list(GROUPS) if group == "overall" else [group]
            x, y = a[columns].to_numpy().ravel(), b[columns].to_numpy().ravel()
            rows.append({"annotator_left": left, "annotator_right": right, "group": group, "n_cells": len(x),
                         "weighted_kappa": weighted_kappa(x, y), "raw_agreement": float((x == y).mean()),
                         "relevant_agreement": float(((x >= 1) == (y >= 1)).mean())})
    return pd.DataFrame(rows)


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Probability that a positive scores above a negative (ties 1/2); NaN if either set is empty."""

    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    difference = np.subtract.outer(positive, negative)
    return float(((difference > 0) + 0.5 * (difference == 0)).mean())


def pattern_tables(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every class and every relevance pattern: AUROC, and whether the top group is in the pattern.

    Args:
        scores: ``(n_classes, n_groups)`` score matrix.

    Returns:
        ``(auroc[c, p], hit[c, p])``; the top group is the highest score, ties
        broken by ``GROUPS`` order.
    """

    top = scores.argmax(axis=1)
    aurocs = np.array([[auroc(row[pattern], row[~pattern]) for pattern in PATTERNS] for row in scores])
    hits = PATTERNS[:, top].T.astype(float)
    return aurocs, hits


def pattern_index(mask: np.ndarray) -> np.ndarray:
    """Row index into ``PATTERNS`` of each boolean row of ``mask``."""

    return (mask.astype(int) * (1 << np.arange(len(GROUPS))[::-1])).sum(axis=1)


def statistics(relevant: np.ndarray, primary: np.ndarray, scores: np.ndarray, *, seed: int = SEED,
               n_permutations: int = N_PERMUTATIONS, n_bootstrap: int = N_BOOTSTRAP) -> dict[str, Any]:
    """Mean AUROC with CI and permutation p; top-1 hit rate with chance and permutation p."""

    rng = np.random.default_rng(seed)
    aurocs, hits = pattern_tables(scores)
    classes = np.arange(len(scores))
    rel_index, pri_index = pattern_index(relevant), pattern_index(primary)
    class_auroc = aurocs[classes, rel_index]
    eligible = ~np.isnan(class_auroc)
    has_primary = primary.any(axis=1)
    hit_rate = float(hits[classes, pri_index][has_primary].mean()) if has_primary.any() else float("nan")
    chance = float(primary[has_primary].mean()) if has_primary.any() else float("nan")
    mean_auroc = float(class_auroc[eligible].mean()) if eligible.any() else float("nan")
    boot = rng.choice(class_auroc[eligible], size=(n_bootstrap, int(eligible.sum())), replace=True).mean(axis=1) if eligible.any() else np.array([np.nan])
    null_auroc, null_hit = np.empty(n_permutations), np.empty(n_permutations)
    for index in range(n_permutations):
        order = rng.permutation(len(scores))
        permuted = aurocs[classes, rel_index[order]]
        null_auroc[index] = np.nanmean(permuted) if (~np.isnan(permuted)).any() else np.nan
        hit_mask = has_primary[order]
        null_hit[index] = hits[classes, pri_index[order]][hit_mask].mean() if hit_mask.any() else np.nan
    return {
        "mean_auroc": mean_auroc, "auroc_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "auroc_permutation_p": float((1 + np.sum(null_auroc >= mean_auroc)) / (1 + n_permutations)),
        "n_classes_eligible": int(eligible.sum()), "n_excluded_all_relevant": int(relevant.all(axis=1).sum()),
        "n_excluded_none_relevant": int((~relevant.any(axis=1)).sum()),
        "top1_hit_rate": hit_rate, "top1_chance": chance, "n_classes_with_primary": int(has_primary.sum()),
        "top1_permutation_p": float((1 + np.sum(null_hit >= hit_rate)) / (1 + n_permutations)),
        "class_auroc": class_auroc, "null_mean_auroc": null_auroc,
    }


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Spearman correlation with average ranks; NaN if either vector is constant."""

    a, b = pd.Series(first).rank(), pd.Series(second).rank()
    return float("nan") if a.nunique() < 2 or b.nunique() < 2 else float(a.corr(b))


def score_matrix(scores: pd.DataFrame, variant: Variant) -> pd.DataFrame:
    """``(label_id x GROUPS)`` score matrix of ``variant``."""

    rows = scores[(scores["run"] == variant.run) & (scores["population"] == variant.population) & (scores["score"] == variant.score)
                  & scores["group"].isin(GROUPS)]
    return rows.pivot(index="label_id", columns="group", values="value")[list(GROUPS)].sort_index()


def evaluate_variant(variant: Variant, ratings: Mapping[str, pd.DataFrame], scores: pd.DataFrame, **kwargs: Any) -> tuple[dict[str, Any], pd.DataFrame]:
    """Statistics and the per-class table for one variant."""

    chosen = {name: frame for name, frame in ratings.items() if variant.annotators is None or name in variant.annotators}
    rating = consensus(chosen)
    matrix = score_matrix(scores, variant)
    classes = pd.DataFrame(class_rows()).set_index("label_id")
    labels = [label for label in matrix.index if not (variant.single_person_only and classes.loc[label, "two_person"])]
    rating, matrix = rating.loc[labels], matrix.loc[labels]
    relevant, primary = (rating >= 1).to_numpy(), (rating == 2).to_numpy()
    values = matrix.to_numpy(dtype=float)
    stats = statistics(relevant, primary, values, **kwargs)
    top = [GROUPS[index] for index in values.argmax(axis=1)]
    per_class = pd.DataFrame({
        "label_id": labels, "action_name": classes.loc[labels, "action_name"].to_numpy(),
        "two_person": classes.loc[labels, "two_person"].to_numpy(),
        "relevant_groups": [";".join(g for g, r in zip(GROUPS, row) if r) for row in relevant],
        "primary_groups": [";".join(g for g, p in zip(GROUPS, row) if p) for row in primary],
        "top_group": top, "auroc": stats["class_auroc"],
        "top1_hit": [float(row[GROUPS.index(t)]) if row.any() else np.nan for row, t in zip(primary, top)],
        "spearman": [spearman(r, s) for r, s in zip(rating.to_numpy(), values)],
    })
    stats["mean_spearman"] = float(per_class["spearman"].mean())
    stats["n_classes_spearman"] = int(per_class["spearman"].notna().sum())
    return stats, per_class


def representative_classes(per_class: pd.DataFrame, n: int = N_EXAMPLES) -> dict[str, list[int]]:
    """Top and bottom ``n`` classes by AUROC among eligible classes; ties by label id."""

    eligible = per_class.dropna(subset=["auroc"])
    top = eligible.sort_values(["auroc", "label_id"], ascending=[False, True]).head(n)
    bottom = eligible.sort_values(["auroc", "label_id"], ascending=[True, True]).head(n)
    return {"top": top["label_id"].astype(int).tolist(), "bottom": bottom["label_id"].astype(int).tolist()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--annotators", nargs="+", default=["A", "B"])
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    args = parser.parse_args(argv)
    ratings = require_blind_annotations(args.annotators)
    summary = args.summary_dir
    scores_path = summary / "class_group_scores.csv"
    scores = pd.read_csv(scores_path)
    outputs: dict[str, Path] = {}
    if len(ratings) > 1:
        agreement_rows(ratings).to_csv(outputs.setdefault("agreement", summary / "annotator_agreement.csv"), index=False)
    rating = consensus(ratings)
    rating.assign(relevant=(rating >= 1).sum(axis=1), primary=(rating == 2).sum(axis=1)).to_csv(
        outputs.setdefault("consensus", summary / "consensus.csv"))
    robustness, result = [], {}
    for variant in variants(args.annotators):
        stats, per_class = evaluate_variant(variant, ratings, scores)
        if variant.name == "primary":
            per_class.to_csv(outputs.setdefault("by_class", summary / "alignment_by_class.csv"), index=False)
            pd.DataFrame({"null_mean_auroc": stats["null_mean_auroc"]}).to_csv(
                outputs.setdefault("null", summary / "permutation_null.csv"), index=False)
            result = {key: value for key, value in stats.items() if key not in {"class_auroc", "null_mean_auroc"}}
            result["representative_classes"] = representative_classes(per_class)
        robustness.append({"variant": variant.name, **{key: value for key, value in stats.items()
                                                        if key not in {"class_auroc", "null_mean_auroc", "auroc_ci95"}},
                           "auroc_ci95_low": stats["auroc_ci95"][0], "auroc_ci95_high": stats["auroc_ci95"][1]})
    pd.DataFrame(robustness).to_csv(outputs.setdefault("robustness", summary / "robustness.csv"), index=False)
    raw = scores[(scores["run"] == "exact") & (scores["population"] == "all") & (scores["score"] == "raw")]
    result.update(annotators=list(args.annotators), n_permutations=N_PERMUTATIONS, n_bootstrap=N_BOOTSTRAP, seed=SEED,
                  raw_group_mean_across_classes=raw.groupby("group")["value"].mean().to_dict())
    (summary / "alignment_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in summary.iterdir():
        path.chmod(0o644)
    write_provenance(summary, inputs={"scores": scores_path, **{f"annotator_{name}": B2_DIR / "annotation" / f"annotator_{name}.csv"
                                                                for name in args.annotators}},
                     extra={"n_permutations": N_PERMUTATIONS, "n_bootstrap": N_BOOTSTRAP, "seed": SEED},
                     filename="evaluate_alignment.provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
