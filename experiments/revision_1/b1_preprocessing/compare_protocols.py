"""B1-2/B1-3: ranking and grade changes between P_exact and P_cf (and the model yardstick).

For each pre-registered pair of runs, compares the per-sample 16-cell
``margin_drop`` profiles (``spatial_profile.parquet``) and the stored
reliability grades. Protocol pairs (exact vs crop-free, same model) are the
comparison of interest; yardstick pairs (mnv2_050 vs mnv2_100, same
protocol) give the size of an ordinary model change on the same metrics.
Requires ``area_tables.py`` output (``cell_area.csv``) in the summary dir.

Example:
    python experiments/revision_1/b1_preprocessing/compare_protocols.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.revision_1.a1_threshold.build_features import cell_columns
from experiments.revision_1.common.agreement import GRADE_ORDER, grade_agreement, spearman, transition_matrix
from experiments.revision_1.common.loading import load_reliability, load_sample_meta, load_spatial_profile
from experiments.revision_1.common.provenance import write_provenance
from experiments.revision_1.common.runs import BASELINE_RUNS, RunRef

B1_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = B1_DIR / "protocol.json"
SUMMARY_DIR = B1_DIR / "summary"
KEYS = ["sample_id", "region_key"]
SPEARMAN_QUANTILES = (10, 25, 50, 75, 90)
HIST_EDGES = np.round(np.linspace(-1.0, 1.0, 41), 3)


def profile_matrix(run: RunRef) -> pd.DataFrame:
    """Return samples x cells ``margin_drop`` (columns sorted by region_key)."""

    profile = load_spatial_profile(run)
    matrix = profile.pivot(index="sample_id", columns="region_key", values="degradation")
    return matrix.reindex(columns=sorted(matrix.columns)).sort_index()


def descending_ranks(matrix: pd.DataFrame) -> np.ndarray:
    """Rank cells within each sample, 1 = largest value; ties broken by region_key ascending."""

    order = np.argsort(-matrix.to_numpy(), axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, order.shape[1] + 1)
    return ranks


def row_spearman(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Per-sample Spearman over cells (average ranks); NaN when either profile is constant."""

    rank_a = a.rank(axis=1, method="average").to_numpy()
    rank_b = b.rank(axis=1, method="average").to_numpy()
    centered_a = rank_a - rank_a.mean(axis=1, keepdims=True)
    centered_b = rank_b - rank_b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((centered_a**2).sum(axis=1) * (centered_b**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (centered_a * centered_b).sum(axis=1) / denominator
    return pd.Series(np.where(denominator > 0, rho, np.nan), index=a.index)


def top_share(ranks: np.ndarray, cells: Sequence[str]) -> pd.Series:
    """Share of samples whose rank-1 cell is each cell."""

    top = np.asarray(cells)[np.argmin(ranks, axis=1)]
    return pd.Series(top).value_counts(normalize=True).reindex(cells, fill_value=0.0)


def ranking_summary(a: pd.DataFrame, b: pd.DataFrame) -> tuple[dict[str, object], pd.Series, np.ndarray, np.ndarray]:
    """Dataset- and sample-level ranking agreement between two aligned profile matrices."""

    rho = row_spearman(a, b)
    ranks_a, ranks_b = descending_ranks(a), descending_ranks(b)
    top3_a = ranks_a <= 3
    top3_b = ranks_b <= 3
    valid = rho.dropna()
    summary = {
        "n_samples": len(a),
        "n_constant_profile_excluded": int(rho.isna().sum()),
        "profile_spearman": spearman(a.mean().to_numpy(), b.mean().to_numpy()),
        **{f"sample_spearman_p{q}": float(np.percentile(valid, q)) for q in SPEARMAN_QUANTILES},
        "sample_spearman_mean": float(valid.mean()),
        "share_sample_spearman_lt_0_5": float((valid < 0.5).mean()),
        "top1_agreement": float((ranks_a.argmin(axis=1) == ranks_b.argmin(axis=1)).mean()),
        "top3_overlap_mean": float((top3_a & top3_b).sum(axis=1).mean() / 3),
    }
    return summary, rho, ranks_a, ranks_b


def grade_summary(rel_a: pd.DataFrame, rel_b: pd.DataFrame) -> dict[str, object]:
    """Agreement and change decomposition between two aligned reliability frames."""

    summary = grade_agreement(rel_a["reliability_grade"], rel_b["reliability_grade"])
    changed = rel_a["reliability_grade"] != rel_b["reliability_grade"]
    sign_changed = rel_a["sign_consistent"] != rel_b["sign_consistent"]
    exceeds_changed = rel_a["exceeds_control"] != rel_b["exceeds_control"]
    summary.update(
        {
            "changed_with_sign_change": float((changed & sign_changed).mean()),
            "changed_with_only_exceeds_change": float((changed & ~sign_changed & exceeds_changed).mean()),
            "changed_other": float((changed & ~sign_changed & ~exceeds_changed).mean()),
            **{f"{grade}_share_a": float((rel_a["reliability_grade"] == grade).mean()) for grade in GRADE_ORDER},
            **{f"{grade}_share_b": float((rel_b["reliability_grade"] == grade).mean()) for grade in GRADE_ORDER},
            "area_matched_true_a": float((rel_a["area_matched"] == "true").mean()),
            "area_matched_true_b": float((rel_b["area_matched"] == "true").mean()),
        }
    )
    return summary


def compare_pair(
    pair_type: str, run_a: RunRef, run_b: RunRef, cell_area: pd.DataFrame
) -> dict[str, list[dict[str, object]]]:
    """Compute every B1-2/B1-3 table for one pair of runs."""

    profile_a, profile_b = profile_matrix(run_a), profile_matrix(run_b)
    if not profile_a.index.equals(profile_b.index) or not profile_a.columns.equals(profile_b.columns):
        raise ValueError(f"{run_a.name} / {run_b.name}: profiles cover different samples or cells")
    if profile_a.isna().any().any() or profile_b.isna().any().any():
        raise ValueError(f"{run_a.name} / {run_b.name}: missing margin_drop values in spatial_profile")
    rel_a = load_reliability(run_a).set_index(KEYS).sort_index()
    rel_b = load_reliability(run_b).set_index(KEYS).sort_index()
    if not rel_a.index.equals(rel_b.index):
        raise ValueError(f"{run_a.name} / {run_b.name}: reliability rows differ")
    correct = (
        load_sample_meta(run_a).set_index("sample_id")["clean_correct"]
        & load_sample_meta(run_b).set_index("sample_id")["clean_correct"]
    ).reindex(profile_a.index)

    cells = list(profile_a.columns)
    cell_info = cell_columns(pd.Series(cells)).set_index(pd.Index(cells))
    base = {"pair_type": pair_type, "run_a": run_a.name, "run_b": run_b.name}
    out: dict[str, list[dict[str, object]]] = {
        key: [] for key in ("ranking", "cells", "top_share", "transitions", "grade", "by_cell_type", "hist", "rho")
    }
    for population, mask in (("all", pd.Series(True, index=profile_a.index)), ("both_correct", correct.astype(bool))):
        samples = profile_a.index[mask.to_numpy()]
        a, b = profile_a.loc[samples], profile_b.loc[samples]
        summary, rho, ranks_a, ranks_b = ranking_summary(a, b)
        anchors = rel_a.index.get_level_values("sample_id").isin(samples)
        ga, gb = rel_a[anchors], rel_b[anchors]
        grades = grade_summary(ga, gb)
        changed = (ga["reliability_grade"] != gb["reliability_grade"]).groupby(level="region_key").mean()
        row = {**base, "population": population}

        share_a, share_b = top_share(ranks_a, cells), top_share(ranks_b, cells)
        cell_rows = pd.DataFrame(
            {
                "cell": cell_info["region_cell"],
                "cell_type": cell_info["cell_type"],
                "mean_margin_drop_a": a.mean(),
                "mean_margin_drop_b": b.mean(),
                "mean_rank_a": ranks_a.mean(axis=0),
                "mean_rank_b": ranks_b.mean(axis=0),
                "top_share_a": share_a,
                "top_share_b": share_b,
                "grade_change_rate": changed.reindex(cells),
                "high_share_a": (ga["reliability_grade"] == "high").groupby(level="region_key").mean().reindex(cells),
                "high_share_b": (gb["reliability_grade"] == "high").groupby(level="region_key").mean().reindex(cells),
            }
        )
        cell_rows["mean_rank_change"] = cell_rows["mean_rank_b"] - cell_rows["mean_rank_a"]
        area = cell_area.set_index(["model", "protocol", "cell"])["fraction_mean"]
        cell_rows["area_fraction_a"] = [area[(run_a.model, run_a.protocol, cell)] for cell in cell_rows["cell"]]
        cell_rows["area_fraction_b"] = [area[(run_b.model, run_b.protocol, cell)] for cell in cell_rows["cell"]]
        cell_rows["area_ratio_b_over_a"] = cell_rows["area_fraction_b"] / cell_rows["area_fraction_a"]
        if pair_type == "protocol":
            summary["cell_spearman_area_ratio_vs_rank_change"] = spearman(
                cell_rows["area_ratio_b_over_a"], cell_rows["mean_rank_change"]
            )
            summary["cell_spearman_area_ratio_vs_grade_change"] = spearman(
                cell_rows["area_ratio_b_over_a"], cell_rows["grade_change_rate"]
            )
        out["ranking"].append({**row, **summary})
        out["grade"].append({**row, "cell_type": "all", **grades})
        out["cells"] += [{**row, **record} for record in cell_rows.to_dict("records")]
        for _, record in cell_rows.iterrows():
            out["top_share"].append({**row, "cell": record["cell"], "top_share_a": record["top_share_a"], "top_share_b": record["top_share_b"]})

        matrix = transition_matrix(ga["reliability_grade"], gb["reliability_grade"])
        for from_grade in GRADE_ORDER:
            for to_grade in GRADE_ORDER:
                out["transitions"].append({**row, "from_grade": from_grade, "to_grade": to_grade, "n": int(matrix.loc[from_grade, to_grade])})

        anchor_type = cell_info["cell_type"].reindex(ga.index.get_level_values("region_key")).to_numpy()
        for cell_type in ("corner", "edge", "center"):
            in_type = anchor_type == cell_type
            type_grades = grade_summary(ga[in_type], gb[in_type])
            cells_of_type = cell_rows[cell_rows["cell_type"] == cell_type]
            out["grade"].append({**row, "cell_type": cell_type, **type_grades})
            out["by_cell_type"].append(
                {
                    **row,
                    "cell_type": cell_type,
                    "agreement": type_grades["agreement"],
                    "kappa_linear": type_grades["kappa_linear"],
                    "delta_ge2": type_grades["delta_ge2"],
                    "high_share_a": type_grades["high_share_a"],
                    "high_share_b": type_grades["high_share_b"],
                    "top_share_a": float(cells_of_type["top_share_a"].sum()),
                    "top_share_b": float(cells_of_type["top_share_b"].sum()),
                    "mean_rank_a": float(cells_of_type["mean_rank_a"].mean()),
                    "mean_rank_b": float(cells_of_type["mean_rank_b"].mean()),
                    "mean_margin_drop_a": float(cells_of_type["mean_margin_drop_a"].mean()),
                    "mean_margin_drop_b": float(cells_of_type["mean_margin_drop_b"].mean()),
                    "area_fraction_a": float(cells_of_type["area_fraction_a"].mean()),
                    "area_fraction_b": float(cells_of_type["area_fraction_b"].mean()),
                }
            )
        counts, edges = np.histogram(rho.dropna(), bins=HIST_EDGES)
        for left, right, count in zip(edges[:-1], edges[1:], counts):
            out["hist"].append({**row, "bin_left": float(left), "bin_right": float(right), "count": int(count)})
        if population == "all":
            out["rho"] = [{"sample_id": sample_id, "rho": value} for sample_id, value in rho.items()]
    return out


def representative_cases(rho: pd.Series, pair: tuple[str, str]) -> list[dict[str, object]]:
    """Pick the samples closest to the p10/p50/p90 per-sample Spearman (ties: sample_id ascending)."""

    valid = rho.dropna().sort_index()
    rows = []
    for q in (10, 50, 90):
        target = float(np.percentile(valid, q))
        distance = (valid - target).abs()
        chosen = distance[distance == distance.min()].index.min()
        rows.append({"run_a": pair[0], "run_b": pair[1], "quantile": q, "target_rho": target, "sample_id": chosen, "rho": float(valid[chosen])})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, default=SUMMARY_DIR)
    args = parser.parse_args(argv)
    protocol = json.loads(PROTOCOL_PATH.read_text())
    cell_area = pd.read_csv(args.output_dir / "cell_area.csv")

    tables: dict[str, list[dict[str, object]]] = {}
    cases: list[dict[str, object]] = []
    for pair_type, key in (("protocol", "protocol_comparison"), ("yardstick", "yardstick_model_comparison")):
        for name_a, name_b in protocol["pairs"][key]:
            out = compare_pair(pair_type, BASELINE_RUNS[name_a], BASELINE_RUNS[name_b], cell_area)
            rho = pd.Series({row["sample_id"]: row["rho"] for row in out.pop("rho")})
            if (name_a, name_b) == ("imagenet_mnv2_050_exact_k3", "imagenet_mnv2_050_crop_free_k3"):
                cases = representative_cases(rho, (name_a, name_b))
            for table, rows in out.items():
                tables.setdefault(table, []).extend(rows)
            print(f"{pair_type}: {name_a} vs {name_b}")

    out_dir = args.output_dir
    ranking = pd.DataFrame(tables["ranking"])
    grade = pd.DataFrame(tables["grade"])
    ranking.to_csv(out_dir / "ranking_change.csv", index=False)
    pd.DataFrame(tables["cells"]).to_csv(out_dir / "cell_rank_change.csv", index=False)
    pd.DataFrame(tables["top_share"]).to_csv(out_dir / "top_region_share.csv", index=False)
    transitions = pd.DataFrame(tables["transitions"])
    transitions[transitions["pair_type"] == "protocol"].to_csv(out_dir / "grade_transition_exact_to_cf.csv", index=False)
    transitions[transitions["pair_type"] == "yardstick"].to_csv(out_dir / "grade_transition_yardstick.csv", index=False)
    grade.to_csv(out_dir / "grade_change.csv", index=False)
    pd.DataFrame(tables["by_cell_type"]).to_csv(out_dir / "by_cell_type.csv", index=False)
    pd.DataFrame(tables["hist"]).to_csv(out_dir / "sample_spearman_hist.csv", index=False)
    pd.DataFrame(cases).to_csv(out_dir / "representative_cases.csv", index=False)

    columns = ["profile_spearman", "sample_spearman_p50", "top1_agreement", "top3_overlap_mean"]
    grade_columns = ["agreement", "kappa_linear", "delta_ge2"]
    yardstick = ranking[ranking["population"] == "all"][["pair_type", "run_a", "run_b", *columns]].merge(
        grade[(grade["population"] == "all") & (grade["cell_type"] == "all")][["run_a", "run_b", *grade_columns]],
        on=["run_a", "run_b"],
    )
    yardstick.to_csv(out_dir / "yardstick.csv", index=False)

    runs = {name for pair in protocol["pairs"].values() for names in pair for name in names}
    write_provenance(
        out_dir,
        inputs={
            "protocol": PROTOCOL_PATH,
            "cell_area": out_dir / "cell_area.csv",
            **{f"{name}:metrics": BASELINE_RUNS[name].metrics for name in sorted(runs)},
            **{f"{name}:analysis": BASELINE_RUNS[name].analysis for name in sorted(runs)},
        },
        filename="compare_protocols.provenance.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
