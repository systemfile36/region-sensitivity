#!/usr/bin/env python3
"""A2 step 2: control-count ablation over subsets of each run's K=20 controls.

For every registered K and subset scheme (``protocol.json`` ``ablation``),
recomputes the control mean/std from the control tensor, regrades every
anchor with the A1 grade engine (only ``exceeds_control`` depends on the
controls), and compares with the run's K=20 reference.

Parity gates (``--no-check-parity`` skips the second one):

* prefix K=20 with ddof 0 must reproduce ``analysis/reliability.parquet``;
* prefix K=3 (and K=2 for synthetic) must reproduce the official
  ``ComparisonIndexer -> compare_to_controls -> compute_reliability`` path
  on the item values filtered to those control indices.

Writes the ``outputs`` tables of ``protocol.json`` (except figures) and
``ablation_parity.json`` to ``summary/``.

Example:
    python experiments/revision_1/a2_control_count/run_ablation.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from experiments.revision_1.a1_threshold.build_features import build_features, feature_paths, load_features
from experiments.revision_1.a2_control_count.build_control_tensor import (
    DEFAULT_OUTPUT_DIR as TENSOR_DIR,
    ControlTensor,
    load_tensor,
    subset_stats,
)
from experiments.revision_1.common.agreement import GRADE_ORDER, grade_agreement, spearman
from experiments.revision_1.common.grade_engine import (
    GRADE_CODES,
    GradeParams,
    compute_scales,
    grade_anchors,
    grade_codes,
    grade_labels,
    with_control_stats,
)
from experiments.revision_1.common.loading import CONTEXT_COLUMNS, latest_index, load_item_values
from experiments.revision_1.common.provenance import write_json_shared, write_provenance
from experiments.revision_1.common.runs import A2_RESULTS_DIR, A2_RUNS, RunRef
from ssat.analysis.control import compare_to_controls
from ssat.analysis.indexer import ComparisonIndexer
from ssat.analysis.reliability import compute_reliability
from ssat.analysis.stability import compute_seed_stability, compute_strategy_stability
from ssat.analysis.store import _table_to_intervals

A2_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = A2_DIR / "protocol.json"
SUMMARY_DIR = A2_DIR / "summary"
FEATURE_DIR = A2_RESULTS_DIR / "features"
LOG_PATH = A2_RESULTS_DIR / "run_matrix_log.jsonl"
KEY = ["sample_id", "region_key", "invert_mask"]
SCHEMES = ("prefix", "global_random")
CONVERGENCE_K = (1, 2, 3, 5, 10)
CELL_TYPES = ("all", "corner", "edge", "center")
Z_QUANTILES = (10, 25, 50, 75, 90, 99)
PATCH_CELL = "r0/c0"
HIGH = int(GRADE_CODES["high"])
N_REPLICATES = 200
SEED_BASE = 20260923


# --- subsets and summaries ------------------------------------------------------


def draw_subsets(scheme: str, k: int, n_controls: int, n_replicates: int, seed_base: int) -> list[tuple[int, ...]]:
    """Return the control-index subsets of one scheme and K (protocol ``subset_schemes``)."""

    if scheme == "prefix" or k == n_controls:
        return [tuple(range(k))]
    if scheme != "global_random":
        raise ValueError(f"unknown scheme {scheme!r}")
    return [
        tuple(sorted(int(index) for index in np.random.default_rng(seed_base + r).choice(n_controls, k, replace=False)))
        for r in range(n_replicates)
    ]


def band(values: Iterable[float]) -> dict[str, float]:
    """Return the mean and the 5th/95th percentiles of per-replicate values."""

    array = np.asarray([value for value in values if value is not None], dtype=float)
    if array.size == 0:
        return {"mean": float("nan"), "p05": float("nan"), "p95": float("nan")}
    return {"mean": float(array.mean()), "p05": float(np.percentile(array, 5)), "p95": float(np.percentile(array, 95))}


def replicate_disagreement(grades: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-anchor grade instability across replicates.

    Args:
        grades: Grade codes, shape ``(replicates, anchors)``.

    Returns:
        ``1 - sum_g p_g^2`` (pairwise disagreement) and ``1 - max_g p_g``,
        where ``p_g`` is the share of replicates giving the anchor grade ``g``.
    """

    shares = np.stack([(grades == code).mean(axis=0) for code in range(len(GRADE_ORDER))])
    return 1.0 - (shares**2).sum(axis=0), 1.0 - shares.max(axis=0)


def cell_profile(mean_excess: np.ndarray, cells: pd.Series) -> pd.Series:
    """Dataset-level mean of per-anchor excess by grid cell."""

    return pd.Series(mean_excess, index=cells.index).groupby(cells.to_numpy()).mean().sort_index()


def top1_cells(mean_excess: np.ndarray, sample_ids: pd.Series, cells: pd.Series) -> pd.Series:
    """Per-sample cell with the largest mean excess (first cell in sorted order on ties)."""

    table = pd.DataFrame({"sample_id": sample_ids.to_numpy(), "cell": cells.to_numpy(), "value": mean_excess})
    wide = table.pivot(index="sample_id", columns="cell", values="value").sort_index(axis=1)
    return pd.Series(wide.columns[np.argmax(wide.to_numpy(), axis=1)], index=wide.index)


def cell_masks(cell_types: pd.Series) -> dict[str, np.ndarray]:
    """Anchor masks for ``all`` and each cell type present."""

    masks = {"all": np.ones(len(cell_types), bool)}
    for cell_type in CELL_TYPES[1:]:
        mask = (cell_types == cell_type).to_numpy()
        if mask.any():
            masks[cell_type] = mask
    return masks


def agreement_row(reference: np.ndarray, other: np.ndarray) -> dict[str, float | None]:
    """``grade_agreement`` from ``other`` (K) to ``reference`` (K=20), with explicit direction names."""

    stats = grade_agreement(pd.Series(grade_labels(other)), pd.Series(grade_labels(reference)))
    return {
        "agreement": stats["agreement"],
        "kappa_linear": stats["kappa_linear"],
        "delta_ge2": stats["delta_ge2"],
        "k20_higher": stats["up"],
        "k20_lower": stats["down"],
    }


def grade_shares(grades: np.ndarray) -> dict[str, float]:
    """Share of each grade."""

    return {f"{grade}_share": float((grades == code).mean()) for code, grade in enumerate(GRADE_ORDER)}


# --- official path (parity) -------------------------------------------------------


def control_index_column(region_params_json: pd.Series) -> np.ndarray:
    """Parse ``control_index`` from the JSON column (-1 for targets)."""

    parsed = region_params_json.map(lambda text: json.loads(text).get("control_index", -1) if text else -1)
    return parsed.to_numpy(dtype=np.int64)


def official_grades(item_values: pd.DataFrame, analysis_dir: Path, metric: str, k: int) -> pd.Series:
    """Grades from the ssat analysis path on the items with ``control_index < k``.

    The K-independent intervals are the run's stored ones (they use target
    items only).
    """

    index = control_index_column(item_values["region_params_json"])
    subset = item_values[(~item_values["is_control"].to_numpy()) | (index < k)].reset_index(drop=True)
    indexer = ComparisonIndexer(subset[list(CONTEXT_COLUMNS)])
    control_rows = compare_to_controls(subset, indexer.control_pairs, metric_names=[metric])
    seed_rows = compute_seed_stability(subset)
    strategy_rows, _ = compute_strategy_stability(subset, primary_metric=metric)
    intervals = [row for row in _table_to_intervals(pq.read_table(analysis_dir / "intervals.parquet")) if row.metric == metric]
    rows = compute_reliability(control_rows, seed_rows, strategy_rows, intervals)
    return pd.Series(
        [row.reliability_grade.value for row in rows],
        index=pd.MultiIndex.from_tuples(
            [(row.anchor_key.sample_id, row.anchor_key.region_key, row.anchor_key.invert_mask) for row in rows], names=KEY
        ),
    )


# --- per-run ablation ------------------------------------------------------------


def aligned_inputs(run: RunRef, metric: str) -> tuple[pd.DataFrame, ControlTensor, dict]:
    """Load the run's feature table and tensor and check that their anchors and conditions align."""

    table_path, _ = feature_paths(FEATURE_DIR, run.name, metric)
    if not table_path.is_file():
        features, meta = build_features(run, metric)
        FEATURE_DIR.mkdir(parents=True, exist_ok=True)
        features.to_parquet(table_path, index=False)
        table_path.chmod(0o644)
        write_json_shared(table_path.with_suffix(".json"), meta)
    features, feature_meta = load_features(FEATURE_DIR, run.name, metric)
    tensor, tensor_meta = load_tensor(TENSOR_DIR, run.name, metric)
    same_keys = (
        np.array_equal(features["sample_id"].to_numpy(dtype=object), tensor.sample_id)
        and np.array_equal(features["region_key"].to_numpy(dtype=object), tensor.region_key)
        and np.array_equal(features["invert_mask"].to_numpy(dtype=bool), tensor.invert_mask)
    )
    if not same_keys:
        raise ValueError(f"{run.name}: feature and tensor anchors differ")
    feature_conditions = [(c["perturb_op"], c["perturb_params_hash"]) for c in feature_meta["conditions"]]
    if feature_conditions != list(tensor.conditions):
        raise ValueError(f"{run.name}: feature and tensor conditions differ")
    return features, tensor, {"features": feature_meta, "tensor": tensor_meta}


def _z(target: np.ndarray, mean: np.ndarray, std0: np.ndarray, n: int, ddof: int) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        std = std0 if ddof == 0 else std0 * np.sqrt(n / (n - ddof))
        return np.where((std != 0) & (n >= 2), (target - mean) / std, np.nan)


def ablate_run(run: RunRef, protocol: dict, metric: str, *, check_official: bool) -> tuple[dict[str, list[dict]], dict]:
    """Compute every A2 table for one run; returns ``(tables, parity)``."""

    ablation = protocol["ablation"]
    k_values = [int(k) for k in ablation["k_values"]]
    ddofs = (int(ablation["ddof"]["primary"]), int(ablation["ddof"]["secondary"]))
    registered = ablation["subset_schemes"]["global_random"]
    if f"R = {N_REPLICATES};" not in registered or f"default_rng({SEED_BASE} + r)" not in registered:
        raise ValueError("N_REPLICATES / SEED_BASE differ from protocol.json")
    n_replicates, seed_base = N_REPLICATES, SEED_BASE

    features, tensor, meta = aligned_inputs(run, metric)
    scales = compute_scales(features)
    target = tensor.target
    cells = features["region_cell"]
    cell_types = features["cell_type"]
    masks = cell_masks(cell_types)
    n_all = tensor.n_controls

    def grade(mean: np.ndarray, std0: np.ndarray, n: int, ddof: int) -> pd.DataFrame:
        return grade_anchors(with_control_stats(features, target, mean, std0, n), GradeParams(ddof=ddof), scales=scales)

    ref_mean, ref_std0, _ = subset_stats(tensor, range(n_all))
    ref_sigma1 = ref_std0 * np.sqrt(n_all / (n_all - 1))
    reference = {d: grade(ref_mean, ref_std0, n_all, d) for d in ddofs}
    ref_z = {d: _z(target, ref_mean, ref_std0, n_all, d) for d in ddofs}
    ref_excess = (target - ref_mean).mean(axis=1)
    ref_profile = cell_profile(ref_excess, cells)
    ref_top1 = top1_cells(ref_excess, features["sample_id"], cells)

    stored = grade_codes(features["stored__reliability_grade"])
    parity: dict[str, object] = {
        "tensor_stored_parity": meta["tensor"]["stored_parity"],
        "k20_prefix_vs_reliability_parquet": {
            "n_anchors": int(len(stored)),
            "n_grade_mismatch": int((reference[0]["grade"].to_numpy() != stored).sum()),
            "n_exceeds_mismatch": int(
                (reference[0]["exceeds_control"].to_numpy()
                 != np.array([{"true": 1, "false": 0, "unavailable": -1}[v] for v in features["stored__exceeds_control"]])).sum()
            ),
        },
    }
    if parity["k20_prefix_vs_reliability_parquet"]["n_grade_mismatch"] or parity["k20_prefix_vs_reliability_parquet"]["n_exceeds_mismatch"]:
        raise ValueError(f"{run.name}: prefix K={n_all} grades differ from reliability.parquet: {parity}")

    tables: dict[str, list[dict]] = defaultdict(list)
    base = {"run": run.name, "dataset": run.dataset}
    grades_by: dict[tuple[str, int, int], np.ndarray] = {}

    for scheme in SCHEMES:
        # Control-mean convergence (K = 1 included; no z needed).
        for k in CONVERGENCE_K:
            errors_per_rep = []
            for subset in draw_subsets(scheme, k, n_all, n_replicates, seed_base):
                mean, _, _ = subset_stats(tensor, subset)
                with np.errstate(divide="ignore", invalid="ignore"):
                    error = np.abs(mean - ref_mean) / ref_sigma1
                errors_per_rep.append(error[ref_sigma1 > 0])
            pooled = np.concatenate(errors_per_rep)
            medians = [float(np.median(e)) for e in errors_per_rep]
            tables["control_mean_convergence"].append({
                **base, "scheme": scheme, "k": k, "n_replicates": len(errors_per_rep),
                "n_anchor_conditions": int(errors_per_rep[0].size),
                "median": float(np.median(pooled)), "p90": float(np.percentile(pooled, 90)),
                **{f"replicate_median_{key}": value for key, value in band(medians).items()},
            })

        for k in k_values:
            subsets = draw_subsets(scheme, k, n_all, n_replicates, seed_base)
            per_rep: dict[int, list[dict]] = defaultdict(list)
            rank_rows = []
            for subset in subsets:
                mean, std0, n = subset_stats(tensor, subset)
                excess = (target - mean).mean(axis=1)
                top1 = top1_cells(excess, features["sample_id"], cells)
                rank_rows.append({"profile_spearman": spearman(cell_profile(excess, cells), ref_profile),
                                  "top1_agreement": float((top1 == ref_top1.reindex(top1.index)).mean())})
                for d in ddofs:
                    graded = grade(mean, std0, n, d)
                    codes = graded["grade"].to_numpy()
                    z = _z(target, mean, std0, n, d)
                    both = np.isfinite(z) & np.isfinite(ref_z[d])
                    record = {
                        "codes": codes,
                        "max_z": graded["max_z"].to_numpy(),
                        "abs_dz_median": float(np.median(np.abs(z - ref_z[d])[both])) if both.any() else np.nan,
                        "abs_dz_p90": float(np.percentile(np.abs(z - ref_z[d])[both], 90)) if both.any() else np.nan,
                        "abs_dexcess_median": float(np.median(np.abs(mean - ref_mean))),
                        "abs_dexcess_p90": float(np.percentile(np.abs(mean - ref_mean), 90)),
                        "max_z_spearman": spearman(graded["max_z"].to_numpy(), reference[d]["max_z"].to_numpy()),
                    }
                    per_rep[d].append(record)
            tables["ranking_vs_k"].append({
                **base, "scheme": scheme, "k": k, "n_replicates": len(subsets),
                **{f"{stat}_{key}": value for stat in ("profile_spearman", "top1_agreement")
                   for key, value in band(row[stat] for row in rank_rows).items()},
            })
            for d in ddofs:
                reps = per_rep[d]
                codes = np.stack([rep["codes"] for rep in reps])
                grades_by[(scheme, k, d)] = codes
                ref_codes = reference[d]["grade"].to_numpy()
                tables["effect_convergence"].append({
                    **base, "scheme": scheme, "k": k, "ddof": d, "n_replicates": len(reps),
                    **{f"{stat}_{key}": value
                       for stat in ("abs_dz_median", "abs_dz_p90", "abs_dexcess_median", "abs_dexcess_p90", "max_z_spearman")
                       for key, value in band(rep[stat] for rep in reps).items()},
                })
                for cell_type, mask in masks.items():
                    rows = [
                        {**agreement_row(ref_codes[mask], rep_codes[mask]), **grade_shares(rep_codes[mask])}
                        for rep_codes in codes
                    ]
                    tables["grade_vs_k"].append({
                        **base, "scheme": scheme, "k": k, "ddof": d, "cell_type": cell_type,
                        "n_anchors": int(mask.sum()), "n_replicates": len(reps),
                        **{f"{stat}_{key}": value for stat in rows[0] for key, value in band(row[stat] for row in rows).items()},
                        **{f"k20_{grade}_share": float((ref_codes[mask] == code).mean()) for code, grade in enumerate(GRADE_ORDER)},
                    })
                shares = np.zeros((len(GRADE_ORDER), len(GRADE_ORDER)))
                for rep_codes in codes:
                    np.add.at(shares, (rep_codes, ref_codes), 1.0 / (len(ref_codes) * len(codes)))
                for i, from_grade in enumerate(GRADE_ORDER):
                    for j, to_grade in enumerate(GRADE_ORDER):
                        tables["transitions_vs_k"].append({**base, "scheme": scheme, "k": k, "ddof": d,
                                                           "from_grade_k": from_grade, "to_grade_k20": to_grade,
                                                           "share": float(shares[i, j])})
                max_z = np.concatenate([rep["max_z"] for rep in reps])
                finite = max_z[np.isfinite(max_z)]
                tables["ddof_effect"].append({
                    **base, "scheme": scheme, "k": k, "ddof": d, "n_replicates": len(reps),
                    "z_undefined_share": float(1 - finite.size / max_z.size),
                    **{f"max_z_p{q}": float(np.percentile(finite, q)) for q in Z_QUANTILES},
                    **{f"high_share_{key}": value for key, value in band((c == HIGH).mean() for c in codes).items()},
                })
                if run.dataset == "synthetic":
                    patch = (cells == PATCH_CELL).to_numpy()
                    tables["synthetic_gt_vs_k"].append({
                        **base, "scheme": scheme, "k": k, "ddof": d, "n_replicates": len(reps),
                        **{f"patch_high_share_{key}": value for key, value in band((c[patch] == HIGH).mean() for c in codes).items()},
                        **{f"other_high_share_{key}": value for key, value in band((c[~patch] == HIGH).mean() for c in codes).items()},
                    })

    # Replicate variability (global_random, K < K_max).
    for (scheme, k, d), codes in sorted(grades_by.items()):
        if scheme != "global_random" or k == n_all:
            continue
        pairwise, one_minus_max = replicate_disagreement(codes)
        for cell_type, mask in masks.items():
            row = {**base, "k": k, "ddof": d, "cell_type": cell_type, "n_replicates": int(codes.shape[0]),
                   "n_anchors": int(mask.sum()),
                   "high_share_replicate_std": float((codes[:, mask] == HIGH).mean(axis=1).std(ddof=1))}
            for name, values in (("flip_pairwise", pairwise[mask]), ("flip_vs_mode", one_minus_max[mask])):
                row.update({f"{name}_mean": float(values.mean()), f"{name}_share_gt0": float((values > 0).mean()),
                            f"{name}_p50": float(np.percentile(values, 50)), f"{name}_p90": float(np.percentile(values, 90))})
            tables["replicate_variability"].append(row)

    tables["marginal_gain"].extend(marginal_gain(run, tables["grade_vs_k"], len(features), n_all))

    if check_official:
        item_values = load_item_values(run, (metric,))
        official = {}
        for k in ([2, 3] if run.dataset == "synthetic" else [3]):
            expected = official_grades(item_values, run.analysis, metric, k)
            prefix = grades_by[("prefix", k, 0)][0]
            ours = pd.Series(grade_labels(prefix), index=pd.MultiIndex.from_frame(features[KEY]))
            joined = pd.concat({"official": expected, "engine": ours}, axis=1, join="outer")
            official[f"k{k}"] = {"n_anchors": int(len(joined)),
                                 "n_missing": int(joined.isna().any(axis=1).sum()),
                                 "n_grade_mismatch": int((joined["official"] != joined["engine"]).sum())}
            if official[f"k{k}"]["n_missing"] or official[f"k{k}"]["n_grade_mismatch"]:
                raise ValueError(f"{run.name}: prefix K={k} grades differ from the official path: {official}")
        parity["prefix_vs_official_path"] = official
    return tables, parity


def marginal_gain(run: RunRef, grade_rows: list[dict], n_anchors: int, n_all: int) -> list[dict]:
    """Agreement gained per added control and per added item, with run-time estimates."""

    total_items = len(latest_index(run.dump))
    per_anchor = total_items / (n_anchors * (1 + n_all))
    if per_anchor != int(per_anchor):
        raise ValueError(f"{run.name}: {total_items} items do not split into {n_anchors} anchors x {1 + n_all}")
    per_anchor = int(per_anchor)
    elapsed = None
    if LOG_PATH.is_file():
        for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["run"] == run.name and record["step"] == "run" and record["returncode"] == 0:
                elapsed = float(record["elapsed_s"])
    items_per_s = total_items / elapsed if elapsed else None
    rows = []
    for (scheme, ddof), group in pd.DataFrame(
        [row for row in grade_rows if row["cell_type"] == "all"]
    ).groupby(["scheme", "ddof"]):
        group = group.sort_values("k")
        previous, previous_items = None, 0
        for row in group.itertuples(index=False):
            items = n_anchors * per_anchor * (1 + row.k)
            out = {"run": run.name, "scheme": scheme, "ddof": ddof, "k": row.k, "agreement_mean": row.agreement_mean,
                   "perturbed_items": items, "items_per_s": items_per_s,
                   "run_hours": items / items_per_s / 3600 if items_per_s else None}
            if previous is not None:
                gain = row.agreement_mean - previous.agreement_mean
                out["agreement_gain_per_control"] = gain / (row.k - previous.k)
                out["agreement_gain_per_million_items"] = gain / ((items - previous_items) / 1e6)
            rows.append(out)
            previous, previous_items = row, items
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", nargs="+", default=list(A2_RUNS))
    parser.add_argument("--metric", default="margin_drop")
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    parser.add_argument("--no-check-parity", action="store_true", help="skip the official-path parity (slow)")
    args = parser.parse_args(argv)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    all_tables: dict[str, list[dict]] = defaultdict(list)
    parities = {}
    for name in args.runs:
        tables, parity = ablate_run(A2_RUNS[name], protocol, args.metric, check_official=not args.no_check_parity)
        for table, rows in tables.items():
            all_tables[table].extend(rows)
        parities[name] = parity
        print(f"{name}: parity {json.dumps(parity['k20_prefix_vs_reliability_parquet'])} "
              f"{json.dumps(parity.get('prefix_vs_official_path'))}", flush=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    for table, rows in all_tables.items():
        path = args.summary_dir / f"{table}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        path.chmod(0o644)
    write_json_shared(args.summary_dir / "ablation_parity.json", parities)
    inputs = {}
    for name in args.runs:
        run = A2_RUNS[name]
        inputs[name] = run.dump
        inputs[f"{name}:tensor"] = TENSOR_DIR / f"{name}__{args.metric}.npz"
        inputs[f"{name}:features"] = feature_paths(FEATURE_DIR, name, args.metric)[0]
    write_provenance(args.summary_dir, inputs={**inputs, "protocol": PROTOCOL_PATH},
                     extra={"metric": args.metric, "runs": args.runs, "official_parity": not args.no_check_parity},
                     filename="run_ablation.provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
