#!/usr/bin/env python3
"""A3 analysis: per-model summaries and cross-model comparison (``protocol.json`` ``analyses``).

Reads the four MobileNetV2 baselines and the four A3 runs (stored metrics and
analysis stores only) and writes to ``summary/``:

* ``model_info.csv``: model, weights, preprocessing, clean accuracy;
* ``region_profile.csv``: 16-cell mean ``margin_drop``, within-model rank and
  z, and top-region share per run and population;
* ``top_region_share.csv``: top-region share by cell and by cell type;
* ``grade_distribution.csv``, ``flag_rates.csv``, ``strategy_rank_corr.csv``;
* ``cross_model.csv``: pairwise ranking and grade agreement per protocol;
* ``common_patterns.csv``: center-leading pattern and HIGH share by cell type;
* ``deit_exact_area.csv``: ``deit_small`` exact cell areas against
  mobilenetv2_050 exact.

Cross-model pairs follow the comparison rule: every pair in crop-free; in
exact only the models sharing mobilenetv2_050's geometry.

Example:
    python experiments/revision_1/a3_multi_arch/compare_models.py
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.revision_1.a1_threshold.build_features import cell_columns
from experiments.revision_1.b1_preprocessing.area_tables import item_areas
from experiments.revision_1.b1_preprocessing.compare_protocols import (
    KEYS,
    descending_ranks,
    grade_summary,
    profile_matrix,
    ranking_summary,
)
from experiments.revision_1.common.agreement import GRADE_ORDER, cohen_kappa
from experiments.revision_1.common.loading import load_reliability, load_sample_meta
from experiments.revision_1.common.provenance import write_provenance
from experiments.revision_1.common.runs import A3_RUNS, BASELINE_RUNS, RunRef
from ssat.utils.io import load_json

A3_DIR = Path(__file__).resolve().parent
SUMMARY_DIR = A3_DIR / "summary"
SELECTION_PATH = SUMMARY_DIR / "model_selection.json"
MODELS = ("mobilenetv2_050", "mobilenetv2_100", "convnext_tiny", "deit_small")
PROTOCOLS = ("crop_free", "exact")
EXACT_COMPARABLE = ("mobilenetv2_050", "mobilenetv2_100", "convnext_tiny")
"""Models sharing mobilenetv2_050's exact geometry (``protocol.json`` ``comparison_rule``; checked against ``model_selection.json``)."""
CELL_TYPES = ("corner", "edge", "center")
FLAGS = ("sign_consistent", "exceeds_control", "seed_stable", "multi_strategy", "ci_excludes_zero", "area_matched")
SAMPLE_QUANTILES = (10, 25, 50, 75, 90)
INPUT_PX = 224 * 224


@dataclass(frozen=True)
class RunData:
    """Loaded per-run inputs, aligned on sorted sample ids and cells.

    Attributes:
        run: The run.
        profile: Samples x cells ``margin_drop`` (``spatial_profile``).
        reliability: Stored reliability rows indexed by ``(sample_id, region_key)``.
        correct: Clean top-1 correctness per sample.
    """

    run: RunRef
    profile: pd.DataFrame
    reliability: pd.DataFrame
    correct: pd.Series


def model_runs() -> dict[tuple[str, str], RunRef]:
    """Return ``{(model, protocol): run}`` for the eight runs compared."""

    runs = {(run.model, run.protocol): run for run in A3_RUNS.values()}
    for width in ("050", "100"):
        for protocol in PROTOCOLS:
            run = BASELINE_RUNS[f"imagenet_mnv2_{width}_{protocol}_k3"]
            runs[(run.model, run.protocol)] = run
    return runs


def load_run(run: RunRef) -> RunData:
    """Load and validate one run's profile, reliability, and correctness."""

    profile = profile_matrix(run)
    if profile.isna().any().any():
        raise ValueError(f"{run.name}: missing margin_drop values in spatial_profile")
    correct = load_sample_meta(run).set_index("sample_id")["clean_correct"].astype(bool).reindex(profile.index)
    if correct.isna().any():
        raise ValueError(f"{run.name}: samples without clean correctness")
    return RunData(run, profile, load_reliability(run).set_index(KEYS).sort_index(), correct)


def cell_info(cells: Sequence[str]) -> pd.DataFrame:
    """Return ``region_cell`` and ``cell_type`` indexed by region key."""

    return cell_columns(pd.Series(list(cells))).set_index(pd.Index(list(cells)))[["region_cell", "cell_type"]]


def top_cells(profile: pd.DataFrame) -> np.ndarray:
    """Return each sample's largest-``margin_drop`` region key (ties by region key ascending)."""

    return np.asarray(profile.columns)[np.argmin(descending_ranks(profile), axis=1)]


def profile_rows(data: RunData, samples: pd.Index, base: dict[str, object]) -> list[dict[str, object]]:
    """Per-cell mean, within-model rank and z, and top-region share."""

    profile = data.profile.loc[samples]
    means = profile.mean()
    rank = means.rank(ascending=False, method="first").astype(int)
    z = (means - means.mean()) / means.std(ddof=0)
    share = pd.Series(top_cells(profile)).value_counts(normalize=True).reindex(profile.columns, fill_value=0.0)
    info = cell_info(profile.columns)
    return [
        {**base, "cell": info.loc[key, "region_cell"], "cell_type": info.loc[key, "cell_type"],
         "mean_margin_drop": float(means[key]), "rank_in_model": int(rank[key]), "z_in_model": float(z[key]),
         "top_share": float(share[key])}
        for key in profile.columns
    ]


def grade_rows(data: RunData, samples: pd.Index, base: dict[str, object]) -> list[dict[str, object]]:
    """Grade shares overall and per cell type."""

    rel = data.reliability[data.reliability.index.get_level_values("sample_id").isin(samples)]
    types = cell_info(data.profile.columns)["cell_type"].reindex(rel.index.get_level_values("region_key")).to_numpy()
    rows = []
    for cell_type in ("all", *CELL_TYPES):
        grades = rel["reliability_grade"] if cell_type == "all" else rel["reliability_grade"][types == cell_type]
        shares = grades.value_counts(normalize=True).reindex(GRADE_ORDER, fill_value=0.0)
        rows.append({**base, "cell_type": cell_type, "n_anchors": len(grades), **{grade: float(shares[grade]) for grade in GRADE_ORDER}})
    return rows


def flag_rows(data: RunData, samples: pd.Index, base: dict[str, object]) -> list[dict[str, object]]:
    """Share of each flag value over the anchors of ``samples``."""

    rel = data.reliability[data.reliability.index.get_level_values("sample_id").isin(samples)]
    rows = []
    for flag in FLAGS:
        shares = rel[flag].astype(str).value_counts(normalize=True)
        rows += [{**base, "flag": flag, "value": value, "share": float(share)} for value, share in shares.sort_index().items()]
    return rows


def pair_row(a: RunData, b: RunData, samples: pd.Index, base: dict[str, object]) -> dict[str, object]:
    """Ranking and grade agreement between two runs on ``samples``."""

    pa_, pb = a.profile.loc[samples], b.profile.loc[samples]
    summary, _, _, _ = ranking_summary(pa_, pb)
    anchors_a = a.reliability[a.reliability.index.get_level_values("sample_id").isin(samples)]
    anchors_b = b.reliability.loc[anchors_a.index]
    grades = grade_summary(anchors_a, anchors_b)
    return {
        **base, **summary,
        "top1_kappa": cohen_kappa(top_cells(pa_), top_cells(pb)),
        **{key: grades[key] for key in ("n", "agreement", "kappa_linear", "delta_ge2", "changed_with_sign_change",
                                        "changed_with_only_exceeds_change", "high_share_a", "high_share_b")},
    }


def common_pattern_rows(data: RunData, samples: pd.Index, base: dict[str, object]) -> dict[str, object]:
    """Center-leading pattern, top share and HIGH share by cell type for one run."""

    profile = data.profile.loc[samples]
    info = cell_info(profile.columns)
    means = profile.mean()
    top4 = set(means.sort_values(ascending=False, kind="stable").index[:4])
    top = pd.Series(info["cell_type"].reindex(top_cells(profile)).to_numpy()).value_counts(normalize=True)
    rel = data.reliability[data.reliability.index.get_level_values("sample_id").isin(samples)]
    types = info["cell_type"].reindex(rel.index.get_level_values("region_key")).to_numpy()
    high = rel["reliability_grade"] == "high"
    row = {**base, "center_cells_top4": top4 == set(info.index[info["cell_type"] == "center"])}
    for cell_type in CELL_TYPES:
        row[f"top_share_{cell_type}"] = float(top.get(cell_type, 0.0))
        row[f"high_share_{cell_type}"] = float(high[types == cell_type].mean())
        row[f"mean_rank_{cell_type}"] = float(means.rank(ascending=False, method="first")[info["cell_type"] == cell_type].mean())
    return row


def populations(data: dict[str, RunData]) -> dict[str, pd.Index]:
    """Return the shared populations of one protocol: ``all`` and ``common_correct``."""

    indexes = [item.profile.index for item in data.values()]
    if any(not index.equals(indexes[0]) for index in indexes):
        raise ValueError("runs of one protocol cover different samples")
    common = np.logical_and.reduce([item.correct.to_numpy() for item in data.values()])
    return {"all": indexes[0], "common_correct": indexes[0][common]}


def comparable_pairs(protocol: str, models: Iterable[str]) -> list[tuple[str, str]]:
    """Return the model pairs compared in ``protocol`` (comparison rule)."""

    allowed = [model for model in models if protocol == "crop_free" or model in EXACT_COMPARABLE]
    return list(itertools.combinations(allowed, 2))


def analyze_protocol(protocol: str, data: dict[str, RunData]) -> dict[str, list[dict[str, object]]]:
    """Compute every per-run and pairwise table of one protocol."""

    shared = populations(data)
    out: dict[str, list[dict[str, object]]] = {key: [] for key in ("profile", "grades", "flags", "cross", "patterns")}
    for model, item in data.items():
        own = {**shared, "model_correct": item.profile.index[item.correct.to_numpy()]}
        for population, samples in own.items():
            base = {"protocol": protocol, "model": model, "run": item.run.name, "population": population, "n_samples": len(samples)}
            out["profile"] += profile_rows(item, samples, base)
            out["grades"] += grade_rows(item, samples, base)
            out["flags"] += flag_rows(item, samples, base)
            if population != "model_correct":
                out["patterns"].append(common_pattern_rows(item, samples, base))
    for model_a, model_b in comparable_pairs(protocol, data):
        a, b = data[model_a], data[model_b]
        pair_type = "yardstick" if {model_a, model_b} == {"mobilenetv2_050", "mobilenetv2_100"} else "cross_architecture"
        both = a.profile.index[(a.correct & b.correct).to_numpy()]
        for population, samples in (*shared.items(), ("pair_correct", both)):
            base = {"protocol": protocol, "model_a": model_a, "model_b": model_b, "pair_type": pair_type,
                    "population": population}
            out["cross"].append(pair_row(a, b, samples, base))
    return out


def model_info_rows(runs: dict[tuple[str, str], RunRef], data: dict[tuple[str, str], RunData], selection: dict) -> list[dict[str, object]]:
    """Model identity, preprocessing, and clean accuracy per run."""

    import timm

    records = {selection["reference"]["model"]: selection["reference"], **selection["candidates"]}
    rows = []
    for (model, protocol), run in runs.items():
        spec = load_json(run.dump / "run_manifest.json")["adapter_spec"]
        record = records.get(spec["model_name"], {})
        correct = data[(model, protocol)].correct
        rows.append({
            "model": model, "protocol": protocol, "run": run.name, "timm_name": spec["model_name"],
            "n_parameters": int(sum(p.numel() for p in timm.create_model(spec["model_name"], pretrained=False).parameters())),
            "weights_id": spec["weights_id"], "weights_sha256": record.get("weights", {}).get("sha256", ""),
            "preprocessing_desc": spec["preprocessing_desc"], "n_controls": run.n_controls,
            "n_samples": len(correct), "n_correct": int(correct.sum()), "accuracy": float(correct.mean()),
        })
    return rows


def area_rows(run: RunRef, reference: RunRef) -> list[dict[str, object]]:
    """Mean target-cell area fraction of ``run`` against ``reference``, per cell and per cell type."""

    frames = {}
    for label, item in (("run", run), ("reference", reference)):
        targets, _ = item_areas(item)
        frames[label] = targets.set_index(["sample_id", "region_key"])["area"]
    joined = pd.concat(frames, axis=1, join="inner")
    info = cell_info(sorted(joined.index.get_level_values("region_key").unique()))
    joined["cell_type"] = info["cell_type"].reindex(joined.index.get_level_values("region_key")).to_numpy()
    joined["cell"] = info["region_cell"].reindex(joined.index.get_level_values("region_key")).to_numpy()
    spread = joined.groupby(level="sample_id")[["run", "reference"]].agg(["max", "min"])
    rows = []
    for level, key in (("cell", "cell"), ("cell_type", "cell_type")):
        for value, group in joined.groupby(key):
            rows.append({"run": run.name, "reference": reference.name, "level": level, "group": value,
                         "fraction_run": float(group["run"].mean() / INPUT_PX),
                         "fraction_reference": float(group["reference"].mean() / INPUT_PX),
                         "ratio_of_means": float(group["run"].mean() / group["reference"].mean()),
                         "share_cells_differ": float((group["run"] != group["reference"]).mean())})
    for label in ("run", "reference"):
        visible = spread[(label, "min")] > 0
        ratio = spread.loc[visible, (label, "max")] / spread.loc[visible, (label, "min")]
        rows.append({"run": run.name, "reference": reference.name, "level": "within_sample_max_over_min", "group": label,
                     **{f"p{q}": float(np.percentile(ratio, q)) for q in (5, 50, 95)},
                     "n_samples_with_zero_area_cell": int((~visible).sum())})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, default=SUMMARY_DIR)
    args = parser.parse_args(argv)
    out_dir = args.output_dir

    selection = load_json(SELECTION_PATH)
    candidates = selection["candidates"]
    if not candidates["convnext_tiny.fb_in1k"]["condition_3_geometry_equal_reference"] or \
            candidates["deit_small_patch16_224.fb_in1k"]["condition_3_geometry_equal_reference"]:
        raise ValueError("EXACT_COMPARABLE disagrees with model_selection.json")
    runs = model_runs()
    data = {key: load_run(run) for key, run in runs.items()}
    print(f"loaded {len(data)} runs", flush=True)

    tables: dict[str, list[dict[str, object]]] = {}
    for protocol in PROTOCOLS:
        per_model = {model: data[(model, protocol)] for model in MODELS}
        for key, rows in analyze_protocol(protocol, per_model).items():
            tables.setdefault(key, []).extend(rows)
        print(f"analyzed {protocol}", flush=True)

    profile = pd.DataFrame(tables["profile"])
    profile.to_csv(out_dir / "region_profile.csv", index=False)
    by_type = profile.groupby(["protocol", "model", "run", "population", "cell_type"], sort=False)["top_share"].sum().reset_index()
    pd.concat([profile.assign(level="cell", group=profile["cell"]), by_type.assign(level="cell_type", group=by_type["cell_type"])])[
        ["protocol", "model", "run", "population", "level", "group", "top_share"]
    ].to_csv(out_dir / "top_region_share.csv", index=False)
    pd.DataFrame(tables["grades"]).to_csv(out_dir / "grade_distribution.csv", index=False)
    pd.DataFrame(tables["flags"]).to_csv(out_dir / "flag_rates.csv", index=False)
    pd.DataFrame(tables["cross"]).to_csv(out_dir / "cross_model.csv", index=False)
    pd.DataFrame(tables["patterns"]).to_csv(out_dir / "common_patterns.csv", index=False)
    pd.DataFrame(model_info_rows(runs, data, selection)).to_csv(out_dir / "model_info.csv", index=False)

    strategy = []
    for (model, protocol), run in runs.items():
        frame = pd.read_parquet(run.analysis / "rank_correlation.parquet")
        strategy.append(frame.assign(model=model, protocol=protocol, run=run.name))
    pd.concat(strategy).to_csv(out_dir / "strategy_rank_corr.csv", index=False)
    pd.DataFrame(area_rows(runs[("deit_small", "exact")], runs[("mobilenetv2_050", "exact")])).to_csv(
        out_dir / "deit_exact_area.csv", index=False)

    write_provenance(
        out_dir,
        inputs={"protocol": A3_DIR / "protocol.json", "model_selection": SELECTION_PATH,
                **{f"{run.name}:metrics": run.metrics for run in runs.values()},
                **{f"{run.name}:analysis": run.analysis for run in runs.values()}},
        extra={"exact_comparable": list(EXACT_COMPARABLE)},
        filename="compare_models.provenance.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
