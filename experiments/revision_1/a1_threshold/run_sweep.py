"""A1 steps 3-4: parity gate and threshold sweep over the pre-registered settings.

Reads ``protocol.json`` and the feature tables from ``build_features.py``.
With ``--check-parity`` (the default), every run's default-setting flags and
grades must equal the stored ``reliability.parquet`` before anything is
swept. Writes long-format CSVs to ``a1_threshold/summary/``.

Example:
    python experiments/revision_1/a1_threshold/run_sweep.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.revision_1.a1_threshold.build_features import (
    DEFAULT_OUTPUT_DIR as FEATURE_DIR,
    feature_paths,
    load_features,
)
from experiments.revision_1.common.agreement import GRADE_ORDER, grade_agreement, transition_matrix
from experiments.revision_1.common.grade_engine import (
    FLAG_COLUMNS,
    TRUE,
    UNAVAILABLE,
    GradeParams,
    RunScales,
    combine_grade,
    exceeds_flag,
    flag_labels,
    grade_anchors,
    grade_labels,
)
from experiments.revision_1.common.provenance import write_json_shared, write_provenance
from experiments.revision_1.common.runs import BASELINE_RUNS

A1_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = A1_DIR / "protocol.json"
SUMMARY_DIR = A1_DIR / "summary"
PATCH_CELL = "r0/c0"
Z_PARAMS = ("z_threshold", "ddof", "std_floor_beta")
HIST_EDGES = np.round(np.arange(-3.0, 10.0 + 1e-9, 0.1), 2)


def load_protocol(path: Path = PROTOCOL_PATH) -> dict:
    """Return the pre-registered protocol."""

    return json.loads(path.read_text())


def _params(values: dict) -> GradeParams:
    return GradeParams(
        z_threshold=float(values["z_threshold"]),
        sign_alpha=float(values["sign_alpha"]),
        min_multi_strategy=int(values["min_multi_strategy"]),
        ci_level=str(values["ci_level"]),
        ddof=int(values["ddof"]),
        std_floor_beta=float(values["std_floor_beta"]),
    )


def settings_for(protocol: dict, dataset: str) -> Iterator[tuple[str, str, object, GradeParams]]:
    """Yield ``(setting_id, param, value, params)`` for one dataset, default first."""

    default = _params(protocol["presets"]["default"])
    yield "default", "default", None, default
    has_controls = dataset != "ntu60"
    for param, spec in protocol["parameters"].items():
        if param in Z_PARAMS and not has_controls:
            continue
        values = spec["oat"][dataset] if isinstance(spec["oat"], dict) else spec["oat"]
        for value in values:
            yield f"oat:{param}={value}", param, value, replace(default, **{param: type(getattr(default, param))(value)})
    for name, values in protocol["presets"].items():
        if name != "default":
            yield f"preset:{name}", "preset", name, _params(values)
    for alpha in protocol["parameters"]["sign_alpha"]["oat"]:
        if alpha > 0:
            yield (
                f"alt:zero_abstain:sign_alpha={alpha}",
                "zero_abstain_alpha",
                alpha,
                replace(default, sign_alpha=float(alpha), zero_abstain=True),
            )


def z_curve_values(protocol: dict) -> np.ndarray:
    """Return the registered tau_z curve grid."""

    curve = protocol["parameters"]["z_threshold"]["curve"]
    return np.round(np.arange(curve["start"], curve["stop"] + curve["step"] / 2, curve["step"]), 4)


def check_parity(features: pd.DataFrame, graded: pd.DataFrame) -> dict[str, int]:
    """Count default-setting mismatches against the stored reliability rows."""

    mismatches = {
        flag: int((flag_labels(graded[flag]) != features[f"stored__{flag}"].to_numpy()).sum())
        for flag in FLAG_COLUMNS
    }
    mismatches["grade"] = int((grade_labels(graded["grade"]) != features["stored__reliability_grade"].to_numpy()).sum())
    return mismatches


def _grade_shares(grades: np.ndarray) -> dict[str, float]:
    counts = np.bincount(grades.astype(np.int64), minlength=len(GRADE_ORDER))
    return {grade: counts[index] / grades.size for index, grade in enumerate(GRADE_ORDER)}


def _flag_rates(graded: pd.DataFrame) -> Iterator[dict[str, object]]:
    for flag in FLAG_COLUMNS:
        codes = graded[flag].to_numpy()
        yield {
            "flag": flag,
            "true": float((codes == 1).mean()),
            "false": float((codes == 0).mean()),
            "unavailable": float((codes == -1).mean()),
        }


def _by_cell(features: pd.DataFrame, default: np.ndarray, grades: np.ndarray) -> Iterator[dict[str, object]]:
    frame = pd.DataFrame(
        {
            "region_cell": features["region_cell"],
            "cell_type": features["cell_type"],
            "same": default == grades,
            "high_default": default == GRADE_ORDER.index("high"),
            "high": grades == GRADE_ORDER.index("high"),
        }
    )
    for level, column in (("cell", "region_cell"), ("cell_type", "cell_type")):
        summary = frame.groupby(column, sort=True)[["same", "high_default", "high"]].mean()
        counts = frame.groupby(column, sort=True).size()
        for key, row in summary.iterrows():
            yield {
                "level": level,
                "cell": key,
                "n": int(counts[key]),
                "agreement": float(row["same"]),
                "high_share_default": float(row["high_default"]),
                "high_share": float(row["high"]),
            }


def _synthetic_gt(features: pd.DataFrame, grades: np.ndarray) -> dict[str, float]:
    patch = (features["region_cell"] == PATCH_CELL).to_numpy()
    high = grades == GRADE_ORDER.index("high")
    return {
        "high_share_patch": float(high[patch].mean()),
        "high_share_non_patch": float(high[~patch].mean()),
        "n_patch": int(patch.sum()),
        "n_non_patch": int((~patch).sum()),
    }


def sweep_run(name: str, protocol: dict, feature_dir: Path, *, check: bool) -> dict[str, list[dict[str, object]]]:
    """Run every registered setting on one run and return the output rows by table."""

    run = BASELINE_RUNS[name]
    metric = protocol["primary_metric"]
    features, meta = load_features(feature_dir, name, metric)
    scales = RunScales(**meta["scales"])
    tables: dict[str, list[dict[str, object]]] = {key: [] for key in (
        "grade_distribution", "agreement", "transitions", "flag_rates", "by_cell", "z_curve",
        "synthetic_gt", "boundary_mass", "max_z_hist", "seed_cv_flag_rates", "parity",
    )}
    base = {"run": name, "dataset": run.dataset, "metric": metric}
    anchor_index = pd.MultiIndex.from_frame(features[["sample_id", "region_key", "invert_mask"]])

    settings = list(settings_for(protocol, run.dataset))
    default_params = settings[0][3]
    default = grade_anchors(features, default_params, scales=scales)
    mismatches = check_parity(features, default)
    tables["parity"].append({**base, **mismatches, "n_anchors": len(features), "interval_parity_95": meta["interval_parity_95"]["passed"]})
    if check and any(mismatches.values()):
        raise SystemExit(f"{name}: default grades differ from reliability.parquet: {mismatches}")
    default_grades = default["grade"].to_numpy()
    default_labels = pd.Series(grade_labels(default_grades), index=anchor_index)
    is_grid = features["region_cell"].notna().all()

    for setting_id, param, value, params in settings:
        graded = default if setting_id == "default" else grade_anchors(features, params, scales=scales)
        grades = graded["grade"].to_numpy()
        labels = pd.Series(grade_labels(grades), index=anchor_index)
        row = {**base, "setting_id": setting_id, "param": param, "value": value}
        counts = np.bincount(grades.astype(np.int64), minlength=len(GRADE_ORDER))
        for index, grade in enumerate(GRADE_ORDER):
            tables["grade_distribution"].append({**row, "grade": grade, "n": int(counts[index]), "share": counts[index] / grades.size})
        tables["agreement"].append({**row, **grade_agreement(default_labels, labels)})
        matrix = transition_matrix(default_labels, labels)
        for from_grade in GRADE_ORDER:
            for to_grade in GRADE_ORDER:
                tables["transitions"].append({**row, "from_grade": from_grade, "to_grade": to_grade, "n": int(matrix.loc[from_grade, to_grade])})
        for rates in _flag_rates(graded):
            tables["flag_rates"].append({**row, **rates})
        if is_grid:
            for cell_row in _by_cell(features, default_grades, grades):
                tables["by_cell"].append({**row, **cell_row})
        if run.dataset == "synthetic" and (param in ("default", "sign_alpha", "preset")):
            tables["synthetic_gt"].append({**row, **_synthetic_gt(features, grades)})

    z = default["max_z"].to_numpy()
    evaluable = ~np.isnan(z)
    if evaluable.any():
        consistent = default["sign_consistent"].to_numpy()
        multi = default["multi_strategy"].to_numpy()
        ci = default["ci_excludes_zero"].to_numpy()
        for tau in z_curve_values(protocol):
            grades = combine_grade(consistent, exceeds_flag(z, float(tau)), multi, ci)
            curve_row = {**base, "z_threshold": float(tau)}
            tables["z_curve"].append({**curve_row, **_grade_shares(grades)})
            if run.dataset == "synthetic":
                tables["synthetic_gt"].append({**base, "setting_id": f"curve:z_threshold={tau:g}", "param": "z_curve", "value": float(tau), **_synthetic_gt(features, grades)})
        threshold = default_params.z_threshold
        for delta in protocol["summaries"]["boundary_mass"]["deltas"]:
            near = evaluable & (np.abs(z - threshold) <= delta)
            tables["boundary_mass"].append({
                **base,
                "z_threshold": threshold,
                "delta": delta,
                "share_of_all": float(near.mean()),
                "share_of_evaluable": float(near.sum() / evaluable.sum()),
                "n_evaluable": int(evaluable.sum()),
                "n_anchors": int(z.size),
                **{f"max_z_p{q}": float(np.percentile(z[evaluable], q)) for q in (10, 25, 50, 75, 90, 99)},
            })
        clipped = np.clip(z[evaluable], HIST_EDGES[0], HIST_EDGES[-1])
        counts, edges = np.histogram(clipped, bins=HIST_EDGES)
        for left, right, count in zip(edges[:-1], edges[1:], counts):
            tables["max_z_hist"].append({**base, "bin_left": float(left), "bin_right": float(right), "count": int(count)})

    seed_cv = features["seed_cv_max"].to_numpy(dtype=float)
    for threshold in protocol["structurally_inert"]["seed_cv_threshold"]["values"]:
        codes = np.where(np.isnan(seed_cv), UNAVAILABLE, np.where(seed_cv < threshold, TRUE, 0))
        entry = {**base, "seed_cv_threshold": threshold, "true": float((codes == 1).mean()), "false": float((codes == 0).mean()), "unavailable": float((codes == -1).mean())}
        if threshold == 0.2:
            entry["stored_mismatches"] = int((flag_labels(codes) != features["stored__seed_stable"].to_numpy()).sum())
        tables["seed_cv_flag_rates"].append(entry)
    return tables


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", nargs="+", default=None, help="subset of protocol runs")
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=SUMMARY_DIR)
    parser.add_argument("--no-check-parity", dest="check_parity", action="store_false")
    args = parser.parse_args(argv)

    protocol = load_protocol()
    names = args.runs or protocol["runs"]
    tables: dict[str, list[dict[str, object]]] = {}
    for name in names:
        for key, rows in sweep_run(name, protocol, args.feature_dir, check=args.check_parity).items():
            tables.setdefault(key, []).extend(rows)
        print(f"{name}: swept")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key, rows in tables.items():
        if key != "parity":
            pd.DataFrame(rows).to_csv(args.output_dir / f"{key}.csv", index=False)
    write_json_shared(args.output_dir / "parity.json", {"check_parity": args.check_parity, "runs": tables["parity"]})
    inputs = {"protocol": PROTOCOL_PATH}
    for name in names:
        table_path, meta_path = feature_paths(args.feature_dir, name, protocol["primary_metric"])
        inputs[f"features:{name}"] = table_path
        inputs[f"features_meta:{name}"] = meta_path
    write_provenance(args.output_dir, inputs=inputs, extra={"runs": names, "check_parity": args.check_parity}, filename="run_sweep.provenance.json")
    print(f"wrote {len(tables) - 1} tables to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
