"""Aggregation, reliability analysis, measurement, and reporting."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import tokenize
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # Support both ``python run.py`` and package-style test imports.
    from .workflow import canonical_json, load_raw, region_key
except ImportError:  # pragma: no cover - exercised by the documented script entrypoint
    from workflow import canonical_json, load_raw, region_key

CAPABILITIES = (
    ("model wrapper", "custom", "built-in"),
    ("dataset iteration", "custom", "built-in"),
    ("region mask generation", "custom", "built-in"),
    ("multiple fill strategies", "custom", "built-in"),
    ("output to task metric", "Captum primitive + custom wrapper", "metric interface"),
    ("per-sample serialization", "custom", "standard raw schema"),
    ("matched random control", "custom", "built-in"),
    ("control normalization", "custom", "built-in"),
    ("seed repeat", "custom", "built-in"),
    ("bootstrap uncertainty", "custom", "built-in"),
    ("operator consistency", "custom", "built-in"),
    ("sample aggregation", "custom", "built-in"),
    ("region aggregation", "custom", "built-in"),
    ("class aggregation", "custom", "built-in"),
    ("dataset aggregation", "custom", "built-in"),
    ("preprocessing validation", "custom", "built-in"),
    ("mask-area validation", "custom", "built-in"),
    ("resolved configuration", "custom", "automatic"),
    ("provenance", "custom", "automatic"),
    ("cache", "custom", "built-in"),
    ("resume", "custom", "built-in"),
    ("report generation", "custom", "built-in"),
)

USER_DECISIONS = (
    "raw result schema",
    "aggregation convention",
    "degradation sign",
    "matched-control definition",
    "confidence interval method",
    "reproducibility metadata",
    "cache identity and retry policy",
    "resume conflict handling",
)


def _save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.reindex(sorted(frame.columns), axis=1)
    ordered.to_parquet(path.with_suffix(".parquet"), index=False)
    ordered.to_csv(path.with_suffix(".csv"), index=False)


def canonical_table_hash(frame: pd.DataFrame) -> str:
    """Hash a table after stable ordering and float normalization."""

    normalized = frame.copy()
    for column in normalized.select_dtypes(include=["float", "float32", "float64"]).columns:
        normalized[column] = normalized[column].round(10)
    columns = sorted(normalized.columns)
    normalized = normalized.reindex(columns=columns)
    if columns:
        normalized = normalized.sort_values(columns, kind="stable", na_position="last")
    payload = normalized.to_json(orient="records", double_precision=10)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rank(frame: pd.DataFrame, patch_key: str) -> int:
    ordered = frame.sort_values(["degradation", "region_key"], ascending=[False, True])
    matches = np.flatnonzero(ordered["region_key"].to_numpy() == patch_key)
    if len(matches) != 1:
        raise ValueError(f"expected one patch row for {patch_key}")
    return int(matches[0]) + 1


def _bootstrap_intervals(
    raw: pd.DataFrame,
    *,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> pd.DataFrame:
    target = raw.loc[~raw["is_control"]]
    sample_values = (
        target.groupby(
            ["model", "dataset", "sample_id", "region_key", "perturbation"], as_index=False
        )["degradation"]
        .mean()
    )
    alpha = (1.0 - confidence_level) / 2.0
    rows: list[dict[str, Any]] = []
    for key, group in sample_values.groupby(
        ["model", "dataset", "region_key", "perturbation"], sort=True
    ):
        values = group["degradation"].to_numpy(dtype=np.float64)
        local_seed = int.from_bytes(
            hashlib.sha256(canonical_json((seed, key)).encode()).digest()[:8], "big"
        )
        rng = np.random.default_rng(local_seed)
        indices = rng.integers(0, len(values), size=(resamples, len(values)))
        means = values[indices].mean(axis=1)
        rows.append(
            {
                "model": key[0],
                "dataset": key[1],
                "region_key": key[2],
                "perturbation": key[3],
                "mean": float(values.mean()),
                "ci_low": float(np.quantile(means, alpha)),
                "ci_high": float(np.quantile(means, 1.0 - alpha)),
                "n_samples": len(values),
                "resamples": resamples,
            }
        )
    return pd.DataFrame(rows)


def _control_comparison(raw: pd.DataFrame, ratio_zero_threshold: float) -> pd.DataFrame:
    target = raw.loc[~raw["is_control"]].copy()
    controls = raw.loc[raw["is_control"]].copy()
    control_stats = (
        controls.groupby(
            [
                "model",
                "dataset",
                "sample_id",
                "target_region_key",
                "perturbation",
                "seed_salt",
            ],
            as_index=False,
        )["degradation"]
        .agg(control_mean="mean", control_std=lambda values: float(np.std(values, ddof=0)), n_controls="size")
    )
    joined = target.merge(
        control_stats,
        left_on=["model", "dataset", "sample_id", "region_key", "perturbation", "seed_salt"],
        right_on=[
            "model",
            "dataset",
            "sample_id",
            "target_region_key",
            "perturbation",
            "seed_salt",
        ],
        how="left",
        suffixes=("", "_control"),
    )
    joined["excess"] = joined["degradation"] - joined["control_mean"]
    joined["ratio"] = np.where(
        joined["control_mean"].abs() >= ratio_zero_threshold,
        joined["degradation"] / joined["control_mean"],
        np.nan,
    )
    joined["z_vs_control"] = np.where(
        (joined["n_controls"] >= 2) & (joined["control_std"] != 0),
        joined["excess"] / joined["control_std"],
        np.nan,
    )
    return joined[
        [
            "model",
            "dataset",
            "sample_id",
            "region_key",
            "perturbation",
            "seed_salt",
            "degradation",
            "control_mean",
            "control_std",
            "n_controls",
            "excess",
            "ratio",
            "z_vs_control",
        ]
    ]


def _seed_stability(raw: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model",
        "dataset",
        "sample_id",
        "region_key",
        "target_region_key",
        "is_control",
        "perturbation",
    ]
    result = raw.groupby(keys, as_index=False)["degradation"].agg(
        seed_mean="mean", seed_std=lambda values: float(np.std(values, ddof=0)), n_seeds="size"
    )
    result["seed_cv"] = np.where(
        result["seed_mean"].abs() > 1.0e-12,
        result["seed_std"] / result["seed_mean"].abs(),
        np.nan,
    )
    return result


def _operator_consistency(region: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, dataset), group in region.groupby(["model", "dataset"], sort=True):
        pivot = group.pivot(index="region_key", columns="perturbation", values="degradation")
        for left, right in combinations(sorted(pivot.columns), 2):
            pair = pivot[[left, right]].dropna()
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "operator_left": left,
                    "operator_right": right,
                    "spearman": float(pair[left].rank().corr(pair[right].rank())),
                    "top_region_agrees": bool(pair[left].idxmax() == pair[right].idxmax()),
                    "n_regions": len(pair),
                }
            )
    return pd.DataFrame(rows)


def _area_sanity(raw: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    target = raw.loc[~raw["is_control"]]
    source_areas = sorted(int(value) for value in target["source_area"].unique())
    model_areas = sorted(int(value) for value in target["model_area"].unique())
    source_shape = tuple(config["preprocessing"]["input_size"])
    output_shape = tuple(config["preprocessing"]["output_size"])
    expected_source = (source_shape[0] // config["regions"]["rows"]) * (
        source_shape[1] // config["regions"]["cols"]
    )
    expected_model = (output_shape[0] // config["regions"]["rows"]) * (
        output_shape[1] // config["regions"]["cols"]
    )
    passed = source_areas == [expected_source] and model_areas == [expected_model]
    return {
        "pass": passed,
        "source_areas": source_areas,
        "model_areas": model_areas,
        "expected_source_area": expected_source,
        "expected_model_area": expected_model,
        "preprocessing": "resize(224,224)->float->normalize->channels_first",
        "preprocessing_fingerprint": hashlib.sha256(
            canonical_json(config["preprocessing"]).encode()
        ).hexdigest(),
    }


def _verdicts(
    canonical_region: pd.DataFrame,
    accuracy: Mapping[str, Mapping[str, float]],
    thresholds: Mapping[str, float],
    patch_key: str,
) -> dict[str, Any]:
    def select(model: str, dataset: str, operator: str) -> pd.DataFrame:
        return canonical_region.loc[
            (canonical_region["model"] == model)
            & (canonical_region["dataset"] == dataset)
            & (canonical_region["perturbation"] == operator)
        ]

    baseline = select("shortcut", "A", "constant_fill")
    patch_rank = _rank(baseline, patch_key)
    patch_mean = float(baseline.loc[baseline["region_key"] == patch_key, "degradation"].iloc[0])
    other_mean = float(baseline.loc[baseline["region_key"] != patch_key, "degradation"].mean())
    multiplier = patch_mean / other_mean if other_mean != 0 else float("inf")
    normal_rank = _rank(select("normal", "A", "constant_fill"), patch_key)
    fills = sorted(canonical_region.loc[canonical_region["model"] == "shortcut", "perturbation"].unique())
    reproduced = [fill for fill in fills if _rank(select("shortcut", "A", fill), patch_key) == 1]
    shortcut_drop = accuracy["shortcut"]["A"] - accuracy["shortcut"]["C"]
    normal_drop = accuracy["normal"]["A"] - accuracy["normal"]["C"]
    margin_points = (shortcut_drop - normal_drop) * 100.0
    auxiliary = _rank(select("shortcut", "B", "constant_fill"), patch_key)
    return {
        "Q1_identifies_patch_region": {"pass": patch_rank == 1, "patch_region_rank": patch_rank},
        "Q2_separated_from_baseline": {
            "pass": multiplier >= thresholds["q2_multiplier"],
            "multiplier": multiplier,
            "threshold": thresholds["q2_multiplier"],
        },
        "Q3_distinguishes_normal_model": {
            "pass": normal_rank != 1,
            "patch_region_rank_in_m_normal": normal_rank,
        },
        "Q4_robust_to_fill_strategy": {
            "pass": len(reproduced) >= thresholds["q4_min_fill_strategies"],
            "reproduced_in": reproduced,
            "min_required": thresholds["q4_min_fill_strategies"],
        },
        "Q5_predicts_generalization_gap": {
            "pass": margin_points >= thresholds["q5_min_margin_points"],
            "shortcut_accuracy_drop_points": shortcut_drop * 100.0,
            "normal_accuracy_drop_points": normal_drop * 100.0,
            "margin_points": margin_points,
            "threshold_points": thresholds["q5_min_margin_points"],
        },
        "B_auxiliary_control": {"patch_region_rank": auxiliary},
    }


def _ssat_parity(
    config: Mapping[str, Any],
    captum_region: pd.DataFrame,
    captum_verdicts: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the independent result with the existing crop-free result files."""

    root = (
        Path(config["repo_root"])
        / "experiments"
        / "synthetic_shortcut"
        / "results_crop_free"
    )
    run_specs = {
        "shortcut_A_constant_fill": ("shortcut", "A", "constant_fill"),
        "shortcut_A_mean_fill": ("shortcut", "A", "mean_fill"),
        "shortcut_A_blur": ("shortcut", "A", "blur"),
        "shortcut_A_gaussian_noise": ("shortcut", "A", "gaussian_noise"),
        "shortcut_A_patch_shuffle": ("shortcut", "A", "patch_shuffle"),
        "normal_A_constant_fill": ("normal", "A", "constant_fill"),
        "shortcut_B_constant_fill": ("shortcut", "B", "constant_fill"),
    }
    missing = [
        str(root / f"region_metrics_{run_id}.csv")
        for run_id in run_specs
        if not (root / f"region_metrics_{run_id}.csv").is_file()
    ]
    if missing or not (root / "accuracy.json").is_file():
        return {"available": False, "missing": missing}

    reference_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    deterministic_runs = {
        "shortcut_A_constant_fill",
        "normal_A_constant_fill",
        "shortcut_B_constant_fill",
    }
    for run_id, (model, dataset, operator) in run_specs.items():
        reference = pd.read_csv(root / f"region_metrics_{run_id}.csv").rename(
            columns={"metric_mean": "reference_degradation"}
        )
        reference_frames.append(
            reference[["region_key", "reference_degradation"]]
            .rename(columns={"reference_degradation": "degradation"})
            .assign(model=model, dataset=dataset, perturbation=operator)
        )
        current = captum_region.loc[
            (captum_region["model"] == model)
            & (captum_region["dataset"] == dataset)
            & (captum_region["perturbation"] == operator),
            ["region_key", "degradation"],
        ]
        joined = current.merge(reference[["region_key", "reference_degradation"]], on="region_key")
        current_order = list(
            joined.sort_values(["degradation", "region_key"], ascending=[False, True])["region_key"]
        )
        reference_order = list(
            joined.sort_values(
                ["reference_degradation", "region_key"], ascending=[False, True]
            )["region_key"]
        )
        run_rows.append(
            {
                "run_id": run_id,
                "deterministic": run_id in deterministic_runs,
                "rank_order_identical": current_order == reference_order,
                "spearman": float(
                    joined["degradation"].rank().corr(joined["reference_degradation"].rank())
                ),
                "max_abs_degradation_difference": float(
                    (joined["degradation"] - joined["reference_degradation"]).abs().max()
                ),
            }
        )
    reference_region = pd.concat(reference_frames, ignore_index=True)
    reference_accuracy = json.loads((root / "accuracy.json").read_text(encoding="utf-8"))
    patch = region_key(int(config["regions"]["patch_row"]), int(config["regions"]["patch_col"]))
    reference_verdicts = _verdicts(
        reference_region, reference_accuracy, config["thresholds"], patch
    )
    question_keys = [key for key in captum_verdicts if key.startswith("Q")]
    return {
        "available": True,
        "all_q_pass_fail_match": all(
            bool(captum_verdicts[key]["pass"]) == bool(reference_verdicts[key]["pass"])
            for key in question_keys
        ),
        "q1_rank_match": (
            captum_verdicts["Q1_identifies_patch_region"]["patch_region_rank"]
            == reference_verdicts["Q1_identifies_patch_region"]["patch_region_rank"]
        ),
        "q3_rank_match": (
            captum_verdicts["Q3_distinguishes_normal_model"]["patch_region_rank_in_m_normal"]
            == reference_verdicts["Q3_distinguishes_normal_model"]["patch_region_rank_in_m_normal"]
        ),
        "q4_reproduced_fill_match": (
            captum_verdicts["Q4_robust_to_fill_strategy"]["reproduced_in"]
            == reference_verdicts["Q4_robust_to_fill_strategy"]["reproduced_in"]
        ),
        "q2_multiplier_abs_difference": abs(
            captum_verdicts["Q2_separated_from_baseline"]["multiplier"]
            - reference_verdicts["Q2_separated_from_baseline"]["multiplier"]
        ),
        "q5_margin_points_abs_difference": abs(
            captum_verdicts["Q5_predicts_generalization_gap"]["margin_points"]
            - reference_verdicts["Q5_predicts_generalization_gap"]["margin_points"]
        ),
        "runs": run_rows,
        "reference_verdicts": reference_verdicts,
    }


def count_python_sloc(paths: Iterable[Path]) -> int:
    """Count physical lines containing meaningful Python tokens."""

    lines: set[tuple[str, int]] = set()
    ignored = {
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.COMMENT,
    }
    for path in paths:
        source = path.read_bytes()
        for token in tokenize.tokenize(io.BytesIO(source).readline):
            if token.type not in ignored and token.string.strip():
                lines.add((str(path), token.start[0]))
    return len(lines)


def function_sloc(path: Path, names: Sequence[str]) -> int:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    ranges = [
        (node.lineno, node.end_lineno or node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    meaningful: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in {
            tokenize.ENDMARKER,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.COMMENT,
        }:
            continue
        if any(start <= token.start[0] <= end for start, end in ranges):
            meaningful.add(token.start[0])
    return len(meaningful)


def comparison_metrics(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    repo = Path(config["repo_root"])
    baseline_dir = Path(__file__).resolve().parent
    baseline_paths = sorted(
        path for path in baseline_dir.glob("*.py") if path.name != "__init__.py"
    )
    glue_paths = [
        repo / "experiments/synthetic_shortcut/run_audit.py",
        repo / "experiments/synthetic_shortcut/evaluate_accuracy.py",
        repo / "experiments/synthetic_shortcut/evaluate.py",
        repo / "experiments/synthetic_shortcut/run_threshold_validation_full.py",
        repo / "experiments/synthetic_shortcut/generate_report.py",
    ]
    run_audit_path = glue_paths[0]
    threshold_path = glue_paths[3]
    return {
        "loc_method": "physical lines containing non-comment Python tokens; tests and outputs excluded",
        "loc": {
            "captum_custom_workflow": count_python_sloc(baseline_paths),
            "ssat_existing_experiment_glue": count_python_sloc(glue_paths),
            "ssat_audit_config_builders": function_sloc(
                run_audit_path, ("_build_audit_config",)
            )
            + function_sloc(threshold_path, ("_build_config",)),
        },
        "semantic_execution_steps": {
            "captum_custom_workflow": 8,
            "captum_steps": [
                "prepare masks/baselines",
                "write evaluation loop",
                "serialize raw observations",
                "run matched controls/seeds",
                "aggregate levels",
                "compute bootstrap/stability",
                "validate preprocessing/area",
                "generate report",
            ],
            "ssat": 3,
            "ssat_steps": ["ssat run", "ssat analyze", "ssat report"],
        },
        "user_decisions": {
            "captum_custom_workflow": len(USER_DECISIONS),
            "ssat": 1,
            "items": list(USER_DECISIONS),
        },
        "capabilities": [
            {"capability": name, "captum_workflow": captum_value, "ssat": ssat_value}
            for name, captum_value, ssat_value in CAPABILITIES
        ],
        "reproducibility": {
            "single_config": True,
            "raw_results_preserved": True,
            "reanalyze_without_inference": True,
            "provenance_automatic": True,
            "canonical_hash": json.loads(
                (output_dir / "analysis" / "summary.json").read_text(encoding="utf-8")
            )["canonical_hash"],
            "raw_canonical_hash": json.loads(
                (output_dir / "analysis" / "summary.json").read_text(encoding="utf-8")
            )["raw_canonical_hash"],
        },
    }


def analyze(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Compute every comparison and reliability table from cached raw rows."""

    raw = load_raw(output_dir)
    expected_rows = 0
    manifest_sizes = {
        name: len(json.loads(Path(path).read_text(encoding="utf-8"))["samples"])
        for name, path in config["data"]["manifests"].items()
    }
    regions = int(config["regions"]["rows"]) * int(config["regions"]["cols"])
    controls = regions * int(config["regions"]["controls_per_region"])
    for run in config["audit_runs"]:
        sample_count = manifest_sizes[run["manifest"]]
        op_count = len(config["perturbations"]) if run["operators"] == "all" else len(run["operators"])
        seed_count = len(config["seed_salts"]) if run["operators"] == "all" else 1
        expected_rows += sample_count * (regions + (controls if run["controls"] else 0)) * op_count * seed_count
    if len(raw) != expected_rows:
        raise RuntimeError(f"audit is incomplete: expected {expected_rows} rows, found {len(raw)}")

    analysis_dir = output_dir / "analysis"
    target = raw.loc[~raw["is_control"]]
    sample = target.groupby(
        ["model", "dataset", "sample_id", "gt_label", "perturbation"], as_index=False
    )["degradation"].mean()
    region = target.groupby(
        ["model", "dataset", "region_key", "perturbation"], as_index=False
    )["degradation"].mean()
    class_level = target.groupby(
        ["model", "dataset", "gt_label", "perturbation"], as_index=False
    )["degradation"].mean()
    dataset_level = target.groupby(
        ["model", "dataset", "perturbation"], as_index=False
    )["degradation"].mean()
    canonical_region = (
        target.loc[target["seed_salt"] == 0]
        .groupby(["model", "dataset", "region_key", "perturbation"], as_index=False)[
            "degradation"
        ]
        .mean()
    )
    controls = _control_comparison(raw, float(config["thresholds"]["ratio_zero_threshold"]))
    seed_stability = _seed_stability(raw)
    intervals = _bootstrap_intervals(
        raw,
        seed=int(config["bootstrap_seed"]),
        resamples=int(config["bootstrap_resamples"]),
        confidence_level=float(config["confidence_level"]),
    )
    operator_consistency = _operator_consistency(region)
    tables = {
        "sample_metrics": sample,
        "region_metrics": region,
        "class_metrics": class_level,
        "dataset_metrics": dataset_level,
        "canonical_region_metrics": canonical_region,
        "control_comparison": controls,
        "seed_stability": seed_stability,
        "bootstrap_intervals": intervals,
        "operator_consistency": operator_consistency,
    }
    for name, frame in tables.items():
        _save_table(frame, analysis_dir / name)
    accuracy = json.loads((output_dir / "accuracy.json").read_text(encoding="utf-8"))
    patch = region_key(int(config["regions"]["patch_row"]), int(config["regions"]["patch_col"]))
    verdicts = _verdicts(canonical_region, accuracy, config["thresholds"], patch)
    area = _area_sanity(raw, config)
    hashes = {name: canonical_table_hash(frame) for name, frame in tables.items()}
    raw_canonical_hash = canonical_table_hash(raw)
    canonical_hash = hashlib.sha256(canonical_json(hashes).encode()).hexdigest()
    summary = {
        "schema_version": "captum-reference-analysis-v1",
        "raw_rows": len(raw),
        "expected_rows": expected_rows,
        "verdicts": verdicts,
        "ssat_parity": _ssat_parity(config, canonical_region, verdicts),
        "area_sanity": area,
        "table_hashes": hashes,
        "raw_canonical_hash": raw_canonical_hash,
        "canonical_hash": canonical_hash,
    }
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def render_report(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Write local Markdown/JSON/CSV comparison reports."""

    summary_path = output_dir / "analysis" / "summary.json"
    if not summary_path.is_file():
        analyze(config, output_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = comparison_metrics(config, output_dir)
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame(metrics["capabilities"]).to_csv(output_dir / "capability_comparison.csv", index=False)
    lines = [
        "# Captum Reference Workflow Comparison",
        "",
        "Captum supplies the region-ablation primitive. Dataset iteration, controls, repeated seeds, uncertainty, aggregation, caching, provenance, and reporting are custom workflow code.",
        "",
        "## Q1-Q5",
        "",
        "| Question | Result | Detail |",
        "|---|---|---|",
    ]
    for key, value in summary["verdicts"].items():
        if key == "B_auxiliary_control":
            continue
        detail = ", ".join(f"{name}={item}" for name, item in value.items() if name != "pass")
        lines.append(f"| {key} | **{'PASS' if value['pass'] else 'FAIL'}** | {detail} |")
    lines.extend(
        [
            "",
            "## Quantitative comparison",
            "",
            "| Measure | Captum custom workflow | SSAT |",
            "|---|---:|---:|",
            f"| User-authored Python SLOC | {metrics['loc']['captum_custom_workflow']} | {metrics['loc']['ssat_existing_experiment_glue']} experiment glue |",
            f"| Audit-config builder SLOC | included above | {metrics['loc']['ssat_audit_config_builders']} |",
            f"| Semantic execution steps | {metrics['semantic_execution_steps']['captum_custom_workflow']} | {metrics['semantic_execution_steps']['ssat']} |",
            f"| Explicit workflow decisions | {metrics['user_decisions']['captum_custom_workflow']} | {metrics['user_decisions']['ssat']} |",
            "",
            "## Capability matrix",
            "",
            "| Capability | Captum custom workflow | SSAT |",
            "|---|---|---|",
        ]
    )
    lines.extend(
        f"| {row['capability']} | {row['captum_workflow']} | {row['ssat']} |"
        for row in metrics["capabilities"]
    )
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Raw rows: {summary['raw_rows']}",
            f"- Raw canonical hash: `{summary['raw_canonical_hash']}`",
            f"- Canonical analysis hash: `{summary['canonical_hash']}`",
            f"- Area sanity: **{'PASS' if summary['area_sanity']['pass'] else 'FAIL'}** (source={summary['area_sanity']['source_areas']}, model={summary['area_sanity']['model_areas']})",
            "- A completed output can be reanalyzed without model inference; cache identity binds config, manifests, checkpoints, and dataset statistics.",
            "",
        ]
    )
    parity = summary.get("ssat_parity", {"available": False})
    lines.extend(["## SSAT result parity", ""])
    if parity["available"]:
        lines.extend(
            [
                f"- Q1-Q5 pass/fail match: **{parity['all_q_pass_fail_match']}**",
                f"- Q1 patch rank match: **{parity['q1_rank_match']}**",
                f"- Q3 normal-model patch rank match: **{parity['q3_rank_match']}**",
                f"- Q4 reproduced-fill set match: **{parity['q4_reproduced_fill_match']}**",
                f"- Q2 multiplier absolute difference: {parity['q2_multiplier_abs_difference']:.6g}",
                f"- Q5 margin absolute difference: {parity['q5_margin_points_abs_difference']:.6g} points",
                "",
                "| Run | Deterministic | Identical ranking | Spearman | Max absolute degradation difference |",
                "|---|---|---|---:|---:|",
            ]
        )
        lines.extend(
            f"| {row['run_id']} | {row['deterministic']} | {row['rank_order_identical']} | {row['spearman']:.6f} | {row['max_abs_degradation_difference']:.6g} |"
            for row in parity["runs"]
        )
    else:
        lines.append("Reference result files were unavailable; parity was not computed.")
    lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"summary": summary, "comparison": metrics}
