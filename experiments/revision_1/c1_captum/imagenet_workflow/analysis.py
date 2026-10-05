"""Aggregation, control comparison, uncertainty, and reporting for the ImageNet audit."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:  # Support both ``python run.py`` and package-style test imports.
    from .workflow import canonical_json, load_raw, perturbation_variants
except ImportError:  # pragma: no cover - exercised by the documented script entrypoint
    from workflow import canonical_json, load_raw, perturbation_variants


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


def _bootstrap_intervals(
    raw: pd.DataFrame,
    *,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> pd.DataFrame:
    target = raw.loc[~raw["is_control"]]
    sample_values = (
        target.groupby(["sample_id", "region_key", "perturbation"], as_index=False)["degradation"].mean()
    )
    alpha = (1.0 - confidence_level) / 2.0
    rows: list[dict[str, Any]] = []
    for key, group in sample_values.groupby(["region_key", "perturbation"], sort=True):
        values = group["degradation"].to_numpy(dtype=np.float64)
        local_seed = int.from_bytes(
            hashlib.sha256(canonical_json((seed, key)).encode()).digest()[:8], "big"
        )
        rng = np.random.default_rng(local_seed)
        indices = rng.integers(0, len(values), size=(resamples, len(values)))
        means = values[indices].mean(axis=1)
        low, high = float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))
        rows.append(
            {
                "region_key": key[0],
                "perturbation": key[1],
                "mean": float(values.mean()),
                "ci_low": low,
                "ci_high": high,
                "excludes_zero": bool(low > 0.0 or high < 0.0),
                "n_samples": len(values),
                "resamples": resamples,
            }
        )
    return pd.DataFrame(rows)


def _control_comparison(raw: pd.DataFrame, ratio_zero_threshold: float) -> pd.DataFrame:
    target = raw.loc[~raw["is_control"]].copy()
    controls = raw.loc[raw["is_control"]].copy()
    keys = ["sample_id", "target_region_key", "perturbation", "seed_salt"]
    control_stats = controls.groupby(keys, as_index=False)["degradation"].agg(
        control_mean="mean", control_std=lambda values: float(np.std(values, ddof=0)), n_controls="size"
    )
    joined = target.merge(
        control_stats,
        left_on=["sample_id", "region_key", "perturbation", "seed_salt"],
        right_on=keys,
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


def _control_summary(controls: pd.DataFrame, z_threshold: float) -> pd.DataFrame:
    """Per region and operator: mean excess and the share of target rows with z above the threshold."""

    frame = controls.assign(exceeds=controls["z_vs_control"] > z_threshold, z_defined=controls["z_vs_control"].notna())
    return frame.groupby(["region_key", "perturbation"], as_index=False).agg(
        mean_excess=("excess", "mean"),
        share_z_above_threshold=("exceeds", "mean"),
        share_z_defined=("z_defined", "mean"),
        rows=("excess", "size"),
    )


def _seed_stability(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["sample_id", "region_key", "target_region_key", "is_control", "perturbation"]
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
    pivot = region.pivot(index="region_key", columns="perturbation", values="degradation")
    for left, right in combinations(sorted(pivot.columns), 2):
        pair = pivot[[left, right]].dropna()
        rows.append(
            {
                "operator_left": left,
                "operator_right": right,
                "spearman": float(pair[left].rank().corr(pair[right].rank())),
                "top_region_agrees": bool(pair[left].idxmax() == pair[right].idxmax()),
                "n_regions": len(pair),
            }
        )
    return pd.DataFrame(rows)


def _region_profile(target: pd.DataFrame) -> pd.DataFrame:
    """Dataset-level mean degradation per region, per operator and pooled over all variants, with ranks."""

    per_operator = target.groupby(["region_key", "perturbation"], as_index=False)["degradation"].mean()
    pooled = target.groupby("region_key", as_index=False)["degradation"].mean().assign(perturbation="all")
    profile = pd.concat([per_operator, pooled], ignore_index=True)
    profile["rank"] = (
        profile.sort_values(["degradation", "region_key"], ascending=[False, True])
        .groupby("perturbation")
        .cumcount()
        .add(1)
        .reindex(profile.index)
    )
    return profile


def _top_region_share(target: pd.DataFrame) -> pd.DataFrame:
    """Share of samples whose largest variant-pooled degradation is in each region."""

    per_sample = target.groupby(["sample_id", "region_key"], as_index=False)["degradation"].mean()
    top = per_sample.sort_values(["degradation", "region_key"], ascending=[False, True]).drop_duplicates("sample_id")
    counts = top["region_key"].value_counts()
    return pd.DataFrame(
        {"region_key": counts.index, "samples": counts.to_numpy(), "share": (counts / len(top)).to_numpy()}
    )


def _area_sanity(raw: pd.DataFrame) -> dict[str, Any]:
    """Controls keep their target's source area; model-space areas of target cells are summarized."""

    target_area = (
        raw.loc[~raw["is_control"], ["sample_id", "region_key", "source_area"]]
        .drop_duplicates()
        .rename(columns={"region_key": "target_region_key", "source_area": "target_source_area"})
    )
    controls = raw.loc[raw["is_control"], ["sample_id", "target_region_key", "source_area", "model_area"]]
    joined = controls.merge(target_area, on=["sample_id", "target_region_key"], how="left")
    target_model = raw.loc[~raw["is_control"]].drop_duplicates(["sample_id", "region_key"])["model_area"]
    matched = bool((joined["source_area"] == joined["target_source_area"]).all())
    return {
        "pass": matched,
        "controls_match_target_source_area": matched,
        "target_model_area": {
            "min": int(target_model.min()),
            "median": float(target_model.median()),
            "max": int(target_model.max()),
        },
        "control_model_area": {
            "min": int(controls["model_area"].min()),
            "median": float(controls["model_area"].median()),
            "max": int(controls["model_area"].max()),
        },
        "model_area_definition": "nearest-neighbour count of 224x224 model-input pixels mapped from the source rectangle",
    }


def _findings(
    profile: pd.DataFrame,
    top_share: pd.DataFrame,
    control_summary: pd.DataFrame,
    intervals: pd.DataFrame,
    operator_consistency: pd.DataFrame,
    accuracy: Mapping[str, Any],
) -> dict[str, Any]:
    pooled = profile.loc[profile["perturbation"] == "all"].sort_values("rank")
    return {
        "clean_top1": accuracy["top1"],
        "pooled_region_order": pooled["region_key"].tolist(),
        "pooled_top_region": pooled["region_key"].iloc[0],
        "top_region_share": dict(zip(top_share["region_key"], top_share["share"].round(6))),
        "share_z_above_threshold_by_operator": (
            control_summary.groupby("perturbation")["share_z_above_threshold"].mean().round(6).to_dict()
        ),
        "regions_with_ci_excluding_zero": int(intervals["excludes_zero"].sum()),
        "region_operator_rows": int(len(intervals)),
        "min_operator_spearman": float(operator_consistency["spearman"].min()),
    }


def analyze(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Compute every aggregation and control table from cached raw rows."""

    raw = load_raw(output_dir)
    samples = sum(1 for line in Path(config["data"]["annotation"]).read_text().splitlines() if line.strip())
    regions = int(config["regions"]["rows"]) * int(config["regions"]["cols"])
    controls_per_region = int(config["regions"]["controls_per_region"])
    expected_rows = samples * regions * (1 + controls_per_region) * len(perturbation_variants(config))
    if len(raw) != expected_rows:
        raise RuntimeError(f"audit is incomplete: expected {expected_rows} rows, found {len(raw)}")

    analysis_dir = output_dir / "analysis"
    target = raw.loc[~raw["is_control"]]
    sample = target.groupby(["sample_id", "gt_label", "perturbation"], as_index=False)["degradation"].mean()
    region = target.groupby(["region_key", "perturbation"], as_index=False)["degradation"].mean()
    class_level = target.groupby(["gt_label", "perturbation"], as_index=False)["degradation"].mean()
    dataset_level = target.groupby("perturbation", as_index=False)["degradation"].mean()
    profile = _region_profile(target)
    top_share = _top_region_share(target)
    controls = _control_comparison(raw, float(config["thresholds"]["ratio_zero_threshold"]))
    control_summary = _control_summary(controls, float(config["thresholds"]["z_vs_control"]))
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
        "region_profile": profile,
        "top_region_share": top_share,
        "control_comparison": controls,
        "control_summary": control_summary,
        "seed_stability": seed_stability,
        "bootstrap_intervals": intervals,
        "operator_consistency": operator_consistency,
    }
    for name, frame in tables.items():
        _save_table(frame, analysis_dir / name)
    accuracy = json.loads((output_dir / "accuracy.json").read_text(encoding="utf-8"))
    hashes = {name: canonical_table_hash(frame) for name, frame in tables.items()}
    raw_canonical_hash = canonical_table_hash(raw)
    summary = {
        "schema_version": "captum-reference-imagenet-analysis-v1",
        "raw_rows": len(raw),
        "expected_rows": expected_rows,
        "findings": _findings(profile, top_share, control_summary, intervals, operator_consistency, accuracy),
        "area_sanity": _area_sanity(raw),
        "table_hashes": hashes,
        "raw_canonical_hash": raw_canonical_hash,
        "canonical_hash": hashlib.sha256(canonical_json(hashes).encode()).hexdigest(),
    }
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def render_report(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Write a local Markdown report."""

    summary_path = output_dir / "analysis" / "summary.json"
    if not summary_path.is_file():
        analyze(config, output_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    findings = summary["findings"]
    profile = pd.read_parquet(output_dir / "analysis" / "region_profile.parquet")
    pivot = profile.pivot(index="region_key", columns="perturbation", values="degradation").sort_index()
    lines = [
        "# Captum Reference Workflow: ImageNet Region Audit",
        "",
        "Captum supplies the region-ablation primitive. Data iteration, controls, repeated seeds, uncertainty, aggregation, caching, provenance, and reporting are custom workflow code.",
        "",
        f"- Model: `{config['model']['timm_name']}`, crop-free squash to {config['preprocessing']['output_size']}",
        f"- Clean top-1 on the audited samples: {findings['clean_top1']:.4f}",
        f"- Pooled top region: `{findings['pooled_top_region']}`",
        f"- Region x operator rows whose bootstrap CI excludes zero: {findings['regions_with_ci_excluding_zero']} / {findings['region_operator_rows']}",
        f"- Minimum operator-pair Spearman of the region profile: {findings['min_operator_spearman']:.3f}",
        "",
        "## Mean margin drop by region",
        "",
        "| Region | " + " | ".join(pivot.columns) + " |",
        "|---|" + "---:|" * len(pivot.columns),
    ]
    lines.extend(
        f"| {region} | " + " | ".join(f"{value:.4f}" for value in row) + " |"
        for region, row in zip(pivot.index, pivot.to_numpy())
    )
    lines.extend(["", "## Share of target rows with z above the control threshold", ""])
    lines.extend(f"- {operator}: {share:.4f}" for operator, share in findings["share_z_above_threshold_by_operator"].items())
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Raw rows: {summary['raw_rows']}",
            f"- Raw canonical hash: `{summary['raw_canonical_hash']}`",
            f"- Canonical analysis hash: `{summary['canonical_hash']}`",
            f"- Area sanity: **{'PASS' if summary['area_sanity']['pass'] else 'FAIL'}**",
            "- A completed output can be reanalyzed without model inference; cache identity binds the config and the annotation file.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"summary": summary}
