"""C1-std summaries from ``results/c1_std/measurements.jsonl`` and the kept outputs.

Writes to ``summary_std/``:
- ``measurements.csv``, ``runs.csv``, ``resources.csv``, ``storage.csv``:
  as in C1-min (``summarize.py``);
- ``parity_items.csv``, ``parity_profile.csv``, ``parity.json``: Captum vs
  SSAT results on the same samples (protocol_std.json ``parity``);
- ``verification.json``: completeness and repeat-identity checks, and the
  informational identity check against the stored 10k crop-free run;
- ``engineering.json``: quantitative indicators (SLOC of the Captum
  ImageNet workflow, lines changed from the synthetic reference, the SSAT
  config change, commands) and the qualitative capability table. Setup time
  is not measured (protocol_std.json decisions.setup_time).

Example:
    python experiments/revision_1/c1_captum/summarize_std.py
"""

from __future__ import annotations

import argparse
import difflib
import io
import json
import re
import tokenize
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.revision_1.c1_captum import summarize
from experiments.revision_1.c1_captum.measure_std import C1_STD_RESULTS_DIR, CAPTUM_WORKFLOW, SSAT_CONFIG, workflow_steps
from experiments.revision_1.common.agreement import spearman
from experiments.revision_1.common.loading import load_clean, load_item_values
from experiments.revision_1.common.provenance import write_provenance
from experiments.revision_1.common.runs import CASE_STUDY_DIR, REPO_ROOT, RunRef, get_run

C1_DIR = Path(__file__).resolve().parent
SUMMARY_DIR = C1_DIR / "summary_std"
PROTOCOL = C1_DIR / "protocol_std.json"
SYNTHETIC_CAPTUM = REPO_ROOT / "experiments" / "reference_comparison" / "captum_baseline"
CASE_STUDY_CONFIG = CASE_STUDY_DIR / "configs" / "imagenet_mnv2_050_crop_free.yaml"
BASELINE_RUN = "imagenet_mnv2_050_crop_free_k3"
DETERMINISTIC_OPERATORS = ("mean_fill", "blur")
CONTENT = {
    ("captum", "raw"): "raw observations: one row per item (ids, clean and perturbed margin, degradation, areas, status)",
    ("captum", "analysis"): "analysis tables (sample/region/class/dataset, region profile, controls, seed stability, bootstrap CI, operator consistency; Parquet + CSV)",
    ("captum", "files"): "run manifest, provenance, accuracy, report.md",
    ("ssat", "clean"): "clean logits (1,000 classes) per sample",
    ("ssat", "perturbed"): "per-item logits (1,000 classes) with region/operator/seed/area metadata",
    ("ssat", "index"): "item index",
    ("ssat", "metrics"): "per-item metrics and sample/region aggregates",
    ("ssat", "analysis"): "reliability grades, controls, seed and strategy stability, intervals, rank correlation",
    ("ssat", "report"): "HTML report with data CSVs and figures",
    ("ssat", "other"): "run manifest with resolved config, logs, other manifests",
}
_CELL = re.compile(r"/r(\d+)/c(\d+)$")


def ssat_run(output: Path) -> RunRef:
    return RunRef.co_located("c1_std_ssat", output, dataset="imagenet", model="mobilenetv2_050",
                             protocol="crop_free", n_controls=3)


def perturbed_items(workflow: str, output: Path) -> int:
    """Perturbed model evaluations: Captum raw rows, or SSAT dump items minus clean rows."""

    if workflow == "captum":
        return int(json.loads((output / "analysis" / "summary.json").read_text(encoding="utf-8"))["raw_rows"])
    counts = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))["counts_by_status"]
    return sum(counts.values()) - len(load_clean(ssat_run(output)))


def run_rows(records: Sequence[Mapping[str, Any]], steps: pd.DataFrame) -> pd.DataFrame:
    """One row per workflow x repeat (same columns as C1-min)."""

    rows = []
    for record in records:
        own = steps[(steps["workflow"] == record["workflow"]) & (steps["repeat"] == record["repeat"])]
        audit = own[own["stage"] == "audit"]
        items = perturbed_items(record["workflow"], Path(record["output"]))
        total_s = float(own["elapsed_s"].sum())
        gpu_peak = float(own["gpu_peak_mib"].max())
        rows.append({
            "workflow": record["workflow"], "repeat": record["repeat"], "perturbed_items": items,
            "total_s": total_s, "audit_s": float(audit["elapsed_s"].sum()),
            "analysis_report_s": float(own.loc[own["stage"] == "analysis_report", "elapsed_s"].sum()),
            "ms_per_item": 1000 * total_s / items, "audit_ms_per_item": 1000 * float(audit["elapsed_s"].sum()) / items,
            "peak_rss_gib": float(own["peak_rss_gib"].max()), "tree_rss_peak_gib": float(own["tree_rss_peak_gib"].max()),
            "gpu_idle_mib": record["gpu_idle_mib"], "gpu_peak_mib": gpu_peak, "gpu_above_idle_mib": gpu_peak - record["gpu_idle_mib"],
            "audit_gpu_util_mean": float((audit["gpu_util_mean"] * audit["elapsed_s"]).sum() / audit["elapsed_s"].sum()),
            "total_bytes": record["bytes"]["total"],
        })
    return pd.DataFrame(rows).sort_values(["workflow", "repeat"])


def storage_rows(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = summarize.storage_rows(records)
    frame["content"] = [CONTENT.get((workflow, entry), "") for workflow, entry in zip(frame["workflow"], frame["entry"])]
    return frame


# --- Captum vs SSAT results -------------------------------------------------------------------------------------


def captum_targets(output: Path) -> pd.DataFrame:
    """Captum target rows, seed-averaged: ``sample_id, row, col, perturbation, degradation``."""

    raw = pd.concat([pd.read_parquet(path) for path in sorted((output / "raw").glob("part-*.parquet"))], ignore_index=True)
    target = raw.loc[~raw["is_control"]]
    cells = target["region_key"].str.extract(_CELL).astype(int)
    frame = target.assign(row=cells[0], col=cells[1])
    return frame.groupby(["sample_id", "row", "col", "perturbation"], as_index=False)["degradation"].mean()


def captum_clean(output: Path) -> pd.DataFrame:
    raw = pd.read_parquet(output / "raw", columns=["sample_id", "clean_margin", "clean_correct"])
    return raw.drop_duplicates("sample_id").reset_index(drop=True)


def ssat_targets(output: Path) -> pd.DataFrame:
    """SSAT margin_drop target rows, seed-averaged, in the same layout as ``captum_targets``."""

    values = load_item_values(ssat_run(output), ("margin_drop",))
    target = values.loc[~values["is_control"].astype(bool) & values["available"].astype(bool)]
    params = target["region_params_json"].map(json.loads)
    frame = target.assign(row=params.map(lambda p: int(p["row_index"])), col=params.map(lambda p: int(p["col_index"])),
                          perturbation=target["perturb_op"])
    return frame.groupby(["sample_id", "row", "col", "perturbation"], as_index=False)["degradation"].mean()


def ssat_clean(output: Path) -> pd.DataFrame:
    """Clean margin and top-1 correctness per sample from the stored clean logits."""

    clean = load_clean(ssat_run(output), with_logits=True)
    rows = []
    for sample_id, label, logits in zip(clean["sample_id"], clean["gt_label"], clean["logits"]):
        logits = np.asarray(logits, dtype=np.float64)
        others = np.delete(logits, int(label))
        rows.append({"sample_id": sample_id, "clean_margin": float(logits[int(label)] - others.max()),
                     "clean_correct": bool(int(np.argmax(logits)) == int(label))})
    return pd.DataFrame(rows)


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def item_parity(captum: pd.DataFrame, ssat: pd.DataFrame) -> pd.DataFrame:
    """Per operator: agreement of the (sample, cell) seed-mean degradations."""

    joined = captum.merge(ssat, on=["sample_id", "row", "col", "perturbation"], suffixes=("_captum", "_ssat"))
    rows = []
    for operator, group in joined.groupby("perturbation", sort=True):
        a, b = group["degradation_captum"].to_numpy(), group["degradation_ssat"].to_numpy()
        diff = np.abs(a - b)
        rows.append({"perturbation": operator, "deterministic": operator in DETERMINISTIC_OPERATORS, "n": len(group),
                     "pearson": _pearson(a, b), "spearman": spearman(a, b), "max_abs_diff": float(diff.max()),
                     "median_abs_diff": float(np.median(diff)), "mean_diff": float((a - b).mean())})
    return pd.DataFrame(rows)


def _profile(frame: pd.DataFrame) -> pd.DataFrame:
    """16-cell dataset mean per operator and pooled over operators (``all``), weighted as the workflows weight items."""

    per_operator = frame.groupby(["perturbation", "row", "col"], as_index=False)["degradation"].mean()
    pooled = per_operator.groupby(["row", "col"], as_index=False)["degradation"].mean().assign(perturbation="all")
    return pd.concat([per_operator, pooled], ignore_index=True)


def profile_parity(captum: pd.DataFrame, ssat: pd.DataFrame) -> pd.DataFrame:
    """Per operator and pooled: dataset-level 16-cell profile agreement and per-sample top-cell agreement."""

    a, b = _profile(captum), _profile(ssat)
    rows = []
    for operator in sorted(a["perturbation"].unique()):
        left = a[a["perturbation"] == operator].set_index(["row", "col"])["degradation"]
        right = b[b["perturbation"] == operator].set_index(["row", "col"])["degradation"].reindex(left.index)
        order = lambda s: list(s.sort_values(ascending=False, kind="stable").index)  # noqa: E731
        rows.append({"perturbation": operator, "deterministic": operator in DETERMINISTIC_OPERATORS,
                     "profile_spearman": spearman(left.to_numpy(), right.to_numpy()),
                     "rank_order_identical": order(left) == order(right),
                     "top_cell_identical": order(left)[0] == order(right)[0],
                     "max_abs_cell_mean_diff": float((left - right).abs().max()),
                     "top_cell_agreement_per_sample": _top_cell_agreement(captum, ssat, operator)})
    return pd.DataFrame(rows)


def _top_cell_agreement(captum: pd.DataFrame, ssat: pd.DataFrame, operator: str) -> float:
    def top(frame: pd.DataFrame) -> pd.Series:
        own = frame if operator == "all" else frame[frame["perturbation"] == operator]
        per_cell = own.groupby(["sample_id", "row", "col"], as_index=False)["degradation"].mean()
        best = per_cell.sort_values(["degradation", "row", "col"], ascending=[False, True, True]).drop_duplicates("sample_id")
        return best.set_index("sample_id")[["row", "col"]].apply(tuple, axis=1)

    left, right = top(captum), top(ssat)
    common = left.index.intersection(right.index)
    return float((left[common] == right[common]).mean())


def clean_parity(captum: pd.DataFrame, ssat: pd.DataFrame) -> dict[str, Any]:
    joined = captum.merge(ssat, on="sample_id", suffixes=("_captum", "_ssat"))
    diff = (joined["clean_margin_captum"] - joined["clean_margin_ssat"]).abs()
    return {"samples": int(len(joined)), "top1_captum": float(joined["clean_correct_captum"].mean()),
            "top1_ssat": float(joined["clean_correct_ssat"].mean()),
            "correctness_agreement": float((joined["clean_correct_captum"] == joined["clean_correct_ssat"]).mean()),
            "clean_margin_max_abs_diff": float(diff.max()), "clean_margin_median_abs_diff": float(diff.median())}


def control_summary(captum_output: Path, ssat_output: Path, z_threshold: float) -> dict[str, Any]:
    """Informational: share of target rows with z above the threshold in each workflow (placements differ by design)."""

    captum = pd.read_parquet(captum_output / "analysis" / "control_comparison.parquet")
    ssat = pd.read_parquet(ssat_output / "analysis" / "control_comparison.parquet")
    ssat = ssat[ssat["metric_name"] == "margin_drop"]
    return {
        "captum_unit": "target row per (sample, cell, operator, seed); controls per seed",
        "ssat_unit": "target anchor per (sample, cell, condition); seed-averaged anchors",
        "captum_share_z_above": float((captum["z_vs_control"] > z_threshold).mean()),
        "ssat_share_z_above": float((ssat["z_vs_control"].astype(float) > z_threshold).mean()),
    }


def parity(captum_output: Path, ssat_output: Path, z_threshold: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """All Captum-vs-SSAT comparisons for one pair of outputs."""

    captum, ssat = captum_targets(captum_output), ssat_targets(ssat_output)
    items, profile = item_parity(captum, ssat), profile_parity(captum, ssat)
    deterministic = profile[profile["perturbation"].isin(DETERMINISTIC_OPERATORS)]
    document = {
        "captum_output": str(captum_output), "ssat_output": str(ssat_output),
        "primary_deterministic_rank_order_identical": bool(deterministic["rank_order_identical"].all()),
        "clean": clean_parity(captum_clean(captum_output), ssat_clean(ssat_output)),
        "controls_informational": control_summary(captum_output, ssat_output, z_threshold),
        "items": items.to_dict(orient="records"), "profile": profile.to_dict(orient="records"),
    }
    return items, profile, document


# --- verification -----------------------------------------------------------------------------------------------


def grade_counts(analysis_dir: Path) -> dict[str, int]:
    frame = pd.read_parquet(analysis_dir / "reliability.parquet", columns=["metric_name", "reliability_grade"])
    return dict(sorted(Counter(map(str, frame.loc[frame["metric_name"] == "margin_drop", "reliability_grade"])).items()))


def check_captum(output: Path, expected_rows: int) -> dict[str, Any]:
    summary = json.loads((output / "analysis" / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    checks = {"complete": manifest["status"] == "complete", "raw_rows": summary["raw_rows"] == expected_rows,
              "area_sanity": bool(summary["area_sanity"]["pass"])}
    return {"checks": checks, "pass": all(checks.values()), "raw_canonical_hash": summary["raw_canonical_hash"],
            "analysis_canonical_hash": summary["canonical_hash"]}


def check_ssat(output: Path, expected_items: int) -> dict[str, Any]:
    counts = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))["counts_by_status"]
    failed = {status: count for status, count in counts.items() if status != "ok" and count}
    checks = {"no_failed_items": not failed, "perturbed_items": perturbed_items("ssat", output) == expected_items}
    return {"checks": checks, "pass": all(checks.values()), "counts_by_status": counts,
            "grade_counts": grade_counts(output / "analysis")}


def baseline_identity(ssat_output: Path, baseline: RunRef) -> dict[str, Any]:
    """Informational: the fresh items equal the stored 10k run's items for the same samples (deviations.md D-001)."""

    fresh = load_item_values(ssat_run(ssat_output), ("margin_drop",))
    stored = load_item_values(baseline, ("margin_drop",))
    stored = stored[stored["sample_id"].isin(set(fresh["sample_id"]))]
    joined = fresh.merge(stored, on="item_id", suffixes=("_fresh", "_stored"))
    diff = (joined["degradation_fresh"] - joined["degradation_stored"]).abs()
    return {"baseline": baseline.name, "fresh_items": int(len(fresh)), "stored_items_same_samples": int(len(stored)),
            "item_id_sets_equal": set(fresh["item_id"]) == set(stored["item_id"]),
            "margin_drop_max_abs_diff": float(diff.max()) if len(diff) else None,
            "margin_drop_bitwise_equal_share": float((diff == 0).mean()) if len(diff) else None}


def verification(records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], *,
                 baseline: RunRef | None) -> dict[str, Any]:
    expected = protocol["setting"]
    runs: dict[str, Any] = {}
    for record in sorted(records, key=lambda r: (r["workflow"], r["repeat"])):
        output = Path(record["output"])
        result = (check_captum(output, expected["captum_raw_rows"]) if record["workflow"] == "captum"
                  else check_ssat(output, expected["ssat_perturbed_items"]))
        result["checks"]["returncodes_zero"] = all(step["returncode"] == 0 for step in record["steps"])
        result["pass"] = all(result["checks"].values())
        runs[f"{record['workflow']}_r{record['repeat']}"] = result
    captum = [run for key, run in runs.items() if key.startswith("captum")]
    ssat = [run for key, run in runs.items() if key.startswith("ssat")]
    repeats = {
        "captum_raw_hash_identical": len({run["raw_canonical_hash"] for run in captum}) == 1,
        "captum_analysis_hash_identical": len({run["analysis_canonical_hash"] for run in captum}) == 1,
        "ssat_counts_identical": len({json.dumps(run["counts_by_status"], sort_keys=True) for run in ssat}) == 1,
        "ssat_grade_counts_identical": len({json.dumps(run["grade_counts"], sort_keys=True) for run in ssat}) == 1,
    }
    document = {"all_pass": all(run["pass"] for run in runs.values()) and all(repeats.values()),
                "runs": runs, "repeats": repeats}
    if baseline is not None:
        first_ssat = next(Path(r["output"]) for r in sorted(records, key=lambda r: r["repeat"]) if r["workflow"] == "ssat")
        document["baseline_identity_informational"] = baseline_identity(first_ssat, baseline)
    return document


# --- engineering ------------------------------------------------------------------------------------------------


def meaningful_lines(path: Path) -> list[str]:
    """Stripped source lines that contain non-comment Python tokens (the ``count_python_sloc`` rule)."""

    ignored = {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE,
               tokenize.NL, tokenize.COMMENT}
    source = path.read_bytes()
    numbers = sorted({token.start[0] for token in tokenize.tokenize(io.BytesIO(source).readline)
                      if token.type not in ignored and token.string.strip()})
    lines = source.decode("utf-8").splitlines()
    return [lines[number - 1].strip() for number in numbers]


def config_lines(path: Path) -> list[str]:
    """Non-blank, non-comment YAML lines."""

    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def line_changes(old: Sequence[str], new: Sequence[str]) -> dict[str, int]:
    """Lines of ``new`` that are added or changed, and lines of ``old`` removed, by ``difflib`` opcodes."""

    added = removed = unchanged = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=list(old), b=list(new), autojunk=False).get_opcodes():
        if tag == "equal":
            unchanged += j2 - j1
        else:
            added += j2 - j1
            removed += i2 - i1
    return {"old": len(old), "new": len(new), "unchanged": unchanged, "added_or_changed": added, "removed": removed}


CAPABILITIES = (
    ("model wrapper: timm model, squash resize, margin output", "custom", "built in (timm adapter, geometry_mode)"),
    ("dataset iteration: file list, variable image sizes", "custom", "built in (imagenet source)"),
    ("region mask generation: grid cells", "custom", "built in"),
    ("perturbation operators and seeds", "custom (full-frame candidates)", "built in"),
    ("matched random controls", "custom", "built in"),
    ("control normalization: excess, z", "custom", "built in"),
    ("seed-repeat aggregation", "custom", "built in"),
    ("bootstrap uncertainty", "custom", "built in"),
    ("operator consistency", "custom", "built in"),
    ("sample / region / class / dataset aggregation", "custom", "built in"),
    ("preprocessing validation", "custom (check against timm data config)", "built in (preprocessing fingerprint)"),
    ("model-space region area", "custom", "built in (effective_area_px)"),
    ("per-item raw storage", "custom (one scalar row per item)", "built in (full logits)"),
    ("resolved configuration and provenance", "custom", "automatic"),
    ("cache identity and resume", "custom", "built in"),
    ("reliability grades", "not implemented", "built in"),
    ("report", "custom (Markdown)", "built in (HTML)"),
)
"""Qualitative indicator: how each capability is obtained in this setting (protocol_std.json reported_measures)."""


def engineering() -> dict[str, Any]:
    files = ("workflow.py", "analysis.py", "run.py")
    per_file = {name: line_changes(meaningful_lines(SYNTHETIC_CAPTUM / name), meaningful_lines(CAPTUM_WORKFLOW / name))
                for name in files}
    total = {key: sum(change[key] for change in per_file.values()) for key in next(iter(per_file.values()))}
    return {
        "loc_method": "physical lines containing non-comment Python tokens (count_python_sloc); "
                      "changes by difflib over those lines",
        "captum_imagenet_workflow_sloc": total["new"],
        "captum_synthetic_reference_sloc": total["old"],
        "captum_changes_from_synthetic_reference": {"total": total, "per_file": per_file},
        "ssat_config_lines": len(config_lines(SSAT_CONFIG)),
        "ssat_config_changes_from_case_study": line_changes(config_lines(CASE_STUDY_CONFIG), config_lines(SSAT_CONFIG)),
        "shared_input_script_sloc": len(meaningful_lines(C1_DIR / "make_inputs_std.py")),
        "commands": {workflow: len(workflow_steps(workflow, Path("<out>"))) for workflow in ("captum", "ssat")},
        "capabilities": [{"capability": name, "captum_workflow": captum, "ssat": ssat} for name, captum, ssat in CAPABILITIES],
        "setup_time": "not measured (protocol_std.json decisions.setup_time)",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-dir", type=Path, default=C1_STD_RESULTS_DIR)
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    parser.add_argument("--skip-baseline-check", action="store_true",
                        help="skip the informational identity check against the stored 10k crop-free run")
    args = parser.parse_args(argv)
    measurements = args.results_dir / "measurements.jsonl"
    records = summarize.load_records(measurements)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    summary = args.summary_dir
    summary.mkdir(parents=True, exist_ok=True)
    steps = summarize.step_rows(records)
    runs = run_rows(records, steps)
    steps.to_csv(summary / "measurements.csv", index=False)
    runs.to_csv(summary / "runs.csv", index=False)
    summarize.resource_rows(runs).to_csv(summary / "resources.csv", index=False)
    storage_rows(records).to_csv(summary / "storage.csv", index=False)
    checks = verification(records, protocol, baseline=None if args.skip_baseline_check else get_run(BASELINE_RUN))
    (summary / "verification.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    first = {workflow: Path(next(r["output"] for r in sorted(records, key=lambda r: r["repeat"]) if r["workflow"] == workflow))
             for workflow in ("captum", "ssat")}
    items, profile, document = parity(first["captum"], first["ssat"], float(protocol["parity"]["z_threshold"]))
    items.to_csv(summary / "parity_items.csv", index=False)
    profile.to_csv(summary / "parity_profile.csv", index=False)
    (summary / "parity.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (summary / "engineering.json").write_text(json.dumps(engineering(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in summary.iterdir():
        path.chmod(0o644)
    write_provenance(summary, inputs={"measurements": measurements, "protocol": PROTOCOL,
                                      "ssat_config": SSAT_CONFIG, "captum_config": CAPTUM_WORKFLOW / "config.yaml"})
    print(json.dumps({"all_pass": checks["all_pass"],
                      "deterministic_rank_order_identical": document["primary_deterministic_rank_order_identical"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
