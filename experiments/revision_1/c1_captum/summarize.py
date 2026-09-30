"""C1-min summaries from ``results/c1/measurements.jsonl`` and the kept outputs.

Writes to ``summary/``:
- ``measurements.csv``: one row per workflow x repeat x command;
- ``runs.csv``: one row per workflow x repeat (stage times, memory, GPU, items);
- ``resources.csv``: measure x workflow, mean / min / max over repeats;
- ``storage.csv``: bytes by output entry and content, mean over repeats;
- ``verification.json``: the protocol's checks against the stored runs.

Example:
    python experiments/revision_1/c1_captum/summarize.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from experiments.revision_1.c1_captum.measure_resources import C1_RESULTS_DIR, CAPTUM_DIR
from experiments.revision_1.common.provenance import write_provenance
from experiments.revision_1.common.runs import SYNTHETIC_DIR

C1_DIR = Path(__file__).resolve().parent
SUMMARY_DIR = C1_DIR / "summary"
PROTOCOL = C1_DIR / "protocol.json"
STORED_SSAT = SYNTHETIC_DIR / "results_crop_free"
Q2_REL_TOLERANCE = 1e-3
GIB = 2**30
CONTENT = {
    ("captum", "raw"): "raw observations: one row per item (ids, clean and perturbed margin, degradation, areas, status)",
    ("captum", "analysis"): "analysis tables (sample/region/class/dataset, controls, seed stability, bootstrap CI, operator consistency; Parquet + CSV)",
    ("captum", "files"): "run manifest, provenance, accuracy, report.md, comparison metrics",
    ("ssat", "dumps"): "raw observations: clean and per-item logits (10 classes), region/operator/seed/area metadata, item index, run manifest with resolved config",
    ("ssat", "metrics"): "per-item margin-drop metrics and region aggregates",
    ("ssat", "analysis"): "reliability grades, controls, seed and strategy stability, intervals, rank correlation",
    ("ssat", "report"): "HTML reports with data CSVs and figures",
    ("ssat", "heatmaps"): "representative heatmaps (evaluate.py)",
    ("ssat", "files"): "accuracy, verdicts, report.md, region CSVs",
}


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def step_rows(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """One row per workflow x repeat x command."""

    return pd.DataFrame([
        {"workflow": record["workflow"], "repeat": record["repeat"], "step": step["step"], "stage": step["stage"],
         "command": " ".join(Path(part).name if index < 2 else part for index, part in enumerate(step["argv"][:4])),
         "elapsed_s": step["elapsed_s"], "peak_rss_gib": step["peak_rss_kb"] / 2**20,
         "tree_rss_peak_gib": step["tree_rss_peak_bytes"] / GIB, "tree_processes": step["tree_rss_peak_processes"],
         "gpu_peak_mib": step["gpu_peak_mib"], "gpu_util_mean": step["gpu_util_mean"], "returncode": step["returncode"]}
        for record in records for step in record["steps"]
    ])


def dump_dirs(output: Path) -> list[Path]:
    """SSAT run directories under ``output/dumps`` (skipping ``.ssat.lock`` files)."""

    return sorted(path.parent for path in (output / "dumps").glob("*/run_manifest.json"))


def perturbed_items(workflow: str, output: Path) -> int:
    """Perturbed model evaluations: Captum raw rows, or SSAT dump items minus clean rows."""

    if workflow == "captum":
        return int(json.loads((output / "analysis" / "summary.json").read_text(encoding="utf-8"))["raw_rows"])
    total = 0
    for dump in dump_dirs(output):
        counts = json.loads((dump / "run_manifest.json").read_text(encoding="utf-8"))["counts_by_status"]
        clean = sum(pq.ParquetFile(path).metadata.num_rows for path in (dump / "clean").rglob("*.parquet"))
        total += sum(counts.values()) - clean
    return total


def run_rows(records: Sequence[Mapping[str, Any]], steps: pd.DataFrame) -> pd.DataFrame:
    """One row per workflow x repeat."""

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


RESOURCE_MEASURES = (
    ("total_s", "end-to-end wall time", "s"),
    ("audit_s", "audit stage (all model evaluations)", "s"),
    ("analysis_report_s", "analysis + report stage", "s"),
    ("perturbed_items", "perturbed model evaluations", "items"),
    ("ms_per_item", "end-to-end wall time per perturbed item", "ms"),
    ("audit_ms_per_item", "audit wall time per perturbed item", "ms"),
    ("peak_rss_gib", "peak host RSS, largest process", "GiB"),
    ("tree_rss_peak_gib", "peak host RSS, summed process tree", "GiB"),
    ("gpu_peak_mib", "peak GPU memory, device total", "MiB"),
    ("gpu_above_idle_mib", "peak GPU memory above the idle baseline", "MiB"),
    ("audit_gpu_util_mean", "mean GPU utilization during the audit stage", "%"),
    ("total_bytes", "total storage", "bytes"),
)


def resource_rows(runs: pd.DataFrame) -> pd.DataFrame:
    """Measure x workflow with mean / min / max over repeats."""

    rows = []
    for column, measure, unit in RESOURCE_MEASURES:
        for workflow, group in runs.groupby("workflow"):
            values = group[column].astype(float)
            rows.append({"measure": measure, "unit": unit, "workflow": workflow, "mean": values.mean(),
                         "min": values.min(), "max": values.max(), "n_repeats": len(values)})
    return pd.DataFrame(rows)


def storage_rows(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Bytes by top-level entry and content, mean over repeats, with the share of the workflow total."""

    frame = pd.DataFrame([{"workflow": record["workflow"], "repeat": record["repeat"], "entry": entry, "bytes": size}
                          for record in records for entry, size in record["bytes"].items() if entry != "total"])
    means = frame.groupby(["workflow", "entry"], as_index=False)["bytes"].mean()
    means["share"] = means["bytes"] / means.groupby("workflow")["bytes"].transform("sum")
    means["content"] = [CONTENT.get((workflow, entry), "") for workflow, entry in zip(means["workflow"], means["entry"])]
    return means


def check_captum(output: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    """Protocol checks for one Captum output against ``measured_results.json`` values."""

    summary = json.loads((output / "analysis" / "summary.json").read_text(encoding="utf-8"))
    checks = {
        "raw_rows": summary["raw_rows"] == expected["raw_rows"],
        "raw_canonical_hash": summary["raw_canonical_hash"] == expected["raw_canonical_hash"],
        "analysis_canonical_hash": summary["canonical_hash"] == expected["analysis_canonical_hash"],
        "all_q1_q5_pass": all(value["pass"] for key, value in summary["verdicts"].items() if key.startswith("Q")),
        "area_sanity": bool(summary["area_sanity"]["pass"]),
    }
    return {"checks": checks, "pass": all(checks.values())}


def compare_verdicts(new: Mapping[str, Any], stored: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two ``evaluate.py`` verdict files; Q2's multiplier within ``Q2_REL_TOLERANCE``."""

    questions = [key for key in stored if key.startswith("Q")]
    q2_new, q2_stored = new["Q2_separated_from_baseline"]["multiplier"], stored["Q2_separated_from_baseline"]["multiplier"]
    q2_rel = abs(q2_new - q2_stored) / abs(q2_stored)
    checks = {
        "pass_flags": all(new[key]["pass"] == stored[key]["pass"] for key in questions),
        "q1_rank": new["Q1_identifies_patch_region"]["patch_region_rank"] == stored["Q1_identifies_patch_region"]["patch_region_rank"],
        "q3_rank": new["Q3_distinguishes_normal_model"] == stored["Q3_distinguishes_normal_model"],
        "q4_reproduced_in": new["Q4_robust_to_fill_strategy"] == stored["Q4_robust_to_fill_strategy"],
        "q5": new["Q5_predicts_generalization_gap"] == stored["Q5_predicts_generalization_gap"],
        "b_auxiliary_rank": new["B_auxiliary_control"] == stored["B_auxiliary_control"],
        "q2_multiplier_within_tolerance": q2_rel <= Q2_REL_TOLERANCE,
    }
    return {"checks": checks, "q2_multiplier_relative_difference": q2_rel}


def grade_counts(analysis_dir: Path) -> dict[str, int]:
    grades = pd.read_parquet(analysis_dir / "reliability.parquet", columns=["reliability_grade"])["reliability_grade"]
    return dict(sorted(Counter(map(str, grades)).items()))


def check_ssat(output: Path, stored: Path) -> dict[str, Any]:
    """Protocol checks for one SSAT output against the stored crop-free results."""

    read = lambda path: json.loads(path.read_text(encoding="utf-8"))  # noqa: E731
    dumps = [path.name for path in dump_dirs(stored)]
    counts = {name: read(output / "dumps" / name / "run_manifest.json")["counts_by_status"]
              == read(stored / "dumps" / name / "run_manifest.json")["counts_by_status"]
              if (output / "dumps" / name).is_dir() else False for name in dumps}
    verdicts = compare_verdicts(read(output / "verdicts.json"), read(stored / "verdicts.json"))
    checks = {"counts_by_status": all(counts.values()), **verdicts["checks"],
              "accuracy": read(output / "accuracy.json") == read(stored / "accuracy.json")}
    grades = {name: {"fresh": grade_counts(output / "analysis" / name), "stored": grade_counts(stored / "analysis" / name)}
              for name in sorted(path.parent.name for path in (stored / "analysis").glob("*/reliability.parquet"))}
    for value in grades.values():
        value["identical"] = value["fresh"] == value["stored"]
    return {"checks": checks, "pass": all(checks.values()), "dump_counts": counts,
            "q2_multiplier_relative_difference": verdicts["q2_multiplier_relative_difference"],
            "grade_counts_informational": grades}


def verification(records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], stored: Path) -> dict[str, Any]:
    expected = protocol["verification"]["captum_required"]
    runs = {f"{record['workflow']}_r{record['repeat']}": (check_captum(Path(record["output"]), expected)
                                                          if record["workflow"] == "captum" else check_ssat(Path(record["output"]), stored))
            for record in sorted(records, key=lambda r: (r["workflow"], r["repeat"]))}
    return {"all_pass": all(run["pass"] for run in runs.values()), "runs": runs}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-dir", type=Path, default=C1_RESULTS_DIR)
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    parser.add_argument("--stored-ssat", type=Path, default=STORED_SSAT)
    args = parser.parse_args(argv)
    measurements = args.results_dir / "measurements.jsonl"
    records = load_records(measurements)
    summary = args.summary_dir
    summary.mkdir(parents=True, exist_ok=True)
    steps = step_rows(records)
    runs = run_rows(records, steps)
    steps.to_csv(summary / "measurements.csv", index=False)
    runs.to_csv(summary / "runs.csv", index=False)
    resource_rows(runs).to_csv(summary / "resources.csv", index=False)
    storage_rows(records).to_csv(summary / "storage.csv", index=False)
    checks = verification(records, json.loads(PROTOCOL.read_text(encoding="utf-8")), args.stored_ssat)
    (summary / "verification.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in summary.iterdir():
        path.chmod(0o644)
    write_provenance(summary, inputs={"measurements": measurements, "protocol": PROTOCOL,
                                      "captum_measured_results": CAPTUM_DIR / "measured_results.json",
                                      "stored_ssat_verdicts": args.stored_ssat / "verdicts.json"})
    print(json.dumps({"all_pass": checks["all_pass"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
