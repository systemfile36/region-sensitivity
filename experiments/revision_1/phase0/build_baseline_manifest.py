#!/usr/bin/env python3
"""P0-1: freeze a description of every baseline run revision-1 builds on.

For each run in ``BASELINE_RUNS``, ``K1_PARITY_RUNS``, and
``ORIGINAL_SMALL_RUNS`` (the NTU/synthetic stores as originally stored,
described but not gated) this records the
config (current and launch-time SHA-256), the ``run_manifest.json``/
``metrics_manifest.json``/``analysis_manifest.json`` hashes and whether they
chain (metrics computed from this dump, analysis from these metrics), the
resolved controls/perturbations/regions/seed, adapter spec and preprocessing
fingerprint, run environment and timing, analysis thresholds, the
``margin_drop`` grade and flag distribution, on-disk size, and whether the
stored analysis predates the last commit touching ``ssat/analysis`` or
``ssat/metrics``. Output: ``phase0/summary/baseline_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.revision_1.common.loading import load_reliability
from experiments.revision_1.common.provenance import REPO_ROOT, _git, git_blob_sha256, git_state, write_json_shared
from experiments.revision_1.common.runs import BASELINE_RUNS, K1_PARITY_RUNS, ORIGINAL_SMALL_RUNS, RunRef
from ssat.utils.io import load_json, sha256_bytes, sha256_file

SUMMARY_DIR = Path(__file__).resolve().parent / "summary"
_FLAGS = ("sign_consistent", "exceeds_control", "multi_strategy", "ci_excludes_zero", "seed_stable", "area_matched")
_CONTAINER_ROOT = Path("/workspace")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=SUMMARY_DIR / "baseline_manifest.json")
    parser.add_argument("--skip-sizes", action="store_true", help="Do not walk run directories for byte sizes.")
    return parser.parse_args(argv)


def _relative(path: Path | str | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    for root in (REPO_ROOT, _CONTAINER_ROOT):
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
    return str(path)


def _to_repo_path(recorded: str | None) -> Path | None:
    """Map a path recorded inside the container (``/workspace/...``) onto this checkout."""

    if recorded is None:
        return None
    path = Path(recorded)
    try:
        return REPO_ROOT / path.relative_to(_CONTAINER_ROOT)
    except ValueError:
        return path


def _dir_bytes(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _last_commit_time(*paths: str) -> str | None:
    return _git("log", "-1", "--format=%cI", "--", *paths)


def _margin_drop_summary(run: RunRef) -> dict[str, Any]:
    reliability = load_reliability(run, "margin_drop")
    grades = reliability["reliability_grade"].value_counts()
    n = int(len(reliability))
    flag_rates = {
        flag: {value: round(float(share), 6) for value, share in reliability[flag].value_counts(normalize=True).items()}
        for flag in _FLAGS
    }
    return {
        "n_anchors": n,
        "grade_counts": {grade: int(count) for grade, count in grades.items()},
        "grade_pct": {grade: round(100.0 * count / n, 4) for grade, count in grades.items()},
        "flag_rates": flag_rates,
    }


def describe_run(run: RunRef, *, sizes: bool, analysis_code_changed_at: str | None) -> dict[str, Any]:
    """Collect the frozen facts for one run."""

    run_manifest_path = run.dump / "run_manifest.json"
    metrics_manifest_path = run.metrics / "metrics_manifest.json"
    analysis_manifest_path = run.analysis / "analysis_manifest.json"
    run_manifest = load_json(run_manifest_path)
    metrics_manifest = load_json(metrics_manifest_path)
    analysis_manifest = load_json(analysis_manifest_path)
    resolved = run_manifest["resolved_config"]
    run_manifest_sha = sha256_file(run_manifest_path)
    metrics_manifest_sha = sha256_file(metrics_manifest_path)

    source = resolved.get("source_provenance") or {}
    source_manifest = _to_repo_path(source.get("manifest"))
    source_manifest_now = sha256_file(source_manifest) if source_manifest and source_manifest.is_file() else None

    config: dict[str, Any] = {"path": _relative(run.config), "revision": run.config_revision}
    if run.config is not None and run.config.is_file():
        config["sha256_current"] = sha256_file(run.config)
        if run.config_revision is not None:
            config["sha256_at_revision"] = git_blob_sha256(run.config_revision, run.config)
    config["recorded_config_source"] = _relative(resolved.get("config_source"))

    started = datetime.fromisoformat(run_manifest["started_at"])
    finished = datetime.fromisoformat(run_manifest["finished_at"]) if run_manifest.get("finished_at") else None
    controls = resolved.get("controls") or []
    n_controls_recorded = sum(int(control["n_samples"]) for control in controls)

    record: dict[str, Any] = {
        "name": run.name,
        "dataset": run.dataset,
        "model": run.model,
        "protocol": run.protocol,
        "n_controls": run.n_controls,
        "paths": {"dump": _relative(run.dump), "metrics": _relative(run.metrics), "analysis": _relative(run.analysis)},
        "config": config,
        "run_manifest": {
            "sha256": run_manifest_sha,
            "schema_version": run_manifest["schema_version"],
            "code_version": run_manifest.get("code_version"),
            "started_at": run_manifest["started_at"],
            "finished_at": run_manifest.get("finished_at"),
            "run_duration_s": (finished - started).total_seconds() if finished else None,
            "counts_by_status": run_manifest["counts_by_status"],
            "resume_events": len(run_manifest.get("resume_events") or []),
            "environment": run_manifest.get("environment"),
        },
        "resolved_config": {
            "sha256_canonical": sha256_bytes(json.dumps(resolved, sort_keys=True).encode("utf-8")),
            "controls": controls,
            "perturbations": resolved.get("perturbations"),
            "regions": resolved.get("regions"),
            "runtime": resolved.get("runtime"),
            "dataset_stats": resolved.get("dataset_stats"),
            "skeleton_source": resolved.get("skeleton_source"),
            "source_provenance": source,
        },
        "adapter_spec": run_manifest.get("adapter_spec"),
        "preprocessing_fingerprint": (run_manifest.get("adapter_spec") or {}).get("preprocessing_fingerprint"),
        "metrics_manifest": {
            "sha256": metrics_manifest_sha,
            "schema_version": metrics_manifest.get("metrics_schema_version"),
            "computed_at": metrics_manifest.get("computed_at"),
            "metric_config": metrics_manifest.get("metric_config"),
            "exclusion_summary": metrics_manifest.get("exclusion_summary"),
        },
        "analysis_manifest": {
            "sha256": sha256_file(analysis_manifest_path),
            "schema_version": analysis_manifest.get("analysis_schema_version"),
            "computed_at": analysis_manifest.get("computed_at"),
            "thresholds": analysis_manifest.get("thresholds"),
            "n_bootstrap": analysis_manifest.get("n_bootstrap"),
            "random_seed": analysis_manifest.get("random_seed"),
            "available_analyses": analysis_manifest.get("available_analyses"),
            "grade_distribution_all_metrics": analysis_manifest.get("grade_distribution"),
        },
        "margin_drop": _margin_drop_summary(run),
        "checks": {
            "metrics_from_this_dump": metrics_manifest.get("source_run_manifest_hash") == run_manifest_sha,
            "analysis_from_these_metrics": analysis_manifest.get("source_metrics_manifest_hash") == metrics_manifest_sha,
            "run_finished": finished is not None,
            "all_items_ok": all(
                count == 0 for status, count in run_manifest["counts_by_status"].items() if status != "ok"
            ),
            "n_controls_matches_registry": n_controls_recorded == run.n_controls,
            "source_manifest_unchanged": (
                None if source_manifest_now is None else source_manifest_now == source.get("manifest_hash")
            ),
            "analysis_predates_last_analysis_or_metrics_commit": (
                None
                if analysis_code_changed_at is None
                else datetime.fromisoformat(analysis_manifest["computed_at"])
                < datetime.fromisoformat(analysis_code_changed_at)
            ),
        },
    }
    if sizes:
        record["bytes"] = {
            child.name: _dir_bytes(child) for child in (run.dump, run.metrics, run.analysis) if child.is_dir()
        }
        record["bytes"]["dump_total"] = _dir_bytes(run.dump)
    return record


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    analysis_code_changed_at = _last_commit_time("ssat/analysis", "ssat/metrics")
    runs = {"baseline": BASELINE_RUNS, "k1_parity": K1_PARITY_RUNS, "original_small_stores": ORIGINAL_SMALL_RUNS}
    payload: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "git": git_state(),
        "baseline_tag": "v1.0.0",
        "baseline_commit": _git("rev-list", "-n", "1", "v1.0.0"),
        "last_commit_touching_ssat_analysis_or_metrics": analysis_code_changed_at,
        "primary_metric": "margin_drop",
    }
    # Only stores the revision reads are gated; the original small stores
    # are described so their known defects stay on record (see P0-7).
    failures, original_findings = [], []
    for role, registry in runs.items():
        payload[role] = {}
        for name, run in registry.items():
            print(f"describing {role}/{name} ...", flush=True)
            record = describe_run(run, sizes=not args.skip_sizes, analysis_code_changed_at=analysis_code_changed_at)
            payload[role][name] = record
            for check in ("metrics_from_this_dump", "analysis_from_these_metrics", "run_finished",
                          "all_items_ok", "n_controls_matches_registry"):
                if not record["checks"][check]:
                    target = original_findings if role == "original_small_stores" else failures
                    target.append(f"{name}: {check}")
    payload["check_failures"] = failures
    payload["original_small_store_findings"] = original_findings
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_shared(args.output, payload)
    table = pd.DataFrame(
        [
            {"role": role, "run": name,
             **{f"{g}%": record["margin_drop"]["grade_pct"].get(g, 0.0) for g in ("high", "moderate", "low", "unreliable")},
             **record["checks"]}
            for role in runs
            for name, record in payload[role].items()
        ]
    )
    print(table.to_string(index=False))
    print(f"wrote {args.output}; check failures: {failures or 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
