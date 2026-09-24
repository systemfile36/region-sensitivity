#!/usr/bin/env python3
"""A2 nested parity: the K=20 runs contain their baseline's items unchanged.

For each A2 run and its baseline (``runs.A2_BASELINES``), restricted to the
A2 run's samples:

* identity: every baseline item (targets and controls, whose
  ``control_index`` is below the baseline K) reappears in the K=20 run with
  the same ``region_params_json``, ``effective_area_px``, and ``seed_used``;
  every K=20 item with ``control_index`` below the baseline K is a baseline
  item; no K=20 item with a larger index is;
* logits: bitwise equality is reported with a max-|diff| histogram and top-1
  changes, not gated (``deviations.md`` D-001);
* clean rows: labels, status, and logits.

Writes ``summary/nested_parity.json``; the exit code reflects identity only.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from experiments.revision_1.common.loading import iter_latest_perturbed, load_clean
from experiments.revision_1.common.provenance import write_json_shared, write_provenance
from experiments.revision_1.common.runs import A2_BASELINES, A2_RUNS, BASELINE_RUNS, RunRef
from experiments.revision_1.phase0.verify_target_parity import (
    _accumulate,
    _control_index,
    _logits_matrix,
    _new_stats,
)

SUMMARY_DIR = Path(__file__).resolve().parent / "summary"
_COLUMNS = ("sample_id", "is_control", "region_params_json", "effective_area_px", "seed_used", "logits")
_MATCH_COLUMNS = ("sample_id", "is_control", "region_params_json", "effective_area_px", "seed_used")


def _sample_filter(sample_ids: set[str]) -> pc.Expression:
    return pc.field("sample_id").isin(pa.array(sorted(sample_ids)))


def _load(run: RunRef, sample_ids: set[str]) -> tuple[pd.DataFrame, np.ndarray]:
    frames, logits = [], []
    for table in iter_latest_perturbed(run.dump, _COLUMNS, filter_expression=_sample_filter(sample_ids)):
        logits.append(_logits_matrix(table.column("logits")))
        frames.append(table.drop_columns(["logits"]).to_pandas())
    frame = pd.concat(frames, ignore_index=True)
    frame["control_index"] = _control_index(frame["region_params_json"]).fillna(-1).astype(np.int64)
    return frame, np.concatenate(logits)


def compare_clean(baseline: RunRef, candidate: RunRef, sample_ids: set[str]) -> dict[str, object]:
    """Compare clean labels, status, and logits on ``sample_ids``."""

    a = load_clean(baseline, with_logits=True)
    b = load_clean(candidate, with_logits=True)
    a = a[a["sample_id"].isin(sample_ids)].set_index("sample_id").sort_index()
    b = b.set_index("sample_id").sort_index()
    result: dict[str, object] = {"n_baseline": len(a), "n_candidate": len(b), "sample_ids_equal": a.index.equals(b.index)}
    if result["sample_ids_equal"]:
        logits_a = np.stack(a["logits"].to_numpy()).astype(np.float32)
        logits_b = np.stack(b["logits"].to_numpy()).astype(np.float32)
        result["gt_label_equal"] = bool((a["gt_label"] == b["gt_label"]).all())
        result["status_equal"] = bool((a["status"] == b["status"]).all())
        result["logits_bitwise_equal"] = bool(np.array_equal(logits_a.view(np.uint32), logits_b.view(np.uint32)))
        result["max_abs_diff"] = float(np.abs(logits_a - logits_b).max())
        result["n_top1_mismatch"] = int((logits_a.argmax(1) != logits_b.argmax(1)).sum())
    return result


def compare_nested(candidate: RunRef, baseline: RunRef) -> dict[str, object]:
    """Compare one K=20 run with the baseline run it extends."""

    started = time.perf_counter()
    sample_ids = set(load_clean(candidate)["sample_id"])
    reference, reference_logits = _load(baseline, sample_ids)
    k_baseline = int(reference["control_index"].max()) + 1
    position = pd.Series(np.arange(len(reference)), index=pd.Index(reference["item_id"]))
    if not position.index.is_unique:
        raise ValueError(f"{baseline.name}: duplicate item_id after latest-row selection")
    seen = np.zeros(len(reference), dtype=bool)
    targets, controls = _new_stats(), _new_stats()
    counts = {"candidate_items": 0, "candidate_targets_missing_in_baseline": 0,
              "candidate_nested_controls_missing_in_baseline": 0,
              "candidate_new_controls_present_in_baseline": 0}
    index_counts: dict[int, int] = {}
    for table in iter_latest_perturbed(candidate.dump, _COLUMNS):
        frame = table.drop_columns(["logits"]).to_pandas()
        logits = _logits_matrix(table.column("logits"))
        control_index = _control_index(frame["region_params_json"]).fillna(-1).to_numpy(dtype=np.int64)
        rows = position.reindex(frame["item_id"]).to_numpy()
        present = ~np.isnan(rows)
        is_control = frame["is_control"].to_numpy()
        for value, count in zip(*np.unique(control_index[is_control], return_counts=True)):
            index_counts[int(value)] = index_counts.get(int(value), 0) + int(count)
        nested = is_control & (control_index < k_baseline)
        counts["candidate_items"] += len(frame)
        counts["candidate_targets_missing_in_baseline"] += int((~is_control & ~present).sum())
        counts["candidate_nested_controls_missing_in_baseline"] += int((nested & ~present).sum())
        counts["candidate_new_controls_present_in_baseline"] += int((is_control & ~nested & present).sum())
        for stats, selector in ((targets, ~is_control & present), (controls, nested & present)):
            if not selector.any():
                continue
            ref_rows = rows[selector].astype(np.int64)
            seen[ref_rows] = True
            ref_context = reference.iloc[ref_rows][list(_MATCH_COLUMNS)].reset_index(drop=True)
            cand_context = frame.loc[selector, list(_MATCH_COLUMNS)].reset_index(drop=True)
            context_ok = (ref_context == cand_context).all(axis=1).to_numpy()
            _accumulate(stats, context_ok, reference_logits[ref_rows], logits[selector])
    n_ref_targets = int((~reference["is_control"]).sum())
    n_ref_controls = int(reference["is_control"].sum())
    identity = (
        bool(seen.all())
        and counts["candidate_targets_missing_in_baseline"] == 0
        and counts["candidate_nested_controls_missing_in_baseline"] == 0
        and counts["candidate_new_controls_present_in_baseline"] == 0
        and targets["n_matched"] == n_ref_targets
        and controls["n_matched"] == n_ref_controls
        and targets["n_context_mismatch"] == 0
        and controls["n_context_mismatch"] == 0
    )
    clean = compare_clean(baseline, candidate, sample_ids)
    clean_identity = bool(clean["sample_ids_equal"] and clean.get("gt_label_equal") and clean.get("status_equal"))
    return {
        "run": candidate.name,
        "baseline": baseline.name,
        "n_samples": len(sample_ids),
        "baseline_k": k_baseline,
        "baseline_items": len(reference),
        "baseline_targets": n_ref_targets,
        "baseline_controls": n_ref_controls,
        "baseline_items_missing_in_candidate": int((~seen).sum()),
        **counts,
        "candidate_control_index_counts": {str(key): value for key, value in sorted(index_counts.items())},
        "targets": targets,
        "nested_controls": controls,
        "clean": clean,
        "identity_pass": bool(identity and clean_identity),
        "target_logits_bitwise_pass": targets["n_bitwise_equal"] == n_ref_targets,
        "control_logits_bitwise_pass": controls["n_bitwise_equal"] == n_ref_controls,
        "clean_logits_bitwise_pass": bool(clean.get("logits_bitwise_equal")),
        "top1_unchanged": targets["n_top1_mismatch"] == 0 and controls["n_top1_mismatch"] == 0
        and clean.get("n_top1_mismatch", 1) == 0,
        "elapsed_s": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=list(A2_RUNS))
    parser.add_argument("--output", type=Path, default=SUMMARY_DIR / "nested_parity.json")
    args = parser.parse_args(argv)

    results = []
    for name in args.runs:
        print(f"comparing {name} ...", flush=True)
        result = compare_nested(A2_RUNS[name], BASELINE_RUNS[A2_BASELINES[name]])
        print(
            f"  identity={result['identity_pass']} targets {result['targets']['n_bitwise_equal']}/"
            f"{result['baseline_targets']} bitwise (max diff {result['targets']['max_abs_diff']:.3g}), "
            f"nested controls {result['nested_controls']['n_bitwise_equal']}/{result['baseline_controls']} "
            f"(max diff {result['nested_controls']['max_abs_diff']:.3g}) ({result['elapsed_s']:.0f}s)",
            flush=True,
        )
        results.append(result)
    payload = {
        "comparison": (
            "identity = item_id sets plus region_params_json/effective_area_px/seed_used equality on the "
            "A2 run's samples; logits compared as bitwise float32 equality, unequal items bucketed by max |diff|"
        ),
        "identity_pass": all(r["identity_pass"] for r in results),
        "logits_bitwise_pass": all(
            r["target_logits_bitwise_pass"] and r["control_logits_bitwise_pass"] and r["clean_logits_bitwise_pass"]
            for r in results
        ),
        "runs": results,
    }
    write_json_shared(args.output, payload)
    inputs = {}
    for name in args.runs:
        inputs[name] = A2_RUNS[name].dump
        inputs[A2_BASELINES[name]] = BASELINE_RUNS[A2_BASELINES[name]].dump
    write_provenance(args.output.parent, inputs=inputs, filename=f"{args.output.stem}.provenance.json")
    print(f"wrote {args.output} (identity_pass={payload['identity_pass']})")
    return 0 if payload["identity_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
