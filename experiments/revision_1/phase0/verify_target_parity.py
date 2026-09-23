#!/usr/bin/env python3
"""P0-3: check that the K=1 and K=3 ImageNet runs share identical target items.

The paper's accuracy/mean-margin_drop table comes from the K=1 runs and its
reliability numbers from the K=3 runs, so both must describe the same
target evaluations. For each model/protocol pair this compares:

* clean samples: ids, labels, status, and logits;
* target items (``is_control=False``): item-id sets, mask context
  (``region_params_json``, ``effective_area_px``, ``seed_used``), and logits;
* nested controls: every K=1 control must reappear bit-for-bit as the K=3
  control with ``control_index == 0`` (the property A2's K=20 design relies
  on), and K=3 controls with ``control_index > 0`` must be new items.

Identity (same items, masks, and seeds) and logit equality are reported
separately. Logits are compared bitwise as the plan requires; unequal items
are bucketed by their maximum absolute difference and top-1 changes are
counted, because cuDNN convolutions run in TF32 by default and their result
can depend on the batch shape an item happened to be evaluated in. The exit
code reflects identity only; the bitwise outcome is recorded in the JSON.
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
from experiments.revision_1.common.runs import BASELINE_RUNS, RunRef, k1_counterpart
from experiments.revision_1.common.subsets import hash_rank

SUMMARY_DIR = Path(__file__).resolve().parent / "summary"
_COLUMNS = ("sample_id", "is_control", "region_params_json", "effective_area_px", "seed_used", "logits")
_MATCH_COLUMNS = ("sample_id", "is_control", "region_params_json", "effective_area_px", "seed_used")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=None, help="K=3 baseline run names (default: all ImageNet).")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Restrict to the N lowest sha256('rev1-p0:<sample_id>') samples (default: all).",
    )
    parser.add_argument("--output", type=Path, default=SUMMARY_DIR / "target_parity.json")
    return parser.parse_args(argv)


def _logits_matrix(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if array.null_count:
        raise ValueError("null logits encountered (failed items are not expected in these runs)")
    lengths = pc.list_value_length(array).to_numpy()
    width = int(lengths[0]) if len(lengths) else 0
    if len(lengths) and not (lengths == width).all():
        raise ValueError("logit vectors have unequal lengths")
    return array.flatten().to_numpy().reshape(-1, width)


def _control_index(region_params_json: pd.Series) -> pd.Series:
    return region_params_json.str.extract(r'"control_index":(\d+)', expand=False).astype("Int64")


def _sample_filter(sample_ids: set[str] | None) -> pc.Expression | None:
    if sample_ids is None:
        return None
    return pc.field("sample_id").isin(pa.array(sorted(sample_ids)))


def _load_reference(run: RunRef, sample_ids: set[str] | None) -> tuple[pd.DataFrame, np.ndarray]:
    frames, logits = [], []
    for table in iter_latest_perturbed(run.dump, _COLUMNS, filter_expression=_sample_filter(sample_ids)):
        logits.append(_logits_matrix(table.column("logits")))
        frames.append(table.drop_columns(["logits"]).to_pandas())
    frame = pd.concat(frames, ignore_index=True)
    frame["control_index"] = _control_index(frame["region_params_json"])
    return frame, np.concatenate(logits)


_DIFF_BUCKETS = (("le_1e-5", 1e-5), ("le_1e-3", 1e-3), ("le_1e-1", 1e-1), ("gt_1e-1", np.inf))


def _new_stats() -> dict[str, object]:
    return {
        "n_matched": 0,
        "n_bitwise_equal": 0,
        "n_context_mismatch": 0,
        "n_top1_mismatch": 0,
        "max_abs_diff": 0.0,
        # Items whose logits are not bitwise equal, by their max |difference|.
        "unequal_by_max_abs_diff": {name: 0 for name, _ in _DIFF_BUCKETS},
    }


def _accumulate(stats: dict, context_ok: np.ndarray, ref: np.ndarray, cand: np.ndarray) -> None:
    equal_rows = (ref.view(np.uint32) == cand.view(np.uint32)).all(axis=1)
    diff = np.abs(ref - cand).max(axis=1)
    stats["n_matched"] += len(ref)
    stats["n_bitwise_equal"] += int(equal_rows.sum())
    stats["n_context_mismatch"] += int((~context_ok).sum())
    stats["n_top1_mismatch"] += int((ref.argmax(1) != cand.argmax(1)).sum())
    if len(ref):
        stats["max_abs_diff"] = max(stats["max_abs_diff"], float(diff.max()))
    lower = 0.0
    for name, upper in _DIFF_BUCKETS:
        stats["unequal_by_max_abs_diff"][name] += int((~equal_rows & (diff > lower) & (diff <= upper)).sum())
        lower = upper


def compare_clean(k1: RunRef, k3: RunRef, sample_ids: set[str] | None) -> dict[str, object]:
    """Compare clean-sample ids, labels, status, and logits between two runs."""

    a = load_clean(k1, with_logits=True)
    b = load_clean(k3, with_logits=True)
    if sample_ids is not None:
        a = a[a["sample_id"].isin(sample_ids)]
        b = b[b["sample_id"].isin(sample_ids)]
    a = a.set_index("sample_id").sort_index()
    b = b.set_index("sample_id").sort_index()
    same_ids = a.index.equals(b.index)
    result: dict[str, object] = {"n_k1": len(a), "n_k3": len(b), "sample_ids_equal": same_ids}
    if same_ids:
        logits_a = np.stack(a["logits"].to_numpy()).astype(np.float32)
        logits_b = np.stack(b["logits"].to_numpy()).astype(np.float32)
        result["gt_label_equal"] = bool((a["gt_label"] == b["gt_label"]).all())
        result["status_equal"] = bool((a["status"] == b["status"]).all())
        result["logits_bitwise_equal"] = bool(np.array_equal(logits_a.view(np.uint32), logits_b.view(np.uint32)))
        result["max_abs_diff"] = float(np.abs(logits_a - logits_b).max())
    return result


def compare_pair(k3: RunRef, sample_ids: set[str] | None) -> dict[str, object]:
    """Compare one K=3 baseline run against its K=1 counterpart."""

    k1 = k1_counterpart(k3)
    started = time.perf_counter()
    reference, reference_logits = _load_reference(k1, sample_ids)
    position = pd.Series(np.arange(len(reference)), index=pd.Index(reference["item_id"]))
    if not position.index.is_unique:
        raise ValueError(f"{k1.name}: duplicate item_id after latest-row selection")
    seen = np.zeros(len(reference), dtype=bool)
    targets, controls0 = _new_stats(), _new_stats()
    counts = {"k3_targets": 0, "k3_targets_missing_in_k1": 0, "k3_controls": 0,
              "k3_control0_missing_in_k1": 0, "k3_control_gt0_present_in_k1": 0}
    control_index_counts: dict[int, int] = {}
    for table in iter_latest_perturbed(k3.dump, _COLUMNS, filter_expression=_sample_filter(sample_ids)):
        frame = table.drop_columns(["logits"]).to_pandas()
        logits = _logits_matrix(table.column("logits"))
        frame["control_index"] = _control_index(frame["region_params_json"])
        rows = position.reindex(frame["item_id"]).to_numpy()
        present = ~np.isnan(rows)
        is_control = frame["is_control"].to_numpy()
        control_index = frame["control_index"].fillna(-1).to_numpy(dtype=np.int64)
        for value, count in zip(*np.unique(control_index[is_control], return_counts=True)):
            control_index_counts[int(value)] = control_index_counts.get(int(value), 0) + int(count)
        counts["k3_targets"] += int((~is_control).sum())
        counts["k3_targets_missing_in_k1"] += int((~is_control & ~present).sum())
        counts["k3_controls"] += int(is_control.sum())
        counts["k3_control0_missing_in_k1"] += int((is_control & (control_index == 0) & ~present).sum())
        counts["k3_control_gt0_present_in_k1"] += int((is_control & (control_index > 0) & present).sum())
        for stats, selector in ((targets, ~is_control & present), (controls0, is_control & (control_index == 0) & present)):
            if not selector.any():
                continue
            ref_rows = rows[selector].astype(np.int64)
            seen[ref_rows] = True
            ref_context = reference.iloc[ref_rows][list(_MATCH_COLUMNS)].reset_index(drop=True)
            cand_context = frame.loc[selector, list(_MATCH_COLUMNS)].reset_index(drop=True)
            context_ok = (ref_context == cand_context).all(axis=1).to_numpy()
            _accumulate(stats, context_ok, reference_logits[ref_rows], logits[selector])
    unmatched = reference.loc[~seen]
    k1_targets = int((~reference["is_control"]).sum())
    k1_controls = int(reference["is_control"].sum())
    target_identity = (
        counts["k3_targets"] == k1_targets
        and counts["k3_targets_missing_in_k1"] == 0
        and targets["n_matched"] == k1_targets
        and targets["n_context_mismatch"] == 0
    )
    control_identity = (
        counts["k3_control0_missing_in_k1"] == 0
        and counts["k3_control_gt0_present_in_k1"] == 0
        and controls0["n_matched"] == k1_controls
        and controls0["n_context_mismatch"] == 0
    )
    clean = compare_clean(k1, k3, sample_ids)
    clean_identity = bool(clean["sample_ids_equal"] and clean.get("gt_label_equal") and clean.get("status_equal"))
    return {
        "k1_run": k1.name,
        "k3_run": k3.name,
        "k1_items": len(reference),
        "k1_targets": k1_targets,
        "k1_controls": k1_controls,
        "k1_items_unmatched_in_k3": int(len(unmatched)),
        **counts,
        "k3_control_index_counts": {str(key): value for key, value in sorted(control_index_counts.items())},
        "targets": targets,
        "controls_k1_vs_k3_index0": controls0,
        "clean": clean,
        # Identity: same items, masks, and seeds (the paired-design invariant).
        "target_identity_pass": bool(target_identity),
        "control_nested_identity_pass": bool(control_identity),
        "clean_identity_pass": clean_identity,
        # Plan P0-3 criterion as written: logits bitwise equal.
        "target_logits_bitwise_pass": targets["n_bitwise_equal"] == k1_targets,
        "control_logits_bitwise_pass": controls0["n_bitwise_equal"] == k1_controls,
        "clean_logits_bitwise_pass": bool(clean.get("logits_bitwise_equal")),
        "top1_unchanged": targets["n_top1_mismatch"] == 0 and controls0["n_top1_mismatch"] == 0,
        "elapsed_s": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = args.runs or [name for name, run in BASELINE_RUNS.items() if run.dataset == "imagenet"]
    sample_ids = None
    if args.max_samples is not None:
        clean = load_clean(BASELINE_RUNS[names[0]])
        sample_ids = set(hash_rank(clean["sample_id"], "rev1-p0")[: args.max_samples])
    results = []
    for name in names:
        print(f"comparing {name} ...", flush=True)
        result = compare_pair(BASELINE_RUNS[name], sample_ids)
        print(
            f"  targets {result['targets']['n_bitwise_equal']}/{result['k1_targets']} bitwise equal "
            f"(max diff {result['targets']['max_abs_diff']:.3g}), controls idx0 "
            f"{result['controls_k1_vs_k3_index0']['n_bitwise_equal']}/{result['k1_controls']} "
            f"(max diff {result['controls_k1_vs_k3_index0']['max_abs_diff']:.3g}), identity="
            f"{result['target_identity_pass']}/{result['control_nested_identity_pass']}/"
            f"{result['clean_identity_pass']} ({result['elapsed_s']:.0f}s)",
            flush=True,
        )
        results.append(result)
    identity = all(
        r["target_identity_pass"] and r["control_nested_identity_pass"] and r["clean_identity_pass"]
        for r in results
    )
    bitwise = all(
        r["target_logits_bitwise_pass"] and r["control_logits_bitwise_pass"] and r["clean_logits_bitwise_pass"]
        for r in results
    )
    payload = {
        "scope": "all samples" if sample_ids is None else f"{len(sample_ids)} samples (rev1-p0 hash prefix)",
        "comparison": (
            "identity = item_id sets plus region_params_json/effective_area_px/seed_used equality; "
            "logits compared as bitwise float32 equality, with unequal items bucketed by max |diff|"
        ),
        "identity_pass": identity,
        "logits_bitwise_pass": bitwise,
        "top1_unchanged": all(r["top1_unchanged"] for r in results),
        "pairs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_shared(args.output, payload)
    inputs = {}
    for name in names:
        inputs[name] = BASELINE_RUNS[name].dump
        inputs[k1_counterpart(BASELINE_RUNS[name]).name] = k1_counterpart(BASELINE_RUNS[name]).dump
    write_provenance(args.output.parent, inputs=inputs, extra={"output": args.output.name},
                     filename=f"{args.output.stem}.provenance.json")
    print(f"wrote {args.output} (identity_pass={identity}, logits_bitwise_pass={bitwise})")
    return 0 if identity else 1


if __name__ == "__main__":
    sys.exit(main())
