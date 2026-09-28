#!/usr/bin/env python3
"""A3 run checks (``protocol.json`` ``run_checks``) on the finished runs.

For every A3 run, against the mobilenetv2_050 baseline of the same protocol:

* weights: the cached weight file's SHA-256 and the manifest's model name and
  preprocessing fingerprint equal ``model_selection.json``;
* clean: every sample's ``content_hash`` equals the baseline's;
* items: the item_id sets are equal, with equal ``seed_used`` and
  ``region_params_json``; ``effective_area_px`` equals the baseline's for
  every item when the geometry is shared, and for ``deit_small`` exact the
  target areas equal the areas recomputed with its own ``TimmAdapter``
  (control areas are summarized, not gated).

Writes ``summary/run_checks.json``; exits 1 if any gated check fails. Run with
``HF_HOME=/workspace/data/hf_cache``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from experiments.revision_1.a3_multi_arch.inspect_models import GRID, cell_areas, source_shapes, weight_file
from experiments.revision_1.common.loading import iter_latest_perturbed
from experiments.revision_1.common.provenance import write_json_shared, write_provenance
from experiments.revision_1.common.runs import A3_RUNS, BASELINE_RUNS, RunRef
from experiments.revision_1.phase0.verify_imagenet_images import recorded_hashes
from ssat.utils.io import load_json

A3_DIR = Path(__file__).resolve().parent
SUMMARY_DIR = A3_DIR / "summary"
SELECTION_PATH = SUMMARY_DIR / "model_selection.json"
_COLUMNS = ("item_id", "region_instance_id", "region_params_json", "seed_used", "effective_area_px", "is_control", "sample_id")


def baseline_for(run: RunRef) -> RunRef:
    """Return the mobilenetv2_050 K=3 baseline with the same protocol."""

    return BASELINE_RUNS[f"imagenet_mnv2_050_{run.protocol}_k3"]


def _items(run: RunRef) -> pd.DataFrame:
    return pa.concat_tables(iter_latest_perturbed(run.dump, _COLUMNS)).select(list(_COLUMNS)).to_pandas()


def check_weights(run: RunRef, selection: dict) -> dict[str, object]:
    """Compare the manifest and cached weight file with ``model_selection.json``."""

    spec = load_json(run.dump / "run_manifest.json")["adapter_spec"]
    record = selection["candidates"][spec["model_name"]]
    expected = record["adapter"][run.protocol]
    cached = weight_file(record["hf_hub_id"])
    return {
        "model_name": spec["model_name"],
        "weights_sha256": cached["sha256"],
        "sha256_equal": cached["sha256"] == record["weights"]["sha256"],
        "fingerprint_equal": spec["preprocessing_fingerprint"] == expected["preprocessing_fingerprint"],
        "pass": cached["sha256"] == record["weights"]["sha256"] and spec["preprocessing_fingerprint"] == expected["preprocessing_fingerprint"],
    }


def check_clean(run: RunRef, reference: dict[str, str]) -> dict[str, object]:
    """Compare per-sample ``content_hash`` with the reference hashes."""

    hashes = recorded_hashes(run.dump)
    mismatched = sorted(sid for sid, digest in hashes.items() if reference.get(sid) != digest)
    return {"n_samples": len(hashes), "n_reference": len(reference), "n_mismatched": len(mismatched),
            "pass": len(hashes) == len(reference) and not mismatched}


def expected_target_areas(run: RunRef, name: str) -> pd.DataFrame:
    """Recompute ``sample_id, cell, area`` for the run's own geometry from the source image shapes."""

    from ssat.core.adapter.timm_adapter import TimmAdapter

    samples = source_shapes("imagenet_mnv2_050_exact_k3")
    unique = samples[["height", "width"]].drop_duplicates().reset_index(drop=True)
    mode = "model_default" if run.protocol == "exact" else "squash"
    areas = cell_areas(TimmAdapter(name, pretrained=True, device="cpu", geometry_mode=mode), unique.to_numpy())
    per_shape = pd.DataFrame(areas, columns=range(GRID * GRID)).assign(height=unique["height"], width=unique["width"])
    merged = samples.merge(per_shape, on=["height", "width"])
    return merged.melt(id_vars=["sample_id", "height", "width"], var_name="cell", value_name="area")[["sample_id", "cell", "area"]]


def check_items(run: RunRef, baseline: RunRef, name: str, shared_geometry: bool) -> dict[str, object]:
    """Compare item identity and effective areas with the baseline."""

    new, old = _items(run), _items(baseline)
    merged = new.merge(old, on="item_id", how="outer", suffixes=("", "_ref"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    result: dict[str, object] = {
        "n_items": len(new), "n_reference_items": len(old),
        "n_only_in_run": int((merged["_merge"] == "left_only").sum()),
        "n_only_in_reference": int((merged["_merge"] == "right_only").sum()),
        "n_seed_mismatch": int((both["seed_used"] != both["seed_used_ref"]).sum()),
        "n_params_mismatch": int((both["region_params_json"] != both["region_params_json_ref"]).sum()),
        "shared_geometry": shared_geometry,
    }
    differ = both["effective_area_px"] != both["effective_area_px_ref"]
    for label, subset in (("targets", ~both["is_control"]), ("controls", both["is_control"])):
        result[f"n_{label}_area_differ_from_reference"] = int((differ & subset).sum())
    identity = (result["n_only_in_run"] == 0 and result["n_only_in_reference"] == 0
                and result["n_seed_mismatch"] == 0 and result["n_params_mismatch"] == 0)
    if shared_geometry:
        areas_ok = int(differ.sum()) == 0
    else:
        targets = both[~both["is_control"]]
        cells = targets["region_instance_id"].str.extract(r"r(\d+)/c(\d+)$").astype(int)
        stored = pd.DataFrame({"sample_id": targets["sample_id"].to_numpy(),
                               "cell": (cells[0] * GRID + cells[1]).to_numpy(),
                               "area": targets["effective_area_px"].to_numpy()})
        joined = stored.merge(expected_target_areas(run, name), on=["sample_id", "cell"], how="left", suffixes=("", "_expected"))
        result["n_targets_area_differ_from_own_geometry"] = int((joined["area"] != joined["area_expected"]).sum())
        controls = both[both["is_control"]]
        ratio = controls["effective_area_px"] / controls["effective_area_px_ref"].where(controls["effective_area_px_ref"] > 0)
        result["control_area_ratio_to_reference"] = {f"p{q}": float(np.nanpercentile(ratio, q)) for q in (5, 50, 95)}
        areas_ok = result["n_targets_area_differ_from_own_geometry"] == 0
    result["pass"] = bool(identity and areas_ok)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=list(A3_RUNS))
    parser.add_argument("--output", type=Path, default=SUMMARY_DIR / "run_checks.json")
    args = parser.parse_args(argv)

    selection = load_json(SELECTION_PATH)
    reference_hashes = recorded_hashes(BASELINE_RUNS["imagenet_mnv2_050_exact_k3"].dump)
    results = {}
    for name in args.runs:
        started = time.perf_counter()
        run = A3_RUNS[name]
        weights = check_weights(run, selection)
        shared = run.protocol == "crop_free" or selection["candidates"][weights["model_name"]]["condition_3_geometry_equal_reference"]
        results[name] = {
            "baseline": baseline_for(run).name,
            "weights": weights,
            "clean": check_clean(run, reference_hashes),
            "items": check_items(run, baseline_for(run), weights["model_name"], shared),
        }
        results[name]["pass"] = all(results[name][key]["pass"] for key in ("weights", "clean", "items"))
        print(f"{name}: pass={results[name]['pass']} ({time.perf_counter() - started:.0f}s)", flush=True)
    payload = {"all_pass": all(entry["pass"] for entry in results.values()), "runs": results}
    write_json_shared(args.output, payload)
    inputs = {"model_selection": SELECTION_PATH}
    for name in args.runs:
        inputs[name] = A3_RUNS[name].dump
        inputs[baseline_for(A3_RUNS[name]).name] = baseline_for(A3_RUNS[name]).dump
    write_provenance(args.output.parent, inputs=inputs, filename=f"{args.output.stem}.provenance.json")
    print(f"wrote {args.output} (all_pass={payload['all_pass']})")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
