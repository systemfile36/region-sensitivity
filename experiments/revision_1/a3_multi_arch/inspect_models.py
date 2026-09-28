#!/usr/bin/env python3
"""A3 step 0: check the candidate models against the pre-registered selection rule.

Implementation plan section 6.1. For the reference model (``mobilenetv2_050``)
and every candidate, in priority order per role:

* loads the timm pretrained checkpoint and records its data config, parameter
  count, Hugging Face hub id, and the SHA-256 and snapshot revision of the
  weight file;
* judges condition 1 (timm 1.0.x ImageNet-1k checkpoint), 2 (<= 50 M
  parameters, 224 input), and 3 (official eval geometry equal to the
  reference: input size, ``crop_pct``, ``crop_mode``, interpolation);
* builds ``TimmAdapter`` in both geometry modes and transforms the 16
  ``grid_4x4`` cell masks of every source image shape in the 10,000-sample
  set, comparing the model-space cell areas with the reference model's.

The reference model's areas must reproduce the ``effective_area_px`` stored in
the two ``mobilenetv2_050`` baseline dumps exactly; otherwise the script exits
with status 1. Writes ``summary/model_selection.json``, which is committed
before any A3 run.

Run with ``HF_HOME=/workspace/data/hf_cache`` so the weights land in the cache
the runs use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from experiments.revision_1.common.loading import iter_latest_perturbed
from experiments.revision_1.common.provenance import write_json_shared, write_provenance
from experiments.revision_1.common.runs import BASELINE_RUNS
from ssat.core.dump._storage import fragment_files
from ssat.utils.io import sha256_file

SUMMARY_DIR = Path(__file__).resolve().parent / "summary"
REFERENCE = "mobilenetv2_050.lamb_in1k"
REFERENCE_RUNS = {"model_default": "imagenet_mnv2_050_exact_k3", "squash": "imagenet_mnv2_050_crop_free_k3"}
CANDIDATES: dict[str, tuple[str, ...]] = {
    "cnn": ("convnext_tiny.fb_in1k", "resnet50.a1_in1k", "resnet50.tv2_in1k"),
    "transformer": (
        "deit_small_patch16_224.fb_in1k",
        "vit_small_patch16_224.augreg_in21k_ft_in1k",
        "swin_tiny_patch4_window7_224.ms_in1k",
    ),
}
AUTHOR_CHOICE = {"cnn": "convnext_tiny.fb_in1k", "transformer": "deit_small_patch16_224.fb_in1k"}
GEOMETRY_FIELDS = ("input_size", "crop_pct", "crop_mode", "interpolation")
MAX_PARAMS = 50_000_000
GRID = 4
MODES = ("model_default", "squash")
MODE_PROTOCOL = {"model_default": "exact", "squash": "crop_free"}


def cell_type(row: int, col: int) -> str:
    """Classify a 4x4 grid cell as ``corner``, ``edge``, or ``center``."""

    border = (row in (0, GRID - 1), col in (0, GRID - 1))
    return "corner" if all(border) else "edge" if any(border) else "center"


def grid_masks(height: int, width: int) -> np.ndarray:
    """Return the 16 ``grid_4x4`` cell masks, row-major, with ssat's integer boundaries."""

    masks = np.zeros((GRID * GRID, height, width), dtype=np.bool_)
    for row in range(GRID):
        for col in range(GRID):
            masks[row * GRID + col, row * height // GRID:(row + 1) * height // GRID,
                  col * width // GRID:(col + 1) * width // GRID] = True
    return masks


def source_shapes(run_name: str) -> pd.DataFrame:
    """Return ``sample_id, height, width`` from a baseline dump's clean rows."""

    tables = [pq.read_table(path, columns=["sample_id", "original_shape"])
              for _, path in fragment_files(BASELINE_RUNS[run_name].dump / "clean", "part")]
    frame = pa.concat_tables(tables).to_pandas().drop_duplicates("sample_id", keep="last")
    shapes = np.stack(frame["original_shape"].to_numpy())
    if shapes.shape[1] != 4 or not (shapes[:, 0] == 1).all() or not (shapes[:, 3] == 3).all():
        raise ValueError(f"{run_name}: expected original_shape (1, H, W, 3)")
    return pd.DataFrame({"sample_id": frame["sample_id"].to_numpy(), "height": shapes[:, 1], "width": shapes[:, 2]})


def stored_areas(run_name: str) -> pd.DataFrame:
    """Return ``sample_id, cell, area`` for the target cells of a baseline dump (one operator)."""

    selector = (pc.field("is_control") == pc.scalar(False)) & (pc.field("perturb_op") == pc.scalar("mean_fill"))
    columns = ("sample_id", "region_instance_id", "effective_area_px", "is_control", "perturb_op")
    frame = pa.concat_tables(iter_latest_perturbed(BASELINE_RUNS[run_name].dump, columns, filter_expression=selector)).to_pandas()
    frame = frame[["sample_id", "region_instance_id", "effective_area_px"]]
    frame = frame.drop_duplicates(["sample_id", "region_instance_id", "effective_area_px"])
    if frame.duplicated(["sample_id", "region_instance_id"]).any():
        raise ValueError(f"{run_name}: a target cell has more than one effective area")
    parts = frame["region_instance_id"].str.extract(r"r(\d+)/c(\d+)$").astype(int)
    return pd.DataFrame({"sample_id": frame["sample_id"], "cell": parts[0] * GRID + parts[1], "area": frame["effective_area_px"]})


def cell_areas(adapter, shapes: np.ndarray) -> np.ndarray:
    """Return model-space areas, shape ``(n_shapes, 16)``, for each ``(height, width)``."""

    return np.stack([adapter.transform_mask(grid_masks(int(h), int(w))).reshape(GRID * GRID, -1).sum(1) for h, w in shapes])


def weight_file(hf_hub_id: str) -> dict[str, object]:
    """Locate the cached weight file of a hub checkpoint and hash it."""

    from huggingface_hub import try_to_load_from_cache

    for filename in ("model.safetensors", "pytorch_model.bin"):
        path = try_to_load_from_cache(hf_hub_id, filename)
        if isinstance(path, str):
            path = Path(path)
            return {"filename": filename, "snapshot_revision": path.parent.name, "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size}
    raise FileNotFoundError(f"no cached weight file for {hf_hub_id}")


def describe_model(name: str, shapes: np.ndarray) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Load one checkpoint; return its record and per-mode cell areas."""

    import timm
    from timm.data import resolve_model_data_config

    from ssat.core.adapter.timm_adapter import TimmAdapter

    model = timm.create_model(name, pretrained=True)
    data_config = resolve_model_data_config(model)
    pretrained = model.pretrained_cfg
    record: dict[str, object] = {
        "hf_hub_id": pretrained.get("hf_hub_id"),
        "num_classes": int(model.num_classes),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "data_config": {key: (list(value) if isinstance(value, tuple) else value) for key, value in data_config.items()},
        "weights": weight_file(pretrained["hf_hub_id"]),
        "adapter": {},
    }
    del model
    areas = {}
    for mode in MODES:
        adapter = TimmAdapter(name, pretrained=True, device="cpu", geometry_mode=mode)
        spec = adapter.describe()
        record["adapter"][MODE_PROTOCOL[mode]] = {
            "geometry_mode": mode, "model_id": spec.model_id, "weights_id": spec.weights_id,
            "preprocessing_desc": spec.preprocessing_desc, "preprocessing_fingerprint": spec.preprocessing_fingerprint,
        }
        areas[mode] = cell_areas(adapter, shapes)
    return record, areas


def area_comparison(areas: np.ndarray, reference: np.ndarray, weights: np.ndarray, input_px: int) -> dict[str, object]:
    """Summarize model-space cell areas against the reference, weighting shapes by sample count."""

    types = np.array([cell_type(index // GRID, index % GRID) for index in range(GRID * GRID)])
    differ = areas != reference
    result: dict[str, object] = {
        "n_sample_cells_differ": int((differ * weights[:, None]).sum()),
        "share_sample_cells_differ": float((differ * weights[:, None]).sum() / (weights.sum() * GRID * GRID)),
        "by_cell_type": {},
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(reference > 0, areas / reference, np.nan)
    for kind in ("corner", "edge", "center"):
        selected = types == kind
        fraction = (areas[:, selected] * weights[:, None]).sum() / (weights.sum() * selected.sum() * input_px)
        result["by_cell_type"][kind] = {
            "mean_area_fraction": float(fraction),
            "mean_ratio_to_reference": float(np.nansum(ratio[:, selected] * weights[:, None]) / (weights[:, None] * ~np.isnan(ratio[:, selected])).sum()),
        }
    return result


def judge(name: str, record: dict[str, object], reference: dict[str, object], timm_version: str) -> dict[str, object]:
    """Evaluate selection conditions 1-3 for one candidate."""

    config, ref_config = record["data_config"], reference["data_config"]
    mismatched = [field for field in GEOMETRY_FIELDS if config.get(field) != ref_config.get(field)]
    return {
        "condition_1_timm_1_0_imagenet_1k": timm_version.startswith("1.0.") and record["num_classes"] == 1000,
        "condition_2_params_le_50M_and_224": record["n_parameters"] <= MAX_PARAMS and config["input_size"] == [3, 224, 224],
        "condition_3_geometry_equal_reference": not mismatched,
        "condition_3_mismatched_fields": {field: {"candidate": config.get(field), "reference": ref_config.get(field)} for field in mismatched},
    }


def main(argv: list[str] | None = None) -> int:
    import timm

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=SUMMARY_DIR / "model_selection.json")
    args = parser.parse_args(argv)

    samples = source_shapes(REFERENCE_RUNS["model_default"])
    unique = samples.groupby(["height", "width"]).size().rename("n").reset_index()
    shapes = unique[["height", "width"]].to_numpy()
    weights = unique["n"].to_numpy()
    print(f"{len(samples)} samples, {len(shapes)} distinct source shapes", flush=True)

    reference, reference_areas = describe_model(REFERENCE, shapes)
    shape_row = {(h, w): i for i, (h, w) in enumerate(shapes)}
    sample_rows = samples.assign(row=[shape_row[(h, w)] for h, w in zip(samples["height"], samples["width"])])
    reference_check = {}
    for mode, run_name in REFERENCE_RUNS.items():
        stored = stored_areas(run_name).merge(sample_rows[["sample_id", "row"]], on="sample_id", how="left")
        computed = reference_areas[mode][stored["row"].to_numpy(), stored["cell"].to_numpy()]
        reference_check[MODE_PROTOCOL[mode]] = {
            "run": run_name, "n_sample_cells": len(stored), "n_mismatch": int((computed != stored["area"].to_numpy()).sum()),
        }
        print(f"reference {mode}: {reference_check[MODE_PROTOCOL[mode]]}", flush=True)
    reference_ok = all(entry["n_mismatch"] == 0 and entry["n_sample_cells"] == len(samples) * GRID * GRID
                       for entry in reference_check.values())

    candidates: dict[str, dict[str, object]] = {}
    for role, names in CANDIDATES.items():
        for priority, name in enumerate(names, start=1):
            print(f"inspecting {name} ...", flush=True)
            record, areas = describe_model(name, shapes)
            record.update(role=role, priority=priority, **judge(name, record, reference, timm.__version__))
            record["passes"] = all(record[key] for key in (
                "condition_1_timm_1_0_imagenet_1k", "condition_2_params_le_50M_and_224", "condition_3_geometry_equal_reference"))
            _, height, width = record["data_config"]["input_size"]
            record["area_vs_reference"] = {MODE_PROTOCOL[mode]: area_comparison(areas[mode], reference_areas[mode], weights, height * width)
                                           for mode in MODES}
            candidates[name] = record

    selection = {}
    for role, names in CANDIDATES.items():
        first = next((name for name in names if candidates[name]["passes"]), None)
        choice = AUTHOR_CHOICE[role]
        passes = bool(candidates[choice]["passes"])
        selection[role] = {
            "first_passing_candidate": first,
            "selected": choice,
            "selected_passes_all_conditions": passes,
            "cross_model_comparison": "exact and crop_free" if passes else "crop_free only; exact results are per-model reference (implementation plan sections 4.5, 6.1)",
        }

    payload = {
        "rule": "implementation plan section 6.1: (1) timm 1.0.x standard ImageNet-1k pretrained checkpoint, (2) <= 50M parameters and 224 input, "
                "(3) official eval geometry equal to mobilenetv2_050 (input_size, crop_pct, crop_mode, interpolation); the first passing "
                "candidate in priority order is adopted; if none passes, architecture comparison uses crop-free results only (section 4.5).",
        "timm_version": timm.__version__,
        "reference": {"model": REFERENCE, **reference, "stored_area_check": reference_check},
        "source_shapes": {"from_run": REFERENCE_RUNS["model_default"], "n_samples": len(samples), "n_distinct": len(shapes)},
        "candidates": candidates,
        "selection": selection,
        "reference_area_check_pass": reference_ok,
    }
    write_json_shared(args.output, payload)
    write_provenance(args.output.parent, inputs={name: BASELINE_RUNS[name].dump for name in REFERENCE_RUNS.values()},
                     filename=f"{args.output.stem}.provenance.json")
    for role, entry in selection.items():
        print(f"{role}: {entry}")
    print(f"wrote {args.output} (reference_area_check_pass={reference_ok})")
    return 0 if reference_ok else 1


if __name__ == "__main__":
    sys.exit(main())
