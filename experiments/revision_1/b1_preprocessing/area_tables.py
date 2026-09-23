"""B1-1: effective-area distributions of target cells and matched controls.

Reads ``effective_area_px`` from the four ImageNet K=3 dumps and writes:

- ``cell_area.csv``: per model, protocol, and cell, the distribution of the
  target cell's effective fraction of the 224x224 model input
- ``within_sample_area_ratio.csv``: per sample, largest / smallest cell area
- ``control_area_ratio.csv``: control / target effective-area ratios at item
  level and at the anchor level ``ssat.analysis.indexer`` uses, with the share
  within each tolerance; the anchor-level share at 0.05 must reproduce the
  stored ``area_matched`` flag
- ``zero_area_targets.csv``: grades of target cells that fall entirely
  outside the model input (effective area 0)

Example:
    python experiments/revision_1/b1_preprocessing/area_tables.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from experiments.revision_1.a1_threshold.build_features import cell_columns
from experiments.revision_1.common.loading import iter_latest_perturbed, load_reliability, load_spatial_profile
from experiments.revision_1.common.provenance import write_provenance
from experiments.revision_1.common.runs import BASELINE_RUNS, RunRef

B1_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = B1_DIR / "protocol.json"
SUMMARY_DIR = B1_DIR / "summary"
RUNS = (
    "imagenet_mnv2_050_exact_k3",
    "imagenet_mnv2_050_crop_free_k3",
    "imagenet_mnv2_100_exact_k3",
    "imagenet_mnv2_100_crop_free_k3",
)
SQUARE_EXACT_RATIO = (256 / 4) ** 2 / (256 / 4 - (256 - 224) / 2) ** 2
"""Center-cell / corner-cell area for a square image resized to 256 and center-cropped to 224."""
QUANTILES = (5, 25, 50, 75, 95)


def load_protocol() -> dict:
    """Return the pre-registered B1 protocol."""

    return json.loads(PROTOCOL_PATH.read_text())


def item_areas(run: RunRef) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(targets, controls)`` item frames with effective areas.

    ``targets`` has one row per (sample, cell) after checking that every
    target item of a cell reports the same area. ``controls`` has one row
    per control item with its ``target_key`` and control ``region_key``.

    Raises:
        ValueError: If a target cell's area differs across its items.
    """

    columns = ["sample_id", "region_id", "region_instance_id", "region_params_json", "is_control", "effective_area_px"]
    table = pa.concat_tables(iter_latest_perturbed(run.dump, columns)).select(columns).to_pandas()
    table["region_key"] = table["region_id"] + "::" + table["region_instance_id"]

    targets = table[~table["is_control"]]
    per_cell = targets.groupby(["sample_id", "region_key"], sort=False)["effective_area_px"].agg(["min", "max"])
    if (per_cell["min"] != per_cell["max"]).any():
        raise ValueError(f"{run.name}: a target cell reports different effective areas across items")
    targets = per_cell["min"].rename("area").reset_index()

    controls = table[table["is_control"]].copy()
    target_of = {}
    for params in controls["region_params_json"].unique():
        target = json.loads(params)["target_region"]
        target_of[params] = f"{target['region_id']}::{target['region_instance_id']}"
    controls["target_key"] = controls["region_params_json"].map(target_of)
    controls = controls[["sample_id", "region_key", "target_key", "effective_area_px"]].rename(
        columns={"effective_area_px": "area"}
    )
    return targets, controls


def _quantiles(values: pd.Series | np.ndarray, prefix: str = "") -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {
        f"{prefix}mean": float(values.mean()),
        **{f"{prefix}p{q}": float(np.percentile(values, q)) for q in QUANTILES},
        f"{prefix}min": float(values.min()),
        f"{prefix}max": float(values.max()),
    }


def cell_area_rows(run: RunRef, targets: pd.DataFrame, protocol: dict) -> list[dict[str, object]]:
    """Effective-fraction distribution per cell."""

    input_px = protocol["b1_1_area"]["model_input_px"]
    nominal = protocol["b1_1_area"]["nominal_fraction"]
    frame = pd.concat([targets, cell_columns(targets["region_key"])], axis=1)
    frame["fraction"] = frame["area"] / input_px
    rows = []
    for (cell, cell_type), group in frame.groupby(["region_cell", "cell_type"], sort=True):
        stats = _quantiles(group["fraction"], "fraction_")
        rows.append(
            {
                "model": run.model,
                "protocol": run.protocol,
                "cell": cell,
                "cell_type": cell_type,
                "n": len(group),
                "n_zero_area": int((group["area"] == 0).sum()),
                **stats,
                "mean_ratio_to_nominal": stats["fraction_mean"] / nominal,
            }
        )
    return rows


def within_sample_rows(run: RunRef, targets: pd.DataFrame) -> list[dict[str, object]]:
    """Distribution over samples of max cell area / min cell area.

    Samples with a cell entirely outside the model input (area 0) have an
    unbounded ratio; they are counted separately and left out of the quantiles.
    """

    per_sample = targets.groupby("sample_id")["area"].agg(["min", "max"])
    bounded = per_sample[per_sample["min"] > 0]
    ratio = bounded["max"] / bounded["min"]
    return [
        {
            "model": run.model,
            "protocol": run.protocol,
            "n_samples": len(per_sample),
            "n_samples_with_zero_area_cell": int((per_sample["min"] == 0).sum()),
            **_quantiles(ratio, "ratio_"),
            "share_ratio_ge_1_78_of_bounded": float((ratio >= 1.78).mean()),
            "analytic_square_exact": SQUARE_EXACT_RATIO,
        }
    ]


def zero_area_rows(run: RunRef, targets: pd.DataFrame) -> list[dict[str, object]]:
    """Grades and margin_drop of target cells that lie entirely outside the model input."""

    zero = targets[targets["area"] == 0]
    if zero.empty:
        return []
    keys = ["sample_id", "region_key"]
    grades = load_reliability(run).set_index(keys)["reliability_grade"]
    profile = load_spatial_profile(run).set_index(keys)["degradation"]
    frame = zero.join(grades, on=keys).join(profile, on=keys)
    frame["region_cell"] = cell_columns(frame["region_key"])["region_cell"].to_numpy()
    rows = []
    for cell, group in [("all", frame), *frame.groupby("region_cell", sort=True)]:
        counts = group["reliability_grade"].value_counts()
        rows.append(
            {
                "model": run.model,
                "protocol": run.protocol,
                "cell": cell,
                "n_zero_area": len(group),
                "n_samples": group["sample_id"].nunique(),
                **{f"n_{grade}": int(counts.get(grade, 0)) for grade in ("high", "moderate", "low", "unreliable")},
                "margin_drop_mean": float(group["degradation"].mean()),
                "margin_drop_abs_max": float(group["degradation"].abs().max()),
            }
        )
    return rows


def control_ratio_rows(
    run: RunRef, targets: pd.DataFrame, controls: pd.DataFrame, tolerances: Sequence[float]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Item- and anchor-level control/target area ratios and the area_matched parity result."""

    target_area = targets.set_index(["sample_id", "region_key"])["area"]
    item = controls.join(target_area.rename("target_area"), on=["sample_id", "target_key"])
    item["ratio"] = item["area"] / item["target_area"]

    anchors = item.groupby(["sample_id", "region_key", "target_key"], sort=False).agg(
        area=("area", "mean"), target_area=("target_area", "first")
    )
    # ssat.analysis.indexer._mean_area: int(round(mean)) over the anchor's items.
    anchors["area"] = anchors["area"].map(lambda value: int(round(value)))
    anchors["ratio"] = anchors["area"] / anchors["target_area"]
    anchors = anchors.reset_index()

    cells = cell_columns(item["target_key"])
    item["cell_type"] = cells["cell_type"].to_numpy()
    anchors["cell_type"] = cell_columns(anchors["target_key"])["cell_type"].to_numpy()

    rows = []
    for level, frame in (("item", item), ("anchor", anchors)):
        for cell_type, group in [("all", frame), *frame.groupby("cell_type", sort=True)]:
            base = {
                "model": run.model,
                "protocol": run.protocol,
                "level": level,
                "cell_type": cell_type,
                "n": len(group),
                # Ratios against a zero-area target are undefined; ssat counts them as not matched.
                "n_target_area_zero": int((group["target_area"] == 0).sum()),
            }
            stats = _quantiles(group["ratio"], "ratio_")
            for tolerance in tolerances:
                within = (group["ratio"] >= 1 - tolerance) & (group["ratio"] <= 1 + tolerance)
                row = {**base, "tolerance": tolerance, "share_within": float(within.mean()), **stats}
                if level == "anchor":
                    all_within = within.groupby([group["sample_id"], group["target_key"]]).all()
                    row["target_share_all_controls_within"] = float(all_within.mean())
                rows.append(row)

    within = (anchors["ratio"] >= 0.95) & (anchors["ratio"] <= 1.05)
    predicted = within.groupby([anchors["sample_id"], anchors["target_key"]]).all()
    stored = load_reliability(run).set_index(["sample_id", "region_key"])["area_matched"]
    joined = pd.concat({"predicted": predicted, "stored": stored}, axis=1, join="outer")
    mismatches = int((joined["predicted"].map({True: "true", False: "false"}) != joined["stored"]).sum())
    parity = {
        "run": run.name,
        "n_targets": int(len(joined)),
        "area_matched_true_share": float((joined["stored"] == "true").mean()),
        "mismatches": mismatches,
        "passed": mismatches == 0,
    }
    return rows, parity


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, default=SUMMARY_DIR)
    args = parser.parse_args(argv)
    protocol = load_protocol()
    tolerances = protocol["b1_1_area"]["control_area"]["tolerances"]

    cell_rows, sample_rows, ratio_rows, zero_rows, parities = [], [], [], [], []
    target_areas: dict[str, pd.Series] = {}
    for name in RUNS:
        run = BASELINE_RUNS[name]
        targets, controls = item_areas(run)
        target_areas[name] = targets.set_index(["sample_id", "region_key"])["area"].sort_index()
        cell_rows += cell_area_rows(run, targets, protocol)
        sample_rows += within_sample_rows(run, targets)
        zero_rows += zero_area_rows(run, targets)
        rows, parity = control_ratio_rows(run, targets, controls, tolerances)
        ratio_rows += rows
        parities.append(parity)
        print(f"{name}: {len(targets)} target cells, {len(controls)} control items, area_matched parity {parity}")

    for protocol_name in ("exact", "crop_free"):
        a = target_areas[f"imagenet_mnv2_050_{protocol_name}_k3"]
        b = target_areas[f"imagenet_mnv2_100_{protocol_name}_k3"]
        if not a.equals(b):
            raise SystemExit(f"{protocol_name}: target areas differ between the two models")
    failed = [parity for parity in parities if not parity["passed"]]
    if failed:
        raise SystemExit(f"area_matched parity failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cell_rows).to_csv(args.output_dir / "cell_area.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(args.output_dir / "within_sample_area_ratio.csv", index=False)
    pd.DataFrame(ratio_rows).to_csv(args.output_dir / "control_area_ratio.csv", index=False)
    pd.DataFrame(zero_rows).to_csv(args.output_dir / "zero_area_targets.csv", index=False)
    write_provenance(
        args.output_dir,
        inputs={"protocol": PROTOCOL_PATH, **{name: BASELINE_RUNS[name].dump for name in RUNS}},
        extra={"area_matched_parity": parities, "target_areas_identical_across_models": True},
        filename="area_tables.provenance.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
