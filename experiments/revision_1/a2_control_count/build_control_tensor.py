#!/usr/bin/env python3
"""A2 step 1: per-anchor target values and the control tensor of a K=20 run.

For one metric, ``T[a, c]`` is target anchor ``a``'s seed-mean under
condition ``c`` and ``C[a, c, k]`` is the seed-mean of the control anchor with
``control_index`` ``k`` matched to ``a`` (read from the parsed
``region_params_json``). Both use the grouping of
``ssat.analysis.control._anchor_level_means``.

``compare_to_controls`` lists a target's controls in control ``region_key``
order (``0, 1, 10, ..., 19, 2, ...``), and its mean/std depend on that order
in the last bits. ``stat_order`` records it, and ``subset_stats`` uses it, so
the full K=20 set reproduces the stored ``control_comparison`` exactly; the
build fails otherwise.

Output: ``<output-dir>/<run>__<metric>.npz`` and a ``.json`` sidecar.

Example:
    python experiments/revision_1/a2_control_count/build_control_tensor.py --runs all
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.revision_1.common.loading import load_item_values, read_metric_table
from experiments.revision_1.common.provenance import write_json_shared
from experiments.revision_1.common.runs import A2_RESULTS_DIR, A2_RUNS, RunRef
from ssat.analysis.control import _perturb_params_hash_column
from ssat.analysis.types import region_key, region_key_column
from ssat.utils.io import load_json, sha256_file

ANCHOR = ["sample_id", "region_key", "invert_mask"]
GROUP = [*ANCHOR, "perturb_op", "perturb_params_hash", "metric_name"]
DEFAULT_OUTPUT_DIR = A2_RESULTS_DIR / "tensors"


@dataclass(frozen=True)
class ControlTensor:
    """Target values and per-index control values of one run and metric.

    Attributes:
        sample_id, region_key, invert_mask: Target anchor keys, sorted.
        conditions: ``(perturb_op, perturb_params_hash)`` pairs, sorted.
        target: ``T``, shape ``(anchors, conditions)``.
        controls: ``C``, shape ``(anchors, conditions, K)``, indexed by control_index.
        stat_order: Control indices in the order ``compare_to_controls`` lists them.
    """

    sample_id: np.ndarray
    region_key: np.ndarray
    invert_mask: np.ndarray
    conditions: tuple[tuple[str, str], ...]
    target: np.ndarray
    controls: np.ndarray
    stat_order: np.ndarray

    @property
    def n_controls(self) -> int:
        return self.controls.shape[-1]

    def save(self, path: Path) -> None:
        """Write the tensor as an ``.npz`` without pickled objects."""

        np.savez(
            path,
            sample_id=self.sample_id.astype(str),
            region_key=self.region_key.astype(str),
            invert_mask=self.invert_mask,
            condition_op=np.array([op for op, _ in self.conditions], dtype=str),
            condition_hash=np.array([params for _, params in self.conditions], dtype=str),
            target=self.target,
            controls=self.controls,
            stat_order=self.stat_order,
        )

    @classmethod
    def load(cls, path: Path) -> "ControlTensor":
        """Read a tensor written by ``save``."""

        with np.load(path, allow_pickle=False) as data:
            return cls(
                sample_id=data["sample_id"].astype(object),
                region_key=data["region_key"].astype(object),
                invert_mask=data["invert_mask"],
                conditions=tuple(zip(data["condition_op"].tolist(), data["condition_hash"].tolist())),
                target=data["target"],
                controls=data["controls"],
                stat_order=data["stat_order"],
            )


def anchor_means(item_values: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Seed-mean per (anchor, condition, metric), as ``_anchor_level_means`` computes it.

    Returns:
        The means with an ``is_control`` column, and each control anchor's
        ``region_params_json``.
    """

    frame = item_values.assign(
        region_key=region_key_column(item_values),
        perturb_params_hash=_perturb_params_hash_column(item_values),
    )
    means = frame[frame["available"]].groupby(GROUP, sort=False)["degradation"].mean()
    is_control = frame.groupby(ANCHOR, sort=False)["is_control"].first().rename("is_control")
    params = frame[frame["is_control"]].groupby(ANCHOR, sort=False)["region_params_json"].first()
    return means.reset_index().merge(is_control.reset_index(), on=ANCHOR, how="left"), params


def control_targets(params: pd.Series) -> pd.DataFrame:
    """Map each control anchor to its target region key and control index.

    Raises:
        ValueError: If a control has no usable ``target_region`` recipe or a
            ``control_request_index`` other than 0.
    """

    rows = []
    for (sample_id, control_key, invert_mask), text in params.items():
        parsed = json.loads(text)
        target = parsed.get("target_region") or {}
        if not target.get("region_id") or not target.get("region_instance_id"):
            raise ValueError(f"{control_key}: control without a target_region recipe")
        if parsed.get("control_request_index") != 0:
            raise ValueError(f"{control_key}: unexpected control_request_index {parsed.get('control_request_index')}")
        rows.append(
            (sample_id, control_key, invert_mask, region_key(target["region_id"], target["region_instance_id"]),
             int(parsed["control_index"]))
        )
    return pd.DataFrame(rows, columns=["sample_id", "control_key", "invert_mask", "region_key", "control_index"])


def build_tensor(item_values: pd.DataFrame, metric: str) -> ControlTensor:
    """Assemble ``T``, ``C``, and ``stat_order`` from one metric's item values.

    Raises:
        ValueError: If any target or control value is missing, or the control
            listing order differs between targets.
    """

    item_values = item_values[item_values["metric_name"] == metric]
    means, params = anchor_means(item_values)
    mapping = control_targets(params)

    targets = means[~means["is_control"]]
    anchors = targets[ANCHOR].drop_duplicates().sort_values(ANCHOR, kind="stable").reset_index(drop=True)
    conditions = tuple(sorted(set(zip(means["perturb_op"], means["perturb_params_hash"]))))
    anchor_pos = pd.Series(np.arange(len(anchors)), index=pd.MultiIndex.from_frame(anchors))
    condition_pos = {condition: index for index, condition in enumerate(conditions)}
    n_controls = int(mapping["control_index"].max()) + 1

    target = np.full((len(anchors), len(conditions)), np.nan)
    rows = anchor_pos.reindex(pd.MultiIndex.from_frame(targets[ANCHOR])).to_numpy()
    cols = [condition_pos[key] for key in zip(targets["perturb_op"], targets["perturb_params_hash"])]
    target[rows, cols] = targets["degradation"].to_numpy(dtype=float)

    controls_frame = means[means["is_control"]].rename(columns={"region_key": "control_key"})
    controls_frame = controls_frame.merge(mapping, on=["sample_id", "control_key", "invert_mask"], how="left")
    if controls_frame["region_key"].isna().any():
        raise ValueError("control values without a parsed target mapping")
    rows = anchor_pos.reindex(pd.MultiIndex.from_frame(controls_frame[ANCHOR])).to_numpy()
    if np.isnan(rows.astype(float)).any():
        raise ValueError("controls reference a target anchor without values")
    controls = np.full((len(anchors), len(conditions), n_controls), np.nan)
    cols = [condition_pos[key] for key in zip(controls_frame["perturb_op"], controls_frame["perturb_params_hash"])]
    controls[rows.astype(np.int64), cols, controls_frame["control_index"].to_numpy()] = controls_frame["degradation"].to_numpy(dtype=float)

    if np.isnan(target).any() or np.isnan(controls).any():
        raise ValueError(
            f"{metric}: incomplete tensor ({int(np.isnan(target).sum())} target and "
            f"{int(np.isnan(controls).sum())} control values missing)"
        )
    orders = (
        mapping.sort_values(["sample_id", "region_key", "invert_mask", "control_key"], kind="stable")
        .groupby(ANCHOR, sort=False)["control_index"]
        .agg(tuple)
    )
    distinct = set(orders)
    if len(distinct) != 1:
        raise ValueError(f"control listing order differs between targets: {sorted(distinct)[:3]}")
    return ControlTensor(
        sample_id=anchors["sample_id"].to_numpy(dtype=object),
        region_key=anchors["region_key"].to_numpy(dtype=object),
        invert_mask=anchors["invert_mask"].to_numpy(dtype=bool),
        conditions=conditions,
        target=target,
        controls=controls,
        stat_order=np.array(next(iter(distinct)), dtype=np.int64),
    )


def subset_stats(tensor: ControlTensor, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the control mean, ddof-0 std, and count over a set of control indices.

    Values are reduced in ``stat_order`` so that the full set reproduces
    ``compare_to_controls`` bit for bit.
    """

    chosen = set(int(index) for index in indices)
    order = [int(index) for index in tensor.stat_order if int(index) in chosen]
    if len(order) != len(chosen):
        raise ValueError(f"indices {sorted(chosen)} not all in 0..{tensor.n_controls - 1}")
    values = np.ascontiguousarray(tensor.controls[:, :, order])
    return values.mean(axis=-1), values.std(axis=-1), len(order)


def stored_parity(tensor: ControlTensor, analysis_dir: Path, metric: str) -> dict[str, object]:
    """Compare the full-set statistics with the stored ``control_comparison`` rows."""

    stored = read_metric_table(
        analysis_dir / "control_comparison.parquet",
        [metric],
        columns=(*ANCHOR, "perturb_op", "perturb_params_hash", "metric_name", "control_mean",
                 "control_std", "n_controls", "excess", "z_vs_control"),
    )
    mean, std, n = subset_stats(tensor, range(tensor.n_controls))
    with np.errstate(divide="ignore", invalid="ignore"):
        excess = tensor.target - mean
        z = np.where((std != 0) & (n >= 2), excess / std, np.nan)
    anchors = pd.DataFrame({"sample_id": tensor.sample_id, "region_key": tensor.region_key, "invert_mask": tensor.invert_mask})
    frames = []
    for index, (op, params_hash) in enumerate(tensor.conditions):
        frames.append(anchors.assign(perturb_op=op, perturb_params_hash=params_hash, t_mean=mean[:, index],
                                     t_std=std[:, index], t_excess=excess[:, index], t_z=z[:, index]))
    joined = stored.merge(pd.concat(frames, ignore_index=True), on=[*ANCHOR, "perturb_op", "perturb_params_hash"],
                          how="outer", indicator=True)
    both = joined[joined["_merge"] == "both"]
    result: dict[str, object] = {
        "n_stored": int(len(stored)),
        "n_unmatched": int((joined["_merge"] != "both").sum()),
        "n_controls_values": sorted(int(value) for value in both["n_controls"].unique()),
    }
    for field, ours in (("control_mean", "t_mean"), ("control_std", "t_std"), ("excess", "t_excess"), ("z_vs_control", "t_z")):
        a = both[field].to_numpy(dtype=float)
        b = both[ours].to_numpy(dtype=float)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        finite = np.isfinite(a) & np.isfinite(b)
        result[f"{field}_bitwise_equal"] = int(same.sum())
        result[f"{field}_max_abs_diff"] = float(np.abs(a[finite] - b[finite]).max()) if finite.any() else 0.0
        result[f"{field}_nan_mismatch"] = int((np.isnan(a) != np.isnan(b)).sum())
    result["passed"] = (
        result["n_unmatched"] == 0
        and result["n_controls_values"] == [tensor.n_controls]
        and all(result[f"{field}_bitwise_equal"] == len(both) for field in ("control_mean", "control_std", "excess", "z_vs_control"))
    )
    return result


def tensor_paths(output_dir: Path, run_name: str, metric: str) -> tuple[Path, Path]:
    """Return the ``(npz, json)`` paths of one tensor."""

    stem = output_dir / f"{run_name}__{metric}"
    return stem.with_suffix(".npz"), stem.with_suffix(".json")


def build(run: RunRef, metric: str, output_dir: Path) -> dict[str, object]:
    """Build, check, and save one run's tensor; returns the sidecar content.

    Raises:
        ValueError: If the full-set statistics do not reproduce the stored rows.
    """

    started = time.monotonic()
    item_values = load_item_values(run, (metric,))
    tensor = build_tensor(item_values, metric)
    del item_values
    parity = stored_parity(tensor, run.analysis, metric)
    if not parity["passed"]:
        raise ValueError(f"{run.name}: K={tensor.n_controls} statistics differ from control_comparison: {parity}")
    npz_path, json_path = tensor_paths(output_dir, run.name, metric)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor.save(npz_path)
    npz_path.chmod(0o644)
    meta = {
        "run": run.name,
        "metric": metric,
        "n_anchors": int(len(tensor.sample_id)),
        "n_samples": int(len(set(tensor.sample_id))),
        "conditions": [list(condition) for condition in tensor.conditions],
        "n_controls": tensor.n_controls,
        "stat_order": tensor.stat_order.tolist(),
        "stored_parity": parity,
        "tensor_sha256": sha256_file(npz_path),
        "inputs": {
            "run_manifest_sha256": sha256_file(run.dump / "run_manifest.json"),
            "metrics_manifest_sha256": sha256_file(run.metrics / "metrics_manifest.json"),
            "analysis_manifest_sha256": sha256_file(run.analysis / "analysis_manifest.json"),
        },
        "elapsed_s": time.monotonic() - started,
    }
    write_json_shared(json_path, meta)
    return meta


def load_tensor(output_dir: Path, run_name: str, metric: str) -> tuple[ControlTensor, dict[str, object]]:
    """Load a tensor and its sidecar."""

    npz_path, json_path = tensor_paths(output_dir, run_name, metric)
    return ControlTensor.load(npz_path), load_json(json_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", nargs="+", default=["all"], help="run names from A2_RUNS, or 'all'")
    parser.add_argument("--metric", default="margin_drop")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    names = list(A2_RUNS) if args.runs == ["all"] else args.runs
    for name in names:
        meta = build(A2_RUNS[name], args.metric, args.output_dir)
        print(f"{name}: {meta['n_anchors']} anchors x {len(meta['conditions'])} conditions x "
              f"{meta['n_controls']} controls, stored parity ok ({meta['elapsed_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
