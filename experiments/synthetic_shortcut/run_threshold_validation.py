#!/usr/bin/env python3
"""Produce one extra L3 synthetic-shortcut run with control regions and
multiple seeds, to validate ssat.analysis's z_vs_control_threshold /
seed_cv_threshold defaults (2.0 / 0.2) -- something the five existing
shortcut_A_* runs cannot do, since run_audit.py's config has no ``controls``
entry and uses a single ``seed_salt`` (IMPLE_PLAN_CONTROL_STABILITY_v1.md §5
단계9 follow-up to docs/L3_Synthetic-Shortcut Experiment Report.md, whose
dataset answered Q1-Q5 with no control/seed axis at all).

Deliberately NOT added to run_audit.py's RUN_SPECS/_build_audit_config:
run_audit.py's own docstring documents "exactly seven combinations are
needed" for the already-reported L3 verdict, and evaluate.py iterates
RUN_SPECS assuming that fixed set -- adding an eighth, differently-shaped
run there would touch a script whose result is already reported and should
stay untouched. This script is a separate, additive validation run using
its own run_id (shortcut_A_constant_fill_thresholds) under the same
results/ tree, reusing only the constant_fill operator (already the
baseline used everywhere else in this experiment) -- cross-op comparison is
not the point here, only the control/seed axes are.

ssat.core already supports both axes with no code changes needed
(ssat/core/config/schema.py's ControlConfig + PerturbationSpec.seed_salts)
-- run_audit.py's config dict simply never populated them.

Run as: python3 experiments/synthetic_shortcut/run_threshold_validation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import GRID_COLS, GRID_ROWS, NUM_CLASSES, PATCH_REGION_ID
from run_audit import FILL_PARAMS, PRIMARY_METRIC
from ssat.application import AuditApplication, RunRequest
from ssat.core.estimate import EstimateOptions
from ssat.metrics.aggregate import aggregate_item_metrics
from ssat.metrics.builtin_metrics.continuous import MarginDrop
from ssat.metrics.dump_reader import DumpHandle
from ssat.metrics.registry import MetricRegistry
from ssat.metrics.store import save_metrics

RUN_ID = "shortcut_A_constant_fill_thresholds"

# 2 controls/target-region and 3 seeds/item are the smallest values that
# actually exercise both thresholds (z_vs_control needs n_controls>=2 for a
# std; seed_cv needs n_seeds>=2 for a std) while keeping the item count
# increase bounded: 16 regions * 3 seeds * (1 target + 2 controls) = 144
# items/sample (9x run_audit.py's 16 items/sample), vs. e.g. doubling both.
N_CONTROLS = 2
SEED_SALTS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one threshold-validation run."""

    parser = argparse.ArgumentParser(
        description="Run one L3 synthetic-shortcut audit with control regions and multiple seeds."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoints"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    return parser.parse_args()


def _build_config(checkpoint_path: Path, manifest_path: Path) -> dict:
    """Build the audit config: run_audit.py's constant_fill config plus controls + multi-seed.

    Mirrors run_audit.py's ``_build_audit_config`` exactly for the
    source/adapter/regions sections (constant_fill needs no
    ``dataset_stats`` -- only mean_fill does) and adds the two fields this
    script exists to exercise.

    Deliberately does NOT set the newly-available ``preprocessing`` field.
    m_shortcut.pt was trained through squeezenet1_0's stock ImageNet preset
    (train.py's module docstring explains that it replicates the adapter's
    transform on purpose), so declaring a crop-free pipeline here would feed
    the model an input distribution it never saw and make every degradation
    number meaningless. The known consequence is that the preset's
    CenterCrop still makes model-space area depend on position, so the
    control/target area match is imperfect -- A2's ``area_matched`` flag now
    reports that rather than hiding it. Removing the confound outright
    requires retraining under the same crop-free pipeline, which is a
    separate exercise.
    """

    return {
        "source": {"kind": "image_manifest", "manifest": str(manifest_path.resolve())},
        "adapter": {
            "provider": "torchvision",
            "model_name": "squeezenet1_0",
            "checkpoint": {"path": str(checkpoint_path.resolve()), "state_dict_key": "model"},
            "model_kwargs": {"num_classes": NUM_CLASSES},
            "device": "auto",
        },
        "regions": [
            {
                "region_id": PATCH_REGION_ID,
                "kind": "grid",
                "params": {"rows": GRID_ROWS, "cols": GRID_COLS},
            }
        ],
        "perturbations": [
            {
                "op": "constant_fill",
                "params": FILL_PARAMS["constant_fill"],
                "seed_salts": list(SEED_SALTS),
            }
        ],
        "controls": [{"match_area_of": PATCH_REGION_ID, "n_samples": N_CONTROLS}],
    }


def _compute_and_save_metrics(dump_root: Path, metrics_dir: Path) -> None:
    """Compute margin_drop item metrics for one dump and persist the aggregation.

    Duplicated from run_audit.py's helper of the same name rather than
    imported -- that module's own docstring on this function already
    documents why (production experiment code does not import from
    tests/, and there is no shared non-test module for it to live in).
    """

    handle = DumpHandle(dump_root)
    joined = handle.joined()
    resolved_config = handle.manifest.resolved_config

    registry = MetricRegistry()
    registry.register(MarginDrop())

    item_metrics = registry.compute_item_metrics(joined, adapter_spec=resolved_config.adapter_spec)
    result = aggregate_item_metrics(
        item_metrics, joined, registry, resolved_config, primary_metric=PRIMARY_METRIC
    )
    save_metrics(
        metrics_dir,
        item_metrics,
        result,
        registry=registry,
        primary_metric=PRIMARY_METRIC,
        source_run_manifest_path=handle.manifest_path,
        exclusion_summary=handle.summary(),
    )


def main() -> int:
    """Produce the control+multi-seed dump (if missing) and its metrics store (if missing)."""

    args = parse_args()

    output_dir = args.results_dir / "dumps" / RUN_ID
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"[{RUN_ID}] dump already exists, skipping audit run")
    else:
        config = _build_config(
            checkpoint_path=args.checkpoint_dir / "m_shortcut.pt",
            manifest_path=args.data_dir / "manifests" / "A_audit.json",
        )
        request = RunRequest(config, output_dir, estimate_options=EstimateOptions())
        application = AuditApplication()
        with application.prepare_run(request) as prepared:
            # confirmation_granted=True: this script is a scripted,
            # non-interactive experiment runner (there is no terminal to
            # prompt); invoking it is itself the user's confirmation.
            application.execute_run(prepared, confirmation_granted=True)
        print(f"[{RUN_ID}] audit dump written to {output_dir}")

    metrics_dir = args.results_dir / "metrics" / RUN_ID
    if metrics_dir.exists() and any(metrics_dir.iterdir()):
        print(f"[{RUN_ID}] metrics already computed, skipping")
        return 0
    _compute_and_save_metrics(output_dir, metrics_dir)
    print(f"[{RUN_ID}] metrics saved to {metrics_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
