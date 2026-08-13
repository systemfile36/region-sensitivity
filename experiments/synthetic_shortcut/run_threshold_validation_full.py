#!/usr/bin/env python3
"""Produce a crop-free, all-fill-strategy audit run with control regions and
multiple seeds, for the final z_vs_control_threshold/seed_cv_threshold
recalibration required by docs/IMPLE_PLAN_CONTROL_STABILITY_v1.md section 8's
"단계 7, B3 재분석 후 재조정" procedure (proposed defaults: z=2.0, cv=0.2 --
see ssat/analysis/reliability.py).

This supersedes run_threshold_validation.py's provisional, single-op
(constant_fill only) run in two ways:

1. Crop-free preprocessing (--preprocessing crop_free on the adapter,
   checkpoints_crop_free/ models trained the same way) -- the earlier run
   deliberately kept the CenterCrop confound to avoid a train/audit
   mismatch against the (then only available) preset-trained checkpoint;
   that confound is gone now that crop-free retraining exists.
2. All five fill strategies in one config's ``perturbations`` list, not
   just constant_fill. run_audit.py's own module docstring warns against
   mixing perturb_ops in one config *for its own purposes* -- its Aggregator
   (ssat/metrics/aggregate.py) pools RegionMetrics by (region_key,
   metric_name) with no perturb_op axis, so Q1-Q4's region-ranking
   comparison would blur across strategies. That concern does not apply
   here: this script never reads RegionMetrics/region ranking. It only
   feeds ssat.analysis's item-grain pipeline (AnalysisReader.item_context()
   preserves perturb_op per item -- see ssat/analysis/reader.py's
   fill_strategy_stability field, which is explicitly computed from
   ``perturb_op.nunique() >= 2``), and A5/A6's multi_strategy flag is
   *designed* to consume exactly this shape: one run, multiple operators.
   Without it, multi_strategy is structurally FALSE for every anchor
   forever (see validate_reliability_thresholds.py's caveat) and the
   reliability grade can never rise above LOW regardless of evidence.

Run as: python3 experiments/synthetic_shortcut/run_threshold_validation_full.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import CROP_FREE_PREPROCESSING_OPS, GRID_COLS, GRID_ROWS, NUM_CLASSES, PATCH_REGION_ID
from run_audit import FILL_PARAMS, PRIMARY_METRIC
from ssat.application import AuditApplication, RunRequest
from ssat.core.estimate import EstimateOptions
from ssat.metrics.aggregate import aggregate_item_metrics
from ssat.metrics.builtin_metrics.continuous import MarginDrop
from ssat.metrics.dump_reader import DumpHandle
from ssat.metrics.registry import MetricRegistry
from ssat.metrics.store import save_metrics

RUN_ID = "shortcut_A_all_ops_thresholds_crop_free"

# Same minimal values run_threshold_validation.py validated as sufficient to
# exercise both thresholds (z needs n_controls>=2 for a std; seed_cv needs
# n_seeds>=2). Item count this time: 16 regions * 5 ops * 3 seeds *
# (1 target + 2 controls) = 720 items/sample (vs. that script's 144/sample),
# since every fill strategy now needs its own seed/control sweep in the same
# run to give multi_strategy real data to work with.
N_CONTROLS = 2
SEED_SALTS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one full threshold-validation run."""

    parser = argparse.ArgumentParser(
        description=(
            "Run one crop-free, all-fill-strategy L3 audit with control "
            "regions and multiple seeds, for final threshold recalibration."
        )
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints_crop_free",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results_crop_free",
    )
    return parser.parse_args()


def _build_config(
    checkpoint_path: Path, manifest_path: Path, channel_mean: list[float] | None
) -> dict:
    """Build the audit config: all 5 fill strategies, crop-free, controls + multi-seed.

    Mirrors run_audit.py's ``_build_audit_config`` for the source/adapter/
    regions sections (adding ``preprocessing`` for the crop-free pipeline)
    and run_threshold_validation.py's ``controls``/``seed_salts`` additions,
    but lists every ``FILL_PARAMS`` entry instead of one.
    """

    perturbations = [
        {"op": fill, "params": params, "seed_salts": list(SEED_SALTS)}
        for fill, params in FILL_PARAMS.items()
    ]
    config: dict = {
        "source": {"kind": "image_manifest", "manifest": str(manifest_path.resolve())},
        "adapter": {
            "provider": "torchvision",
            "model_name": "squeezenet1_0",
            "checkpoint": {"path": str(checkpoint_path.resolve()), "state_dict_key": "model"},
            "model_kwargs": {"num_classes": NUM_CLASSES},
            "device": "auto",
            "preprocessing": list(CROP_FREE_PREPROCESSING_OPS),
        },
        "regions": [
            {
                "region_id": PATCH_REGION_ID,
                "kind": "grid",
                "params": {"rows": GRID_ROWS, "cols": GRID_COLS},
            }
        ],
        "perturbations": perturbations,
        "controls": [{"match_area_of": PATCH_REGION_ID, "n_samples": N_CONTROLS}],
    }
    if channel_mean is not None:
        config["dataset_stats"] = {"channel_mean": channel_mean}
    return config


def _compute_and_save_metrics(dump_root: Path, metrics_dir: Path) -> None:
    """Compute margin_drop item metrics for one dump and persist the aggregation.

    Duplicated from run_audit.py's helper of the same name; see that
    module's docstring on this function for why (production experiment
    code does not import from tests/, no shared non-test module for it).
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
    """Produce the crop-free, all-ops, control+multi-seed dump and its metrics store."""

    args = parse_args()

    channel_mean: list[float] | None = None
    stats_path = args.data_dir / "dataset_stats.json"
    if stats_path.is_file():
        channel_mean = json.loads(stats_path.read_text(encoding="utf-8"))["channel_mean"]

    output_dir = args.results_dir / "dumps" / RUN_ID
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"[{RUN_ID}] dump already exists, skipping audit run")
    else:
        config = _build_config(
            checkpoint_path=args.checkpoint_dir / "m_shortcut.pt",
            manifest_path=args.data_dir / "manifests" / "A_audit.json",
            channel_mean=channel_mean,
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
