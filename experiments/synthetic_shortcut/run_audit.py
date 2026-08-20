#!/usr/bin/env python3
"""Run ssat's audit pipeline for every (model, dataset, fill strategy) combo the
L3 synthetic-shortcut experiment needs, then compute and store margin_drop
metrics for each.

Exactly seven combinations are needed (see RUN_SPECS below) -- not all 2
models x 2 datasets x 5 fill strategies -- because each question only needs
a specific slice:

    Q1/Q2/fill-strategy sensitivity: M_shortcut audited on A, across all 5
        fill strategies (one audit per strategy -- see the note on grouping
        below).
    Q3: M_normal audited on A, constant_fill only (the reference strategy).
    B control: M_shortcut audited on B, constant_fill only.

Why one ssat run (and one metrics store) per fill strategy, instead of
listing all 5 perturbations in a single audit config: ssat's Aggregator
groups RegionMetrics/SpatialProfile rows by (region_key, metric_name) only
-- there is no perturb_op axis (ssat/metrics/aggregate.py). Packing multiple
fill strategies into one config would average their per-region degradation
together, making it impossible to compare each strategy's own region
ranking against the others (exactly what section 3.5 needs). Running one
audit per strategy keeps each strategy's RegionMetrics independent.

Q5 (accuracy drop) is intentionally absent here: it is a plain top-1
classification accuracy check with no region/perturbation dimension, so it
is computed directly against the raw checkpoints by evaluate_accuracy.py,
without going through this audit pipeline at all.

Run as: python3 experiments/synthetic_shortcut/run_audit.py [--only RUN_ID ...]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from common import GRID_COLS, GRID_ROWS, NUM_CLASSES, PATCH_REGION_ID
from ssat.application import AuditApplication, RunRequest
from ssat.core.estimate import EstimateOptions
from ssat.metrics.aggregate import aggregate_item_metrics
from ssat.metrics.builtin_metrics.continuous import MarginDrop
from ssat.metrics.dump_reader import DumpHandle
from ssat.metrics.registry import MetricRegistry
from ssat.metrics.store import save_metrics

PRIMARY_METRIC = MarginDrop.name

# Fixed, fill-strategy-specific parameters. Values are chosen to be
# meaningfully strong relative to an 8x8-pixel patch on a 32x32 image (e.g.
# a blur too mild to visibly disturb an 8px-wide solid patch would make Q4
# vacuous by construction), while remaining within each operator's normal
# operating range (ssat/core/perturb/operators.py's validate_config).
FILL_PARAMS: dict[str, dict[str, object]] = {
    "constant_fill": {"value": 0.0},
    "mean_fill": {},  # resolved_config_params injects dataset_stats.channel_mean.
    "blur": {"sigma": 4.0},
    "gaussian_noise": {"sigma": 50.0},
    "patch_shuffle": {"patch_size": 4},
}


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One (model checkpoint, dataset variant, fill strategy) audit combination."""

    model: str  # "shortcut" | "normal"
    dataset: str  # "A" | "B"
    fill: str  # a key of FILL_PARAMS

    @property
    def run_id(self) -> str:
        return f"{self.model}_{self.dataset}_{self.fill}"


RUN_SPECS: tuple[RunSpec, ...] = (
    RunSpec("shortcut", "A", "constant_fill"),
    RunSpec("shortcut", "A", "mean_fill"),
    RunSpec("shortcut", "A", "blur"),
    RunSpec("shortcut", "A", "gaussian_noise"),
    RunSpec("shortcut", "A", "patch_shuffle"),
    RunSpec("normal", "A", "constant_fill"),
    RunSpec("shortcut", "B", "constant_fill"),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one (or a filtered subset of) audit sweep."""

    parser = argparse.ArgumentParser(
        description="Run every audit combination the synthetic-shortcut experiment needs."
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
    parser.add_argument(
        "--only",
        action="append",
        choices=[spec.run_id for spec in RUN_SPECS],
        help="restrict the sweep to specific run_ids (repeatable); default: run all seven",
    )
    parser.add_argument(
        "--preprocessing",
        choices=("preset", "crop_free"),
        default="preset",
        help=(
            "'preset' (default) leaves the adapter config's preprocessing "
            "unset, so ssat falls back to squeezenet1_0's ImageNet preset -- "
            "unchanged prior behavior, matching the already-reported 7-run "
            "evidence. 'crop_free' declares common.CROP_FREE_PREPROCESSING_OPS "
            "on the adapter config; must be paired with --checkpoint-dir "
            "models trained via train.py --preprocessing crop_free."
        ),
    )
    return parser.parse_args()


def _build_audit_config(
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    fill: str,
    channel_mean: list[float] | None,
    preprocessing: str = "preset",
) -> dict:
    """Build one ssat audit configuration as a plain dict.

    ssat.application.config.load_application_config accepts a Mapping
    directly (not only a YAML path), so there is no need to render and
    maintain seven near-duplicate YAML files on disk -- this function is the
    single place the audit space (source/adapter/regions/perturbations) is
    defined, parameterized by run.

    ``preprocessing="crop_free"`` declares common.CROP_FREE_PREPROCESSING_OPS
    on the adapter config, removing the CenterCrop-induced model-space area
    confound (see common.py's CROP_FREE_PREPROCESSING_OPS comment). The
    default "preset" leaves the field unset entirely (not set
    to some equivalent preset-describing ops list), so the already-reported
    7-run evidence's config shape is reproduced exactly, byte for byte.
    """

    adapter: dict = {
        "provider": "torchvision",
        "model_name": "squeezenet1_0",
        # checkpoint (not weights=...) loads our from-scratch-trained
        # state dict; see train.py for why num_classes must match here.
        "checkpoint": {"path": str(checkpoint_path.resolve()), "state_dict_key": "model"},
        "model_kwargs": {"num_classes": NUM_CLASSES},
        "device": "auto",
    }
    if preprocessing == "crop_free":
        from common import CROP_FREE_PREPROCESSING_OPS

        adapter["preprocessing"] = list(CROP_FREE_PREPROCESSING_OPS)

    config: dict = {
        "source": {"kind": "image_manifest", "manifest": str(manifest_path.resolve())},
        "adapter": adapter,
        "regions": [
            {
                "region_id": PATCH_REGION_ID,
                "kind": "grid",
                "params": {"rows": GRID_ROWS, "cols": GRID_COLS},
            }
        ],
        "perturbations": [{"op": fill, "params": FILL_PARAMS[fill]}],
    }
    if fill == "mean_fill":
        if channel_mean is None:
            raise ValueError("mean_fill requires dataset_stats.channel_mean from prepare_data.py")
        config["dataset_stats"] = {"channel_mean": channel_mean}
    return config


def _compute_and_save_metrics(dump_root: Path, metrics_dir: Path) -> None:
    """Compute margin_drop item metrics for one dump and persist the aggregation.

    This intentionally reimplements
    tests/fixtures/synthetic_dump_builder.py's compute_and_save_metrics
    against only public ssat.metrics APIs -- this file is production
    experiment code, not a test, so it does not import from tests/.
    """

    handle = DumpHandle(dump_root)
    joined = handle.joined()
    resolved_config = handle.manifest.resolved_config

    registry = MetricRegistry()
    registry.register(MarginDrop())  # margin_drop is the only metric L3 needs.

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


def _run_one(
    application: AuditApplication,
    spec: RunSpec,
    *,
    checkpoint_dir: Path,
    manifests_dir: Path,
    dumps_dir: Path,
    metrics_dir: Path,
    channel_mean: list[float] | None,
    preprocessing: str = "preset",
) -> None:
    """Produce one dump (if missing) and its metrics store (if missing)."""

    output_dir = dumps_dir / spec.run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"[{spec.run_id}] dump already exists, skipping audit run")
    else:
        config = _build_audit_config(
            checkpoint_path=checkpoint_dir / f"m_{spec.model}.pt",
            manifest_path=manifests_dir / f"{spec.dataset}_audit.json",
            fill=spec.fill,
            channel_mean=channel_mean,
            preprocessing=preprocessing,
        )
        request = RunRequest(config, output_dir, estimate_options=EstimateOptions())
        with application.prepare_run(request) as prepared:
            # confirmation_granted=True: this script is a scripted,
            # non-interactive experiment runner (there is no terminal to
            # prompt); invoking it is itself the user's confirmation.
            application.execute_run(prepared, confirmation_granted=True)
        print(f"[{spec.run_id}] audit dump written to {output_dir}")

    run_metrics_dir = metrics_dir / spec.run_id
    if run_metrics_dir.exists() and any(run_metrics_dir.iterdir()):
        print(f"[{spec.run_id}] metrics already computed, skipping")
        return
    _compute_and_save_metrics(output_dir, run_metrics_dir)
    print(f"[{spec.run_id}] metrics saved to {run_metrics_dir}")


def main() -> int:
    """Run the full (or --only-filtered) audit sweep."""

    args = parse_args()
    manifests_dir = args.data_dir / "manifests"

    channel_mean: list[float] | None = None
    stats_path = args.data_dir / "dataset_stats.json"
    if stats_path.is_file():
        channel_mean = json.loads(stats_path.read_text(encoding="utf-8"))["channel_mean"]

    specs = RUN_SPECS if not args.only else tuple(s for s in RUN_SPECS if s.run_id in args.only)
    application = AuditApplication()
    for spec in specs:
        _run_one(
            application,
            spec,
            checkpoint_dir=args.checkpoint_dir,
            manifests_dir=manifests_dir,
            dumps_dir=args.results_dir / "dumps",
            metrics_dir=args.results_dir / "metrics",
            channel_mean=channel_mean,
            preprocessing=args.preprocessing,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
