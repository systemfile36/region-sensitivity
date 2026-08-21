#!/usr/bin/env python3
"""Micro-benchmark the A1-A5 ``ssat.analysis`` hot paths on a synthetic dataset.

``ssat analyze`` was observed pinning one CPU core for a very long time on a
real dataset (item_context ~1.6M rows, item_values ~14.4M rows =
item_context x metric count): ``ssat/analysis/{reader,indexer,control,
stability,interval}.py`` used to rebuild ``AnchorKey``/``ConditionKey`` per
row inside ``itertuples`` loops, several of them re-hashing
``perturb_params_json`` with ``sha256_bytes`` on every row even though that
column only takes a handful of distinct values per run. Those hot loops were
replaced with pandas ``groupby``-based vectorization (2024 perf pass) --
this script exists so a developer can re-measure wall-clock cost on a
synthetic dataset shaped like the real one, without needing to run the
multi-hour real case study, and so a future regression (e.g. someone
re-introducing a per-row ``sha256_bytes`` call) shows up as an obvious time
spike here.

The default scale (2000 samples x 16 regions x 9 metrics) produces
item_context/item_values row counts in the same ballpark as one row of the
suggested micro-benchmark range in the perf plan (~190K / ~1.7M rows) --
large enough to make an O(n) Python-level loop visibly slow relative to a
vectorized one, without needing minutes to generate.

Examples:
    python3 scripts/benchmark_analysis_hot_loops.py
    python3 scripts/benchmark_analysis_hot_loops.py --n-samples 5000 --n-metrics 9
"""

from __future__ import annotations

import argparse
import json
import random
import time

import pandas as pd

from ssat.analysis.control import compare_to_controls
from ssat.analysis.indexer import ComparisonIndexer
from ssat.analysis.interval import compute_intervals

# Private, not part of the public API -- imported here only because this
# script benchmarks the internal hot loop directly rather than going through
# AnalysisReader.available_analyses(), which would require a real
# dump/metrics store on disk.
from ssat.analysis.reader import _has_repeated_condition_group
from ssat.analysis.stability import compute_seed_stability, compute_strategy_stability

_PERTURB_CONDITIONS = [
    # (perturb_op, perturb_params_json, n_seed_repeats) -- mirrors
    # imagenet_mnv2_050_exact's configured operators (experiments/
    # real_dataset_case_study/configs/imagenet_mnv2_050_exact.yaml).
    ("mean_fill", "{}", 1),
    ("blur", json.dumps({"sigma": 3.0}), 1),
    ("gaussian_noise", json.dumps({"sigma": 12.5}), 3),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments controlling the synthetic dataset's scale."""

    parser = argparse.ArgumentParser(
        description=(
            "Time ssat.analysis's A1-A5 entry points on a synthetic "
            "item_context/item_values pair shaped like a real dataset."
        )
    )
    parser.add_argument("--n-samples", type=int, default=2000, help="synthetic samples (default: 2000)")
    parser.add_argument(
        "--n-regions", type=int, default=16, help="grid regions per sample (default: 16)"
    )
    parser.add_argument(
        "--n-metrics", type=int, default=9, help="distinct metric_names per item (default: 9)"
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000, help="compute_intervals bootstrap count (default: 1000)"
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for synthetic degradation values")
    return parser.parse_args()


def _context_row(
    *,
    sample_id: str,
    region_id: str,
    region_instance_id: str,
    perturb_op: str,
    perturb_params_json: str,
    region_kind: str = "grid",
    region_params_json: str = "{}",
    is_control: bool = False,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "region_id": region_id,
        "region_instance_id": region_instance_id,
        "region_kind": region_kind,
        "region_params_json": region_params_json,
        "intended_area_px": 100,
        "effective_area_px": 100,
        "perturb_op": perturb_op,
        "perturb_params_json": perturb_params_json,
        "invert_mask": False,
        "is_control": is_control,
    }


def _item_value_row(
    *,
    sample_id: str,
    region_id: str,
    region_instance_id: str,
    perturb_op: str,
    perturb_params_json: str,
    metric_name: str,
    degradation: float,
    is_control: bool = False,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "region_id": region_id,
        "region_instance_id": region_instance_id,
        "invert_mask": False,
        "perturb_op": perturb_op,
        "perturb_params_json": perturb_params_json,
        "is_control": is_control,
        "metric_name": metric_name,
        "degradation": degradation,
        "available": True,
    }


def _control_region_params_json(*, target_region_id: str, target_region_instance_id: str) -> str:
    # Mirrors PlanBuilder._region_recipe's shape (ssat/core/plan/builder.py),
    # nested under "target_region" -- same shape
    # tests/unit/test_analysis_indexer.py's _control_region_params_json uses.
    return json.dumps(
        {
            "target_region": {
                "region_id": target_region_id,
                "region_instance_id": target_region_instance_id,
                "kind": "grid",
                "params": {},
            },
            "control_request_index": 0,
            "control_index": 0,
        }
    )


def _build_synthetic_frames(
    *, n_samples: int, n_regions: int, n_metrics: int, rng: random.Random
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build item_context/item_values frames shaped like AnalysisReader's output.

    One grid region set per sample, each target region exercised by every
    entry in ``_PERTURB_CONDITIONS`` (seed repeats as separate rows sharing
    one ``perturb_op``/``perturb_params_json``), plus one
    ``random_area_match`` control per region matched by exact target-region
    reference (mirroring one real ``controls: n_samples: 1`` config).
    """

    metric_names = [f"metric_{i}" for i in range(n_metrics)]
    context_rows: list[dict[str, object]] = []
    item_value_rows: list[dict[str, object]] = []

    for sample_index in range(n_samples):
        sample_id = f"sample_{sample_index:07d}"
        for region_index in range(n_regions):
            region_instance_id = f"grid_4x4/cell_{region_index}"

            for perturb_op, perturb_params_json, n_seed_repeats in _PERTURB_CONDITIONS:
                for _ in range(n_seed_repeats):
                    context_rows.append(
                        _context_row(
                            sample_id=sample_id,
                            region_id="grid_4x4",
                            region_instance_id=region_instance_id,
                            perturb_op=perturb_op,
                            perturb_params_json=perturb_params_json,
                        )
                    )
                    for metric_name in metric_names:
                        item_value_rows.append(
                            _item_value_row(
                                sample_id=sample_id,
                                region_id="grid_4x4",
                                region_instance_id=region_instance_id,
                                perturb_op=perturb_op,
                                perturb_params_json=perturb_params_json,
                                metric_name=metric_name,
                                degradation=rng.gauss(0.0, 1.0),
                            )
                        )

            control_region_id = f"control:grid_4x4:{region_index}"
            control_region_instance_id = f"{control_region_id}:0"
            context_rows.append(
                _context_row(
                    sample_id=sample_id,
                    region_id=control_region_id,
                    region_instance_id=control_region_instance_id,
                    region_kind="random_area_match",
                    region_params_json=_control_region_params_json(
                        target_region_id="grid_4x4",
                        target_region_instance_id=region_instance_id,
                    ),
                    perturb_op="mean_fill",
                    perturb_params_json="{}",
                    is_control=True,
                )
            )
            for metric_name in metric_names:
                item_value_rows.append(
                    _item_value_row(
                        sample_id=sample_id,
                        region_id=control_region_id,
                        region_instance_id=control_region_instance_id,
                        perturb_op="mean_fill",
                        perturb_params_json="{}",
                        metric_name=metric_name,
                        degradation=rng.gauss(0.0, 1.0),
                        is_control=True,
                    )
                )

    return pd.DataFrame(context_rows), pd.DataFrame(item_value_rows)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print("Generating synthetic item_context/item_values ...")
    context, item_values = _build_synthetic_frames(
        n_samples=args.n_samples,
        n_regions=args.n_regions,
        n_metrics=args.n_metrics,
        rng=rng,
    )
    print(
        f"  item_context: {len(context):,} rows "
        f"({context['perturb_params_json'].nunique()} distinct perturb_params_json values)"
    )
    print(f"  item_values:  {len(item_values):,} rows")
    print()

    timings: list[tuple[str, float]] = []

    def _time(label: str, fn) -> None:
        start = time.perf_counter()
        fn()
        timings.append((label, time.perf_counter() - start))

    indexer_box: list[ComparisonIndexer] = []
    _time("reader._has_repeated_condition_group", lambda: _has_repeated_condition_group(context))
    _time("indexer.ComparisonIndexer(context)", lambda: indexer_box.append(ComparisonIndexer(context)))
    indexer = indexer_box[0]
    _time(
        "control.compare_to_controls",
        lambda: compare_to_controls(item_values, indexer.control_pairs),
    )
    _time("stability.compute_seed_stability", lambda: compute_seed_stability(item_values))
    _time("stability.compute_strategy_stability", lambda: compute_strategy_stability(item_values))
    _time(
        "interval.compute_intervals",
        lambda: compute_intervals(item_values, n_bootstrap=args.n_bootstrap),
    )

    label_width = max(len(label) for label, _ in timings)
    print(f"{'function':<{label_width}}  seconds")
    for label, elapsed in timings:
        print(f"{label:<{label_width}}  {elapsed:8.3f}")
    print(f"{'total':<{label_width}}  {sum(elapsed for _, elapsed in timings):8.3f}")


if __name__ == "__main__":
    main()
