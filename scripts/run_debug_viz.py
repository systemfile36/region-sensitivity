#!/usr/bin/env python3
"""Manually run one DebugViz view (V1/V2/V3) and save its PNGs for visual inspection.

This is a developer convenience, not a test: pytest already covers each
view's coordinate/order assertions automatically (``tests/unit/
test_metrics_mask_check.py``, ``test_metrics_viz_heatmap.py``,
``test_metrics_viz_ranking.py``). This script exists so a developer can run
DebugViz against a real dump/metrics store inside the
``region-sensitivity-workspace`` container and look at the resulting PNGs
with their own eyes -- DebugViz is a developer diagnostic tool for
tracking down errors, not a user-facing report.

Examples:
    python3 scripts/run_debug_viz.py mask-check \\
        --dump-root /path/to/dump --output-dir /tmp/debug_viz/mask_check

    python3 scripts/run_debug_viz.py heatmap \\
        --dump-root /path/to/dump --metrics-dir /path/to/metrics \\
        --output-dir /tmp/debug_viz/heatmap --metric-name margin_drop

    python3 scripts/run_debug_viz.py ranking \\
        --dump-root /path/to/dump --metrics-dir /path/to/metrics \\
        --output-dir /tmp/debug_viz/ranking --n-top 5 --n-bottom 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ssat.metrics.errors import DebugVizError
from ssat.metrics.viz.heatmap import save_heatmap_views
from ssat.metrics.viz.mask_check import save_mask_check_views
from ssat.metrics.viz.ranking import save_ranking_views

_DEFAULT_METRIC_NAME = "margin_drop"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one DebugViz subcommand."""

    parser = argparse.ArgumentParser(
        description="Render one DebugViz view's PNGs from a dump/metrics store for visual review."
    )
    subparsers = parser.add_subparsers(dest="view", required=True)

    mask_check = subparsers.add_parser("mask-check", help="V1: mask verification (3-panel PNGs)")
    _add_dump_args(mask_check)
    _add_selection_args(mask_check)

    heatmap = subparsers.add_parser("heatmap", help="V2: spatial sensitivity heatmap (2-panel PNGs)")
    _add_dump_args(heatmap)
    _add_metrics_args(heatmap)
    _add_selection_args(heatmap)

    ranking = subparsers.add_parser("ranking", help="V3: vulnerability ranking (top/bottom PNGs)")
    _add_dump_args(ranking)
    _add_metrics_args(ranking)
    ranking.add_argument("--n-top", type=int, default=5, help="most-vulnerable samples (default: 5)")
    ranking.add_argument("--n-bottom", type=int, default=5, help="least-vulnerable samples (default: 5)")

    return parser.parse_args()


def _add_dump_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--dump-root", type=Path, required=True, help="raw audit dump root")
    subparser.add_argument("--output-dir", type=Path, required=True, help="destination for PNGs")


def _add_metrics_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--metrics-dir", type=Path, required=True, help="stored aggregation run (save_metrics output)"
    )
    subparser.add_argument(
        "--metric-name",
        default=_DEFAULT_METRIC_NAME,
        help=f"registered metric to visualize (default: {_DEFAULT_METRIC_NAME})",
    )


def _add_selection_args(subparser: argparse.ArgumentParser) -> None:
    group = subparser.add_mutually_exclusive_group()
    group.add_argument("--n-samples", type=int, default=5, help="number of samples (default: 5)")
    group.add_argument("--sample-ids", help="comma-separated explicit sample_ids, in order")


def main() -> int:
    """Dispatch to the selected DebugViz view and print the saved PNG paths."""

    args = parse_args()
    try:
        paths = _run(args)
    except DebugVizError as error:
        raise SystemExit(f"debug_viz: {error}") from None

    for path in paths:
        print(path)
    return 0


def _run(args: argparse.Namespace) -> tuple[Path, ...]:
    """Call the selected view's save_*_views() and flatten its result to a path tuple."""

    if args.view == "mask-check":
        sample_ids, n_samples = _resolve_selection(args)
        return save_mask_check_views(
            args.dump_root, args.output_dir, n_samples=n_samples, sample_ids=sample_ids
        )
    if args.view == "heatmap":
        sample_ids, n_samples = _resolve_selection(args)
        return save_heatmap_views(
            args.dump_root,
            args.metrics_dir,
            args.output_dir,
            metric_name=args.metric_name,
            n_samples=n_samples,
            sample_ids=sample_ids,
        )
    if args.view == "ranking":
        top_paths, bottom_paths = save_ranking_views(
            args.dump_root,
            args.metrics_dir,
            args.output_dir,
            metric_name=args.metric_name,
            n_top=args.n_top,
            n_bottom=args.n_bottom,
        )
        return (*top_paths, *bottom_paths)
    raise AssertionError(f"unreachable: unknown view {args.view!r}")


def _resolve_selection(args: argparse.Namespace) -> tuple[list[str] | None, int]:
    """Translate the shared --n-samples/--sample-ids pair into (sample_ids, n_samples)."""

    if args.sample_ids:
        return [value.strip() for value in args.sample_ids.split(",")], args.n_samples
    return None, args.n_samples


if __name__ == "__main__":
    raise SystemExit(main())
