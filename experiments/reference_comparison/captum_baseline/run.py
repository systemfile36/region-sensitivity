#!/usr/bin/env python3
"""Command-line entrypoint for the independent Captum reference workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .analysis import analyze, render_report
    from .workflow import load_config, run_audit
except ImportError:  # Direct execution: python run.py ...
    from analysis import analyze, render_report
    from workflow import load_config, run_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_config = Path(__file__).resolve().parent / "config.yaml"
    for command in ("audit", "analyze", "report", "all"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, default=default_config)
        subparser.add_argument("--output", type=Path, required=True)
        if command in {"audit", "all"}:
            subparser.add_argument(
                "--stop-after-items",
                type=int,
                help="write at most this many new raw rows, then leave a resumable incomplete run",
            )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "audit":
        result = run_audit(config, output, args.stop_after_items)
    elif args.command == "analyze":
        result = analyze(config, output)
    elif args.command == "report":
        result = render_report(config, output)
    else:
        audit_result = run_audit(config, output, args.stop_after_items)
        if audit_result["status"] != "complete":
            result = {"audit": audit_result}
        else:
            result = {
                "audit": audit_result,
                "analysis": analyze(config, output),
                "report": render_report(config, output),
            }
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

