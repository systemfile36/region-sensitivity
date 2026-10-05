#!/usr/bin/env python3
"""C1-std: measure the Captum and SSAT ImageNet workflows (implementation plan section 9.3).

The measurement mechanics are those of C1-min (``measure_resources.py``):
every command of ``command_map_std.md`` runs in an isolated ``measure_step``
wrapper process while background threads poll nvidia-smi (every 0.2 s) and
the summed RSS of the process tree (every 0.5 s). Workflows alternate in
order across repeats (Captum first on even repeats).

Outputs go to ``results/c1_std/{captum,ssat}_fresh/r<repeat>`` and are kept
for ``summarize_std.py``. Records are appended to
``results/c1_std/measurements.jsonl``; a (workflow, repeat) already there is
skipped, so an interrupted run can be restarted.

Example:
    python experiments/revision_1/c1_captum/measure_std.py --repeats 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from experiments.revision_1.a4_scaling.run_scaling import GpuPoller, load_benchmark_module, subdir_bytes
from experiments.revision_1.c1_captum.measure_resources import (
    WORKFLOWS,
    TreeRssPoller,
    _done,
    entry_bytes,
    gpu_idle_snapshot,
    workflow_order,
)
from experiments.revision_1.common.provenance import write_provenance
from experiments.revision_1.common.runs import REVISION_RESULTS_DIR

C1_DIR = Path(__file__).resolve().parent
C1_STD_RESULTS_DIR = REVISION_RESULTS_DIR / "c1_std"
CAPTUM_WORKFLOW = C1_DIR / "imagenet_workflow"
CAPTUM_CONFIG = CAPTUM_WORKFLOW / "config.yaml"
SSAT_CONFIG = C1_DIR / "configs" / "imagenet_mnv2_050_crop_free_c1.yaml"


def workflow_steps(
    workflow: str, output: Path, *, captum_config: Path = CAPTUM_CONFIG, ssat_config: Path = SSAT_CONFIG
) -> list[tuple[str, str, list[str]]]:
    """Return ``(step, stage, argv)`` for every command of ``workflow`` (``command_map_std.md``)."""

    py = sys.executable
    out = str(output)
    if workflow == "captum":
        run = str(CAPTUM_WORKFLOW / "run.py")
        return [(f"C-{index}", stage, [py, run, command, "--config", str(captum_config), "--output", out])
                for index, (command, stage) in enumerate((("audit", "audit"), ("analyze", "analysis_report"),
                                                          ("report", "analysis_report")), start=1)]
    if workflow != "ssat":
        raise ValueError(f"unknown workflow {workflow!r}")
    ssat = [py, "-m", "ssat"]
    return [
        ("S-1", "audit", [*ssat, "run", str(ssat_config), "-o", out, "--yes"]),
        ("S-2", "analysis_report", [*ssat, "metrics", out]),
        ("S-3", "analysis_report", [*ssat, "analyze", out]),
        ("S-4", "analysis_report", [*ssat, "report", out]),
    ]


def output_bytes(workflow: str, output: Path) -> dict[str, int]:
    """Bytes per top-level entry (Captum) or per dump subdirectory (SSAT), plus ``total``."""

    return entry_bytes(output) if workflow == "captum" else subdir_bytes(output)


def measure_workflow(bench: object, workflow: str, repeat: int, root: Path, **configs: Path) -> dict[str, object]:
    """Run every step of ``workflow`` into a fresh output directory and return the record."""

    output = root / f"{workflow}_fresh" / f"r{repeat}"
    if output.exists():
        shutil.rmtree(output)
    # ``ssat run`` creates its own output directory; the Captum workflow needs the parent.
    output.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {"workflow": workflow, "repeat": repeat, "output": str(output),
                                 "started_at": datetime.now(timezone.utc).isoformat(), **gpu_idle_snapshot()}
    steps = []
    for step, stage, argv in workflow_steps(workflow, output, **configs):
        started_at = datetime.now(timezone.utc).isoformat()
        with GpuPoller() as gpu, TreeRssPoller() as tree:
            result = bench.measure_step(argv, echo=False)  # type: ignore[attr-defined]
        steps.append({"step": step, "stage": stage, "argv": argv, "started_at": started_at, **result, **gpu.summary(),
                      "tree_rss_peak_bytes": tree.peak_bytes, "tree_rss_peak_processes": tree.peak_processes})
        print(f"[{workflow} r{repeat}] {step} {result['elapsed_s']:.1f} s rc={result['returncode']}", flush=True)
    record.update(finished_at=datetime.now(timezone.utc).isoformat(), steps=steps, bytes=output_bytes(workflow, output))
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=C1_STD_RESULTS_DIR)
    parser.add_argument("--captum-config", type=Path, default=CAPTUM_CONFIG)
    parser.add_argument("--ssat-config", type=Path, default=SSAT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    args = parser.parse_args(argv)
    root = args.output_root
    configs = {"captum_config": args.captum_config, "ssat_config": args.ssat_config}
    if args.dry_run:
        for workflow in WORKFLOWS:
            for step, stage, command in workflow_steps(workflow, root / f"{workflow}_fresh" / "r0", **configs):
                print(f"{step} [{stage}] {' '.join(command)}")
        return 0
    bench = load_benchmark_module()
    path = root / "measurements.jsonl"
    done = _done(path)
    for repeat in range(args.repeats):
        for workflow in workflow_order(repeat):
            if (workflow, repeat) in done:
                continue
            record = measure_workflow(bench, workflow, repeat, root, **configs)
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, sort_keys=True) + "\n")
    write_provenance(root, inputs={"captum_config": args.captum_config, "ssat_config": args.ssat_config},
                     extra={"repeats": args.repeats}, filename="measure.provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
