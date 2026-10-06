#!/usr/bin/env python3
"""C1-min: measure the Captum and SSAT synthetic-shortcut workflows (implementation plan section 9.2).

Runs every command of ``command_map.md`` in an isolated wrapper process
(``measure_step`` from ``experiments/benchmark_runtime_storage/run_benchmark.py``),
which reports wall time and the peak RSS of the largest process. Alongside
each command, a background thread polls nvidia-smi (GPU memory and
utilization, every 0.2 s) and sums the RSS of this process's descendants
(every 0.5 s), so the worker pools are included. The idle GPU memory is read
before each workflow. On-disk bytes are recorded per top-level output entry.

Workflows alternate in order across repeats (Captum first on even repeats).
Outputs go to ``results/c1/{captum,ssat}_fresh/r<repeat>`` and are kept for
verification (``summarize.py``). Records are appended to
``results/c1/measurements.jsonl``; a (workflow, repeat) already there is
skipped, so an interrupted run can be restarted.

Example:
    python experiments/revision_1/c1_captum/measure_resources.py --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from experiments.revision_1.a4_scaling.run_scaling import GpuPoller, load_benchmark_module
from experiments.revision_1.common.provenance import write_provenance
from experiments.revision_1.common.runs import REPO_ROOT, REVISION_RESULTS_DIR, SYNTHETIC_DIR

C1_RESULTS_DIR = REVISION_RESULTS_DIR / "c1"
CAPTUM_DIR = REPO_ROOT / "experiments" / "reference_comparison" / "captum_baseline"
CAPTUM_CONFIG = CAPTUM_DIR / "config.yaml"
CHECKPOINTS = SYNTHETIC_DIR / "checkpoints_crop_free"
TREE_POLL_INTERVAL_S = 0.5
WORKFLOWS = ("captum", "ssat")


def workflow_steps(workflow: str, output: Path) -> list[tuple[str, str, list[str]]]:
    """Return ``(step, stage, argv)`` for every command of ``workflow`` (``command_map.md``)."""

    py = sys.executable
    out = str(output)
    if workflow == "captum":
        return [(f"C-{index}", stage, [py, str(CAPTUM_DIR / "run.py"), command, "--config", str(CAPTUM_CONFIG), "--output", out])
                for index, (command, stage) in enumerate((("audit", "audit"), ("analyze", "analysis_report"),
                                                          ("report", "analysis_report")), start=1)]
    if workflow != "ssat":
        raise ValueError(f"unknown workflow {workflow!r}")
    script = lambda name: str(SYNTHETIC_DIR / name)  # noqa: E731
    checkpoints = ["--checkpoint-dir", str(CHECKPOINTS)]
    results = ["--results-dir", out]
    return [
        ("S-1", "audit", [py, script("run_audit.py"), "--preprocessing", "crop_free", *checkpoints, *results]),
        ("S-2", "audit", [py, script("evaluate_accuracy.py"), "--preprocessing", "crop_free", *checkpoints, *results]),
        ("S-3", "analysis_report", [py, script("evaluate.py"), *results]),
        ("S-4", "audit", [py, script("run_threshold_validation_full.py"), "--model", "shortcut", *checkpoints, *results]),
        ("S-5", "audit", [py, script("run_threshold_validation_full.py"), "--model", "normal", *checkpoints, *results]),
        ("S-6", "analysis_report", [py, script("generate_report.py"), "--model", "shortcut", *results]),
        ("S-7", "analysis_report", [py, script("generate_report.py"), "--model", "normal", *results]),
    ]


def workflow_order(repeat: int) -> tuple[str, str]:
    """Captum first on even repeats, SSAT first on odd ones."""

    return WORKFLOWS if repeat % 2 == 0 else WORKFLOWS[::-1]


def entry_bytes(output: Path) -> dict[str, int]:
    """Bytes per top-level directory of ``output``; top-level files are summed as ``files``; plus ``total``."""

    sizes: dict[str, int] = {"files": 0}
    for entry in sorted(output.iterdir()):
        if entry.is_dir():
            sizes[entry.name] = sum(p.stat().st_size for p in entry.rglob("*") if p.is_file())
        elif entry.is_file():
            sizes["files"] += entry.stat().st_size
    sizes["total"] = sum(sizes.values())
    return sizes


class TreeRssPoller:
    """Poll the summed RSS of this process's descendants in a background thread.

    Shared pages (for example copy-on-write pages of forked workers) are
    counted once per process, so the sum is an upper bound on physical use.
    """

    def __init__(self, interval_s: float = TREE_POLL_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self.peak_bytes = 0
        self.peak_processes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self) -> None:
        # Imported here so the module's other helpers work without the ``reference`` extra.
        import psutil

        me = psutil.Process(os.getpid())
        while not self._stop.is_set():
            total, count = 0, 0
            for child in me.children(recursive=True):
                try:
                    total += child.memory_info().rss
                    count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if total > self.peak_bytes:
                self.peak_bytes, self.peak_processes = total, count
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "TreeRssPoller":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=10)


def gpu_idle_snapshot(samples: int = 10) -> dict[str, object]:
    """Median used GPU memory over ``samples`` polls, plus the nvidia-smi compute-process list."""

    deadline = time.monotonic() + 30
    with GpuPoller() as poller:
        while len(poller.memory) < samples and time.monotonic() < deadline:
            time.sleep(0.1)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    return {"gpu_idle_mib": statistics.median(poller.memory) if poller.memory else None, "gpu_compute_apps": apps}


def measure_workflow(bench: object, workflow: str, repeat: int, root: Path) -> dict[str, object]:
    """Run every step of ``workflow`` into a fresh output directory and return the record."""

    output = root / f"{workflow}_fresh" / f"r{repeat}"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    record: dict[str, object] = {"workflow": workflow, "repeat": repeat, "output": str(output),
                                 "started_at": datetime.now(timezone.utc).isoformat(), **gpu_idle_snapshot()}
    steps = []
    for step, stage, argv in workflow_steps(workflow, output):
        started_at = datetime.now(timezone.utc).isoformat()
        with GpuPoller() as gpu, TreeRssPoller() as tree:
            result = bench.measure_step(argv, echo=False)  # type: ignore[attr-defined]
        steps.append({"step": step, "stage": stage, "argv": argv, "started_at": started_at, **result, **gpu.summary(),
                      "tree_rss_peak_bytes": tree.peak_bytes, "tree_rss_peak_processes": tree.peak_processes})
        print(f"[{workflow} r{repeat}] {step} {result['elapsed_s']:.1f} s", flush=True)
    record.update(finished_at=datetime.now(timezone.utc).isoformat(), steps=steps, bytes=entry_bytes(output))
    return record


def _done(path: Path) -> set[tuple[str, int]]:
    if not path.is_file():
        return set()
    return {(row["workflow"], row["repeat"]) for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if row}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=C1_RESULTS_DIR)
    parser.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    args = parser.parse_args(argv)
    root = args.output_root
    if args.dry_run:
        for workflow in WORKFLOWS:
            for step, stage, command in workflow_steps(workflow, root / f"{workflow}_fresh" / "r0"):
                print(f"{step} [{stage}] {' '.join(command)}")
        return 0
    bench = load_benchmark_module()
    path = root / "measurements.jsonl"
    done = _done(path)
    for repeat in range(args.repeats):
        for workflow in workflow_order(repeat):
            if (workflow, repeat) in done:
                continue
            record = measure_workflow(bench, workflow, repeat, root)
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, sort_keys=True) + "\n")
    write_provenance(root, inputs={"captum_config": CAPTUM_CONFIG, "checkpoints": CHECKPOINTS},
                     extra={"repeats": args.repeats}, filename="measure.provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
