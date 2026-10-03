#!/usr/bin/env python3
"""A4 step 1: measure ``ssat`` cost along the four sweep axes (implementation plan section 7.3).

For every ``(setting, repeat)`` of ``sweep.schedule``, writes the setting's
config, then runs ``ssat run -> metrics -> analyze -> report`` with each
phase in an isolated wrapper process (``measure_step`` from
``experiments/benchmark_runtime_storage/run_benchmark.py``, loaded with
importlib) that reports wall time and peak RSS. During ``run``, nvidia-smi is
polled every 0.2 s for GPU memory and utilization. The ``ssat`` log file
splits ``run`` into preflight and the audit loop (``runtime.started`` to
``runtime.finished``). Output sizes are recorded per subdirectory. The dump
is then deleted unless ``--keep-dumps`` is set.

Before each axis a warm-up run (N=50) is executed and excluded. ``ssat
estimate --json`` is recorded once per setting. Results are appended to
``<output-root>/measurements.jsonl`` (one line per measurement), warm-ups to
``warmups.jsonl``, and estimates to ``estimates.jsonl``. Measurements already
in the file are skipped, so an interrupted sweep can be restarted.
``--settings`` restricts the sweep to the given setting keys (post hoc
re-measurement, ``deviations.md`` D-013).

Example:
    python experiments/revision_1/a4_scaling/run_scaling.py all --repeats 3
    python experiments/revision_1/a4_scaling/run_scaling.py all --settings n1000_g4_v5_k3 --output-root experiments/revision_1/results/a4_plan_cache
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.revision_1.a4_scaling.sweep import (
    A4_RESULTS_DIR,
    AXES,
    BASE_CONFIG,
    WARMUP,
    Setting,
    level_of,
    schedule,
    subset_path,
    write_config,
)
from experiments.revision_1.common.provenance import write_provenance
from experiments.revision_1.common.runs import REPO_ROOT

BENCHMARK_SCRIPT = REPO_ROOT / "experiments" / "benchmark_runtime_storage" / "run_benchmark.py"
GPU_POLL_INTERVAL_S = 0.2
SUBDIRS = ("clean", "perturbed", "index", "metrics", "analysis", "report")
LOG_EVENTS = ("config.resolve_started", "plan.enumeration_started", "plan.enumerated", "runtime.started", "runtime.finished")


def load_benchmark_module() -> Any:
    """Import ``run_benchmark.py`` by path without modifying it."""

    spec = importlib.util.spec_from_file_location("run_benchmark", BENCHMARK_SCRIPT)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def subdir_bytes(output: Path) -> dict[str, int]:
    """Return on-disk bytes per output subdirectory, plus ``other`` (manifests) and ``total``."""

    sizes = {name: sum(p.stat().st_size for p in (output / name).rglob("*") if p.is_file()) if (output / name).is_dir() else 0
             for name in SUBDIRS}
    total = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    sizes["other"] = total - sum(sizes.values())
    sizes["total"] = total
    return sizes


def parse_log_events(path: Path) -> dict[str, str]:
    """Return the first timestamp of each ``LOG_EVENTS`` entry in an ssat log file."""

    events: dict[str, str] = {}
    if not path.is_file():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[3] in LOG_EVENTS and parts[3] not in events:
            events[parts[3]] = parts[0]
    return events


def loop_seconds(events: dict[str, str]) -> float | None:
    """Seconds from ``runtime.started`` to ``runtime.finished`` (1 s log resolution), if both are present."""

    if "runtime.started" not in events or "runtime.finished" not in events:
        return None
    parse = lambda stamp: datetime.fromisoformat(stamp.replace("Z", "+00:00"))  # noqa: E731
    return (parse(events["runtime.finished"]) - parse(events["runtime.started"])).total_seconds()


def manifest_items(output: Path) -> int:
    """Sum ``counts_by_status`` (clean plus perturbed) in the run manifest."""

    counts = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))["counts_by_status"]
    return int(sum(counts.values()))


class GpuPoller:
    """Poll nvidia-smi for used memory (MiB) and utilization (%) in a background thread."""

    def __init__(self, interval_s: float = GPU_POLL_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self.memory: list[int] = []
        self.utilization: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits", "-i", "0"],
                                        capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    memory, utilization = (int(value) for value in result.stdout.strip().splitlines()[0].split(","))
                    self.memory.append(memory)
                    self.utilization.append(utilization)
            except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "GpuPoller":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def summary(self) -> dict[str, object]:
        if not self.memory:
            return {"gpu_peak_mib": None, "gpu_util_mean": None, "gpu_samples": 0}
        return {"gpu_peak_mib": max(self.memory), "gpu_util_mean": sum(self.utilization) / len(self.utilization),
                "gpu_samples": len(self.memory)}


def _append(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def _read_keys(path: Path, fields: tuple[str, ...]) -> set[tuple[object, ...]]:
    if not path.is_file():
        return set()
    return {tuple(json.loads(line)[field] for field in fields) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def measure(bench: Any, setting: Setting, label: str, root: Path, *, keep: bool) -> dict[str, object]:
    """Run all four phases for ``setting`` and return the measurement record."""

    config = write_config(setting, root / "configs")
    output = root / "runs" / label
    log = root / "logs" / f"{label}.log"
    if output.exists():
        shutil.rmtree(output)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.unlink(missing_ok=True)
    ssat = [sys.executable, "-m", "ssat"]
    started_at = datetime.now(timezone.utc).isoformat()
    phases: dict[str, dict[str, object]] = {}
    with GpuPoller() as poller:
        phases["run"] = bench.measure_step([*ssat, "--log-file", str(log), "run", str(config), "-o", str(output), "--yes"], echo=False)
    sizes_after_run = subdir_bytes(output)
    for phase in ("metrics", "analyze", "report"):
        phases[phase] = bench.measure_step([*ssat, phase, str(output)], echo=False)
    events = parse_log_events(log)
    record = {
        "label": label, "setting": setting.key, "n": setting.n, "grid": setting.grid, "regions": setting.regions,
        "v": setting.v, "k": setting.k, "planned_items": setting.planned_items, "manifest_items": manifest_items(output),
        "config": str(config), "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(), "started_at": started_at,
        "phases": phases, "log_events": events, "run_loop_s": loop_seconds(events), **poller.summary(),
        "bytes_after_run": sizes_after_run, "bytes_final": subdir_bytes(output),
    }
    if not keep:
        shutil.rmtree(output)
    return record


def run_estimate(setting: Setting, root: Path) -> dict[str, object]:
    """Record ``ssat estimate --json`` for ``setting``."""

    config = write_config(setting, root / "configs")
    started = time.perf_counter()
    result = subprocess.run([sys.executable, "-m", "ssat", "--log-level", "WARNING", "estimate", str(config), "--json"],
                            capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    record: dict[str, object] = {"setting": setting.key, "elapsed_s": elapsed, "returncode": result.returncode}
    if result.returncode == 0:
        report = json.loads(result.stdout)["report"]
        record.update(estimated_remaining_seconds=report["estimated_remaining_seconds"],
                      estimated_total_dump_bytes=report["estimated_total_dump_bytes"],
                      profile_items_per_second=report["profile"]["items_per_second"],
                      total_perturbed_items=report["total_perturbed_items"], total_clean_samples=report["total_clean_samples"])
    return record


def scheduled(axis: str, repeats: int, only: frozenset[str] | None = None) -> list[tuple[Setting, int]]:
    """``schedule(axis, repeats)``, restricted to the setting keys in ``only`` when given."""

    return [(setting, repeat) for setting, repeat in schedule(axis, repeats) if only is None or setting.key in only]


def run_axis(bench: Any, axis: str, root: Path, *, repeats: int, keep: bool, estimates: bool,
             only: frozenset[str] | None = None) -> None:
    """Warm up, record estimates, then measure every scheduled ``(setting, repeat)`` of ``axis``."""

    measurements = root / "measurements.jsonl"
    done = _read_keys(measurements, ("setting", "repeat"))
    todo = [(setting, repeat) for setting, repeat in scheduled(axis, repeats, only) if (setting.key, repeat) not in done]
    if not todo:
        print(f"[{axis}] nothing to do", flush=True)
        return
    print(f"[{axis}] warm-up {WARMUP.key}", flush=True)
    warm = measure(bench, WARMUP, f"warmup_{axis}", root, keep=False)
    _append(root / "warmups.jsonl", {"axis": axis, **warm})
    if estimates:
        estimated = _read_keys(root / "estimates.jsonl", ("setting",))
        for setting in dict.fromkeys(setting for setting, _ in todo):
            if (setting.key,) not in estimated:
                print(f"[{axis}] estimate {setting.key}", flush=True)
                _append(root / "estimates.jsonl", {"axis": axis, **run_estimate(setting, root)})
    for setting, repeat in todo:
        label = f"{setting.key}_r{repeat}"
        print(f"[{axis}] {label} ({setting.planned_items:,} items) ...", flush=True)
        record = measure(bench, setting, label, root, keep=keep)
        _append(measurements, {"axis": axis, "level": level_of(axis, setting), "repeat": repeat, **record})
        run = record["phases"]["run"]  # type: ignore[index]
        print(f"[{axis}] {label}: run {run['elapsed_s']:.0f}s, loop {record['run_loop_s']}s", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("axis", choices=[*AXES, "all"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=A4_RESULTS_DIR)
    parser.add_argument("--keep-dumps", action="store_true")
    parser.add_argument("--no-estimate", action="store_true")
    parser.add_argument("--settings", nargs="+", metavar="KEY", help="Measure only these setting keys (e.g. n500_g4_v5_k20).")
    parser.add_argument("--dry-run", action="store_true", help="Print the schedule and item totals only.")
    args = parser.parse_args(argv)

    axes = list(AXES) if args.axis == "all" else [args.axis]
    only = None if args.settings is None else frozenset(args.settings)
    if only is not None:
        unknown = only - {setting.key for axis in axes for setting in AXES[axis]}
        if unknown:
            raise SystemExit(f"settings not on the selected axes: {sorted(unknown)}")
        axes = [axis for axis in axes if scheduled(axis, args.repeats, only)]
    if args.dry_run:
        total = 0
        for axis in axes:
            for setting, repeat in scheduled(axis, args.repeats, only):
                total += setting.planned_items
                print(f"{axis:13s} {setting.key:22s} r{repeat} {setting.planned_items:>10,}")
        print(f"total planned items: {total:,}")
        return 0
    for n in sorted({setting.n for axis in axes for setting in AXES[axis]} | {WARMUP.n}):
        if not subset_path(n).is_file():
            raise SystemExit(f"missing {subset_path(n)}; run make_inputs.py")
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    write_provenance(root, inputs={"base_config": BASE_CONFIG, **{f"a4_N{n}": subset_path(n) for n in sorted({s.n for a in axes for s in AXES[a]})}},
                     extra={"axes": axes, "repeats": args.repeats, "settings": None if only is None else sorted(only)}, filename=f"run_scaling_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.provenance.json")
    bench = load_benchmark_module()
    for axis in axes:
        run_axis(bench, axis, root, repeats=args.repeats, keep=args.keep_dumps, estimates=not args.no_estimate, only=only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
