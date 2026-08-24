#!/usr/bin/env python3
"""Measure real end-to-end runtime/throughput/memory/storage/resume behavior.

``ssat/core/estimate/profiler.py`` and ``cost_model.py`` exist for
``ssat estimate``'s *preflight* purpose: they sample a bounded subset of
pending work without writing dump records, then extrapolate. That machinery
is a poor fit for measuring a real, full ``ssat run`` invocation, so this
script instead drives the real ``ssat`` CLI as subprocesses -- the same way
``experiments/real_dataset_case_study/run_matrix.py`` does -- and measures
each phase from the outside.

Isolated peak-RSS measurement: ``resource.getrusage(RUSAGE_CHILDREN)``
accumulates across every child a process has ever reaped, so calling it
after several subprocess phases in one long-lived script would mix their
peaks together. To keep each phase's reading clean, this script re-invokes
itself as a fresh process per measured phase via ``--internal-run-one``; that
wrapper process reaps exactly one child, so its ``RUSAGE_CHILDREN`` reading
reflects only that phase.

Subcommands:
    quickstart    CPU scale using configs/examples/quickstart.yaml.
    real-dataset  GPU scale using the Phase-3 imagenet_mnv2_050_exact config
                  (requires local ImageNet-1k val data and a CUDA GPU).
    resume        CPU scale demonstrating that a resumed run does not redo
                  completed work, using a larger synthetic fixture (see
                  prepare_resume_fixture.py).
    all           Runs quickstart, resume, then real-dataset in sequence.

Examples:
    python3 run_benchmark.py quickstart
    python3 run_benchmark.py resume
    python3 run_benchmark.py real-dataset
    python3 run_benchmark.py all --dry-run
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import threading
import time
from typing import Any

THIS_FILE = Path(__file__).resolve()
EXPERIMENT_DIR = THIS_FILE.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
RESULTS_DIR = EXPERIMENT_DIR / "results"

QUICKSTART_CONFIG = REPO_ROOT / "configs" / "examples" / "quickstart.yaml"
REAL_DATASET_CONFIG = (
    REPO_ROOT
    / "experiments"
    / "real_dataset_case_study"
    / "configs"
    / "imagenet_mnv2_050_exact.yaml"
)
REAL_DATASET_MINIMUM_ACCURACY = 0.50
RESUME_CONFIG = EXPERIMENT_DIR / "configs" / "resume_bench.yaml"
RESUME_FIXTURE_MANIFEST = EXPERIMENT_DIR / "data" / "resume_fixture" / "manifest.json"

DEFAULT_INTERRUPT_FRACTION = 0.5
DEFAULT_POLL_INTERVAL_S = 0.05
DEFAULT_GPU_POLL_INTERVAL_S = 0.2


# --- entry point -------------------------------------------------------


def main() -> int:
    """Dispatch to the internal single-command measurement mode or a subcommand."""

    if len(sys.argv) >= 2 and sys.argv[1] == "--internal-run-one":
        return _internal_run_one(sys.argv[2:])
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "quickstart":
        run_quickstart(args.output, dry_run=args.dry_run)
    elif args.command == "real-dataset":
        run_real_dataset(
            args.output,
            dry_run=args.dry_run,
            minimum_accuracy=args.minimum_accuracy,
            gpu_index=args.gpu_index,
            gpu_poll_interval_s=args.gpu_poll_interval_s,
        )
    elif args.command == "resume":
        run_resume(
            args.output_root,
            dry_run=args.dry_run,
            interrupt_fraction=args.interrupt_fraction,
            poll_interval_s=args.poll_interval_s,
        )
    elif args.command == "all":
        run_quickstart(args.quickstart_output, dry_run=args.dry_run)
        run_resume(
            args.resume_output_root,
            dry_run=args.dry_run,
            interrupt_fraction=args.interrupt_fraction,
            poll_interval_s=args.poll_interval_s,
        )
        run_real_dataset(
            args.real_dataset_output,
            dry_run=args.dry_run,
            minimum_accuracy=args.minimum_accuracy,
            gpu_index=args.gpu_index,
            gpu_poll_interval_s=args.gpu_poll_interval_s,
        )
    return 0


def parse_args() -> argparse.Namespace:
    """Parse the public CLI (the internal wrapper mode bypasses this)."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands that would run without executing them",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    quickstart_parser = subparsers.add_parser("quickstart", parents=[common])
    quickstart_parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "benchmark_runtime_storage" / "quickstart",
    )

    real_dataset_parser = subparsers.add_parser("real-dataset", parents=[common])
    real_dataset_parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "benchmark_runtime_storage" / "imagenet_mnv2_050_exact",
    )
    real_dataset_parser.add_argument("--minimum-accuracy", type=float, default=REAL_DATASET_MINIMUM_ACCURACY)
    real_dataset_parser.add_argument("--gpu-index", type=int, default=0)
    real_dataset_parser.add_argument("--gpu-poll-interval-s", type=float, default=DEFAULT_GPU_POLL_INTERVAL_S)

    resume_parser = subparsers.add_parser("resume", parents=[common])
    resume_parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "benchmark_runtime_storage" / "resume",
    )
    resume_parser.add_argument("--interrupt-fraction", type=float, default=DEFAULT_INTERRUPT_FRACTION)
    resume_parser.add_argument("--poll-interval-s", type=float, default=DEFAULT_POLL_INTERVAL_S)

    all_parser = subparsers.add_parser("all", parents=[common])
    all_parser.add_argument(
        "--quickstart-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "benchmark_runtime_storage" / "quickstart",
    )
    all_parser.add_argument(
        "--real-dataset-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "benchmark_runtime_storage" / "imagenet_mnv2_050_exact",
    )
    all_parser.add_argument(
        "--resume-output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "benchmark_runtime_storage" / "resume",
    )
    all_parser.add_argument("--minimum-accuracy", type=float, default=REAL_DATASET_MINIMUM_ACCURACY)
    all_parser.add_argument("--gpu-index", type=int, default=0)
    all_parser.add_argument("--gpu-poll-interval-s", type=float, default=DEFAULT_GPU_POLL_INTERVAL_S)
    all_parser.add_argument("--interrupt-fraction", type=float, default=DEFAULT_INTERRUPT_FRACTION)
    all_parser.add_argument("--poll-interval-s", type=float, default=DEFAULT_POLL_INTERVAL_S)

    return parser.parse_args()


# --- isolated single-command measurement (recursive self-invocation) ---


def _internal_run_one(cmd: list[str]) -> int:
    """Run exactly one child process and print its measurement as the last JSON line.

    Always launched as a fresh Python process (never called in-process) so
    ``resource.getrusage(RUSAGE_CHILDREN)`` reflects only this one child.
    """

    start = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - start
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(json.dumps({"elapsed_s": elapsed, "returncode": result.returncode, "peak_rss_kb": usage.ru_maxrss}))
    return 0


def measure_step(cmd: list[str], *, dry_run: bool = False, echo: bool = True) -> dict[str, Any]:
    """Run ``cmd`` in an isolated wrapper process and return its measurement.

    On success the wrapper's own JSON line is guaranteed to be the last
    printed line; everything the wrapped command itself printed is echoed
    above it for visibility, then parsed off before returning.
    """

    if echo:
        print(f"+ {' '.join(cmd)}", flush=True)
    if dry_run:
        return {"elapsed_s": 0.0, "returncode": 0, "peak_rss_kb": 0, "dry_run": True}

    wrapper_cmd = [sys.executable, str(THIS_FILE), "--internal-run-one", *cmd]
    proc = subprocess.run(wrapper_cmd, capture_output=True, text=True)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"no output from wrapper for: {' '.join(cmd)}\nstderr:\n{proc.stderr}")
    if echo:
        for line in lines[:-1]:
            print(line)
    try:
        measurement = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"could not parse measurement for: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        ) from error
    if measurement["returncode"] != 0:
        raise RuntimeError(f"command failed (rc={measurement['returncode']}): {' '.join(cmd)}\nstderr:\n{proc.stderr}")
    return measurement


# --- quickstart ----------------------------------------------------------


def run_quickstart(output: Path, *, dry_run: bool) -> dict[str, Any]:
    """Measure the committed quickstart config end to end on CPU."""

    _require_fresh(output, dry_run=dry_run)
    cmd_base = [sys.executable, "-m", "ssat"]
    steps: dict[str, Any] = {}
    steps["run"] = measure_step(
        [*cmd_base, "run", str(QUICKSTART_CONFIG), "-o", str(output), "--yes"], dry_run=dry_run
    )
    raw_dump_bytes = _dump_size_bytes(output, raw_only=True) if not dry_run else 0
    steps["metrics"] = measure_step([*cmd_base, "metrics", str(output)], dry_run=dry_run)
    steps["analyze"] = measure_step([*cmd_base, "analyze", str(output)], dry_run=dry_run)
    steps["report"] = measure_step([*cmd_base, "report", str(output)], dry_run=dry_run)
    total_artifact_bytes = _dump_size_bytes(output, raw_only=False) if not dry_run else 0
    items_per_second, total_items = (0.0, 0) if dry_run else _throughput(output, steps["run"]["elapsed_s"])

    result = {
        "scale": "quickstart",
        "config": str(QUICKSTART_CONFIG.relative_to(REPO_ROOT)),
        "output": str(output),
        "steps": steps,
        "total_items": total_items,
        "items_per_second": items_per_second,
        "raw_dump_bytes": raw_dump_bytes,
        "total_artifact_bytes": total_artifact_bytes,
        "environment": _capture_environment_info(),
        "measured_at": _utc_now_iso(),
    }
    _write_result("quickstart", result)
    _print_scale_summary(result)
    return result


# --- real-dataset ----------------------------------------------------------


def run_real_dataset(
    output: Path,
    *,
    dry_run: bool,
    minimum_accuracy: float,
    gpu_index: int,
    gpu_poll_interval_s: float,
) -> dict[str, Any]:
    """Measure the Phase-3 imagenet_mnv2_050_exact config end to end on GPU."""

    _require_fresh(output, dry_run=dry_run)
    cmd_base = [sys.executable, "-m", "ssat"]
    run_cmd = [
        *cmd_base,
        "run",
        str(REAL_DATASET_CONFIG),
        "-o",
        str(output),
        "--yes",
        "--minimum-accuracy",
        str(minimum_accuracy),
    ]

    stop_event = threading.Event()
    gpu_samples_mib: list[int] = []
    poller = threading.Thread(
        target=_poll_gpu_memory,
        args=(stop_event, gpu_index, gpu_poll_interval_s, gpu_samples_mib),
        daemon=True,
    )
    steps: dict[str, Any] = {}
    if not dry_run:
        poller.start()
    try:
        steps["run"] = measure_step(run_cmd, dry_run=dry_run)
    finally:
        stop_event.set()
        if poller.is_alive():
            poller.join(timeout=5)

    raw_dump_bytes = _dump_size_bytes(output, raw_only=True) if not dry_run else 0
    steps["metrics"] = measure_step([*cmd_base, "metrics", str(output)], dry_run=dry_run)
    steps["analyze"] = measure_step([*cmd_base, "analyze", str(output)], dry_run=dry_run)
    steps["report"] = measure_step([*cmd_base, "report", str(output)], dry_run=dry_run)
    total_artifact_bytes = _dump_size_bytes(output, raw_only=False) if not dry_run else 0
    items_per_second, total_items = (0.0, 0) if dry_run else _throughput(output, steps["run"]["elapsed_s"])

    result = {
        "scale": "real-dataset",
        "config": str(REAL_DATASET_CONFIG.relative_to(REPO_ROOT)),
        "output": str(output),
        "minimum_accuracy": minimum_accuracy,
        "steps": steps,
        "total_items": total_items,
        "items_per_second": items_per_second,
        "raw_dump_bytes": raw_dump_bytes,
        "total_artifact_bytes": total_artifact_bytes,
        "peak_gpu_memory_used_mib": max(gpu_samples_mib) if gpu_samples_mib else None,
        "gpu_memory_note": "sampled via nvidia-smi polling every "
        f"{gpu_poll_interval_s}s during the run step; an approximation, "
        "may miss sub-interval spikes and includes any other process sharing the GPU",
        "environment": _capture_environment_info(),
        "measured_at": _utc_now_iso(),
    }
    _write_result("real-dataset", result)
    _print_scale_summary(result)
    return result


def _poll_gpu_memory(stop_event: threading.Event, gpu_index: int, interval_s: float, samples: list[int]) -> None:
    """Append nvidia-smi's reported used-memory (MiB) until ``stop_event`` is set."""

    while not stop_event.is_set():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(gpu_index),
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                first_line = result.stdout.strip().splitlines()[0]
                samples.append(int(first_line))
        except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
            pass
        stop_event.wait(interval_s)


# --- resume ----------------------------------------------------------------


def run_resume(
    output_root: Path,
    *,
    dry_run: bool,
    interrupt_fraction: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    """Measure a baseline run against an interrupted-then-resumed run."""

    if not dry_run and not RESUME_FIXTURE_MANIFEST.exists():
        raise SystemExit(
            f"resume fixture not found at {RESUME_FIXTURE_MANIFEST}; run "
            "prepare_resume_fixture.py first"
        )
    baseline_output = output_root / "baseline"
    target_output = output_root / "interrupted_then_resumed"
    for path in (baseline_output, target_output):
        _require_fresh(path, dry_run=dry_run)

    cmd_base = [sys.executable, "-m", "ssat", "run", str(RESUME_CONFIG), "--yes"]
    baseline_cmd = [*cmd_base, "-o", str(baseline_output)]
    target_cmd = [*cmd_base, "-o", str(target_output)]

    baseline = measure_step(baseline_cmd, dry_run=dry_run)

    if dry_run:
        total_items_hint = 0
        interrupted: dict[str, Any] = {"elapsed_s": 0.0, "dry_run": True}
        resumed: dict[str, Any] = {"elapsed_s": 0.0, "dry_run": True}
        resume_count = 0
        print(f"+ (interrupt+resume) {' '.join(target_cmd)}", flush=True)
    else:
        total_items_hint = _estimate_total_items(RESUME_CONFIG)
        print(f"+ {' '.join(target_cmd)}  # interrupted at ~{interrupt_fraction:.0%} of items", flush=True)
        interrupted = _run_interrupted(
            target_cmd,
            target_output,
            target_fraction=interrupt_fraction,
            poll_interval_s=poll_interval_s,
            total_items_hint=total_items_hint,
        )
        resumed = measure_step(target_cmd, dry_run=False, echo=True)
        resume_count = _inspect(target_output)["resume_count"]

    result = {
        "scale": "resume",
        "config": str(RESUME_CONFIG.relative_to(REPO_ROOT)),
        "interrupt_fraction": interrupt_fraction,
        "total_items_hint": total_items_hint,
        "baseline": baseline,
        "interrupted": interrupted,
        "resumed": resumed,
        "resume_count": resume_count,
        "environment": _capture_environment_info(),
        "measured_at": _utc_now_iso(),
    }
    _write_result("resume", result)
    _print_resume_summary(result)
    return result


def _run_interrupted(
    cmd: list[str],
    output: Path,
    *,
    target_fraction: float,
    poll_interval_s: float,
    total_items_hint: int,
    grace_period_s: float = 30.0,
) -> dict[str, Any]:
    """Start ``cmd``, poll ``run_manifest.json`` until ``target_fraction`` of
    items complete, then SIGTERM it and report how far it actually got."""

    manifest_path = output / "run_manifest.json"
    target_count = max(1, int(total_items_hint * target_fraction))
    start = time.perf_counter()
    proc = subprocess.Popen(cmd)
    try:
        while proc.poll() is None:
            if _read_manifest_completed_count(manifest_path) >= target_count:
                proc.terminate()
                break
            time.sleep(poll_interval_s)
        try:
            proc.wait(timeout=grace_period_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=grace_period_s)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=grace_period_s)
    elapsed = time.perf_counter() - start
    return {
        "elapsed_s": elapsed,
        "returncode": proc.returncode,
        "was_interrupted": proc.returncode not in (0, None),
        "target_items": target_count,
        "completed_items_at_stop": _read_manifest_completed_count(manifest_path),
    }


def _read_manifest_completed_count(manifest_path: Path) -> int:
    """Sum ``counts_by_status`` from a possibly-in-progress run manifest."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    counts = manifest.get("counts_by_status", {})
    if not isinstance(counts, dict):
        return 0
    return sum(v for v in counts.values() if isinstance(v, int))


def _estimate_total_items(config: Path) -> int:
    """Return the full (non-resume-filtered) clean+perturbed item count."""

    result = subprocess.run(
        [sys.executable, "-m", "ssat", "estimate", str(config), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)["report"]
    return report["total_clean_samples"] + report["total_perturbed_items"]


# --- shared helpers ----------------------------------------------------


def _require_fresh(path: Path, *, dry_run: bool) -> None:
    if not dry_run and path.exists():
        raise SystemExit(f"{path} already exists; pass a fresh output path")


def _inspect(output: Path) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "ssat", "inspect", str(output), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _throughput(output: Path, run_elapsed_s: float) -> tuple[float, int]:
    """Return (items/sec, total items) from a completed dump's real row counts."""

    summary = _inspect(output)
    total_items = summary["clean"]["rows"] + summary["perturbed"]["rows"]
    items_per_second = total_items / run_elapsed_s if run_elapsed_s > 0 else 0.0
    return items_per_second, total_items


def _dump_size_bytes(output: Path, *, raw_only: bool) -> int:
    """Sum on-disk file sizes: raw dump fragments only, or the whole output tree."""

    if raw_only:
        paths: list[Path] = []
        for name in ("clean", "perturbed", "index"):
            sub = output / name
            if sub.exists():
                paths.extend(p for p in sub.rglob("*") if p.is_file())
        manifest_path = output / "run_manifest.json"
        if manifest_path.exists():
            paths.append(manifest_path)
    else:
        paths = [p for p in output.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in paths)


def _capture_environment_info() -> dict[str, Any]:
    """Best-effort CPU/RAM/GPU/software disclosure for the benchmark doc."""

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "mem_total_kb": _read_mem_total_kb(),
        "gpu_names": _read_gpu_names(),
        "torch": _read_torch_versions(),
    }


def _read_mem_total_kb() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _read_gpu_names() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return []


def _read_torch_versions() -> dict[str, str | None]:
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.__version__); print(torch.version.cuda)"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            lines = result.stdout.splitlines()
            torch_version = lines[0] if len(lines) > 0 else None
            cuda_version = lines[1] if len(lines) > 1 and lines[1] != "None" else None
            return {"torch_version": torch_version, "cuda_version": cuda_version}
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"torch_version": None, "cuda_version": None}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_result(scale: str, result: dict[str, Any]) -> None:
    path = RESULTS_DIR / f"{scale}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def _format_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def _format_kb(n: int) -> str:
    return _format_bytes(n * 1024)


def _print_scale_summary(result: dict[str, Any]) -> None:
    print(f"\n== {result['scale']} benchmark ==")
    for name, measurement in result["steps"].items():
        rss = measurement.get("peak_rss_kb")
        rss_text = f"  peak_rss={_format_kb(rss)}" if isinstance(rss, int) else ""
        print(f"  {name:8s} {measurement['elapsed_s']:8.2f}s{rss_text}")
    print(f"  items/sec: {result['items_per_second']:.2f}  (total_items={result['total_items']})")
    print(f"  raw dump size:      {_format_bytes(result['raw_dump_bytes'])}")
    print(f"  total artifact size: {_format_bytes(result['total_artifact_bytes'])}")
    if "peak_gpu_memory_used_mib" in result and result["peak_gpu_memory_used_mib"] is not None:
        print(f"  peak GPU memory (sampled): {result['peak_gpu_memory_used_mib']} MiB")


def _print_resume_summary(result: dict[str, Any]) -> None:
    print("\n== resume benchmark ==")
    print(f"  baseline (fresh, full run):        {result['baseline']['elapsed_s']:8.2f}s")
    print(
        f"  interrupted (partial, ~{result['interrupt_fraction']:.0%} target): "
        f"{result['interrupted']['elapsed_s']:8.2f}s"
    )
    print(f"  resumed (to completion):           {result['resumed']['elapsed_s']:8.2f}s")
    combined = result["interrupted"]["elapsed_s"] + result["resumed"]["elapsed_s"]
    print(f"  interrupted + resumed combined:    {combined:8.2f}s")
    print(f"  resume_count after resume: {result['resume_count']}")


if __name__ == "__main__":
    raise SystemExit(main())
