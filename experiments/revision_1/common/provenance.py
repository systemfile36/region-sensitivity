"""Record code, input, and environment provenance for revision-1 outputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ssat.utils.io import sha256_file, write_json_atomic

REPO_ROOT = Path(__file__).resolve().parents[3]

_MANIFEST_NAMES = (
    "run_manifest.json",
    "metrics_manifest.json",
    "analysis_manifest.json",
    "report_manifest.json",
)
_PACKAGES = ("torch", "torchvision", "timm", "numpy", "pandas", "pyarrow", "decord", "captum")


def _git(*args: str) -> str | None:
    # The container runs as root over a bind mount owned by the host user,
    # so git needs an explicit safe.directory to read the repository.
    try:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={REPO_ROOT}", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def git_state() -> dict[str, Any]:
    """Return HEAD SHA, branch, ``git describe``, and whether tracked files are dirty."""

    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "sha": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git("describe", "--tags", "--always", "--dirty"),
        "dirty": None if status is None else bool(status),
        "dirty_files": [] if not status else status.splitlines(),
    }


def git_blob_sha256(revision: str, path: Path) -> str | None:
    """Return the SHA-256 of ``path`` as committed at ``revision``, or None if absent."""

    relative = path.resolve().relative_to(REPO_ROOT).as_posix()
    try:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={REPO_ROOT}", "-C", str(REPO_ROOT), "show",
             f"{revision}:{relative}"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(completed.stdout).hexdigest()


def _nvidia_smi() -> dict[str, Any] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    query = "--query-gpu=index,name,driver_version,memory.total,memory.used"
    apps = "--query-compute-apps=pid,process_name,used_memory"
    try:
        gpus = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader"], check=True, capture_output=True, text=True
        ).stdout.strip()
        processes = subprocess.run(
            ["nvidia-smi", apps, "--format=csv,noheader"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"gpus": gpus.splitlines(), "compute_processes": processes.splitlines()}


def capture_environment() -> dict[str, Any]:
    """Describe the interpreter, packages, hardware, and container of this process.

    ``SSAT_CONTAINER_IMAGE_ID`` is recorded when the caller exports it from
    the host (``docker inspect``), since the image digest is not visible from
    inside the container.
    """

    packages = {}
    for name in _PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    torch_info: dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "device_names": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        pass
    try:
        ssat_installed_version = importlib.metadata.version("ssat")
    except importlib.metadata.PackageNotFoundError:
        ssat_installed_version = None
    mem_total_kb = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "mem_total_kb": mem_total_kb,
        "packages": packages,
        "ssat_installed_metadata_version": ssat_installed_version,
        "torch": torch_info,
        "nvidia_smi": _nvidia_smi(),
        "container_image_id": os.environ.get("SSAT_CONTAINER_IMAGE_ID"),
    }


def write_json_shared(path: Path, payload: Any) -> None:
    """Write JSON atomically and make it world-readable.

    ``write_json_atomic`` creates files with mode 0600; scripts run as root
    in the container, so summaries meant to be committed from the host
    would otherwise be unreadable there.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload)
    path.chmod(0o644)


def describe_input(path: Path) -> dict[str, Any]:
    """Fingerprint one input: a file's SHA-256, or a run directory's manifest hashes."""

    path = Path(path)
    record: dict[str, Any] = {"path": _display_path(path), "exists": path.exists()}
    if path.is_file():
        record["sha256"] = sha256_file(path)
    elif path.is_dir():
        record["manifest_sha256"] = {
            name: sha256_file(path / name) for name in _MANIFEST_NAMES if (path / name).is_file()
        }
    return record


def write_provenance(
    out_dir: Path,
    *,
    inputs: Mapping[str, Path],
    extra: Mapping[str, Any] | None = None,
    filename: str = "provenance.json",
) -> Path:
    """Write ``<out_dir>/provenance.json`` describing code, inputs, and environment.

    Args:
        out_dir: Directory receiving the file (created if missing).
        inputs: Named input files or run directories to fingerprint.
        extra: Additional JSON-serializable facts (parameters, seeds, notes).
        filename: Output file name.

    Returns:
        The path written.
    """

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "git": git_state(),
        "inputs": {name: describe_input(path) for name, path in inputs.items()},
        "environment": capture_environment(),
        "extra": dict(extra or {}),
    }
    target = out_dir / filename
    write_json_shared(target, payload)
    return target


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)
