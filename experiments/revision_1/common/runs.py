"""Registry of the frozen v1.0.0 runs that revision-1 experiments read from.

``BASELINE_RUNS`` holds the eight runs whose stored outputs back the paper's
reported numbers. ``K1_PARITY_RUNS`` holds the four earlier ImageNet runs
with one control per target, kept only because the paper's accuracy and
mean ``margin_drop`` table was generated from them (Phase 0 P0-3 checks that
their target items are identical to the ``K=3`` baseline's).

The NTU and synthetic baselines read the metrics/analysis stores that Phase 0
P0-7 (``phase0/recompute_small_baselines.py``) recomputed from the unchanged
dumps with the v1.0.0 code: the stored synthetic metrics use a schema the
current code rejects, and ``ntu60_tsm_crop_free``'s run manifest was edited
after its metrics were computed. The originals are left in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_STUDY_DIR = REPO_ROOT / "experiments" / "real_dataset_case_study"
SYNTHETIC_DIR = REPO_ROOT / "experiments" / "synthetic_shortcut"
REVISION_DIR = REPO_ROOT / "experiments" / "revision_1"
REVISION_RESULTS_DIR = REVISION_DIR / "results"
RECOMPUTED_DIR = REVISION_RESULTS_DIR / "phase0" / "recomputed"

Dataset = Literal["synthetic", "imagenet", "ntu60"]
Protocol = Literal["exact", "crop_free"]

K1_CONFIG_REVISION = "486007a^"
"""Last commit whose ImageNet configs still had ``n_samples: 1`` (486007a raised it to 3)."""


@dataclass(frozen=True)
class RunRef:
    """Locate one audit run's dump, metrics store, and analysis store.

    Attributes:
        name: Stable identifier used in every revision-1 output table.
        dump: Raw dump directory (contains ``run_manifest.json``).
        metrics: Metrics store directory.
        analysis: Analysis store directory.
        dataset: Case study the run belongs to.
        model: Short model label.
        protocol: ``exact`` (official preprocessing) or ``crop_free``.
        n_controls: Matched controls per target region (0 if none).
        config: Config YAML, or the script that built the config in code.
        config_revision: Git revision whose copy of ``config`` launched the
            run, when it differs from the current tree.
    """

    name: str
    dump: Path
    metrics: Path
    analysis: Path
    dataset: Dataset
    model: str
    protocol: Protocol
    n_controls: int
    config: Path | None = None
    config_revision: str | None = None

    @classmethod
    def co_located(cls, name: str, dump: Path, **fields: object) -> "RunRef":
        """Build a ref whose metrics/analysis live under ``<dump>/metrics`` and ``<dump>/analysis``."""

        return cls(name=name, dump=dump, metrics=dump / "metrics", analysis=dump / "analysis", **fields)  # type: ignore[arg-type]


def _imagenet(width: str, protocol: Protocol, n_controls: int) -> RunRef:
    run_dir = "results_control_3" if n_controls == 3 else "results"
    stem = f"imagenet_mnv2_{width}_{protocol}"
    return RunRef.co_located(
        f"{stem}_k{n_controls}",
        CASE_STUDY_DIR / run_dir / stem,
        dataset="imagenet",
        model=f"mobilenetv2_{width}",
        protocol=protocol,
        n_controls=n_controls,
        config=CASE_STUDY_DIR / "configs" / f"{stem}.yaml",
        config_revision=None if n_controls == 3 else K1_CONFIG_REVISION,
    )


def _ntu(protocol: Protocol, *, original: bool = False) -> RunRef:
    stem = f"ntu60_tsm_{protocol}"
    dump = CASE_STUDY_DIR / "results" / stem
    stores = dump if original else RECOMPUTED_DIR / stem
    return RunRef(
        name=stem,
        dump=dump,
        metrics=stores / "metrics",
        analysis=stores / "analysis",
        dataset="ntu60",
        model="tsm_resnet50",
        protocol=protocol,
        n_controls=0,
        config=CASE_STUDY_DIR / "configs" / f"{stem}.yaml",
    )


def _synthetic(model: Literal["shortcut", "normal"], *, original: bool = False) -> RunRef:
    stem = f"{model}_A_all_ops_thresholds_crop_free"
    root = SYNTHETIC_DIR / "results_crop_free"
    name = f"synthetic_{model}"
    return RunRef(
        name=name,
        dump=root / "dumps" / stem,
        metrics=root / "metrics" / stem if original else RECOMPUTED_DIR / name / "metrics",
        analysis=root / "analysis" / stem if original else RECOMPUTED_DIR / name / "analysis",
        dataset="synthetic",
        model=f"squeezenet1_0_m_{model}",
        protocol="crop_free",
        n_controls=2,
        config=SYNTHETIC_DIR / "run_threshold_validation_full.py",
    )


_IMAGENET_VARIANTS: tuple[tuple[str, Protocol], ...] = (
    ("050", "exact"),
    ("050", "crop_free"),
    ("100", "exact"),
    ("100", "crop_free"),
)

BASELINE_RUNS: dict[str, RunRef] = {
    run.name: run
    for run in (
        *(_imagenet(width, protocol, 3) for width, protocol in _IMAGENET_VARIANTS),
        _ntu("exact"),
        _ntu("crop_free"),
        _synthetic("shortcut"),
        _synthetic("normal"),
    )
}

K1_PARITY_RUNS: dict[str, RunRef] = {
    run.name: run for run in (_imagenet(width, protocol, 1) for width, protocol in _IMAGENET_VARIANTS)
}

ORIGINAL_SMALL_RUNS: dict[str, RunRef] = {
    run.name: run
    for run in (
        _ntu("exact", original=True),
        _ntu("crop_free", original=True),
        _synthetic("shortcut", original=True),
        _synthetic("normal", original=True),
    )
}
"""The NTU/synthetic runs with their originally stored metrics/analysis (P0-7 input)."""


A2_DIR = REVISION_DIR / "a2_control_count"
A2_RESULTS_DIR = REVISION_RESULTS_DIR / "a2"


def _a2(name: str, config: str, dataset: Dataset, model: str, protocol: Protocol) -> RunRef:
    return RunRef.co_located(
        name,
        A2_RESULTS_DIR / name,
        dataset=dataset,
        model=model,
        protocol=protocol,
        n_controls=20,
        config=A2_DIR / "configs" / config,
    )


A2_RUNS: dict[str, RunRef] = {
    run.name: run
    for run in (
        _a2("a2_imagenet_mnv2_050_cf_k20", "imagenet_mnv2_050_crop_free_k20.yaml", "imagenet", "mobilenetv2_050", "crop_free"),
        _a2("a2_imagenet_mnv2_050_exact_k20", "imagenet_mnv2_050_exact_k20.yaml", "imagenet", "mobilenetv2_050", "exact"),
        _a2("a2_synthetic_shortcut_k20", "synthetic_shortcut_k20.yaml", "synthetic", "squeezenet1_0_m_shortcut", "crop_free"),
    )
}
"""A2's K=20 runs (``a2_control_count/matrix.json``)."""

A2_BASELINES: dict[str, str] = {
    "a2_imagenet_mnv2_050_cf_k20": "imagenet_mnv2_050_crop_free_k3",
    "a2_imagenet_mnv2_050_exact_k20": "imagenet_mnv2_050_exact_k3",
    "a2_synthetic_shortcut_k20": "synthetic_shortcut",
}
"""The ``BASELINE_RUNS`` entry each A2 run extends (same config apart from K and samples)."""


A3_DIR = REVISION_DIR / "a3_multi_arch"
A3_RESULTS_DIR = REVISION_RESULTS_DIR / "a3"


def _a3(model: str, protocol: Protocol) -> RunRef:
    suffix = "cf" if protocol == "crop_free" else protocol
    name = f"a3_imagenet_{model}_{suffix}"
    return RunRef.co_located(
        name,
        A3_RESULTS_DIR / name,
        dataset="imagenet",
        model=model,
        protocol=protocol,
        n_controls=3,
        config=A3_DIR / "configs" / f"imagenet_{model}_{protocol}.yaml",
    )


A3_RUNS: dict[str, RunRef] = {
    run.name: run
    for run in (
        _a3(model, protocol)
        for model in ("convnext_tiny", "deit_small")
        for protocol in ("crop_free", "exact")
    )
}
"""A3's new-architecture runs (``a3_multi_arch/matrix.json``), 10,000 samples with K=3."""


def get_run(name: str) -> RunRef:
    """Look up a run in ``BASELINE_RUNS``, ``K1_PARITY_RUNS``, ``A2_RUNS``, or ``A3_RUNS``.

    Raises:
        KeyError: If ``name`` is in none of the registries.
    """

    registries = (BASELINE_RUNS, K1_PARITY_RUNS, A2_RUNS, A3_RUNS)
    for registry in registries:
        if name in registry:
            return registry[name]
    raise KeyError(f"unknown run {name!r}; known: {sorted(set().union(*registries))}")


def k1_counterpart(run: RunRef) -> RunRef:
    """Return the ``K=1`` ImageNet run sharing ``run``'s model and protocol."""

    if run.dataset != "imagenet":
        raise ValueError(f"{run.name} is not an ImageNet run")
    return K1_PARITY_RUNS[run.name.rsplit("_k", 1)[0] + "_k1"]
