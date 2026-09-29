"""A4 sweep definition shared by ``make_inputs.py``, ``run_scaling.py``, and ``fit_scaling.py``.

The reference workload is the mobilenetv2_050 exact case-study config
(implementation plan section 7.1). Each axis varies one factor around the
center point N=1,000 (samples axis) or N=500 (other axes), 4x4 grid, V=5
operator-seed variants, K=3 controls. Per-sample item count is
``1 + R * V * (1 + K)`` (clean, targets, controls).
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

from experiments.revision_1.common.runs import CASE_STUDY_DIR, REPO_ROOT, REVISION_RESULTS_DIR

A4_DIR = Path(__file__).resolve().parent
A4_RESULTS_DIR = REVISION_RESULTS_DIR / "a4"
BASE_CONFIG = CASE_STUDY_DIR / "configs" / "imagenet_mnv2_050_exact.yaml"
SOURCE_ANNOTATION = REPO_ROOT / "data" / "phase3" / "imagenet" / "val_10_per_class.txt"
SUBSET_DIR = REPO_ROOT / "data" / "revision_1" / "imagenet"
SUBSET_SALT = "rev1-a4"
WARMUP_N = 50
PROFILE_N = 200
SUBSET_SIZES = (WARMUP_N, PROFILE_N, 250, 500, 1000, 2000, 4000)

OPERATORS: tuple[dict[str, object], ...] = (
    {"op": "mean_fill", "params": {}, "seed_salts": [0]},
    {"op": "blur", "params": {"sigma": 3.0}, "seed_salts": [0]},
    {"op": "gaussian_noise", "params": {"sigma": 12.5}, "seed_salts": [0, 1, 2]},
    {"op": "constant_fill", "params": {"value": 0.0}, "seed_salts": [0]},
    {"op": "patch_shuffle", "params": {"patch_size": 4}, "seed_salts": [0]},
)
"""The case-study operators, then constant_fill and patch_shuffle with the synthetic study's ``FILL_PARAMS``."""
V_OPERATORS = {1: 1, 2: 2, 5: 3, 7: 5}
"""Variants per region -> number of leading ``OPERATORS`` entries."""


@dataclass(frozen=True)
class Setting:
    """One workload: samples, grid side, variants per region, controls per target."""

    n: int
    grid: int
    v: int
    k: int

    @property
    def regions(self) -> int:
        return self.grid * self.grid

    @property
    def planned_items(self) -> int:
        """Clean plus perturbed items: ``N * (1 + R * V * (1 + K))``."""

        return self.n * (1 + self.regions * self.v * (1 + self.k))

    @property
    def key(self) -> str:
        return f"n{self.n}_g{self.grid}_v{self.v}_k{self.k}"


WARMUP = Setting(n=WARMUP_N, grid=4, v=5, k=3)
PROFILE = Setting(n=PROFILE_N, grid=4, v=5, k=3)
AXES: dict[str, tuple[Setting, ...]] = {
    "samples": tuple(Setting(n, 4, 5, 3) for n in (250, 500, 1000, 2000, 4000)),
    "regions": tuple(Setting(1000, g, 5, 3) for g in (2, 4, 6, 8)),
    "controls": tuple(Setting(500, 4, 5, k) for k in (0, 1, 3, 10, 20)),
    "perturbations": tuple(Setting(500, 4, v, 3) for v in (1, 2, 5, 7)),
}
AXIS_LEVEL = {"samples": "n", "regions": "regions", "controls": "k", "perturbations": "v"}
SINGLE_REPEAT_MIN_N = 2000
"""Settings with at least this many samples run once; all others run ``repeats`` times."""


def level_of(axis: str, setting: Setting) -> int:
    """Return the value of the axis' varied factor for ``setting``."""

    return int(getattr(setting, AXIS_LEVEL[axis]))


def subset_path(n: int) -> Path:
    return SUBSET_DIR / f"a4_N{n}.txt"


def owner_axis(setting: Setting) -> str:
    """Return the first axis (in ``AXES`` order) that lists ``setting``; shared settings run only there."""

    return next(axis for axis, settings in AXES.items() if setting in settings)


def n_repeats(setting: Setting, repeats: int) -> int:
    return 1 if setting.n >= SINGLE_REPEAT_MIN_N else repeats


def schedule(axis: str, repeats: int) -> Iterator[tuple[Setting, int]]:
    """Yield ``(setting, repeat)`` for one axis, round-robin with the level order rotated by one per repeat.

    Settings owned by an earlier axis are skipped.
    """

    owned = [setting for setting in AXES[axis] if owner_axis(setting) == axis]
    for repeat in range(repeats):
        shift = repeat % len(owned)
        for setting in owned[shift:] + owned[:shift]:
            if repeat < n_repeats(setting, repeats):
                yield setting, repeat


def build_config(setting: Setting) -> dict[str, object]:
    """Return the audit config of ``setting``: the base config with samples, grid, operators, and controls replaced."""

    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["source"]["root"] = str((BASE_CONFIG.parent / config["source"]["root"]).resolve())
    config["source"]["annotation_file"] = str(subset_path(setting.n))
    region_id = f"grid_{setting.grid}x{setting.grid}"
    config["regions"] = [{"region_id": region_id, "kind": "grid", "params": {"rows": setting.grid, "cols": setting.grid}}]
    config["perturbations"] = [copy.deepcopy(op) for op in OPERATORS[: V_OPERATORS[setting.v]]]
    config["controls"] = [] if setting.k == 0 else [{"match_area_of": region_id, "n_samples": setting.k}]
    variants = sum(len(op["seed_salts"]) for op in config["perturbations"])  # type: ignore[arg-type]
    if variants != setting.v:
        raise ValueError(f"{setting.key}: operators give {variants} variants")
    return config


def write_config(setting: Setting, directory: Path) -> Path:
    """Write ``setting``'s config as YAML under ``directory`` and return the path."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{setting.key}.yaml"
    path.write_text(yaml.safe_dump(build_config(setting), sort_keys=False), encoding="utf-8")
    return path
