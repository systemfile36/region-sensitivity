"""End-to-end raw-cache to report smoke test for the Captum reference."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.reference_comparison.captum_baseline.analysis import analyze, render_report
from experiments.reference_comparison.captum_baseline.workflow import RAW_COLUMNS, item_key, region_key


def _manifest(path: Path, sample_id: str) -> None:
    path.write_text(
        json.dumps({"samples": [{"sample_id": sample_id, "path": "unused.png", "gt_label": 0}]}),
        encoding="utf-8",
    )


def _config(tmp_path: Path) -> dict:
    manifests = {}
    for name in ("A_audit", "B_audit", "A_test", "C_test"):
        path = tmp_path / f"{name}.json"
        _manifest(path, f"{name}-sample")
        manifests[name] = str(path)
    repo = Path(__file__).resolve().parents[2]
    return {
        "repo_root": str(repo),
        "preprocessing": {"input_size": [32, 32], "output_size": [224, 224]},
        "regions": {"rows": 2, "cols": 2, "patch_row": 0, "patch_col": 0, "controls_per_region": 1},
        "perturbations": {
            "constant_fill": {},
            "mean_fill": {},
            "blur": {},
            "gaussian_noise": {},
            "patch_shuffle": {},
        },
        "seed_salts": [0, 1, 2],
        "bootstrap_seed": 9,
        "bootstrap_resamples": 20,
        "confidence_level": 0.95,
        "data": {"manifests": manifests},
        "audit_runs": [
            {"model": "shortcut", "dataset": "A", "manifest": "A_audit", "operators": "all", "controls": True},
            {"model": "normal", "dataset": "A", "manifest": "A_audit", "operators": "all", "controls": True},
            {"model": "shortcut", "dataset": "B", "manifest": "B_audit", "operators": ["constant_fill"], "controls": False},
        ],
        "thresholds": {
            "q2_multiplier": 3.0,
            "q4_min_fill_strategies": 2,
            "q5_min_margin_points": 10.0,
            "ratio_zero_threshold": 1.0e-6,
        },
    }


def _raw_rows(config: dict) -> list[dict]:
    rows = []
    region_keys = [region_key(row, col) for row in range(2) for col in range(2)]
    for run in config["audit_runs"]:
        operators = list(config["perturbations"]) if run["operators"] == "all" else run["operators"]
        seeds = config["seed_salts"] if run["operators"] == "all" else [0]
        sample_id = f"{run['manifest']}-sample"
        for operator in operators:
            for seed in seeds:
                for index, target in enumerate(region_keys):
                    if run["model"] == "shortcut" and run["dataset"] == "A":
                        degradation = 10.0 if index == 0 else 1.0 - index * 0.05
                    elif run["model"] == "normal":
                        degradation = -1.0 if index == 0 else float(index)
                    else:
                        degradation = float(index)
                    identity = {
                        "model": run["model"],
                        "dataset": run["dataset"],
                        "sample_id": sample_id,
                        "region_key": target,
                        "target_region_key": target,
                        "is_control": False,
                        "control_index": None,
                        "perturbation": operator,
                        "seed_salt": seed,
                    }
                    rows.append(
                        {
                            "item_key": item_key(**identity),
                            **identity,
                            "gt_label": 0,
                            "clean_margin": 20.0,
                            "perturbed_margin": 20.0 - degradation,
                            "degradation": degradation,
                            "source_area": 256,
                            "model_area": 12544,
                            "status": "complete",
                        }
                    )
                    if run["controls"]:
                        control_identity = {
                            **identity,
                            "region_key": f"control:{target}:0@0,0",
                            "is_control": True,
                            "control_index": 0,
                        }
                        rows.append(
                            {
                                "item_key": item_key(**control_identity),
                                **control_identity,
                                "gt_label": 0,
                                "clean_margin": 20.0,
                                "perturbed_margin": 20.0 - degradation / 2,
                                "degradation": degradation / 2,
                                "source_area": 256,
                                "model_area": 12544,
                                "status": "complete",
                            }
                        )
    return rows


def test_cached_raw_analysis_and_report_are_reproducible(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output = tmp_path / "output"
    (output / "raw").mkdir(parents=True)
    pd.DataFrame(_raw_rows(config), columns=RAW_COLUMNS).to_parquet(
        output / "raw" / "part-000000.parquet", index=False
    )
    (output / "accuracy.json").write_text(
        json.dumps(
            {
                "shortcut": {"A": 0.99, "C": 0.10},
                "normal": {"A": 0.70, "C": 0.74},
            }
        ),
        encoding="utf-8",
    )

    first = analyze(config, output)
    second = analyze(config, output)
    report = render_report(config, output)

    assert first["canonical_hash"] == second["canonical_hash"]
    assert first["raw_rows"] == first["expected_rows"] == 244
    assert first["area_sanity"]["pass"]
    assert all(value["pass"] for key, value in first["verdicts"].items() if key.startswith("Q"))
    assert len(report["comparison"]["capabilities"]) == 22
    assert (output / "report.md").is_file()
    assert (output / "capability_comparison.csv").is_file()

