#!/usr/bin/env python3
"""A3 post-hoc table: per-operator 16-cell mean ``margin_drop`` (``deviations.md`` D-007).

Added after ``strategy_rank_corr.csv`` showed a negative operator-pair
Spearman for ``deit_small`` crop-free. For each of the eight A3 runs, averages
the target items' ``margin_drop`` per operator and cell (all 10,000 samples,
every seed) and records the operator's cell range and its Spearman with the
other operators' profiles.

Writes ``summary/operator_profile.csv``.

Example:
    python experiments/revision_1/a3_multi_arch/operator_profiles.py
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from experiments.revision_1.a3_multi_arch.compare_models import model_runs
from experiments.revision_1.common.agreement import spearman
from experiments.revision_1.common.loading import load_item_values
from experiments.revision_1.common.provenance import write_provenance

SUMMARY_DIR = Path(__file__).resolve().parent / "summary"


def operator_profile(values: pd.DataFrame) -> pd.DataFrame:
    """Return operators x cells mean ``degradation`` over target items (cells ``r<i>/c<j>``)."""

    targets = values[~values["is_control"] & values["available"]]
    cells = targets["region_instance_id"].str.extract(r"(r\d+/c\d+)$")[0]
    return targets.groupby(["perturb_op", cells])["degradation"].mean().unstack().sort_index(axis=1)


def profile_rows(profile: pd.DataFrame, base: dict[str, object]) -> list[dict[str, object]]:
    """Long rows per operator and cell, plus per-operator range and pairwise Spearman rows."""

    rows = [{**base, "kind": "cell", "perturb_op": op, "cell": cell, "value": float(value)}
            for op, series in profile.iterrows() for cell, value in series.items()]
    for op, series in profile.iterrows():
        rows.append({**base, "kind": "range", "perturb_op": op, "cell": "max_minus_min", "value": float(series.max() - series.min())})
    for op_a, op_b in itertools.combinations(profile.index, 2):
        rows.append({**base, "kind": "spearman", "perturb_op": f"{op_a}|{op_b}", "cell": "",
                     "value": spearman(profile.loc[op_a].to_numpy(), profile.loc[op_b].to_numpy())})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=SUMMARY_DIR / "operator_profile.csv")
    args = parser.parse_args(argv)
    rows = []
    runs = model_runs()
    for (model, protocol), run in runs.items():
        profile = operator_profile(load_item_values(run, ("margin_drop",)))
        rows += profile_rows(profile, {"protocol": protocol, "model": model, "run": run.name})
        print(f"{run.name}: done", flush=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    write_provenance(args.output.parent, inputs={f"{run.name}:metrics": run.metrics for run in runs.values()},
                     filename=f"{args.output.stem}.provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
