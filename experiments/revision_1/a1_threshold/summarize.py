"""A1 step 5: derived tables and figures from the sweep CSVs in ``summary/``.

Reads only ``summary/*.csv`` written by ``run_sweep.py`` and writes
``sensitivity_ranking.csv``, ``table_a1.csv``/``.tex``, and the three A1
figures.

Example:
    python experiments/revision_1/a1_threshold/summarize.py
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.revision_1.common.agreement import GRADE_ORDER  # noqa: E402
from experiments.revision_1.common.provenance import write_provenance  # noqa: E402

SUMMARY_DIR = Path(__file__).resolve().parent / "summary"
CURVE_RUNS = (
    "synthetic_shortcut",
    "synthetic_normal",
    "imagenet_mnv2_050_exact_k3",
    "imagenet_mnv2_050_crop_free_k3",
    "imagenet_mnv2_100_exact_k3",
    "imagenet_mnv2_100_crop_free_k3",
)
TRANSITION_RUNS = ("imagenet_mnv2_050_exact_k3", "synthetic_shortcut")
GRADE_COLORS = {"high": "#1b7837", "moderate": "#a6dba0", "low": "#c2a5cf", "unreliable": "#762a83"}
RUN_LABELS = {
    "synthetic_shortcut": "Synthetic M_shortcut",
    "synthetic_normal": "Synthetic M_normal",
    "imagenet_mnv2_050_exact_k3": "ImageNet MNv2-0.5 exact",
    "imagenet_mnv2_050_crop_free_k3": "ImageNet MNv2-0.5 crop-free",
    "imagenet_mnv2_100_exact_k3": "ImageNet MNv2-1.0 exact",
    "imagenet_mnv2_100_crop_free_k3": "ImageNet MNv2-1.0 crop-free",
    "ntu60_tsm_exact": "NTU60 TSM exact",
    "ntu60_tsm_crop_free": "NTU60 TSM crop-free",
}
PARAM_LABELS = {
    "z_threshold": "tau_z",
    "sign_alpha": "alpha",
    "min_multi_strategy": "m",
    "ci_level": "CI",
    "ddof": "ddof",
    "std_floor_beta": "beta",
}


def sensitivity_ranking(agreement: pd.DataFrame) -> pd.DataFrame:
    """Rank registered OAT parameters by their maximum disagreement per run."""

    oat = agreement[agreement["setting_id"].str.startswith("oat:")].copy()
    oat["disagreement"] = 1.0 - oat["agreement"]
    rows = []
    for (run, param), group in oat.groupby(["run", "param"], sort=False):
        worst = group.loc[group["disagreement"].idxmax()]
        rows.append(
            {
                "run": run,
                "dataset": worst["dataset"],
                "param": param,
                "max_disagreement": float(worst["disagreement"]),
                "at_value": worst["value"],
                "max_delta_ge2": float(group["delta_ge2"].max()),
                "min_kappa_linear": float(group["kappa_linear"].min()),
            }
        )
    ranking = pd.DataFrame(rows)
    ranking["rank"] = ranking.groupby("run")["max_disagreement"].rank(ascending=False, method="min").astype(int)
    return ranking.sort_values(["run", "rank", "param"]).reset_index(drop=True)


def preset_table(agreement: pd.DataFrame, distribution: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    """Return per-run preset agreement, weighted kappa, >=2-step changes, HIGH share, and HIGH<->UNRELIABLE flips."""

    high = distribution[distribution["grade"] == "high"].set_index(["run", "setting_id"])["share"]
    flips = transitions.set_index(["run", "setting_id", "from_grade", "to_grade"])["n"]
    totals = agreement.set_index(["run", "setting_id"])["n"]
    rows = []
    for run, group in agreement.groupby("run", sort=False):
        row: dict[str, object] = {"run": run, "dataset": group["dataset"].iloc[0], "high_default": high[(run, "default")]}
        for preset in ("lenient", "conservative"):
            entry = group[group["setting_id"] == f"preset:{preset}"].iloc[0]
            row[f"{preset}_agreement"] = entry["agreement"]
            row[f"{preset}_kappa"] = entry["kappa_linear"]
            row[f"{preset}_delta_ge2"] = entry["delta_ge2"]
            row[f"{preset}_high"] = high[(run, f"preset:{preset}")]
            total = totals[(run, f"preset:{preset}")]
            row[f"{preset}_high_to_unreliable"] = flips[(run, f"preset:{preset}", "high", "unreliable")] / total
            row[f"{preset}_unreliable_to_high"] = flips[(run, f"preset:{preset}", "unreliable", "high")] / total
        rows.append(row)
    return pd.DataFrame(rows)


def write_latex(table: pd.DataFrame, path: Path) -> None:
    """Write ``table_a1`` as a booktabs LaTeX table (percentages, kappa to 3 decimals)."""

    preset_header = r"Agr. (\%) & $\kappa_w$ & $\geq$2 (\%) & HIGH (\%)"
    lines = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r" & Default & \multicolumn{4}{c}{Lenient} & \multicolumn{4}{c}{Conservative} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        rf"Run & HIGH (\%) & {preset_header} & {preset_header} \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        cells = [RUN_LABELS.get(row["run"], row["run"]), f"{100 * row['high_default']:.2f}"]
        for preset in ("lenient", "conservative"):
            kappa = row[f"{preset}_kappa"]
            cells += [
                f"{100 * row[f'{preset}_agreement']:.2f}",
                "--" if pd.isna(kappa) else f"{kappa:.3f}",
                f"{100 * row[f'{preset}_delta_ge2']:.2f}",
                f"{100 * row[f'{preset}_high']:.2f}",
            ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def plot_z_curve(curve: pd.DataFrame, hist: pd.DataFrame, path: Path) -> None:
    """Grade share vs tau_z per run, with a max_z histogram inset."""

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True)
    for ax, run in zip(axes.flat, CURVE_RUNS):
        data = curve[curve["run"] == run].sort_values("z_threshold")
        bottom = np.zeros(len(data))
        for grade in GRADE_ORDER:
            values = data[grade].to_numpy()
            if not values.any():
                continue
            ax.fill_between(data["z_threshold"], bottom, bottom + values, color=GRADE_COLORS[grade], label=grade, linewidth=0)
            bottom += values
        ax.axvline(2.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(RUN_LABELS[run], fontsize=10)
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 1)
        inset = ax.inset_axes([0.52, 0.1, 0.44, 0.33])
        bins = hist[hist["run"] == run]
        inset.bar(bins["bin_left"], bins["count"], width=bins["bin_right"] - bins["bin_left"], align="edge", color="0.4")
        inset.axvline(2.0, color="black", linestyle="--", linewidth=0.8)
        inset.set_xlim(-3, 10)
        inset.set_yticks([])
        inset.tick_params(labelsize=6)
        inset.set_title("max z (clipped to [-3, 10])", fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("z threshold tau_z")
    for ax in axes[:, 0]:
        ax.set_ylabel("share of anchors")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def plot_oat(agreement: pd.DataFrame, path: Path) -> None:
    """Disagreement with the default (1-step and >=2-step) per setting, one panel per dataset."""

    shown = agreement[
        (agreement["setting_id"].str.startswith("oat:") | agreement["setting_id"].str.startswith("preset:"))
    ].copy()
    shown = shown[~shown["setting_id"].isin(_default_oat_ids())]
    datasets = ("synthetic", "imagenet", "ntu60")
    fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharex=True)
    for ax, dataset in zip(axes, datasets):
        data = shown[shown["dataset"] == dataset]
        settings = list(dict.fromkeys(data["setting_id"]))
        runs = list(dict.fromkeys(data["run"]))
        height = 0.8 / len(runs)
        for offset, run in enumerate(runs):
            rows = data[data["run"] == run].set_index("setting_id").reindex(settings)
            positions = np.arange(len(settings)) + offset * height - 0.4 + height / 2
            ax.barh(positions, rows["delta_1"] + rows["delta_ge2"], height=height, color=f"C{offset}", alpha=0.45)
            ax.barh(positions, rows["delta_ge2"], height=height, color=f"C{offset}", label=RUN_LABELS[run])
        ax.set_yticks(np.arange(len(settings)))
        ax.set_yticklabels([_setting_label(setting) for setting in settings], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(dataset, fontsize=10)
        ax.set_xlabel("share of anchors whose grade changes\n(dark: >=2 grade steps)")
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _default_oat_ids() -> set[str]:
    defaults = {"z_threshold": "2.0", "sign_alpha": "0.0", "min_multi_strategy": "2", "ci_level": "95", "ddof": "0", "std_floor_beta": "0.0"}
    return {f"oat:{param}={value}" for param, value in defaults.items()}


def _setting_label(setting_id: str) -> str:
    kind, _, rest = setting_id.partition(":")
    if kind == "preset":
        return f"preset {rest}"
    param, _, value = rest.partition("=")
    return f"{PARAM_LABELS.get(param, param)} = {value}"


def plot_transitions(transitions: pd.DataFrame, path: Path) -> None:
    """Default -> lenient / conservative transition heatmaps (row-normalized)."""

    fig, axes = plt.subplots(len(TRANSITION_RUNS), 2, figsize=(9, 8))
    for row_axes, run in zip(axes, TRANSITION_RUNS):
        for ax, preset in zip(row_axes, ("lenient", "conservative")):
            data = transitions[(transitions["run"] == run) & (transitions["setting_id"] == f"preset:{preset}")]
            matrix = data.pivot(index="from_grade", columns="to_grade", values="n").reindex(index=GRADE_ORDER, columns=GRADE_ORDER)
            totals = matrix.sum(axis=1).replace(0, np.nan)
            shares = matrix.div(totals, axis=0)
            ax.imshow(shares.fillna(0).to_numpy(), cmap="Blues", vmin=0, vmax=1)
            for i, from_grade in enumerate(GRADE_ORDER):
                for j, to_grade in enumerate(GRADE_ORDER):
                    count = int(matrix.loc[from_grade, to_grade])
                    share = shares.loc[from_grade, to_grade]
                    text = f"{count:,}" + ("" if pd.isna(share) else f"\n{100 * share:.1f}%")
                    ax.text(j, i, text, ha="center", va="center", fontsize=7, color="white" if (share or 0) > 0.6 else "black")
            ax.set_xticks(range(len(GRADE_ORDER)), GRADE_ORDER, fontsize=8)
            ax.set_yticks(range(len(GRADE_ORDER)), GRADE_ORDER, fontsize=8)
            ax.set_xlabel(f"{preset} preset")
            ax.set_ylabel("default")
            ax.set_title(RUN_LABELS[run], fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    args = parser.parse_args(argv)
    summary = args.summary_dir

    agreement = pd.read_csv(summary / "agreement.csv")
    distribution = pd.read_csv(summary / "grade_distribution.csv")
    ranking = sensitivity_ranking(agreement)
    ranking.to_csv(summary / "sensitivity_ranking.csv", index=False)
    transitions = pd.read_csv(summary / "transitions.csv")
    table = preset_table(agreement, distribution, transitions)
    table.to_csv(summary / "table_a1.csv", index=False)
    write_latex(table, summary / "table_a1.tex")

    plot_z_curve(pd.read_csv(summary / "z_curve.csv"), pd.read_csv(summary / "max_z_hist.csv"), summary / "fig_a1_z_curve.pdf")
    plot_oat(agreement, summary / "fig_a1_oat.pdf")
    plot_transitions(transitions, summary / "fig_a1_transitions.pdf")

    inputs = {name: summary / f"{name}.csv" for name in ("agreement", "grade_distribution", "z_curve", "max_z_hist", "transitions")}
    write_provenance(summary, inputs=inputs, filename="summarize.provenance.json")
    print(f"wrote ranking, table_a1, and figures to {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
