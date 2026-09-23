"""B1 figures: effective-area distributions, ranking agreement, and representative samples.

``fig_b1_area.pdf`` and ``fig_b1_rank.pdf`` read only ``summary/*.csv``.
``fig_b1_examples.pdf`` additionally reads the three pre-registered
representative samples' images, per-cell ``margin_drop``, and effective areas
from the mnv2_050 runs, and is written to ``results/b1/`` (untracked).

Example:
    python experiments/revision_1/b1_preprocessing/plot_figures.py
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.revision_1.a1_threshold.build_features import cell_columns  # noqa: E402
from experiments.revision_1.common.loading import iter_latest_perturbed, load_spatial_profile  # noqa: E402
from experiments.revision_1.common.provenance import write_provenance  # noqa: E402
from experiments.revision_1.common.runs import BASELINE_RUNS, REPO_ROOT, REVISION_RESULTS_DIR  # noqa: E402
from ssat.core.dump._storage import fragment_files  # noqa: E402
from ssat.utils.io import sha256_file  # noqa: E402

SUMMARY_DIR = Path(__file__).resolve().parent / "summary"
EXAMPLES_PATH = REVISION_RESULTS_DIR / "b1" / "fig_b1_examples.pdf"
"""Kept out of git: the figure embeds ImageNet photographs, which the ImageNet terms do not let us redistribute."""
IMAGE_ROOT = REPO_ROOT / "data" / "imagenet" / "ILSVRC" / "Data" / "CLS-LOC" / "val"
EXAMPLE_RUNS = {"exact": "imagenet_mnv2_050_exact_k3", "crop_free": "imagenet_mnv2_050_crop_free_k3"}
PROTOCOL_COLORS = {"exact": "#d95f02", "crop_free": "#1b9e77"}
CROP_PCT = 0.875
INPUT_SIZE = 224


def plot_area(cell_area: pd.DataFrame, within: pd.DataFrame, path: Path) -> None:
    """Per-cell effective fraction (p5-p95 whiskers, IQR box) for both protocols."""

    data = cell_area[cell_area["model"] == "mobilenetv2_050"]
    cells = sorted(data["cell"].unique())
    fig, (ax, ax_ratio) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [3.2, 1]})
    for offset, protocol in ((-0.18, "exact"), (0.18, "crop_free")):
        rows = data[data["protocol"] == protocol].set_index("cell").loc[cells]
        stats = [
            {
                "med": row["fraction_p50"],
                "q1": row["fraction_p25"],
                "q3": row["fraction_p75"],
                "whislo": row["fraction_p5"],
                "whishi": row["fraction_p95"],
                "mean": row["fraction_mean"],
                "fliers": [],
            }
            for _, row in rows.iterrows()
        ]
        boxes = ax.bxp(
            stats,
            positions=np.arange(len(cells)) + offset,
            widths=0.3,
            showmeans=True,
            patch_artist=True,
            manage_ticks=False,
        )
        for box in boxes["boxes"]:
            box.set_facecolor(PROTOCOL_COLORS[protocol])
            box.set_alpha(0.6)
        ax.plot([], [], color=PROTOCOL_COLORS[protocol], linewidth=6, alpha=0.6, label=protocol.replace("_", "-"))
    ax.axhline(1 / 16, color="black", linestyle="--", linewidth=1, label="nominal 1/16")
    ax.set_xticks(np.arange(len(cells)), cells, rotation=45, fontsize=8)
    ax.set_ylabel("effective fraction of 224x224 input")
    ax.set_title("Target cell effective area (10,000 samples; box IQR, whiskers p5-p95, triangle mean)", fontsize=9)
    ax.legend(fontsize=8)

    exact = within[(within["model"] == "mobilenetv2_050") & (within["protocol"] == "exact")].iloc[0]
    labels = ["p5", "p25", "p50", "p75", "p95"]
    ax_ratio.bar(labels, [exact[f"ratio_{label}"] for label in labels], color=PROTOCOL_COLORS["exact"], alpha=0.7)
    ax_ratio.axhline(exact["analytic_square_exact"], color="black", linestyle="--", linewidth=1)
    ax_ratio.text(0, exact["analytic_square_exact"] * 1.05, "square image 1.78", fontsize=7)
    ax_ratio.set_ylabel("largest / smallest cell area in a sample")
    ax_ratio.set_title(
        f"P_exact ({int(exact['n_samples_with_zero_area_cell'])} samples with a\ncell fully outside the crop excluded)",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_rank(hist: pd.DataFrame, by_cell_type: pd.DataFrame, path: Path) -> None:
    """Per-sample Spearman histograms (protocol vs model pairs) and top-cell share by cell type."""

    fig, (ax_hist, ax_type) = plt.subplots(1, 2, figsize=(12, 4.2))
    data = hist[hist["population"] == "all"]
    for (run_a, run_b), group in data.groupby(["run_a", "run_b"], sort=False):
        pair_type = group["pair_type"].iloc[0]
        label = f"{_short(run_a)} vs {_short(run_b)}"
        centers = (group["bin_left"] + group["bin_right"]) / 2
        shares = group["count"] / group["count"].sum()
        ax_hist.plot(centers, shares, linestyle="-" if pair_type == "protocol" else ":", marker="o", markersize=2, label=label)
    ax_hist.set_xlabel("per-sample Spearman over the 16 cells")
    ax_hist.set_ylabel("share of samples")
    ax_hist.set_title("Within-image ranking agreement (solid: protocol pairs, dotted: model pairs)", fontsize=9)
    ax_hist.legend(fontsize=7)

    rows = by_cell_type[(by_cell_type["population"] == "all") & (by_cell_type["pair_type"] == "protocol")]
    types = ["corner", "edge", "center"]
    width = 0.2
    for index, (run_a, group) in enumerate(rows.groupby("run_a", sort=False)):
        group = group.set_index("cell_type").loc[types]
        positions = np.arange(len(types)) + (index - 0.5) * 2 * width
        ax_type.bar(positions - width / 2, group["top_share_a"], width, color=PROTOCOL_COLORS["exact"], alpha=0.5 + 0.4 * index, label=f"{_model(run_a)} exact")
        ax_type.bar(positions + width / 2, group["top_share_b"], width, color=PROTOCOL_COLORS["crop_free"], alpha=0.5 + 0.4 * index, label=f"{_model(run_a)} crop-free")
    ax_type.set_xticks(np.arange(len(types)), types)
    ax_type.set_ylabel("share of samples whose top cell is of this type")
    ax_type.set_title("Top-1 cell type (4 corner, 8 edge, 4 center cells)", fontsize=9)
    ax_type.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _short(run_name: str) -> str:
    return run_name.replace("imagenet_mnv2_", "").replace("_k3", "").replace("_", "-")


def _model(run_name: str) -> str:
    return "MNv2-0.5" if "_050_" in run_name else "MNv2-1.0"


def _sample_cells(run_name: str, sample_ids: Sequence[str]) -> pd.DataFrame:
    """margin_drop and effective area per (sample, cell) for a few samples."""

    run = BASELINE_RUNS[run_name]
    profile = load_spatial_profile(run)
    profile = profile[profile["sample_id"].isin(sample_ids)][["sample_id", "region_key", "degradation"]]
    selection = pc.field("sample_id").isin(list(sample_ids)) & (pc.field("is_control") == False)  # noqa: E712
    columns = ["sample_id", "region_id", "region_instance_id", "effective_area_px", "is_control"]
    items = pa.concat_tables(iter_latest_perturbed(run.dump, columns, filter_expression=selection)).to_pandas()
    items["region_key"] = items["region_id"] + "::" + items["region_instance_id"]
    areas = items.groupby(["sample_id", "region_key"])["effective_area_px"].first().rename("area").reset_index()
    frame = profile.merge(areas, on=["sample_id", "region_key"])
    cells = cell_columns(frame["region_key"])["region_cell"].str.extract(r"r(\d+)/c(\d+)").astype(int)
    frame["row"], frame["col"] = cells[0].to_numpy(), cells[1].to_numpy()
    return frame


def _grid_overlay(ax, image: Image.Image, cells: pd.DataFrame, vmax: float, *, crop_box: bool) -> None:
    width, height = image.size
    ax.imshow(image)
    grid = np.full((4, 4), np.nan)
    for _, row in cells.iterrows():
        grid[row["row"], row["col"]] = row["degradation"]
    ax.imshow(grid, cmap="magma", alpha=0.55, vmin=0, vmax=vmax, extent=(0, width, height, 0), interpolation="nearest")
    for index in range(1, 4):
        ax.axhline(height * index / 4, color="white", linewidth=0.5)
        ax.axvline(width * index / 4, color="white", linewidth=0.5)
    for _, row in cells.iterrows():
        ax.text(
            width * (row["col"] + 0.5) / 4,
            height * (row["row"] + 0.5) / 4,
            f"{row['degradation']:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.4, "pad": 1, "linewidth": 0},
        )
    if crop_box:
        scale = INPUT_SIZE / CROP_PCT / min(width, height)
        side = INPUT_SIZE / scale
        ax.add_patch(
            patches.Rectangle(((width - side) / 2, (height - side) / 2), side, side, fill=False, edgecolor="cyan", linewidth=1.5)
        )
    ax.set_xticks([])
    ax.set_yticks([])


def verify_images(image_root: Path, sample_ids: Sequence[str]) -> None:
    """Check each image file against the ``content_hash`` recorded in the dump's clean rows.

    Raises:
        ValueError: If an image differs from the one the audit read.
    """

    for run_name in EXAMPLE_RUNS.values():
        tables = [
            pq.read_table(path, columns=["sample_id", "content_hash"])
            for _, path in fragment_files(BASELINE_RUNS[run_name].dump / "clean", "part")
        ]
        recorded = pa.concat_tables(tables).to_pandas().drop_duplicates("sample_id", keep="last").set_index("sample_id")["content_hash"]
        for sample_id in sample_ids:
            if sha256_file(image_root / sample_id) != recorded[sample_id]:
                raise ValueError(f"{image_root / sample_id} differs from the image {run_name} audited")


def plot_examples(cases: pd.DataFrame, path: Path, image_root: Path) -> None:
    """One row per representative sample: image, exact heatmap, crop-free heatmap, areas."""

    sample_ids = list(cases["sample_id"])
    verify_images(image_root, sample_ids)
    frames = {protocol: _sample_cells(run, sample_ids) for protocol, run in EXAMPLE_RUNS.items()}
    fig, axes = plt.subplots(len(sample_ids), 4, figsize=(14, 3.4 * len(sample_ids)))
    for row_axes, (_, case) in zip(axes, cases.iterrows()):
        sample_id = case["sample_id"]
        image = Image.open(image_root / sample_id).convert("RGB")
        exact = frames["exact"][frames["exact"]["sample_id"] == sample_id]
        crop_free = frames["crop_free"][frames["crop_free"]["sample_id"] == sample_id]
        vmax = max(float(exact["degradation"].max()), float(crop_free["degradation"].max()), 1e-6)
        row_axes[0].imshow(image)
        row_axes[0].set_title(f"{sample_id}\nper-sample Spearman {case['rho']:.2f} (p{case['quantile']})", fontsize=8)
        row_axes[0].set_xticks([])
        row_axes[0].set_yticks([])
        _grid_overlay(row_axes[1], image, exact, vmax, crop_box=True)
        row_axes[1].set_title("margin_drop, exact (cyan: center crop)", fontsize=8)
        _grid_overlay(row_axes[2], image, crop_free, vmax, crop_box=False)
        row_axes[2].set_title("margin_drop, crop-free", fontsize=8)
        table = np.full((4, 4), "", dtype=object)
        for (_, e), (_, c) in zip(exact.sort_values(["row", "col"]).iterrows(), crop_free.sort_values(["row", "col"]).iterrows()):
            table[e["row"], e["col"]] = f"{e['area']}\n{c['area']}"
        row_axes[3].axis("off")
        shown = row_axes[3].table(cellText=table.tolist(), loc="center", cellLoc="center")
        shown.scale(1, 2.4)
        shown.set_fontsize(7)
        row_axes[3].set_title("effective area px (top: exact, bottom: crop-free)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    parser.add_argument("--skip-examples", action="store_true")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=IMAGE_ROOT,
        help="directory holding the ImageNet val JPEGs (checked against the dump's content_hash)",
    )
    args = parser.parse_args(argv)
    summary = args.summary_dir

    plot_area(pd.read_csv(summary / "cell_area.csv"), pd.read_csv(summary / "within_sample_area_ratio.csv"), summary / "fig_b1_area.pdf")
    plot_rank(pd.read_csv(summary / "sample_spearman_hist.csv"), pd.read_csv(summary / "by_cell_type.csv"), summary / "fig_b1_rank.pdf")
    inputs = {name: summary / f"{name}.csv" for name in ("cell_area", "within_sample_area_ratio", "sample_spearman_hist", "by_cell_type")}
    if not args.skip_examples:
        cases = pd.read_csv(summary / "representative_cases.csv")
        plot_examples(cases, EXAMPLES_PATH, args.image_root)
        inputs["representative_cases"] = summary / "representative_cases.csv"
        inputs.update({f"image:{sample_id}": args.image_root / sample_id for sample_id in cases["sample_id"]})
    write_provenance(summary, inputs=inputs, filename="plot_figures.provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
