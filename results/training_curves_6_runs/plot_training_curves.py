from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = [
    ("Baseline RESIDE hiện tại", "runs/dehazeformer_t"),
    (
        "Baseline hỗn hợp",
        "runs/dehazeformer_t_reside_nhhaze2_hdnhhaze_bs8_tiles512_w0_clampfix",
    ),
    ("Fusion SCAF", "runs/dehazeformer_t_m1_full50"),
    ("SCAF + bản đồ sương", "runs/dehazeformer_t_m2_full50"),
    ("SCAF + sương + Haar", "runs/dehazeformer_t_m3_full50"),
    ("Loss tán xạ + MCBM", "runs/dehazeformer_t_m6_mcbm_full50"),
]

NAVY = "#123B62"
BLUE = "#2878B5"
ORANGE = "#E7862A"
GRID = "#DCE3EA"
TEXT = "#263746"
EPOCH_LIMIT = 50


def plot_metric(ax, epochs, values, *, color, label, linestyle="-"):
    """Draw a lightly smoothed trend while keeping sampled epoch markers visible."""
    x = np.asarray(epochs, dtype=float)
    y = np.asarray(values, dtype=float)
    if len(x) >= 3:
        trend = gaussian_filter1d(y, sigma=1.15, mode="nearest")
        smooth_x = np.linspace(x.min(), x.max(), max(250, len(x) * 8))
        smooth_y = PchipInterpolator(x, trend)(smooth_x)
    else:
        smooth_x, smooth_y = x, y
    ax.plot(
        smooth_x,
        smooth_y,
        color=color,
        linewidth=1.7,
        linestyle=linestyle,
        label=label,
        solid_capstyle="round",
        dash_capstyle="round",
        antialiased=True,
    )
    # Small markers keep the actual logged measurements visible on the curve.
    ax.plot(
        x,
        y,
        linestyle="None",
        marker="o",
        markersize=1.45,
        markevery=5,
        color=color,
        alpha=0.45,
        label="_nolegend_",
        zorder=3,
    )


def load_runs():
    loaded = []
    for label, relative in RUNS:
        path = ROOT / relative / "metrics.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"No metric rows found: {path}")
        parsed = [{key: value for key, value in row.items()} for row in rows]
        for row in parsed:
            for key, value in tuple(row.items()):
                try:
                    row[key] = float(value) if value not in (None, "") else np.nan
                except ValueError:
                    pass
        best = max(parsed, key=lambda row: row["val_psnr"])
        loaded.append(
            {
                "label": label,
                "path": relative,
                "rows": parsed,
                "best_epoch": int(best["epoch"]),
                "best_val_psnr": best["val_psnr"],
            }
        )
    return loaded


def format_axes(ax, epochs):
    ax.set_xlim(0, EPOCH_LIMIT)
    ax.set_xticks(np.arange(0, EPOCH_LIMIT + 1, 10))
    ax.grid(True, color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#B8C4CF")
        spine.set_linewidth(0.7)


def add_header(fig, title, subtitle):
    header = fig.add_axes([0, 0.93, 1, 0.07])
    header.set_facecolor(NAVY)
    header.set_xticks([])
    header.set_yticks([])
    for spine in header.spines.values():
        spine.set_visible(False)
    header.text(
        0.025,
        0.5,
        title,
        color="white",
        fontsize=17,
        weight="medium",
        va="center",
        ha="left",
    )
    if subtitle:
        fig.text(0.5, 0.91, subtitle, ha="center", va="center", fontsize=9, color=TEXT)


def save_both(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_l1(loaded):
    fig, axes = plt.subplots(3, 2, figsize=(6.2, 3.55), sharex=False)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.98, bottom=0.17, hspace=0.42, wspace=0.22)
    for index, (ax, item) in enumerate(zip(axes.flat, loaded)):
        rows = item["rows"]
        epoch = [row["epoch"] for row in rows]
        train = [row["train_l1"] for row in rows]
        val = [row["val_l1"] for row in rows]
        plot_metric(ax, epoch, train, color=BLUE, label="Train L1")
        plot_metric(ax, epoch, val, color=ORANGE, label="Validation MAE", linestyle="--")
        best = item["best_epoch"]
        best_row = next(row for row in rows if int(row["epoch"]) == best)
        ax.axvline(best, color="#657786", linewidth=0.9, linestyle=":", alpha=0.8)
        ax.scatter([best], [best_row["val_l1"]], marker="D", s=21, color=ORANGE, edgecolor="white", linewidth=0.45, zorder=4)
        ax.set_title(item["label"], fontsize=9.1, color=NAVY, loc="left", pad=4, weight="semibold")
        ax.set_ylabel("L1 / MAE", fontsize=8, color=TEXT)
        ax.set_xlabel("Epoch" if index >= 4 else "", fontsize=8, color=TEXT)
        ax.tick_params(axis="x", labelbottom=index >= 4)
        format_axes(ax, epoch)
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color=BLUE, linewidth=1.7, label="Train L1"),
            plt.Line2D([0], [0], color=ORANGE, linewidth=1.7, linestyle="--", label="Validation MAE"),
            plt.Line2D([0], [0], color="#657786", linewidth=0.9, linestyle=":", label="Best validation PSNR epoch"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3, frameon=False,
        fontsize=7.2, handlelength=1.8, columnspacing=1.1,
    )
    save_both(fig, "train_l1_validation_mae_6_runs")


def plot_ssim(loaded):
    fig, axes = plt.subplots(3, 2, figsize=(6.2, 3.55), sharex=False)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.98, bottom=0.17, hspace=0.42, wspace=0.22)
    for index, (ax, item) in enumerate(zip(axes.flat, loaded)):
        rows = item["rows"]
        epoch = [row["epoch"] for row in rows]
        train_ssim = [row["train_ssim"] for row in rows]
        ssim = [row["val_ssim"] for row in rows]
        plot_metric(ax, epoch, train_ssim, color=BLUE, label="Train SSIM")
        plot_metric(ax, epoch, ssim, color=ORANGE, label="Validation SSIM", linestyle="--")
        best = item["best_epoch"]
        best_row = next(row for row in rows if int(row["epoch"]) == best)
        ax.axvline(best, color="#657786", linewidth=0.9, linestyle=":", alpha=0.8)
        ax.scatter([best], [best_row["val_ssim"]], marker="D", s=28, color=ORANGE, edgecolor="white", linewidth=0.55, zorder=4, label="Best val PSNR")
        ax.set_title(item["label"], fontsize=9.1, color=NAVY, loc="left", pad=4, weight="semibold")
        ax.set_ylabel("SSIM", fontsize=8, color=TEXT)
        ax.set_xlabel("Epoch" if index >= 4 else "", fontsize=8, color=TEXT)
        ax.tick_params(axis="x", labelbottom=index >= 4)
        ax.set_ylim(0.50, 1.00)
        ax.set_yticks(np.arange(0.50, 1.001, 0.10))
        format_axes(ax, epoch)
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color=BLUE, linewidth=1.7, label="Train SSIM"),
            plt.Line2D([0], [0], color=ORANGE, linewidth=1.7, linestyle="--", label="Validation SSIM"),
            plt.Line2D([0], [0], color="#657786", linewidth=0.9, linestyle=":", label="Best validation PSNR epoch"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3, frameon=False,
        fontsize=7.2, handlelength=1.8, columnspacing=1.1,
    )
    save_both(fig, "train_validation_ssim_6_runs")


def export_source(loaded):
    columns = ["run", "run_dir", "epoch", "train_l1", "train_total_loss", "val_l1", "val_ssim", "val_psnr", "learning_rate", "best_val_psnr_epoch"]
    with (OUT / "metrics_source.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for item in loaded:
            for row in item["rows"]:
                writer.writerow({"run": item["label"], "run_dir": item["path"], **row, "best_val_psnr_epoch": item["best_epoch"]})


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    loaded = load_runs()
    plot_l1(loaded)
    plot_ssim(loaded)
    export_source(loaded)
    print(f"Saved charts and source CSV to {OUT}")
    for item in loaded:
        print(f"{item['label']}: {len(item['rows'])} epochs; best validation PSNR at epoch {item['best_epoch']}")


if __name__ == "__main__":
    main()
